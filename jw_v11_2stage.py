import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# ============================================================
# GPU / 백엔드 (cudf: 데이터·전처리·TE, cupy: 벡터 연산)
# 환경변수 JW_V11_USE_GPU_DATA=0 / JW_V11_USE_GPU_NUMERIC=0 으로 끌 수 있음
# ============================================================
USE_GPU_DATA_PIPELINE = os.environ.get('JW_V11_USE_GPU_DATA', '1') != '0'
USE_GPU_NUMERIC = os.environ.get('JW_V11_USE_GPU_NUMERIC', '1') != '0'


def _try_import_cudf():
    try:
        import cudf  # type: ignore

        _ = cudf.Series([1, 2, 3])
        return cudf
    except Exception:
        return None


def _try_import_cupy():
    try:
        import cupy as cp  # type: ignore

        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        return cp
    except Exception:
        return None


CUDF = _try_import_cudf()
CP = _try_import_cupy() if USE_GPU_NUMERIC else None


def _is_cudf_df(obj) -> bool:
    return CUDF is not None and isinstance(obj, CUDF.DataFrame)


def _to_pandas_df(obj):
    if _is_cudf_df(obj):
        return obj.to_pandas()
    return obj


def _mae_vec(y_true, y_pred):
    """OOF/그리드용 MAE. cupy 가능 시 GPU, 아니면 CPU."""
    if CP is not None and USE_GPU_NUMERIC:
        try:
            yt = CP.asarray(np.asarray(y_true), dtype=CP.float64)
            yp = CP.asarray(np.asarray(y_pred), dtype=CP.float64)
            return float(CP.mean(CP.abs(yt - yp)))
        except Exception:
            pass
    return float(mean_absolute_error(np.asarray(y_true), np.asarray(y_pred)))


def _affine_blend(w_a, a, b):
    """w_a * a + (1 - w_a) * b. cupy 시도 후 CPU."""
    if CP is not None and USE_GPU_NUMERIC:
        try:
            ca = CP.asarray(np.asarray(a), dtype=CP.float64)
            cb = CP.asarray(np.asarray(b), dtype=CP.float64)
            return CP.asnumpy(w_a * ca + (1.0 - w_a) * cb)
        except Exception:
            pass
    return (w_a * np.asarray(a, dtype=np.float64) + (1.0 - w_a) * np.asarray(b, dtype=np.float64)).astype(np.float64)


def _clip_predictions(pred, y_train_raw):
    """하한·분위 상한 클립. cupy 시도 후 CPU."""
    if CP is not None and USE_GPU_NUMERIC:
        try:
            p = CP.asarray(np.asarray(pred), dtype=CP.float64)
            p = CP.maximum(p, CLIP_PRED_MIN)
            yref = CP.asarray(np.asarray(y_train_raw), dtype=CP.float64)
            hi = CP.percentile(yref, 100.0 * CLIP_PRED_MAX_Q)
            p = CP.minimum(p, hi)
            return CP.asnumpy(p)
        except Exception:
            pass
    pred = np.maximum(np.asarray(pred, dtype=np.float64), CLIP_PRED_MIN)
    pred_hi = float(np.percentile(np.asarray(y_train_raw), 100.0 * CLIP_PRED_MAX_Q))
    return np.minimum(pred, pred_hi).astype(np.float64)

# ============================================================
# 실행 파라미터
# ============================================================
N_FOLDS = 5
SEED = 42
USE_LOG_TARGET = False

# Stage1 LGB: fold마다 아래 시드들을 각각 학습한 뒤 검증 구간 예측을 평균 (시간 × len(S1_LGB_SEEDS))
# OOF 분산·소폭 개선에 쓰기 좋음. 예: S1_LGB_SEEDS = [42, 142, 242]
S1_LGB_SEEDS = [SEED]

# S1–S2 블렌드·expert_mix 탐색 간격 (0.02 등으로 촘촘히 → OOF 미세 개선, 루프만 증가)
BLEND_ALPHA_STEP = 0.05
EXPERT_MIX_STEP = 0.05

# Fold 3처럼 어려운 시나리오가 한 fold에 몰리는 것 완화 (sklearn>=1.1)
RUN_STRATIFIED_GROUP_CV = True
# 최근 로그: S2 LGB-only OOF가 S1보다 크게 나쁨(8.65 vs 8.57) → 다시 3모델 앙상블 허용
STAGE2_LGB_ONLY = False
# Huber는 fold별 편차가 커질 수 있어 기본은 L1; 필요 시 True
STAGE2_LGB_USE_HUBER = False
# S2 OOF가 S1보다 나쁘면 blend/expert에서 S2 벡터를 S1으로 치환(제출에 악화 경로 차단)
REPLACE_STAGE2_IF_NOT_IMPROVING = True

# 제출 CSV만 바꿔 LB A/B (OOF·로그의 FINAL OOF는 그대로).
# 예: set JW_V11_SUBMIT_PREDICTOR=base_blend
#   auto = OOF에서 고른 best_final_name 그대로 | base_blend | s1_only | expert_only | expert_mix
SUBMIT_PREDICTOR = os.environ.get('JW_V11_SUBMIT_PREDICTOR', 'auto').strip().lower()

CLIP_PRED_MIN = 0.0
CLIP_PRED_MAX_Q = 0.995

TARGET = 'avg_delay_minutes_next_30m'
ID_COLS = ['ID', 'layout_id', 'scenario_id']

PRED_LAG_COLS = [
    'pred_lag1', 'pred_lag2', 'pred_lag3',
    'pred_lag1_log', 'pred_lag_diff', 'pred_lag_roll3',
]


def elapsed(start):
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60:02d}s"


def section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _grid_01(step: float) -> np.ndarray:
    """[0,1] 구간 그리드 (끝점 1.0 포함)."""
    if step <= 0 or step > 1:
        step = 0.05
    n = max(1, int(round(1.0 / step)))
    return np.linspace(0.0, 1.0, n + 1)


def _resolve_data_dir() -> str:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        d = p / 'data'
        if d.is_dir() and (d / 'train.csv').is_file():
            return str(d)
    raise FileNotFoundError('data/train.csv not found')


def to_train_target(y):
    return np.log1p(y) if USE_LOG_TARGET else y


def from_train_pred(p):
    return np.expm1(p) if USE_LOG_TARGET else p


def mae(y_true, y_pred):
    return _mae_vec(y_true, y_pred)


def _ensemble_pred(oof_by_model: dict[str, np.ndarray], maes_by_model: dict[str, float], models: list[str], p: float) -> np.ndarray:
    w = {m: 1.0 / (maes_by_model[m] ** p) for m in models}
    ws = sum(w.values())
    out = np.zeros_like(next(iter(oof_by_model.values())))
    for m in models:
        out += w[m] * oof_by_model[m]
    return out / ws


def _powerset_models_s1():
    return [
        ['lgb'], ['xgb'], ['cat'],
        ['lgb', 'xgb'], ['lgb', 'cat'], ['xgb', 'cat'],
        ['lgb', 'xgb', 'cat'],
    ]


# ============================================================
# 전처리 함수 (Stage 1 & 2 공통; cudf/pandas 동시 지원)
# ============================================================
def handle_missing_values(df):
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    cols = [c for c in df.columns if df[c].isnull().any() and c not in ID_COLS + [TARGET]]
    if cols:
        df[cols] = df.groupby('scenario_id')[cols].ffill()
        df[cols] = df.groupby('scenario_id')[cols].bfill()
        df[cols] = df[cols].fillna(df[cols].median())
    return df


def add_basic_features(df):
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    df['timeslot'] = df.groupby('scenario_id').cumcount()
    df['robot_efficiency'] = df['robot_active'] / (df['robot_total'] + 1e-6)
    df['robot_density'] = df['robot_total'] / (df['floor_area_sqm'] + 1e-6)
    df['order_per_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1e-6)
    df['robot_per_station'] = df['robot_active'] / (df['pack_station_count'] + 1e-6)
    df['cumulative_orders'] = df.groupby('scenario_id')['order_inflow_15m'].cumsum()
    df['order_pressure'] = df['cumulative_orders'] / (df['pack_station_count'] + 1e-6)
    if 'congestion_score' in df.columns:
        df['risk_index'] = df['congestion_score'] * (1 - df['robot_efficiency'])
        df['bottle_neck'] = df['order_per_station'] * df['congestion_score']
    if 'low_battery_ratio' in df.columns:
        df['battery_risk'] = df['low_battery_ratio'] * df['robot_total']
    if 'battery_mean' in df.columns and 'battery_std' in df.columns:
        df['battery_cv'] = df['battery_std'] / (df['battery_mean'] + 1e-6)
    return df


TS_COLS = [
    'order_inflow_15m', 'robot_active', 'robot_idle', 'robot_total',
    'pack_utilization', 'congestion_score', 'avg_trip_distance',
    'low_battery_ratio', 'outbound_truck_wait_min',
    'order_per_station', 'robot_efficiency', 'order_pressure',
    'max_zone_density', 'sku_concentration',
    'battery_risk', 'battery_cv', 'risk_index', 'bottle_neck',
]


def add_timeseries_features(df):
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    for col in TS_COLS:
        if col not in df.columns:
            continue
        g = df.groupby('scenario_id')[col]
        for lag_n in (1, 2, 3, 4, 5):
            df[f'{col}_lag{lag_n}'] = g.shift(lag_n)
        df[f'{col}_diff1'] = g.shift(1) - g.shift(2)
        df[f'{col}_diff2'] = g.shift(2) - g.shift(3)
        for w in (3, 5, 10):
            df[f'{col}_roll{w}_mean'] = g.transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
            df[f'{col}_roll{w}_std'] = g.transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
            df[f'{col}_roll{w}_max'] = g.transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).max())
            df[f'{col}_roll{w}_min'] = g.transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).min())
        df[f'{col}_exp_mean'] = g.transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        df[f'{col}_ewm_mean'] = g.transform(lambda x: x.shift(1).ewm(alpha=0.3, adjust=False).mean())

    lag_cols = [c for c in df.columns if ('_lag' in c or '_diff' in c) and c not in ID_COLS]
    if lag_cols:
        df[lag_cols] = df.groupby('scenario_id')[lag_cols].ffill()
        for c in lag_cols:
            base_col = c.split('_lag')[0].split('_diff')[0]
            if base_col in df.columns:
                scen_mean = df.groupby('scenario_id')[base_col].transform('mean')
                df[c] = df[c].fillna(scen_mean)
        df[lag_cols] = df[lag_cols].fillna(df[lag_cols].median())
    return df


def add_interaction_features(df):
    if 'congestion_score' in df.columns and 'pack_utilization' in df.columns:
        df['cong_x_pack'] = df['congestion_score'] * df['pack_utilization']
    if 'congestion_score' in df.columns and 'avg_trip_distance' in df.columns:
        df['cong_x_trip'] = df['congestion_score'] * df['avg_trip_distance']
    if 'low_battery_ratio' in df.columns and 'robot_efficiency' in df.columns:
        df['lowbat_x_eff'] = df['low_battery_ratio'] * (1 - df['robot_efficiency'])
    if 'order_per_station' in df.columns and 'congestion_score' in df.columns:
        df['ops_x_cong'] = df['order_per_station'] * df['congestion_score']
    if 'order_per_station' in df.columns and 'pack_utilization' in df.columns:
        df['ops_x_pack'] = df['order_per_station'] * df['pack_utilization']
    if 'timeslot' in df.columns:
        for col in ['congestion_score', 'pack_utilization', 'order_per_station', 'low_battery_ratio']:
            if col in df.columns:
                df[f'ts_x_{col}'] = df['timeslot'] * df[col]

    scen_agg_cols = [
        'congestion_score', 'order_inflow_15m', 'battery_mean', 'pack_utilization',
        'avg_trip_distance', 'low_battery_ratio', 'max_zone_density', 'sku_concentration',
        'robot_idle', 'outbound_truck_wait_min', 'order_per_station', 'robot_efficiency',
        'order_pressure', 'risk_index', 'battery_risk', 'battery_cv',
    ]
    for col in scen_agg_cols:
        if col not in df.columns:
            continue
        stats = df.groupby('scenario_id')[col].agg(['mean', 'max', 'min', 'std']).reset_index()
        stats.columns = ['scenario_id'] + [f'{col}_scen_{f}' for f in ['mean', 'max', 'min', 'std']]
        df = df.merge(stats, on='scenario_id', how='left')

    for col in ['congestion_score', 'order_per_station', 'pack_utilization', 'avg_trip_distance']:
        sm = f'{col}_scen_mean'
        if col in df.columns and sm in df.columns:
            df[f'{col}_rel_to_scen'] = df[col] / (df[sm] + 1e-6)

    for col in ['congestion_score', 'order_per_station', 'pack_utilization']:
        if col in df.columns:
            df[f'{col}_scen_rank'] = df.groupby('scenario_id')[col].rank(pct=True)
    return df


def preprocess_all(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')
    df = handle_missing_values(df)
    df = add_basic_features(df)
    df = add_timeseries_features(df)
    df = add_interaction_features(df)
    if 'layout_type' in df.columns:
        if _is_cudf_df(df):
            codes, _ = df['layout_type'].factorize()
            df['layout_type'] = codes
        else:
            df['layout_type'] = pd.factorize(df['layout_type'])[0]
    return df


def _apply_te(df_train, df_test, col_name, group_col_series_tr, group_col_series_te, kf_te, groups_te, global_mean):
    te_col = f'{col_name}_te'
    df_train[te_col] = np.nan
    for tr_idx, val_idx in kf_te.split(df_train, df_train[TARGET], groups=groups_te):
        tr_df = df_train.iloc[tr_idx]
        stats = tr_df.groupby(group_col_series_tr.iloc[tr_idx])[TARGET].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + SMOOTHING * global_mean) / (stats['count'] + SMOOTHING)
        df_train.loc[df_train.index[val_idx], te_col] = group_col_series_tr.iloc[val_idx].map(smooth).fillna(global_mean)
    stats_full = df_train.groupby(group_col_series_tr)[TARGET].agg(['mean', 'count'])
    smooth_full = (stats_full['count'] * stats_full['mean'] + SMOOTHING * global_mean) / (stats_full['count'] + SMOOTHING)
    df_test[te_col] = group_col_series_te.map(smooth_full).fillna(global_mean)


def _run_target_encoding(train_df, test_df):
    kf_te = GroupKFold(n_splits=N_FOLDS)
    groups_te = train_df['scenario_id']
    gm = float(train_df[TARGET].mean())
    TE_COLS_LOCAL = [c for c in ['layout_id', 'timeslot', 'layout_type', 'shift_hour', 'day_of_week'] if c in train_df.columns]
    TE_PAIRS_LOCAL = []
    for a in TE_COLS_LOCAL:
        for b in TE_COLS_LOCAL:
            if a < b:
                TE_PAIRS_LOCAL.append((a, b))
    for col in TE_COLS_LOCAL:
        _apply_te(train_df, test_df, col, train_df[col], test_df[col], kf_te, groups_te, gm)
    for a, b in TE_PAIRS_LOCAL:
        pair_name = f'{a}_X_{b}'
        tr_key = train_df[a].astype(str) + '_' + train_df[b].astype(str)
        te_key = test_df[a].astype(str) + '_' + test_df[b].astype(str)
        _apply_te(train_df, test_df, pair_name, tr_key, te_key, kf_te, groups_te, gm)
    return TE_COLS_LOCAL, TE_PAIRS_LOCAL


# ============================================================
# 1) 데이터 로드  (GPU: cudf 시도 → 실패 시 pandas)
# ============================================================
path = _resolve_data_dir()
project_root = str(Path(path).resolve().parent)
print(f"▶ data dir: {path}")
print(f"▶ project root: {project_root}")

_data_gpu_ok = False
t0 = time.time()
if USE_GPU_DATA_PIPELINE and CUDF is not None:
    try:
        train = CUDF.read_csv(os.path.join(path, 'train.csv'))
        test = CUDF.read_csv(os.path.join(path, 'test.csv'))
        layout = CUDF.read_csv(os.path.join(path, 'layout_info.csv'))
        _data_gpu_ok = True
        print(f"▶ load: GPU (cudf)  train {len(train):,} / test {len(test):,}  ({elapsed(t0)})")
    except Exception as e:
        print(f"▶ load: cudf 실패 → CPU pandas ({e})")
        train = pd.read_csv(os.path.join(path, 'train.csv'))
        test = pd.read_csv(os.path.join(path, 'test.csv'))
        layout = pd.read_csv(os.path.join(path, 'layout_info.csv'))
        print(f"▶ load done ({elapsed(t0)})  train {len(train):,} / test {len(test):,}")
else:
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    test = pd.read_csv(os.path.join(path, 'test.csv'))
    layout = pd.read_csv(os.path.join(path, 'layout_info.csv'))
    print(f"▶ load: CPU (pandas)  train {len(train):,} / test {len(test):,}  ({elapsed(t0)})")


# ============================================================
# 2) 전처리  (GPU: cudf 시도 → 실패 시 CSV에서 pandas로 재시도)
# ============================================================
section('Preprocess')
t0 = time.time()
if _data_gpu_ok:
    try:
        train = preprocess_all(train, layout)
        test = preprocess_all(test, layout)
        print(f"▶ preprocess: GPU (cudf) done ({elapsed(t0)})")
    except Exception as e:
        print(f"▶ preprocess: GPU 실패 → CPU 재로드·pandas ({e})")
        layout_pd = _to_pandas_df(layout)
        train = pd.read_csv(os.path.join(path, 'train.csv'))
        test = pd.read_csv(os.path.join(path, 'test.csv'))
        layout = layout_pd if isinstance(layout_pd, pd.DataFrame) else pd.read_csv(os.path.join(path, 'layout_info.csv'))
        train = preprocess_all(train, layout)
        test = preprocess_all(test, layout)
        _data_gpu_ok = False
        print(f"▶ preprocess done (CPU) ({elapsed(t0)})")
else:
    train = preprocess_all(train, layout)
    test = preprocess_all(test, layout)
    print(f"▶ preprocess done ({elapsed(t0)})")


# ============================================================
# 3) 타겟 인코딩 (OOF)  (GPU: cudf 유지 시도 → 실패 시 pandas로 전환 후 재실행)
# ============================================================
section('Target Encoding')
t0 = time.time()
SMOOTHING = 20
TE_COLS = []
TE_PAIRS = []

if _data_gpu_ok and _is_cudf_df(train):
    try:
        TE_COLS, TE_PAIRS = _run_target_encoding(train, test)
        print(f"▶ target encoding: GPU (cudf) done ({elapsed(t0)})")
    except Exception as e:
        print(f"▶ target encoding: GPU 실패 → CPU pandas ({e})")
        train = pd.read_csv(os.path.join(path, 'train.csv'))
        test = pd.read_csv(os.path.join(path, 'test.csv'))
        layout = pd.read_csv(os.path.join(path, 'layout_info.csv'))
        train = preprocess_all(train, layout)
        test = preprocess_all(test, layout)
        TE_COLS, TE_PAIRS = _run_target_encoding(train, test)
        _data_gpu_ok = False
        print(f"▶ target encoding done (CPU) ({elapsed(t0)})")
else:
    TE_COLS, TE_PAIRS = _run_target_encoding(train, test)
    print(f"▶ target encoding done ({elapsed(t0)})")

# GBDT / sklearn CV는 pandas + CPU 고정
train = _to_pandas_df(train)
test = _to_pandas_df(test)
layout = _to_pandas_df(layout)
print("▶ Stage1/Stage2/Expert LGB·XGB·Cat: CPU (tree_method=hist / task_type=CPU)")
if CP is not None and USE_GPU_NUMERIC:
    print("▶ 벡터 구간(앙상블·블렌드·lag·테스트 후처리): GPU (cupy)")
else:
    print("▶ 벡터 구간: CPU (cupy 없음 또는 JW_V11_USE_GPU_NUMERIC=0)")

global_mean = float(train[TARGET].mean())

# ============================================================
# 4) Stage 1: Base Model (target lag 없음)
# ============================================================
feature_cols_s1 = [c for c in train.columns if c not in ID_COLS + [TARGET]]
print(f"▶ Stage 1 features: {len(feature_cols_s1)}")

y_all = to_train_target(train[TARGET].values)
y_raw = train[TARGET].values
groups = train['scenario_id'].values

# CV용 층화 라벨: 시나리오별 타깃 평균을 분위로 이산화 (같은 시나리오 행은 동일 라벨)
_scen_mean = train.groupby('scenario_id')[TARGET].transform('mean')
try:
    y_strat_cv = pd.qcut(_scen_mean, q=10, labels=False, duplicates='drop')
    y_strat_cv = pd.Series(y_strat_cv).fillna(0).astype(np.int64).values
except Exception:
    y_strat_cv = np.zeros(len(train), dtype=np.int64)

kf_y = y_strat_cv
if RUN_STRATIFIED_GROUP_CV:
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        kf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        kf_y = y_strat_cv
        print('▶ CV: StratifiedGroupKFold (scenario target deciles)')
    except Exception as e:
        kf = GroupKFold(n_splits=N_FOLDS)
        kf_y = y_all
        print(f'▶ CV: GroupKFold (StratifiedGroupKFold unavailable: {e})')
else:
    kf = GroupKFold(n_splits=N_FOLDS)
    kf_y = y_all
    print('▶ CV: GroupKFold')

lgb_params_s1 = dict(
    objective='regression_l1',
    n_estimators=25000,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=2047,
    min_child_samples=60,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.5,
    reg_alpha=0.3,
    reg_lambda=5.0,
    random_state=SEED,
    verbose=-1,
)

xgb_params_s1 = dict(
    objective='reg:absoluteerror',
    n_estimators=20000,
    learning_rate=0.015,
    max_depth=10,
    subsample=0.75,
    colsample_bytree=0.5,
    colsample_bynode=0.5,
    reg_alpha=0.3,
    reg_lambda=3.0,
    random_state=SEED,
    tree_method='hist',
    eval_metric='mae',
    early_stopping_rounds=500,
    verbosity=0,
)

cat_params_s1 = dict(
    iterations=20000,
    learning_rate=0.015,
    depth=10,
    l2_leaf_reg=5.0,
    bootstrap_type='MVS',
    subsample=0.75,
    colsample_bylevel=0.5,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=SEED,
    task_type='CPU',
    early_stopping_rounds=500,
)

section('Stage 1 - Base model (LGB + XGB + Cat + ensemble)')
if len(S1_LGB_SEEDS) > 1:
    print(f"▶ S1 LGB multi-seed (fold 내 평균): {S1_LGB_SEEDS}")
t0 = time.time()
oof_s1_lgb = np.zeros(len(train))
oof_s1_xgb = np.zeros(len(train))
oof_s1_cat = np.zeros(len(train))
models_s1_lgb, models_s1_xgb, models_s1_cat = [], [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):
    X_tr = train.iloc[tr_idx][feature_cols_s1]
    X_va = train.iloc[va_idx][feature_cols_s1]
    y_tr = y_all[tr_idx]
    y_va = y_all[va_idx]

    lgb_va_stack = []
    for rs in S1_LGB_SEEDS:
        lp = {**lgb_params_s1, 'random_state': rs}
        m_lgb = lgb.LGBMRegressor(**lp)
        m_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)],
        )
        lgb_va_stack.append(from_train_pred(m_lgb.predict(X_va)))
        models_s1_lgb.append(m_lgb)
    oof_s1_lgb[va_idx] = np.mean(lgb_va_stack, axis=0)

    try:
        m_xgb = xgb.XGBRegressor(**xgb_params_s1)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    except Exception:
        xgb_fb = dict(xgb_params_s1)
        xgb_fb['objective'] = 'reg:squarederror'
        m_xgb = xgb.XGBRegressor(**xgb_fb)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_s1_xgb[va_idx] = from_train_pred(m_xgb.predict(X_va))
    models_s1_xgb.append(m_xgb)

    m_cat = cb.CatBoostRegressor(**cat_params_s1)
    m_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        verbose=max(1, cat_params_s1['iterations'] // 10),
        use_best_model=True,
    )
    oof_s1_cat[va_idx] = from_train_pred(m_cat.predict(X_va))
    models_s1_cat.append(m_cat)

    ens = (oof_s1_lgb[va_idx] + oof_s1_xgb[va_idx] + oof_s1_cat[va_idx]) / 3
    print(f"  S1 Fold {fold} MAE (avg3): {mae(y_raw[va_idx], ens):.6f}")

mae_s1_lgb = mae(y_raw, oof_s1_lgb)
mae_s1_xgb = mae(y_raw, oof_s1_xgb)
mae_s1_cat = mae(y_raw, oof_s1_cat)
print(f"\n▶ Stage 1 OOF MAE — LGB {mae_s1_lgb:.6f} | XGB {mae_s1_xgb:.6f} | CAT {mae_s1_cat:.6f}")

s1_maes = {'lgb': mae_s1_lgb, 'xgb': mae_s1_xgb, 'cat': mae_s1_cat}
s1_oof_by = {'lgb': oof_s1_lgb, 'xgb': oof_s1_xgb, 'cat': oof_s1_cat}

best_s1_models = ['lgb', 'xgb', 'cat']
best_s1_p = 2.0
best_s1_mae = float('inf')
for models in _powerset_models_s1():
    for p in [1.0, 2.0, 3.0, 4.0]:
        pred_tmp = _ensemble_pred(s1_oof_by, s1_maes, models, p)
        m_val = mae(y_raw, pred_tmp)
        if m_val < best_s1_mae:
            best_s1_mae = float(m_val)
            best_s1_models = list(models)
            best_s1_p = float(p)

print(f"▶ Best Stage1 ensemble: models={best_s1_models}  p={best_s1_p}  OOF_MAE={best_s1_mae:.6f}")

w_s1 = {m: 1.0 / (s1_maes[m] ** best_s1_p) for m in best_s1_models}
ws_s1 = sum(w_s1.values())
oof_s1_pre = sum(w_s1[m] * s1_oof_by[m] for m in best_s1_models) / ws_s1
s1_pre_mae = mae(y_raw, oof_s1_pre)
print(f"▶ Stage 1 ensemble OOF MAE (pre-residual): {s1_pre_mae:.6f}  ({elapsed(t0)})")

# Stage 1b: residual (OOF-safe) — base 예측 오차를 추가로 학습
section('Stage 1b - Residual LGB (stack on Stage1 ensemble)')
resid_params = dict(
    objective='regression_l1',
    n_estimators=8000,
    learning_rate=0.02,
    num_leaves=127,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=SEED,
    verbose=-1,
)
oof_s1_resid = np.zeros(len(train))
models_s1_resid = []
t1b = time.time()
for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):
    X_tr = train.iloc[tr_idx][feature_cols_s1]
    X_va = train.iloc[va_idx][feature_cols_s1]
    r_tr = y_raw[tr_idx] - oof_s1_pre[tr_idx]
    r_va = y_raw[va_idx] - oof_s1_pre[va_idx]
    mr = lgb.LGBMRegressor(**resid_params)
    mr.fit(
        X_tr, r_tr,
        eval_set=[(X_va, r_va)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_s1_resid[va_idx] = mr.predict(X_va)
    models_s1_resid.append(mr)
    print(f"  S1b Fold {fold} residual MAE: {mae(r_va, oof_s1_resid[va_idx]):.6f}")

oof_s1 = oof_s1_pre + oof_s1_resid
s1_mae = mae(y_raw, oof_s1)
print(f"\n▶ Stage 1 after residual OOF MAE: {s1_mae:.6f}  ({elapsed(t1b)})")

X_test_s1 = test[feature_cols_s1]
p_lgb = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_lgb], axis=0)
p_xgb = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_xgb], axis=0)
p_cat = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_cat], axis=0)
pred_s1_pre = sum(w_s1[m] * {'lgb': p_lgb, 'xgb': p_xgb, 'cat': p_cat}[m] for m in best_s1_models) / ws_s1
pred_resid_test = np.mean([m.predict(X_test_s1) for m in models_s1_resid], axis=0)
pred_s1_test = pred_s1_pre + pred_resid_test
print(f"▶ Stage 1 test predictions ready (ensemble + residual)")


# ============================================================
# 5) Target lag 피처 생성 함수 (prediction 기반)
# ============================================================
def build_pred_lag_features(df, pred_col_name, gm):
    """pred_col_name 컬럼의 shifted 값으로 lag 피처 생성.
    df는 scenario_id, ID 순으로 정렬되어 있어야 함.
    GPU: cudf groupby/shift/rolling 시도 → 실패 시 pandas CPU."""
    if CUDF is not None and USE_GPU_NUMERIC:
        try:
            sub = df[['scenario_id', 'ID', pred_col_name]].copy()
            gdf = CUDF.from_pandas(sub)
            g = gdf.groupby('scenario_id')[pred_col_name]
            gdf['pred_lag1'] = g.shift(1).fillna(gm)
            gdf['pred_lag2'] = g.shift(2).fillna(gm)
            gdf['pred_lag3'] = g.shift(3).fillna(gm)
            gdf['pred_lag_diff'] = gdf['pred_lag1'] - gdf['pred_lag2']
            gdf['pred_lag_roll3'] = g.transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            ).fillna(gm)
            lag_pdf = gdf[['pred_lag1', 'pred_lag2', 'pred_lag3', 'pred_lag_diff', 'pred_lag_roll3']].to_pandas()
            for c in lag_pdf.columns:
                df[c] = lag_pdf[c].to_numpy(copy=False)
            pl1 = np.asarray(df['pred_lag1'], dtype=np.float64)
            df['pred_lag1_log'] = np.log1p(np.maximum(pl1, 0.0))
            return df
        except Exception:
            pass
    g = df.groupby('scenario_id')[pred_col_name]
    df['pred_lag1'] = g.shift(1).fillna(gm)
    df['pred_lag2'] = g.shift(2).fillna(gm)
    df['pred_lag3'] = g.shift(3).fillna(gm)
    df['pred_lag1_log'] = np.log1p(df['pred_lag1'].clip(lower=0))
    df['pred_lag_diff'] = df['pred_lag1'] - df['pred_lag2']
    df['pred_lag_roll3'] = g.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(gm)
    return df


# ============================================================
# 6) Stage 2: Lag-Enhanced Model
#    Train AND Val 모두 Stage 1 OOF prediction 기반 lag 사용
#    → train/val/test 분포 완전 일치 → OOF-Public 갭 최소화
# ============================================================
feature_cols_s2 = feature_cols_s1 + PRED_LAG_COLS

lgb_params_s2 = dict(
    objective='huber' if STAGE2_LGB_USE_HUBER else 'regression_l1',
    n_estimators=25000,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=2047,
    min_child_samples=60,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.5,
    reg_alpha=0.3,
    reg_lambda=5.0,
    random_state=SEED,
    verbose=-1,
)
if STAGE2_LGB_USE_HUBER:
    lgb_params_s2['alpha'] = 0.9

xgb_params_s2 = dict(
    objective='reg:absoluteerror',
    n_estimators=20000,
    learning_rate=0.015,
    max_depth=10,
    subsample=0.75,
    colsample_bytree=0.5,
    colsample_bynode=0.5,
    reg_alpha=0.3,
    reg_lambda=3.0,
    random_state=SEED,
    tree_method='hist',
    eval_metric='mae',
    early_stopping_rounds=500,
    verbosity=0,
)

cat_params_s2 = dict(
    iterations=20000,
    learning_rate=0.015,
    depth=10,
    l2_leaf_reg=5.0,
    bootstrap_type='MVS',
    subsample=0.75,
    colsample_bylevel=0.5,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=SEED,
    task_type='CPU',
    early_stopping_rounds=500,
)

section('Stage 2 - Lag-enhanced (LGB only)' if STAGE2_LGB_ONLY else 'Stage 2 - Lag-enhanced (LGB + XGB + CAT)')
t0 = time.time()

train['_s1_pred'] = oof_s1
train = build_pred_lag_features(train, '_s1_pred', global_mean)

print(f"  pred_lag1 stats: mean={train['pred_lag1'].mean():.2f}  std={train['pred_lag1'].std():.2f}")
print(f"  pred_lag1 corr with target: {np.corrcoef(train['pred_lag1'], y_raw)[0,1]:.4f}")

oof_s2_lgb = np.zeros(len(train))
oof_s2_xgb = np.zeros(len(train))
oof_s2_cat = np.zeros(len(train))
models_s2_lgb, models_s2_xgb, models_s2_cat = [], [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):
    print(f"\n  --- Stage 2 Fold {fold} ---")

    X_tr = train.iloc[tr_idx][feature_cols_s2]
    X_va = train.iloc[va_idx][feature_cols_s2]

    y_tr = y_all[tr_idx]
    y_va = y_all[va_idx]

    # LGB
    m_lgb = lgb.LGBMRegressor(**lgb_params_s2)
    m_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_s2_lgb[va_idx] = from_train_pred(m_lgb.predict(X_va))
    models_s2_lgb.append(m_lgb)
    print(f"    LGB MAE: {mae(y_raw[va_idx], oof_s2_lgb[va_idx]):.6f}")

    if not STAGE2_LGB_ONLY:
        # XGB
        try:
            m_xgb = xgb.XGBRegressor(**xgb_params_s2)
            m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        except Exception:
            xgb_fb = dict(xgb_params_s2)
            xgb_fb['objective'] = 'reg:squarederror'
            m_xgb = xgb.XGBRegressor(**xgb_fb)
            m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_s2_xgb[va_idx] = from_train_pred(m_xgb.predict(X_va))
        models_s2_xgb.append(m_xgb)
        print(f"    XGB MAE: {mae(y_raw[va_idx], oof_s2_xgb[va_idx]):.6f}")

        # CAT
        m_cat = cb.CatBoostRegressor(**cat_params_s2)
        m_cat.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            verbose=max(1, cat_params_s2['iterations'] // 5),
            use_best_model=True,
        )
        oof_s2_cat[va_idx] = from_train_pred(m_cat.predict(X_va))
        models_s2_cat.append(m_cat)
        print(f"    CAT MAE: {mae(y_raw[va_idx], oof_s2_cat[va_idx]):.6f}")

        avg3 = (oof_s2_lgb[va_idx] + oof_s2_xgb[va_idx] + oof_s2_cat[va_idx]) / 3
        print(f"    AVG MAE: {mae(y_raw[va_idx], avg3):.6f}")

mae_s2_lgb = mae(y_raw, oof_s2_lgb)
if STAGE2_LGB_ONLY:
    mae_s2_xgb = float('nan')
    mae_s2_cat = float('nan')
    print(f"\n▶ Stage 2 OOF MAE - LGB {mae_s2_lgb:.6f}  (XGB/CAT skipped)")
else:
    mae_s2_xgb = mae(y_raw, oof_s2_xgb)
    mae_s2_cat = mae(y_raw, oof_s2_cat)
    print(f"\n▶ Stage 2 OOF MAE - LGB {mae_s2_lgb:.6f} | XGB {mae_s2_xgb:.6f} | CAT {mae_s2_cat:.6f}")


# ============================================================
# 7) Stage 2 앙상블 최적화
# ============================================================
section('Stage 2 Ensemble search')
if STAGE2_LGB_ONLY:
    oof_s2_ens = oof_s2_lgb.copy()
    s2_ens_mae = mae(y_raw, oof_s2_ens)
    best_s2_models = ['lgb']
    best_s2_p = 1.0
    w_s2 = {'lgb': 1.0}
    ws_s2 = 1.0
    print(f"▶ Stage 2 LGB-only OOF MAE: {s2_ens_mae:.6f}")
else:
    s2_model_maes = {'lgb': mae_s2_lgb, 'xgb': mae_s2_xgb, 'cat': mae_s2_cat}
    s2_oof_by = {'lgb': oof_s2_lgb, 'xgb': oof_s2_xgb, 'cat': oof_s2_cat}

    def _powerset():
        return [
            ["lgb"], ["xgb"], ["cat"],
            ["lgb", "xgb"], ["lgb", "cat"], ["xgb", "cat"],
            ["lgb", "xgb", "cat"],
        ]

    best_s2_models = None
    best_s2_p = None
    best_s2_mae = float('inf')
    for models in _powerset():
        for p in [1.0, 2.0, 3.0, 4.0]:
            w = {m: 1.0 / (s2_model_maes[m] ** p) for m in models}
            ws = sum(w.values())
            pred_tmp = sum(w[m] * s2_oof_by[m] for m in models) / ws
            m_val = mae(y_raw, pred_tmp)
            if m_val < best_s2_mae:
                best_s2_mae = float(m_val)
                best_s2_models = list(models)
                best_s2_p = float(p)

    print(f"▶ Best S2 ensemble: models={best_s2_models}  p={best_s2_p}  OOF_MAE={best_s2_mae:.6f}")

    w_s2 = {m: 1.0 / (s2_model_maes[m] ** best_s2_p) for m in best_s2_models}
    ws_s2 = sum(w_s2.values())
    oof_s2_ens = sum(w_s2[m] * s2_oof_by[m] for m in best_s2_models) / ws_s2
    s2_ens_mae = mae(y_raw, oof_s2_ens)
    print(f"▶ Stage 2 Ensemble OOF MAE: {s2_ens_mae:.6f}")

print(f"\n▶ Stage 1 OOF MAE: {s1_mae:.6f}")
print(f"▶ Stage 2 OOF MAE: {s2_ens_mae:.6f}")
print(f"▶ Improvement: {s1_mae - s2_ens_mae:.6f}")

oof_s2_used = oof_s2_ens
use_s1_instead_of_s2 = False
if REPLACE_STAGE2_IF_NOT_IMPROVING and s2_ens_mae >= s1_mae:
    print("▶ REPLACE: Stage2 OOF >= Stage1 → 이후 blend/expert/S2 제출경로는 S1과 동일 OOF 사용")
    oof_s2_used = oof_s1.copy()
    use_s1_instead_of_s2 = True


# ============================================================
# 8) Stage1 vs Stage2 블렌딩 탐색
# ============================================================
section('Stage 1 + Stage 2 blending')
best_alpha = 0.0
best_blend_mae = mae(y_raw, oof_s2_used)
_alpha_grid = _grid_01(BLEND_ALPHA_STEP)
for i_alpha, alpha in enumerate(_alpha_grid):
    blend_pred = _affine_blend(alpha, oof_s1, oof_s2_used)
    m_val = mae(y_raw, blend_pred)
    if m_val < best_blend_mae:
        best_blend_mae = float(m_val)
        best_alpha = float(alpha)
    if i_alpha in (0, len(_alpha_grid) // 2, len(_alpha_grid) - 1):
        print(f"  alpha={alpha:.4f} MAE={m_val:.6f}")

print(f"\n▶ Best blend: alpha={best_alpha:.2f} (S1 weight)  MAE={best_blend_mae:.6f}")

# ============================================================
# 8-B) Timeslot expert heads (slot0 vs slot1+)
# ============================================================
section('Timeslot expert heads (LGB 학습·추론: CPU / OOF·그리드 MAE: cupy 선택)')

slot0_mask = (train['timeslot'].values == 0)
slotn_mask = ~slot0_mask

expert_params_slot0 = dict(
    objective='regression_l1',
    n_estimators=6000,
    learning_rate=0.02,
    num_leaves=127,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=SEED,
    verbose=-1,
)

expert_params_slotn = dict(
    objective='regression_l1',
    n_estimators=10000,
    learning_rate=0.02,
    num_leaves=255,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.2,
    reg_lambda=3.0,
    random_state=SEED,
    verbose=-1,
)

oof_expert = np.zeros(len(train))
models_slot0 = []
models_slotn = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):
    tr_df = train.iloc[tr_idx]
    va_df = train.iloc[va_idx]

    tr_slot0 = tr_df['timeslot'].values == 0
    va_slot0 = va_df['timeslot'].values == 0
    tr_slotn = ~tr_slot0
    va_slotn = ~va_slot0

    # slot 0: lag 정보가 약하므로 base 피처 위주
    m0 = lgb.LGBMRegressor(**expert_params_slot0)
    m0.fit(
        tr_df.loc[tr_slot0, feature_cols_s1],
        y_all[tr_idx][tr_slot0],
        eval_set=[(va_df.loc[va_slot0, feature_cols_s1], y_all[va_idx][va_slot0])],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
    )
    pred0 = from_train_pred(m0.predict(va_df.loc[va_slot0, feature_cols_s1]))

    # slot 1+: lag 정보가 강하므로 stage2 피처 사용
    mn = lgb.LGBMRegressor(**expert_params_slotn)
    mn.fit(
        tr_df.loc[tr_slotn, feature_cols_s2],
        y_all[tr_idx][tr_slotn],
        eval_set=[(va_df.loc[va_slotn, feature_cols_s2], y_all[va_idx][va_slotn])],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
    )
    predn = from_train_pred(mn.predict(va_df.loc[va_slotn, feature_cols_s2]))

    fold_pred = np.zeros(len(va_df))
    fold_pred[va_slot0] = pred0
    fold_pred[va_slotn] = predn
    oof_expert[va_idx] = fold_pred

    models_slot0.append(m0)
    models_slotn.append(mn)
    print(f"  Expert fold {fold} MAE: {mae(y_raw[va_idx], fold_pred):.6f}")

expert_mae = mae(y_raw, oof_expert)
print(f"▶ Expert OOF MAE: {expert_mae:.6f}")

# 최종 후보 비교: 기존 blend vs expert vs 둘의 blend
best_final_name = "base_stage2_blend"
best_final_oof = _affine_blend(best_alpha, oof_s1, oof_s2_used)
best_final_mae = best_blend_mae

if expert_mae < best_final_mae:
    best_final_name = "expert_only"
    best_final_oof = oof_expert
    best_final_mae = expert_mae

best_mix_w = 0.0
base_b = _affine_blend(best_alpha, oof_s1, oof_s2_used)
for w_mix in _grid_01(EXPERT_MIX_STEP):
    mix_pred = _affine_blend(w_mix, oof_expert, base_b)
    mix_mae = mae(y_raw, mix_pred)
    if mix_mae < best_final_mae:
        best_final_mae = mix_mae
        best_final_oof = mix_pred
        best_final_name = "expert_mix"
        best_mix_w = float(w_mix)

print(f"▶ Best final head: {best_final_name}  MAE={best_final_mae:.6f}")
if best_final_name == "expert_mix":
    print(f"  expert mix weight={best_mix_w:.2f}")

final_oof_mae = best_final_mae
print(f"\n▶▶ FINAL OOF MAE: {final_oof_mae:.6f}")


# ============================================================
# 9) Test 예측 + 제출
# ============================================================
section('Predict test + submit')

# Stage 2 test: Stage 1 predictions을 lag source로 사용
test = test.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
test['_s1_pred'] = pred_s1_test
test = build_pred_lag_features(test, '_s1_pred', global_mean)

X_test_s2 = test[feature_cols_s2]

# Stage 2 각 모델의 fold별 예측 평균
p_s2_lgb = np.mean([from_train_pred(m.predict(X_test_s2)) for m in models_s2_lgb], axis=0)
if STAGE2_LGB_ONLY:
    pred_s2_test = p_s2_lgb
else:
    p_s2_xgb = np.mean([from_train_pred(m.predict(X_test_s2)) for m in models_s2_xgb], axis=0)
    p_s2_cat = np.mean([from_train_pred(m.predict(X_test_s2)) for m in models_s2_cat], axis=0)
    pred_s2_test = sum(w_s2[m] * {'lgb': p_s2_lgb, 'xgb': p_s2_xgb, 'cat': p_s2_cat}[m]
                       for m in best_s2_models) / ws_s2

pred_s2_test_blend = pred_s1_test if use_s1_instead_of_s2 else pred_s2_test

# 기존 최적 blend 예측 (벡터: cupy 선택)
pred_blend = _affine_blend(best_alpha, pred_s1_test, pred_s2_test_blend)

# expert test 예측
test_slot0 = (test['timeslot'].values == 0)
test_slotn = ~test_slot0
pred_expert = np.zeros(len(test))
if len(models_slot0) > 0:
    pred_expert[test_slot0] = np.mean(
        [from_train_pred(m.predict(test.loc[test_slot0, feature_cols_s1])) for m in models_slot0],
        axis=0,
    )
if len(models_slotn) > 0:
    pred_expert[test_slotn] = np.mean(
        [from_train_pred(m.predict(test.loc[test_slotn, feature_cols_s2])) for m in models_slotn],
        axis=0,
    )

def _pred_for_submit_auto():
    if best_final_name == "expert_only":
        return pred_expert
    if best_final_name == "expert_mix":
        return _affine_blend(best_mix_w, pred_expert, pred_blend)
    return pred_blend


_submit_override = {
    'base_blend': lambda: pred_blend,
    's1_only': lambda: pred_s1_test,
    'expert_only': lambda: pred_expert,
    'expert_mix': lambda: _affine_blend(best_mix_w, pred_expert, pred_blend),
}
if SUBMIT_PREDICTOR == 'auto':
    pred = _pred_for_submit_auto()
elif SUBMIT_PREDICTOR in _submit_override:
    pred = _submit_override[SUBMIT_PREDICTOR]()
    print(
        f"▶ JW_V11_SUBMIT_PREDICTOR={SUBMIT_PREDICTOR!r} → 제출만 해당 경로 "
        f"(OOF에서 고른 head={best_final_name!r}, FINAL OOF={final_oof_mae:.6f}는 변경 없음)"
    )
else:
    print(f"▶ JW_V11_SUBMIT_PREDICTOR={SUBMIT_PREDICTOR!r} 미지정 값 → OOF와 동일(auto)")
    pred = _pred_for_submit_auto()

# 후처리 (cupy 선택) + 제출: pandas CSV (호환)
pred = _clip_predictions(pred, y_raw)

sub = pd.DataFrame({'ID': test['ID'].values, TARGET: np.asarray(pred)})
save_path = os.path.join(project_root, 'submission_v11_2stage.csv')
if CUDF is not None and USE_GPU_NUMERIC:
    try:
        CUDF.DataFrame({'ID': sub['ID'], TARGET: CUDF.Series(sub[TARGET].values)}).to_csv(save_path, index=False)
        print(f"▶ saved (cudf to_csv) -> {save_path}")
    except Exception as e:
        print(f"▶ cudf to_csv 실패 → pandas ({e})")
        sub.to_csv(save_path, index=False)
        print(f"▶ saved -> {save_path}")
else:
    sub.to_csv(save_path, index=False)
    print(f"▶ saved -> {save_path}")
print(f"\n▶▶ DONE - FINAL OOF MAE: {final_oof_mae:.6f}")
