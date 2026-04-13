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
# jw_v13_feature_importance_only
#   · 예측/제출 없이 Stage1·Stage2 중요 피처만 저장
#   · 기본: 빠른 실행 (5폴드, S1 LGB 단일 시드)
#   · 본선/최대 설정: JW_V13_FULL=1
# ============================================================
# 실행 파라미터
# ============================================================
SEED = 42
USE_LOG_TARGET = False  # target 왜도가 높아 log 고려할 수 있으나 일단 유지

# 빠른 실험(기본): N_FOLDS=5, S1_LGB_SEEDS=[SEED]
# 풀 설정: PowerShell에서 `$env:JW_V13_FULL='1'` 후 실행
_V13_FULL = os.environ.get('JW_V13_FULL', '0').strip() == '1'
N_FOLDS = 10 if _V13_FULL else 5
S1_LGB_SEEDS = [SEED, SEED + 100, SEED + 200] if _V13_FULL else [SEED]

RUN_STRATIFIED_GROUP_CV = True
BLEND_ALPHA_STEP = 0.02 if _V13_FULL else 0.05
REPLACE_STAGE2_IF_NOT_IMPROVING = True
TOP_IMPORTANCE_K = 30
IMPORTANCE_OUTPUT_PREFIX = 'fi_only'

CLIP_PRED_MIN = 0.0
CLIP_PRED_MAX_Q = 0.995

TARGET = 'avg_delay_minutes_next_30m'
ID_COLS = ['ID', 'layout_id', 'scenario_id']

# ★ 핵심 변경: 실제 target lag 피처 컬럼명 정의
#   Stage 1 OOF 예측 기반 lag (corr ~0.64) → 실제 target lag (corr ~0.86)
TRUE_LAG_COLS = [
    'target_lag1', 'target_lag2', 'target_lag3', 'target_lag4', 'target_lag5',
    'target_roll3_mean', 'target_roll5_mean', 'target_roll10_mean',
    'target_ewm3', 'target_ewm5',
    'target_diff1', 'target_diff2',
    'target_lag1_log',
    'target_lag_max3', 'target_lag_min3', 'target_lag_std3',
]

def elapsed(start):
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60:02d}s"


def section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _grid_01(step: float) -> np.ndarray:
    if step <= 0 or step > 1:
        step = 0.05
    n = max(1, int(round(1.0 / step)))
    return np.linspace(0.0, 1.0, n + 1)


def _safe_feature_importance(model, feature_cols):
    """모델별 피처 중요도를 동일한 길이 벡터로 반환."""
    try:
        if hasattr(model, 'feature_importances_'):
            imp = np.asarray(model.feature_importances_, dtype=np.float64)
            if imp.shape[0] == len(feature_cols):
                return imp
    except Exception:
        pass
    try:
        if hasattr(model, 'get_feature_importance'):
            imp = np.asarray(model.get_feature_importance(), dtype=np.float64)
            if imp.shape[0] == len(feature_cols):
                return imp
    except Exception:
        pass
    return np.zeros(len(feature_cols), dtype=np.float64)


def _aggregate_importance(models, feature_cols):
    if not models:
        return pd.DataFrame(columns=['feature', 'importance'])
    arr = np.vstack([_safe_feature_importance(m, feature_cols) for m in models])
    mean_imp = arr.mean(axis=0)
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': mean_imp})
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    return imp_df


def _report_importance(stage_name, feature_cols, model_groups, top_k, project_root):
    """
    model_groups: {'lgb': [models...], 'xgb': [...], 'cat': [...]}
    """
    section(f'{stage_name} Feature Importance')
    merged = None
    for model_name, models in model_groups.items():
        imp = _aggregate_importance(models, feature_cols)
        if imp.empty:
            continue
        col_name = f'{model_name}_importance'
        imp = imp.rename(columns={'importance': col_name})
        save_path = os.path.join(
            project_root,
            f'{IMPORTANCE_OUTPUT_PREFIX}_feature_importance_{stage_name.lower()}_{model_name}.csv',
        )
        imp.to_csv(save_path, index=False)
        print(f"▶ saved {stage_name}/{model_name} importance -> {save_path}")
        print(imp.head(top_k).to_string(index=False))
        merged = imp if merged is None else merged.merge(imp, on='feature', how='outer')
    if merged is not None:
        score_cols = [c for c in merged.columns if c.endswith('_importance')]
        merged[score_cols] = merged[score_cols].fillna(0.0)
        merged['mean_importance'] = merged[score_cols].mean(axis=1)
        merged = merged.sort_values('mean_importance', ascending=False).reset_index(drop=True)
        save_path = os.path.join(
            project_root,
            f'{IMPORTANCE_OUTPUT_PREFIX}_feature_importance_{stage_name.lower()}_merged.csv',
        )
        merged.to_csv(save_path, index=False)
        print(f"▶ saved {stage_name} merged importance -> {save_path}")
        print(merged[['feature', 'mean_importance']].head(top_k).to_string(index=False))


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
    return mean_absolute_error(y_true, y_pred)


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
# 공통 유틸: lag 업데이트 + 순차 예측
# OOF(Method B)와 Test 추론에서 동일 로직을 사용 → 갭 제거
# ============================================================
def _update_lag_from_history(df, row_indices, scenario_history, gm):
    """
    slot > 0 일 때 row_indices 의 각 행의 TRUE_LAG_COLS를
    해당 시나리오의 예측 히스토리로 갱신한다.
    df는 float64 타입이어야 하며 인덱스가 연속 정수여야 한다.
    """
    for pos in row_indices:
        scen_id = df.at[pos, 'scenario_id']
        hist = scenario_history.get(scen_id, [])
        n = len(hist)

        # 기본 lag
        for lag_i in [1, 2, 3, 4, 5]:
            df.at[pos, f'target_lag{lag_i}'] = hist[-lag_i] if n >= lag_i else gm

        # rolling mean (3, 5, 10)
        for w in [3, 5, 10]:
            df.at[pos, f'target_roll{w}_mean'] = float(np.mean(hist[-w:])) if n > 0 else gm

        # rolling max/min/std (3슬롯)
        df.at[pos, 'target_lag_max3'] = float(np.max(hist[-3:])) if n > 0 else gm
        df.at[pos, 'target_lag_min3'] = float(np.min(hist[-3:])) if n > 0 else gm
        df.at[pos, 'target_lag_std3'] = float(np.std(hist[-3:])) if n >= 2 else 0.0

        # EWM (α=0.3, 0.5)
        if n == 0:
            ewm3, ewm5 = gm, gm
        else:
            ewm3 = ewm5 = hist[0]
            for v in hist[1:]:
                ewm3 = 0.3 * v + 0.7 * ewm3
                ewm5 = 0.5 * v + 0.5 * ewm5
        df.at[pos, 'target_ewm3'] = ewm3
        df.at[pos, 'target_ewm5'] = ewm5

        # 차분
        df.at[pos, 'target_diff1'] = (hist[-1] - hist[-2]) if n >= 2 else 0.0
        df.at[pos, 'target_diff2'] = (hist[-2] - hist[-3]) if n >= 3 else 0.0

        # log lag
        df.at[pos, 'target_lag1_log'] = (
            float(np.log1p(max(hist[-1], 0))) if n >= 1 else float(np.log1p(max(gm, 0)))
        )


def sequential_predict(df_sorted, models_list, feature_cols, gm, from_pred_fn, n_slots=25):
    """
    슬롯별 순차 예측 함수.

    Parameters
    ----------
    df_sorted  : scenario_id, timeslot 순으로 정렬된 DataFrame.
                 TRUE_LAG_COLS 컬럼이 존재해야 하며 gm 으로 초기화된 상태여야 함.
                 인덱스는 호출 측에서 원본 위치를 복원할 수 있도록 보존되어야 한다.
    models_list: fold 모델 리스트 (앙상블 평균).
    feature_cols: 예측에 사용할 컬럼 리스트.
    gm         : global mean (lag 초기값).
    from_pred_fn: 모델 출력 후처리 함수 (log target이면 expm1, 아니면 identity).
    n_slots    : 타임슬롯 수 (default 25).

    Returns
    -------
    preds_out : np.ndarray, df_sorted 행 순서와 동일한 예측값.
    """
    df = df_sorted.copy()
    # lag 피처 초기화 (slot 0 에는 과거 정보 없음)
    for col in TRUE_LAG_COLS:
        df[col] = gm

    preds_out = np.zeros(len(df))
    scenario_history = {}  # {scenario_id: [pred_t0, pred_t1, ...]}

    for slot in range(n_slots):
        slot_mask = df['timeslot'] == slot
        if not slot_mask.any():
            continue
        slot_idx = df.index[slot_mask].tolist()

        # slot > 0: 이전 슬롯 예측값으로 lag 갱신
        if slot > 0:
            _update_lag_from_history(df, slot_idx, scenario_history, gm)

        # 앙상블 평균 예측
        slot_data = df.loc[slot_idx, feature_cols]
        slot_preds = np.mean(
            [from_pred_fn(m.predict(slot_data)) for m in models_list], axis=0
        )
        preds_out[slot_mask.values] = slot_preds

        # 히스토리 누적
        for i, pos in enumerate(slot_idx):
            scen_id = df.at[pos, 'scenario_id']
            scenario_history.setdefault(scen_id, []).append(float(slot_preds[i]))

    return preds_out


# ============================================================
# 1) 데이터 로드
# ============================================================
path = _resolve_data_dir()
project_root = str(Path(path).resolve().parent)
print(f"▶ data dir: {path}")
print(f"▶ project root: {project_root}")

t0 = time.time()
train = pd.read_csv(os.path.join(path, 'train.csv'))
test = pd.read_csv(os.path.join(path, 'test.csv'))
layout = pd.read_csv(os.path.join(path, 'layout_info.csv'))
print(f"▶ load done ({elapsed(t0)})  train {len(train):,} / test {len(test):,}")
print(
    f"▶ jw_v13: mode={'FULL(JW_V13_FULL=1)' if _V13_FULL else 'quick(default)'}, "
    f"N_FOLDS={N_FOLDS}, S1_LGB seeds={len(S1_LGB_SEEDS)}, blend_step={BLEND_ALPHA_STEP}, true-lag+MethodB"
)


# ============================================================
# 2) 전처리  (Stage 1 & 2 공통)
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


def add_extra_features_v13(df):
    """시나리오 내 진행도·sqrt·timeslot 주기 (train/test 공통)."""
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    gsz = df.groupby('scenario_id')['scenario_id'].transform('size')
    gpos = df.groupby('scenario_id').cumcount()
    df['v13_row_frac_in_scen'] = (gpos + 1) / (gsz + 1e-6)
    if 'order_inflow_15m' in df.columns:
        df['v13_sqrt_order_inflow'] = np.sqrt(np.maximum(df['order_inflow_15m'].astype(np.float64), 0.0))
    if 'robot_total' in df.columns:
        df['v13_sqrt_robot_total'] = np.sqrt(np.maximum(df['robot_total'].astype(np.float64), 0.0))
    if 'timeslot' in df.columns:
        ang = 2 * np.pi * df['timeslot'].astype(np.float64) / 24.0
        df['v13_ts_sin'] = np.sin(ang)
        df['v13_ts_cos'] = np.cos(ang)
    return df


def preprocess_all(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')
    df = handle_missing_values(df)
    df = add_basic_features(df)
    df = add_timeseries_features(df)
    df = add_interaction_features(df)
    df = add_extra_features_v13(df)
    if 'layout_type' in df.columns:
        df['layout_type'] = pd.factorize(df['layout_type'])[0]
    return df


section('Preprocess')
t0 = time.time()
train = preprocess_all(train, layout)
test = preprocess_all(test, layout)
print(f"▶ preprocess done ({elapsed(t0)})")


# ============================================================
# 3) 타겟 인코딩 (OOF)
# ============================================================
section('Target Encoding')
t0 = time.time()
TE_COLS = [c for c in ['layout_id', 'timeslot', 'layout_type', 'shift_hour', 'day_of_week'] if c in train.columns]
TE_PAIRS = []
for a in TE_COLS:
    for b in TE_COLS:
        if a < b:
            TE_PAIRS.append((a, b))

SMOOTHING = 20
kf_te = GroupKFold(n_splits=N_FOLDS)
groups_te = train['scenario_id']
global_mean = float(train[TARGET].mean())


def _apply_te(df_train, df_test, col_name, group_col_series_tr, group_col_series_te):
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


for col in TE_COLS:
    _apply_te(train, test, col, train[col], test[col])

for a, b in TE_PAIRS:
    pair_name = f'{a}_X_{b}'
    tr_key = train[a].astype(str) + '_' + train[b].astype(str)
    te_key = test[a].astype(str) + '_' + test[b].astype(str)
    _apply_te(train, test, pair_name, tr_key, te_key)

print(f"▶ target encoding done ({elapsed(t0)})")


# ============================================================
# 3-B) 핵심 추가: 실제 Target Lag 피처 생성 (train only)
#   Stage 1 OOF pred 기반 lag corr ~0.64 -> 실제 target lag1 corr ~0.86
#   GroupKFold는 scenario 단위 분할 -> 시계열 lag은 leakage 아님
# ============================================================
section('True Target Lag Feature Engineering (train only)')
t0 = time.time()

train = train.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
g_target = train.groupby('scenario_id')[TARGET]

for lag in [1, 2, 3, 4, 5]:
    train[f'target_lag{lag}'] = g_target.shift(lag).fillna(global_mean)

for w in [3, 5, 10]:
    train[f'target_roll{w}_mean'] = g_target.transform(
        lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean()
    ).fillna(global_mean)

train['target_lag_max3'] = g_target.transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).max()
).fillna(global_mean)
train['target_lag_min3'] = g_target.transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).min()
).fillna(global_mean)
train['target_lag_std3'] = g_target.transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0)
).fillna(0)

train['target_ewm3'] = g_target.transform(
    lambda x: x.shift(1).ewm(alpha=0.3, adjust=False).mean()
).fillna(global_mean)
train['target_ewm5'] = g_target.transform(
    lambda x: x.shift(1).ewm(alpha=0.5, adjust=False).mean()
).fillna(global_mean)

train['target_diff1'] = train['target_lag1'] - train['target_lag2']
train['target_diff2'] = train['target_lag2'] - train['target_lag3']
train['target_lag1_log'] = np.log1p(train['target_lag1'].clip(lower=0))

for col in TRUE_LAG_COLS:
    if col in train.columns:
        train[col] = train[col].fillna(global_mean)

print(f"  target_lag1 corr with target: {train['target_lag1'].corr(train[TARGET]):.4f}")
print(f"  target_lag2 corr with target: {train['target_lag2'].corr(train[TARGET]):.4f}")
print(f"  target_ewm3 corr with target: {train['target_ewm3'].corr(train[TARGET]):.4f}")
print(f"  TRUE_LAG_COLS count: {len(TRUE_LAG_COLS)}")
print(f"▶ True target lag features done ({elapsed(t0)})")


# ============================================================
# 4) Stage 1: Base Model (target lag 없음)
#    ★ TRUE_LAG_COLS를 명시적으로 제외해야 함
#      (3-B 섹션에서 train에 lag 컬럼이 이미 추가된 상태이므로
#       제외하지 않으면 Stage 1이 real lag로 학습 → OOF-LB 갭 발생)
# ============================================================
feature_cols_s1 = [
    c for c in train.columns
    if c not in ID_COLS + [TARGET] + TRUE_LAG_COLS
]
print(f"▶ Stage 1 features: {len(feature_cols_s1)}  (TRUE_LAG_COLS excluded)")

y_all = to_train_target(train[TARGET].values)
y_raw = train[TARGET].values
groups = train['scenario_id'].values

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
_report_importance(
    stage_name='stage1',
    feature_cols=feature_cols_s1,
    model_groups={
        'lgb': models_s1_lgb,
        'xgb': models_s1_xgb,
        'cat': models_s1_cat,
    },
    top_k=TOP_IMPORTANCE_K,
    project_root=project_root,
)

X_test_s1 = test[feature_cols_s1]
p_lgb = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_lgb], axis=0)
p_xgb = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_xgb], axis=0)
p_cat = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_cat], axis=0)
pred_s1_pre = sum(w_s1[m] * {'lgb': p_lgb, 'xgb': p_xgb, 'cat': p_cat}[m] for m in best_s1_models) / ws_s1
pred_resid_test = np.mean([m.predict(X_test_s1) for m in models_s1_resid], axis=0)
pred_s1_test = pred_s1_pre + pred_resid_test
print(f"▶ Stage 1 test predictions ready (ensemble + residual)")


# ============================================================
# 5) 구버전 pred lag 함수 제거 → sequential_predict 함수로 통합
#    (상단 공통 유틸 섹션에 정의됨)
# ============================================================

# ============================================================
# 6) Stage 2: True Target Lag 기반 모델 + Method B OOF
#
#    [학습]  training fold  → 실제 target lag (corr ~0.86) 사용
#              → 모델이 강한 시계열 신호를 학습
#
#    [OOF 평가] Method B 적용
#              → validation fold도 슬롯별 순차 예측으로 lag 전파
#              → OOF 조건 = Test 조건 → OOF-LB 갭 제거
#
#    [Test 추론] 동일한 sequential_predict() 함수 재사용
# ============================================================
feature_cols_s2 = feature_cols_s1 + TRUE_LAG_COLS

lgb_params_s2 = dict(
    objective='regression_l1',
    n_estimators=25000,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=2047,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_alpha=0.1,
    reg_lambda=3.0,
    random_state=SEED,
    verbose=-1,
)

xgb_params_s2 = dict(
    objective='reg:absoluteerror',
    n_estimators=20000,
    learning_rate=0.015,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.6,
    colsample_bynode=0.5,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=SEED,
    tree_method='hist',
    eval_metric='mae',
    early_stopping_rounds=500,
    verbosity=0,
)

cat_params_s2 = dict(
    iterations=20000,
    learning_rate=0.015,
    depth=8,
    l2_leaf_reg=3.0,
    bootstrap_type='MVS',
    subsample=0.8,
    colsample_bylevel=0.6,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=SEED,
    task_type='CPU',
    early_stopping_rounds=500,
)

section('Stage 2 - True Target Lag model + Method B OOF (LGB + XGB + CAT)')
t0 = time.time()

print(f"  target_lag1 stats: mean={train['target_lag1'].mean():.2f}  std={train['target_lag1'].std():.2f}")
print(f"  target_lag1 corr with target: {np.corrcoef(train['target_lag1'], y_raw)[0,1]:.4f}")
print(f"  [Method B] OOF는 slotwise sequential 예측으로 평가 → OOF-LB 갭 제거")

oof_s2_lgb = np.zeros(len(train))
oof_s2_xgb = np.zeros(len(train))
oof_s2_cat = np.zeros(len(train))
models_s2_lgb, models_s2_xgb, models_s2_cat = [], [], []

# train은 3-B 섹션에서 (scenario_id, ID) 순으로 정렬됨
# timeslot = groupby(scenario_id).cumcount() → 동일 정렬에서 0,1,...,24

for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):
    print(f"\n  --- Stage 2 Fold {fold} ---")

    # ── 학습: training fold의 실제 target lag 사용 ──
    X_tr = train.iloc[tr_idx][feature_cols_s2]
    y_tr = y_all[tr_idx]

    # early stopping 용 val에도 실제 lag 사용 (학습 품질 극대화)
    X_va_real = train.iloc[va_idx][feature_cols_s2]
    y_va_real = y_all[va_idx]

    # LGB 학습
    m_lgb = lgb.LGBMRegressor(**lgb_params_s2)
    m_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va_real, y_va_real)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)],
    )
    models_s2_lgb.append(m_lgb)

    # XGB 학습
    try:
        m_xgb = xgb.XGBRegressor(**xgb_params_s2)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va_real, y_va_real)], verbose=False)
    except Exception:
        xgb_fb = dict(xgb_params_s2)
        xgb_fb['objective'] = 'reg:squarederror'
        m_xgb = xgb.XGBRegressor(**xgb_fb)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va_real, y_va_real)], verbose=False)
    models_s2_xgb.append(m_xgb)

    # CAT 학습
    m_cat = cb.CatBoostRegressor(**cat_params_s2)
    m_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va_real, y_va_real),
        verbose=max(1, cat_params_s2['iterations'] // 10),
        use_best_model=True,
    )
    models_s2_cat.append(m_cat)

    # ── Method B: validation fold를 slotwise 순차 예측으로 OOF 평가 ──
    # va_df: validation fold 행들을 (scenario_id, timeslot) 순으로 정렬
    # 인덱스 = 원본 train 위치 (reset_index 완료 상태) → 나중에 oof 배열 복원 가능
    va_df = train.iloc[va_idx].sort_values(['scenario_id', 'timeslot'])

    # LGB sequential OOF
    lgb_seq = sequential_predict(va_df, [m_lgb], feature_cols_s2, global_mean, from_train_pred)
    for orig_pos, pred in zip(va_df.index, lgb_seq):
        oof_s2_lgb[orig_pos] = pred

    # XGB sequential OOF
    xgb_seq = sequential_predict(va_df, [m_xgb], feature_cols_s2, global_mean, from_train_pred)
    for orig_pos, pred in zip(va_df.index, xgb_seq):
        oof_s2_xgb[orig_pos] = pred

    # CAT sequential OOF
    cat_seq = sequential_predict(va_df, [m_cat], feature_cols_s2, global_mean, from_train_pred)
    for orig_pos, pred in zip(va_df.index, cat_seq):
        oof_s2_cat[orig_pos] = pred

    avg3 = (oof_s2_lgb[va_idx] + oof_s2_xgb[va_idx] + oof_s2_cat[va_idx]) / 3
    print(f"    LGB MAE (sequential): {mae(y_raw[va_idx], oof_s2_lgb[va_idx]):.6f}")
    print(f"    XGB MAE (sequential): {mae(y_raw[va_idx], oof_s2_xgb[va_idx]):.6f}")
    print(f"    CAT MAE (sequential): {mae(y_raw[va_idx], oof_s2_cat[va_idx]):.6f}")
    print(f"    AVG MAE (sequential): {mae(y_raw[va_idx], avg3):.6f}")

mae_s2_lgb = mae(y_raw, oof_s2_lgb)
mae_s2_xgb = mae(y_raw, oof_s2_xgb)
mae_s2_cat = mae(y_raw, oof_s2_cat)
print(f"\n▶ Stage 2 OOF MAE (Method B) - LGB {mae_s2_lgb:.6f} | XGB {mae_s2_xgb:.6f} | CAT {mae_s2_cat:.6f}")


# ============================================================
# 7) Stage 2 앙상블 최적화
# ============================================================
section('Stage 2 Ensemble search')
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

# build S2 ensemble OOF
w_s2 = {m: 1.0 / (s2_model_maes[m] ** best_s2_p) for m in best_s2_models}
ws_s2 = sum(w_s2.values())
oof_s2_ens = sum(w_s2[m] * s2_oof_by[m] for m in best_s2_models) / ws_s2
s2_ens_mae = mae(y_raw, oof_s2_ens)
print(f"▶ Stage 2 Ensemble OOF MAE: {s2_ens_mae:.6f}")

print(f"\n▶ Stage 1 OOF MAE: {s1_mae:.6f}")
print(f"▶ Stage 2 OOF MAE: {s2_ens_mae:.6f}")
print(f"▶ Improvement: {s1_mae - s2_ens_mae:.6f}")
_report_importance(
    stage_name='stage2',
    feature_cols=feature_cols_s2,
    model_groups={
        'lgb': models_s2_lgb,
        'xgb': models_s2_xgb,
        'cat': models_s2_cat,
    },
    top_k=TOP_IMPORTANCE_K,
    project_root=project_root,
)

section('Feature importance export complete')
print('▶ 예측/제출 단계 없이 종료합니다.')
raise SystemExit(0)

oof_s2_used = oof_s2_ens
use_s1_instead_of_s2 = False
if REPLACE_STAGE2_IF_NOT_IMPROVING and s2_ens_mae >= s1_mae:
    print('▶ REPLACE: Stage2 OOF >= Stage1 → 블렌드/제출은 S2 대신 S1 OOF 사용')
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
    blend_pred = alpha * oof_s1 + (1 - alpha) * oof_s2_used
    m_val = mae(y_raw, blend_pred)
    if m_val < best_blend_mae:
        best_blend_mae = float(m_val)
        best_alpha = float(alpha)
    if i_alpha in (0, len(_alpha_grid) // 2, len(_alpha_grid) - 1):
        print(f"  alpha={alpha:.4f} MAE={m_val:.6f}")

print(f"\n▶ Best blend: alpha={best_alpha:.2f} (S1 weight)  MAE={best_blend_mae:.6f}")

# ============================================================
# 8-B) Expert Head 제거
#   이유: slot 1+ OOF 평가에 real target lag 사용 → 동일한 OOF-LB 갭 문제 존재
#        Method B를 적용하면 Stage 2 ensemble이 이미 honest OOF를 반영하므로
#        expert head 없이 Stage1+Stage2 blend가 최종 출력
# ============================================================
best_final_name = "stage1_stage2_blend"
best_final_mae = best_blend_mae
print(f"\n▶ Best final: {best_final_name}  MAE={best_final_mae:.6f}")
print(f"\n▶▶ FINAL OOF MAE: {best_final_mae:.6f}")


# ============================================================
# 9) Test 예측 + 제출
#    OOF와 동일한 sequential_predict() 재사용 → 조건 완전 일치
# ============================================================
section('Predict test + submit (sequential — same as Method B OOF)')

test = test.sort_values(['scenario_id', 'timeslot', 'ID']).reset_index(drop=True)
for col in TRUE_LAG_COLS:
    test[col] = global_mean

# Stage 1 test 예측 (lag 없이, feature_cols_s1 에 TRUE_LAG_COLS 없음)
print("  Stage 1 test prediction (no lag)...")
X_test_s1 = test[feature_cols_s1]
p_lgb_s1 = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_lgb], axis=0)
p_xgb_s1 = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_xgb], axis=0)
p_cat_s1 = np.mean([from_train_pred(m.predict(X_test_s1)) for m in models_s1_cat], axis=0)
pred_s1_pre = sum(w_s1[m] * {'lgb': p_lgb_s1, 'xgb': p_xgb_s1, 'cat': p_cat_s1}[m]
                  for m in best_s1_models) / ws_s1
pred_resid_test = np.mean([m.predict(X_test_s1) for m in models_s1_resid], axis=0)
pred_s1_test = pred_s1_pre + pred_resid_test
print(f"  Stage 1 test pred: mean={pred_s1_test.mean():.2f}  std={pred_s1_test.std():.2f}")

# Stage 2 test 예측: sequential_predict() — OOF와 동일 로직
# 5-fold 모델 앙상블을 한꺼번에 넘겨 슬롯별 평균 예측
print(f"  Stage 2 sequential test prediction (ensemble of {N_FOLDS} folds)...")

# LGB, XGB, CAT 각각 sequential
pred_s2_lgb_test = sequential_predict(test, models_s2_lgb, feature_cols_s2, global_mean, from_train_pred)
pred_s2_xgb_test = sequential_predict(test, models_s2_xgb, feature_cols_s2, global_mean, from_train_pred)
pred_s2_cat_test = sequential_predict(test, models_s2_cat, feature_cols_s2, global_mean, from_train_pred)

# Stage 2 앙상블 가중합
pred_s2_test = sum(
    w_s2[m] * {'lgb': pred_s2_lgb_test, 'xgb': pred_s2_xgb_test, 'cat': pred_s2_cat_test}[m]
    for m in best_s2_models
) / ws_s2
print(f"  Stage 2 test pred: mean={pred_s2_test.mean():.2f}  std={pred_s2_test.std():.2f}")

pred_s2_test_blend = pred_s1_test if use_s1_instead_of_s2 else pred_s2_test

# Stage 1 + Stage 2 블렌드
pred_blend_test = best_alpha * pred_s1_test + (1 - best_alpha) * pred_s2_test_blend

# 후처리
pred = np.maximum(pred_blend_test, CLIP_PRED_MIN)
pred_hi = float(np.percentile(y_raw, 100 * CLIP_PRED_MAX_Q))
pred = np.minimum(pred, pred_hi)

sub = pd.DataFrame({'ID': test['ID'], TARGET: pred})
save_path = os.path.join(project_root, 'submission_v13_deepstack.csv')
sub.to_csv(save_path, index=False)
print(f"▶ saved -> {save_path}")
print(f"\n▶▶ DONE - FINAL OOF MAE: {best_final_mae:.6f}")