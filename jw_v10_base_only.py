import os
import sys
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
# 실행/튜닝 파라미터
# ============================================================
N_FOLDS = 5
SEED = 42

# (1) 원스케일 MAE
USE_LOG_TARGET = False

# (A) 앙상블 구성/가중치 탐색
RUN_ENSEMBLE_SEARCH = True
P_GRID = [1.0, 2.0, 3.0, 4.0]  # w = 1/mae^p

# (B) LGB 파라미터 미니 그리드
RUN_LGB_GRID = True
LGB_GRID = [
    # learning_rate, num_leaves, min_child_samples, reg_lambda
    dict(learning_rate=0.03, num_leaves=511,  min_child_samples=20, reg_lambda=1.0),
    dict(learning_rate=0.02, num_leaves=511,  min_child_samples=20, reg_lambda=2.0),
    dict(learning_rate=0.02, num_leaves=1023, min_child_samples=20, reg_lambda=2.0),
    dict(learning_rate=0.02, num_leaves=1023, min_child_samples=40, reg_lambda=3.0),
    dict(learning_rate=0.015, num_leaves=2047, min_child_samples=40, reg_lambda=3.0),
    dict(learning_rate=0.01, num_leaves=2047, min_child_samples=60, reg_lambda=5.0),
]

# (3) 앙상블 가중치 강화: w = 1/mae^p
WEIGHT_POWER = 2.0

# (3) 약한 모델 자동 제외(옵션)
AUTO_DROP_WORST = True
DROP_WORST_IF_GAP_GT = 0.08

# 예측값 후처리
CLIP_PRED_MIN = 0.0
CLIP_PRED_MAX_Q = 0.995


def elapsed(start):
    s = int(time.time() - start)
    return f"{s//60}m {s%60:02d}s"


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def _resolve_data_dir() -> str:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        d = p / 'data'
        if d.is_dir() and (d / 'train.csv').is_file():
            return str(d)
    raise FileNotFoundError('data/train.csv 를 찾을 수 없습니다. 프로젝트 루트에서 실행하세요.')


def to_train_target(y):
    return np.log1p(y) if USE_LOG_TARGET else y


def from_train_pred(p):
    return np.expm1(p) if USE_LOG_TARGET else p


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def _ensemble_pred(oof_by_model: dict[str, np.ndarray], maes_by_model: dict[str, float], models: list[str], p: float) -> np.ndarray:
    w = {m: 1.0 / (maes_by_model[m] ** p) for m in models}
    w_sum = sum(w.values())
    out = np.zeros_like(next(iter(oof_by_model.values())))
    for m in models:
        out += w[m] * oof_by_model[m]
    return out / w_sum


def _powerset_models():
    return [
        ["lgb"],
        ["xgb"],
        ["cat"],
        ["lgb", "xgb"],
        ["lgb", "cat"],
        ["xgb", "cat"],
        ["lgb", "xgb", "cat"],
    ]


# ============================================================
# 1) 데이터 로드
# ============================================================
path = _resolve_data_dir()
project_root = str(Path(path).resolve().parent)
print(f"▶ data dir: {path}")
print(f"▶ project root: {project_root}")

TARGET = 'avg_delay_minutes_next_30m'
ID_COLS = ['ID', 'layout_id', 'scenario_id']

t0 = time.time()
train = pd.read_csv(os.path.join(path, 'train.csv'))
test  = pd.read_csv(os.path.join(path, 'test.csv'))
layout = pd.read_csv(os.path.join(path, 'layout_info.csv'))
print(f"▶ load done ({elapsed(t0)})  train {len(train):,} / test {len(test):,}")


# ============================================================
# 2) 전처리
# ============================================================
def handle_missing_values(df):
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    cols_to_fix = [c for c in df.columns if df[c].isnull().any() and c not in (ID_COLS + [TARGET])]
    if cols_to_fix:
        df[cols_to_fix] = df.groupby('scenario_id')[cols_to_fix].ffill()
        df[cols_to_fix] = df.groupby('scenario_id')[cols_to_fix].bfill()
        df[cols_to_fix] = df[cols_to_fix].fillna(df[cols_to_fix].median())
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
]


def add_timeseries_features(df):
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    for col in TS_COLS:
        if col not in df.columns:
            continue
        g = df.groupby('scenario_id')[col]
        for lag in (1, 2, 3):
            df[f'{col}_lag{lag}'] = g.shift(lag)
        df[f'{col}_diff1'] = g.shift(1) - g.shift(2)
        df[f'{col}_roll3_mean'] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df[f'{col}_roll5_mean'] = g.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df[f'{col}_roll3_std']  = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0))
        # 과거 누적/지수이동(미래 누수 방지: shift(1) 후 누적)
        df[f'{col}_exp_mean'] = g.transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        df[f'{col}_exp_std']  = g.transform(lambda x: x.shift(1).expanding(min_periods=2).std()).fillna(0)
        df[f'{col}_ewm_mean'] = g.transform(lambda x: x.shift(1).ewm(alpha=0.3, adjust=False).mean())

    lag_cols = [c for c in df.columns if ('_lag' in c or '_diff' in c) and c not in ID_COLS]
    if lag_cols:
        # NOTE: bfill은 미래 시점의 값을 끌어오므로 사용하지 않음.
        df[lag_cols] = df.groupby('scenario_id')[lag_cols].ffill()
        # 초기 슬롯(shift로 생긴 NaN)만 안전하게 채우기: 해당 피처의 시나리오 평균 -> 전체 중앙값
        for c in lag_cols:
            base_col = c.split('_lag')[0].split('_diff')[0]
            if base_col in df.columns:
                scen_mean = df.groupby('scenario_id')[base_col].transform('mean')
                df[c] = df[c].fillna(scen_mean)
        df[lag_cols] = df[lag_cols].fillna(df[lag_cols].median())
    return df


def preprocess_all(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')
    df = handle_missing_values(df)
    df = add_basic_features(df)
    df = add_timeseries_features(df)
    if 'layout_type' in df.columns:
        df['layout_type'] = pd.factorize(df['layout_type'])[0]
    return df


section('Preprocess')
t0 = time.time()
train = preprocess_all(train, layout)
test  = preprocess_all(test, layout)
print(f"▶ preprocess done ({elapsed(t0)})")


# ============================================================
# 3) 타겟 인코딩 (OOF)
# ============================================================
section('Target Encoding')
t0 = time.time()
TE_COLS = [c for c in ['layout_id', 'timeslot', 'layout_type', 'shift_hour', 'day_of_week'] if c in train.columns]
SMOOTHING = 20
kf_te = GroupKFold(n_splits=N_FOLDS)
groups_te = train['scenario_id']
for col in TE_COLS:
    te_col = f'{col}_te'
    train[te_col] = np.nan
    global_mean = train[TARGET].mean()
    for tr_idx, val_idx in kf_te.split(train, train[TARGET], groups=groups_te):
        tr_df = train.iloc[tr_idx]
        stats = tr_df.groupby(col)[TARGET].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + SMOOTHING * global_mean) / (stats['count'] + SMOOTHING)
        train.loc[val_idx, te_col] = train.iloc[val_idx][col].map(smooth).fillna(global_mean)
    stats_full = train.groupby(col)[TARGET].agg(['mean', 'count'])
    smooth_full = (stats_full['count'] * stats_full['mean'] + SMOOTHING * global_mean) / (stats_full['count'] + SMOOTHING)
    test[te_col] = test[col].map(smooth_full).fillna(global_mean)
print(f"▶ target encoding done ({elapsed(t0)})")


# ============================================================
# 4) 학습/OOF
# ============================================================
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
print(f"▶ features: {len(feature_cols)}")

y_all = to_train_target(train[TARGET].values)
groups = train['scenario_id'].values
kf = GroupKFold(n_splits=N_FOLDS)

lgb_params = dict(
    objective='regression_l1',
    n_estimators=20000,
    learning_rate=0.02,
    max_depth=10,
    num_leaves=1023,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.2,
    reg_lambda=3.0,
    random_state=SEED,
    verbose=-1,
)


def run_lgb_oof(params_override: dict) -> tuple[np.ndarray, float]:
    """LGB만 빠르게 OOF 계산 (grid 용)."""
    params = dict(lgb_params)
    params.update(params_override)
    oof = np.zeros(len(train))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y_all, groups=groups), 1):
        X_tr = train.iloc[tr_idx][feature_cols]
        X_va = train.iloc[va_idx][feature_cols]
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[va_idx] = from_train_pred(m.predict(X_va))
    return oof, mae(train[TARGET].values, oof)


if RUN_LGB_GRID:
    section("LGB mini grid (OOF)")
    grid_results = []
    for i, cfg in enumerate(LGB_GRID, 1):
        t0 = time.time()
        oof_tmp, m = run_lgb_oof(cfg)
        grid_results.append((m, cfg))
        print(f"  [{i}/{len(LGB_GRID)}] OOF MAE {m:.6f}  cfg={cfg}  ({elapsed(t0)})")
    grid_results.sort(key=lambda x: x[0])
    best_m, best_cfg = grid_results[0]
    print(f"\n▶ Best LGB grid OOF MAE: {best_m:.6f}\n  cfg={best_cfg}")
    # 아래 한 줄로 이후 전체 학습에 best 설정 반영
    lgb_params.update(best_cfg)

xgb_params = dict(
    objective='reg:absoluteerror',
    n_estimators=12000,
    learning_rate=0.01,
    max_depth=9,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=SEED,
    tree_method='hist',
    eval_metric='mae',
    early_stopping_rounds=300,
    verbosity=0,
)

cat_params = dict(
    iterations=12000,
    learning_rate=0.01,
    depth=9,
    l2_leaf_reg=6.0,
    bootstrap_type='MVS',
    subsample=0.8,
    colsample_bylevel=0.7,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=SEED,
    task_type='CPU',
    early_stopping_rounds=300,
)

section('Train OOF')
oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
models_lgb, models_xgb, models_cat = [], [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y_all, groups=groups), 1):
    X_tr = train.iloc[tr_idx][feature_cols]
    X_va = train.iloc[va_idx][feature_cols]
    y_tr = y_all[tr_idx]
    y_va = y_all[va_idx]

    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[va_idx] = from_train_pred(m.predict(X_va))
    models_lgb.append(m)

    try:
        mx = xgb.XGBRegressor(**xgb_params)
        mx.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    except Exception:
        xgb_fb = dict(xgb_params)
        xgb_fb['objective'] = 'reg:squarederror'
        mx = xgb.XGBRegressor(**xgb_fb)
        mx.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = from_train_pred(mx.predict(X_va))
    models_xgb.append(mx)

    mc = cb.CatBoostRegressor(**cat_params)
    mc.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        verbose=max(1, cat_params['iterations'] // 5),
        use_best_model=True,
    )
    oof_cat[va_idx] = from_train_pred(mc.predict(X_va))
    models_cat.append(mc)

    fold_mae = mae(train.iloc[va_idx][TARGET].values, (oof_lgb[va_idx] + oof_xgb[va_idx] + oof_cat[va_idx]) / 3)
    print(f"\n  ✔ Fold {fold}  avg(M3) MAE {fold_mae:.4f}")

mae_lgb = mae(train[TARGET].values, oof_lgb)
mae_xgb = mae(train[TARGET].values, oof_xgb)
mae_cat = mae(train[TARGET].values, oof_cat)
print(f"\n▶ OOF MAE — LGB {mae_lgb:.6f} | XGB {mae_xgb:.6f} | CAT {mae_cat:.6f}")

model_maes = {'lgb': mae_lgb, 'xgb': mae_xgb, 'cat': mae_cat}
oof_by_model = {'lgb': oof_lgb, 'xgb': oof_xgb, 'cat': oof_cat}

if RUN_ENSEMBLE_SEARCH:
    section("Ensemble search (models × p)")
    best_models = None
    best_p = None
    best_search_mae = float('inf')
    for models in _powerset_models():
        for p in P_GRID:
            pred_tmp = _ensemble_pred(oof_by_model, model_maes, models, p)
            m = mae(train[TARGET].values, pred_tmp)
            if m < best_search_mae:
                best_search_mae = float(m)
                best_models = list(models)
                best_p = float(p)
    print(f"▶ Best ensemble: models={best_models}  p={best_p}  OOF_MAE={best_search_mae:.6f}")
best_name = min(model_maes, key=model_maes.get)
worst_name = max(model_maes, key=model_maes.get)
best_mae = model_maes[best_name]
worst_mae = model_maes[worst_name]

use_models = ['lgb', 'xgb', 'cat']
if AUTO_DROP_WORST and worst_mae > best_mae * (1 + DROP_WORST_IF_GAP_GT):
    use_models.remove(worst_name)
    print(f"▶ drop worst model: {worst_name} (best={best_name})")

def weight(name):
    return 1.0 / (model_maes[name] ** WEIGHT_POWER)

w = {m: weight(m) for m in use_models}
w_sum = sum(w.values())

oof_ens = 0.0
if 'lgb' in use_models:
    oof_ens += w['lgb'] * oof_lgb
if 'xgb' in use_models:
    oof_ens += w['xgb'] * oof_xgb
if 'cat' in use_models:
    oof_ens += w['cat'] * oof_cat
oof_ens = oof_ens / w_sum

ens_mae = mae(train[TARGET].values, oof_ens)
print(f"▶ Ensemble OOF MAE (models={use_models}, p={WEIGHT_POWER}): {ens_mae:.6f}")


# ============================================================
# 5) Test 예측 + 제출
# ============================================================
section('Predict test + submit')
X_test = test[feature_cols]

p_lgb = np.mean([from_train_pred(m.predict(X_test)) for m in models_lgb], axis=0)
p_xgb = np.mean([from_train_pred(m.predict(X_test)) for m in models_xgb], axis=0)
p_cat = np.mean([from_train_pred(m.predict(X_test)) for m in models_cat], axis=0)

pred = 0.0
if 'lgb' in use_models:
    pred += w['lgb'] * p_lgb
if 'xgb' in use_models:
    pred += w['xgb'] * p_xgb
if 'cat' in use_models:
    pred += w['cat'] * p_cat
pred = pred / w_sum

pred = np.maximum(pred, CLIP_PRED_MIN)
pred_hi = float(np.percentile(train[TARGET].values, 100 * CLIP_PRED_MAX_Q))
pred = np.minimum(pred, pred_hi)

sub = pd.DataFrame({'ID': test['ID'], TARGET: pred})
save_path = os.path.join(project_root, 'submission_v10_base_only.csv')
sub.to_csv(save_path, index=False)
print(f"▶ saved -> {save_path}")
