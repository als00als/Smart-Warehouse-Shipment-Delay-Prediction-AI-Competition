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
# 실행 파라미터
# ============================================================
N_FOLDS = 5
SEED = 42
USE_LOG_TARGET = False

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


def preprocess_all(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')
    df = handle_missing_values(df)
    df = add_basic_features(df)
    df = add_timeseries_features(df)
    df = add_interaction_features(df)
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
global_mean = train[TARGET].mean()


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
# 4) Stage 1: Base Model (target lag 없음)
# ============================================================
feature_cols_s1 = [c for c in train.columns if c not in ID_COLS + [TARGET]]
print(f"▶ Stage 1 features: {len(feature_cols_s1)}")

y_all = to_train_target(train[TARGET].values)
y_raw = train[TARGET].values
groups = train['scenario_id'].values
kf = GroupKFold(n_splits=N_FOLDS)

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
t0 = time.time()
oof_s1_lgb = np.zeros(len(train))
oof_s1_xgb = np.zeros(len(train))
oof_s1_cat = np.zeros(len(train))
models_s1_lgb, models_s1_xgb, models_s1_cat = [], [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y_all, groups=groups), 1):
    X_tr = train.iloc[tr_idx][feature_cols_s1]
    X_va = train.iloc[va_idx][feature_cols_s1]
    y_tr = y_all[tr_idx]
    y_va = y_all[va_idx]

    m_lgb = lgb.LGBMRegressor(**lgb_params_s1)
    m_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_s1_lgb[va_idx] = from_train_pred(m_lgb.predict(X_va))
    models_s1_lgb.append(m_lgb)

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
s1_mae = mae(y_raw, oof_s1_pre)
print(f"▶ Stage 1 ensemble OOF MAE: {s1_mae:.6f}  ({elapsed(t0)})")

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
for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y_all, groups=groups), 1):
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
s1_after_resid = mae(y_raw, oof_s1)
print(f"\n▶ Stage 1 after residual OOF MAE: {s1_after_resid:.6f}  ({elapsed(t1b)})")

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
    df는 scenario_id, ID 순으로 정렬되어 있어야 함."""
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

section('Stage 2 - Lag-enhanced model (LGB + XGB + CAT)')
t0 = time.time()

train['_s1_pred'] = oof_s1
train = build_pred_lag_features(train, '_s1_pred', global_mean)

print(f"  pred_lag1 stats: mean={train['pred_lag1'].mean():.2f}  std={train['pred_lag1'].std():.2f}")
print(f"  pred_lag1 corr with target: {np.corrcoef(train['pred_lag1'], y_raw)[0,1]:.4f}")

oof_s2_lgb = np.zeros(len(train))
oof_s2_xgb = np.zeros(len(train))
oof_s2_cat = np.zeros(len(train))
models_s2_lgb, models_s2_xgb, models_s2_cat = [], [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y_all, groups=groups), 1):
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
mae_s2_xgb = mae(y_raw, oof_s2_xgb)
mae_s2_cat = mae(y_raw, oof_s2_cat)
print(f"\n▶ Stage 2 OOF MAE - LGB {mae_s2_lgb:.6f} | XGB {mae_s2_xgb:.6f} | CAT {mae_s2_cat:.6f}")


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


# ============================================================
# 8) Stage1 vs Stage2 블렌딩 탐색
# ============================================================
section('Stage 1 + Stage 2 blending')
best_alpha = 0.0
best_blend_mae = s2_ens_mae
for alpha in np.arange(0.0, 1.01, 0.05):
    blend_pred = alpha * oof_s1 + (1 - alpha) * oof_s2_ens
    m_val = mae(y_raw, blend_pred)
    if m_val < best_blend_mae:
        best_blend_mae = float(m_val)
        best_alpha = float(alpha)
    if abs(alpha - 0.0) < 0.01 or abs(alpha - 1.0) < 0.01 or abs(alpha - 0.5) < 0.01:
        print(f"  alpha={alpha:.2f} MAE={m_val:.6f}")

print(f"\n▶ Best blend: alpha={best_alpha:.2f} (S1 weight)  MAE={best_blend_mae:.6f}")

# ============================================================
# 8-B) Timeslot expert heads (slot0 vs slot1+)
# ============================================================
section('Timeslot expert heads')

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

for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y_all, groups=groups), 1):
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
best_final_oof = best_alpha * oof_s1 + (1 - best_alpha) * oof_s2_ens
best_final_mae = best_blend_mae

if expert_mae < best_final_mae:
    best_final_name = "expert_only"
    best_final_oof = oof_expert
    best_final_mae = expert_mae

best_mix_w = 0.0
for w_mix in np.arange(0.0, 1.01, 0.05):
    mix_pred = w_mix * oof_expert + (1 - w_mix) * (best_alpha * oof_s1 + (1 - best_alpha) * oof_s2_ens)
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
p_s2_xgb = np.mean([from_train_pred(m.predict(X_test_s2)) for m in models_s2_xgb], axis=0)
p_s2_cat = np.mean([from_train_pred(m.predict(X_test_s2)) for m in models_s2_cat], axis=0)

# Stage 2 앙상블
pred_s2_test = sum(w_s2[m] * {'lgb': p_s2_lgb, 'xgb': p_s2_xgb, 'cat': p_s2_cat}[m]
                   for m in best_s2_models) / ws_s2

# 기존 최적 blend 예측
pred_blend = best_alpha * pred_s1_test + (1 - best_alpha) * pred_s2_test

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

if best_final_name == "expert_only":
    pred = pred_expert
elif best_final_name == "expert_mix":
    pred = best_mix_w * pred_expert + (1 - best_mix_w) * pred_blend
else:
    pred = pred_blend

# 후처리
pred = np.maximum(pred, CLIP_PRED_MIN)
pred_hi = float(np.percentile(y_raw, 100 * CLIP_PRED_MAX_Q))
pred = np.minimum(pred, pred_hi)

sub = pd.DataFrame({'ID': test['ID'], TARGET: pred})
save_path = os.path.join(project_root, 'submission_v11_2stage.csv')
sub.to_csv(save_path, index=False)
print(f"▶ saved -> {save_path}")
print(f"\n▶▶ DONE - FINAL OOF MAE: {final_oof_mae:.6f}")
