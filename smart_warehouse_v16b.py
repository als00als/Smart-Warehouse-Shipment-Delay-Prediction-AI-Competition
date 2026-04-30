import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import time
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def elapsed(start):
    s = int(time.time() - start)
    return f"{s//60}m {s%60:02d}s"

def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")

class LGBProgressCallback:
    def __init__(self, fold, n_folds, total_rounds, log_every=200):
        self.fold = fold; self.n_folds = n_folds
        self.total = total_rounds; self.log_every = log_every
        self.start = time.time(); self.best_mae = float('inf'); self.best_iter = 0
    def __call__(self, env):
        it  = env.iteration + 1
        mae = env.evaluation_result_list[0][2]
        if mae < self.best_mae: self.best_mae = mae; self.best_iter = it
        if it % self.log_every == 0 or it == self.total:
            filled = int(30 * it / self.total)
            bar = '█'*filled + '░'*(30-filled)
            print(f"\r  Fold {self.fold}/{self.n_folds}  [{bar}] {it/self.total*100:5.1f}%"
                  f"  iter {it:>5}  val_MAE {mae:.4f}"
                  f"  best {self.best_mae:.4f}@{self.best_iter}  {elapsed(self.start)}",
                  end='', flush=True)

class XGBProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self, fold, n_folds, total_rounds, log_every=200):
        self.fold = fold; self.n_folds = n_folds
        self.total = total_rounds; self.log_every = log_every
        self.start = time.time(); self.best_mae = float('inf'); self.best_iter = 0
    def after_iteration(self, model, epoch, evals_log):
        it = epoch + 1
        try: mae = list(evals_log.values())[0]['mae'][-1]
        except: return False
        if mae < self.best_mae: self.best_mae = mae; self.best_iter = it
        if it % self.log_every == 0:
            filled = int(30 * it / self.total)
            bar = '█'*filled + '░'*(30-filled)
            print(f"\r  Fold {self.fold}/{self.n_folds}  [{bar}] {it/self.total*100:5.1f}%"
                  f"  iter {it:>5}  val_MAE {mae:.4f}"
                  f"  best {self.best_mae:.4f}@{self.best_iter}  {elapsed(self.start)}",
                  end='', flush=True)
        return False

# ============================================================
# 1. 데이터 로드
# ============================================================
path = 'C:/Users/82108/Downloads/open/'

print("▶ 데이터 로드 중...", end=' ', flush=True)
t0 = time.time()
train  = pd.read_csv(path + 'train.csv')
test   = pd.read_csv(path + 'test.csv')
layout = pd.read_csv(path + 'layout_info.csv')
print(f"완료 ({elapsed(t0)})  train {len(train):,}행 / test {len(test):,}행")

TARGET  = 'avg_delay_minutes_next_30m'
ID_COLS = ['ID', 'layout_id', 'scenario_id']
N_FOLDS = 5

# ============================================================
# 2. 결측치 처리
# ============================================================
def handle_missing_values(df):
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    cols_to_fix = [c for c in df.columns
                   if df[c].isnull().any() and c not in ['ID', 'scenario_id', 'layout_id']]
    df[cols_to_fix] = df.groupby('scenario_id')[cols_to_fix].ffill()
    df[cols_to_fix] = df.groupby('scenario_id')[cols_to_fix].bfill()
    df[cols_to_fix] = df[cols_to_fix].fillna(df[cols_to_fix].median())
    return df

# ============================================================
# 3. 피처 엔지니어링 (v15 동일)
# ============================================================
def add_features(df):
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)

    df['timeslot']          = df.groupby('scenario_id').cumcount()
    df['robot_efficiency']  = df['robot_active'] / (df['robot_total'] + 1e-6)
    df['robot_density']     = df['robot_total']  / (df['floor_area_sqm'] + 1e-6)
    df['order_per_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1e-6)
    df['robot_per_station'] = df['robot_active']  / (df['pack_station_count'] + 1e-6)
    df['cumulative_orders'] = df.groupby('scenario_id')['order_inflow_15m'].cumsum()
    df['order_pressure']    = df['cumulative_orders'] / (df['pack_station_count'] + 1e-6)

    if 'congestion_score' in df.columns:
        df['risk_index']  = df['congestion_score'] * (1 - df['robot_efficiency'])
        df['bottle_neck'] = df['order_per_station'] * df['congestion_score']
    if 'low_battery_ratio' in df.columns:
        df['battery_risk'] = df['low_battery_ratio'] * df['robot_total']
    if 'battery_mean' in df.columns and 'battery_std' in df.columns:
        df['battery_cv'] = df['battery_std'] / (df['battery_mean'] + 1e-6)
    if 'avg_trip_distance' in df.columns:
        df['trip_per_robot']  = df['avg_trip_distance'] / (df['robot_active'] + 1e-6)
        df['trip_congestion'] = df['avg_trip_distance'] * df.get('congestion_score', 0)
    if 'pack_utilization' in df.columns:
        df['pack_order_ratio'] = df['pack_utilization'] / (df['order_per_station'] + 1e-6)

    roll_cols = ['order_per_station', 'congestion_score',
                 'pack_utilization', 'avg_trip_distance', 'low_battery_ratio']
    for col in roll_cols:
        if col not in df.columns: continue
        grp = df.groupby('scenario_id')[col]
        df[f'{col}_roll3_mean'] = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df[f'{col}_roll5_mean'] = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df[f'{col}_roll3_std']  = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0))

    for lag in [1, 2]:
        if 'order_per_station' in df.columns:
            df[f'order_per_station_lag{lag}'] = (
                df.groupby('scenario_id')['order_per_station'].shift(lag).bfill())

    scen_agg_cols = list(dict.fromkeys([
        'congestion_score', 'order_inflow_15m', 'battery_mean',
        'pack_utilization', 'avg_trip_distance', 'low_battery_ratio',
        'max_zone_density', 'sku_concentration', 'robot_idle',
        'outbound_truck_wait_min', 'order_per_station', 'robot_efficiency',
        'order_pressure', 'risk_index', 'battery_risk', 'battery_cv',
        'zone_dispersion', 'avg_items_per_order', 'cold_chain_ratio',
        'manual_override_ratio', 'robot_total', 'battery_std',
        'bottle_neck', 'trip_congestion', 'pack_order_ratio',
        'robot_per_station',
    ]))
    for col in scen_agg_cols:
        if col not in df.columns: continue
        stats = df.groupby('scenario_id')[col].agg(['mean', 'max', 'min', 'std']).reset_index()
        stats.columns = ['scenario_id'] + [f'{col}_scen_{f}' for f in ['mean', 'max', 'min', 'std']]
        df = df.merge(stats, on='scenario_id', how='left')

    if 'congestion_score_scen_mean' in df.columns and 'pack_utilization_scen_mean' in df.columns:
        df['cong_pack_interaction']    = df['congestion_score_scen_mean'] * df['pack_utilization_scen_mean']
    if 'avg_trip_distance_scen_mean' in df.columns and 'congestion_score_scen_mean' in df.columns:
        df['trip_cong_interaction']    = df['avg_trip_distance_scen_mean'] * df['congestion_score_scen_mean']
    if 'low_battery_ratio_scen_mean' in df.columns and 'robot_efficiency_scen_mean' in df.columns:
        df['battery_efficiency_risk']  = df['low_battery_ratio_scen_mean'] * (1 - df['robot_efficiency_scen_mean'])
    if 'pack_utilization_scen_mean' in df.columns and 'avg_trip_distance_scen_mean' in df.columns:
        df['pack_trip_interaction']    = df['pack_utilization_scen_mean'] * df['avg_trip_distance_scen_mean']
    if 'low_battery_ratio_scen_mean' in df.columns and 'congestion_score_scen_mean' in df.columns:
        df['battery_cong_interaction'] = df['low_battery_ratio_scen_mean'] * df['congestion_score_scen_mean']
    if 'max_zone_density_scen_mean' in df.columns and 'congestion_score_scen_mean' in df.columns:
        df['density_cong_interaction'] = df['max_zone_density_scen_mean'] * df['congestion_score_scen_mean']

    return df

def preprocess_all(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')
    df = handle_missing_values(df)
    df = add_features(df)
    df['layout_type'] = pd.factorize(df['layout_type'])[0]
    return df

print("▶ 전처리 중...", end=' ', flush=True)
t0 = time.time()
train = preprocess_all(train, layout)
test  = preprocess_all(test,  layout)
print(f"완료 ({elapsed(t0)})")

# 하위 피처 제거 (v15 동일)
DROP_FEATURES = [
    'timeslot_ratio', 'task_reassign_15m', 'blocked_path_15m',
    'charge_queue_length', 'near_collision_15m', 'avg_charge_wait',
    'fault_count_15m', 'avg_recovery_time', 'congestion_score_diff1',
    'congestion_score_diff2', 'robot_efficiency_diff1', 'robot_efficiency_diff2',
    'congestion_score_lag2', 'robot_efficiency_lag1', 'congestion_score_roll3_std',
    'robot_efficiency_lag2',
]
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET] + DROP_FEATURES]
print(f"▶ 총 피처 수: {len(feature_cols)}개")

# ============================================================
# ★ 4. Stratified GroupKFold
# 시나리오별 타겟 평균을 기준으로 분위수 그룹을 만들어
# 각 fold에 낮은/중간/높은 지연 시나리오가 골고루 들어가도록 함
# → 특정 fold만 어려운 시나리오를 몰아받는 문제 해결
# ============================================================
print("▶ Stratified GroupKFold 구성 중...", end=' ', flush=True)

# 시나리오별 타겟 평균 계산
scen_target_mean = train.groupby('scenario_id')[TARGET].mean()

# 타겟 평균을 5분위로 나눠서 stratum 생성
scen_stratum = pd.qcut(scen_target_mean, q=5, labels=False)
train['_stratum'] = train['scenario_id'].map(scen_stratum)

# 각 stratum 내에서 GroupKFold → fold 인덱스 수동 생성
train['_fold'] = -1
for stratum_id in range(5):
    stratum_mask    = train['_stratum'] == stratum_id
    stratum_df      = train[stratum_mask]
    stratum_groups  = stratum_df['scenario_id']
    stratum_idx     = stratum_df.index

    inner_kf = GroupKFold(n_splits=N_FOLDS)
    for fold_i, (_, val_inner) in enumerate(
            inner_kf.split(stratum_df, stratum_df[TARGET], groups=stratum_groups)):
        real_val_idx = stratum_idx[val_inner]
        train.loc[real_val_idx, '_fold'] = fold_i

# fold 할당 검증
assert (train['_fold'] == -1).sum() == 0, "일부 행에 fold 미할당"
print("완료")

# fold별 타겟 평균 출력 (균등한지 확인)
print("  fold별 타겟 평균:")
for f in range(N_FOLDS):
    m = train.loc[train['_fold'] == f, TARGET].mean()
    print(f"    Fold {f+1}: {m:.3f}")

y      = np.log1p(train[TARGET])
groups = train['scenario_id']

# Stratified fold 인덱스 생성
fold_indices = []
for f in range(N_FOLDS):
    val_idx = train.index[train['_fold'] == f].tolist()
    tr_idx  = train.index[train['_fold'] != f].tolist()
    fold_indices.append((tr_idx, val_idx))

oof_lgb  = np.zeros(len(train))
oof_xgb  = np.zeros(len(train))
oof_cat  = np.zeros(len(train))
test_lgb = np.zeros(len(test))
test_xgb = np.zeros(len(test))
test_cat = np.zeros(len(test))

# ============================================================
# ★ 5. Optuna로 LightGBM 하이퍼파라미터 최적화
# 전체 학습 전에 빠른 탐색 (n_estimators 낮게, 2-fold로 빠르게)
# ============================================================
OPTUNA_TRIALS = 15
section(f"Optuna LightGBM 최적화 ({OPTUNA_TRIALS} trials, 샘플 40%)")

# 전체 데이터의 40%만 샘플링해서 탐색 속도 향상
sample_scens = train['scenario_id'].unique()
rng          = np.random.default_rng(42)
sample_scens = rng.choice(sample_scens, size=int(len(sample_scens)*0.4), replace=False)
sample_mask  = train['scenario_id'].isin(sample_scens)
train_sample = train[sample_mask].reset_index(drop=True)
y_sample     = np.log1p(train_sample[TARGET])

# 샘플 데이터 1-fold 인덱스
sample_kf    = GroupKFold(n_splits=3)
sample_folds = list(sample_kf.split(train_sample, y_sample,
                                     groups=train_sample['scenario_id']))

def lgb_objective(trial):
    params = dict(
        n_estimators      = 1500,
        learning_rate     = trial.suggest_float('learning_rate', 0.003, 0.02, log=True),
        max_depth         = trial.suggest_int('max_depth', 5, 10),
        num_leaves        = trial.suggest_int('num_leaves', 31, 255),
        min_child_samples = trial.suggest_int('min_child_samples', 10, 80),
        subsample         = trial.suggest_float('subsample', 0.6, 0.95),
        subsample_freq    = 1,
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 0.9),
        reg_alpha         = trial.suggest_float('reg_alpha', 0.0, 0.5),
        reg_lambda        = trial.suggest_float('reg_lambda', 0.0, 3.0),
        random_state      = 42,
        verbose           = -1,
    )
    tr_idx, val_idx = sample_folds[0]
    X_tr  = train_sample.iloc[tr_idx][feature_cols]
    y_tr  = y_sample.iloc[tr_idx]
    X_val = train_sample.iloc[val_idx][feature_cols]
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_sample.iloc[val_idx])],
          eval_metric='mae',
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return mean_absolute_error(
        train_sample.iloc[val_idx][TARGET], np.expm1(m.predict(X_val)))

optuna_start = time.time()
study = optuna.create_study(direction='minimize')

def print_callback(study, trial):
    print(f"  trial {trial.number+1:>2}/{OPTUNA_TRIALS}"
          f"  MAE {trial.value:.4f}"
          f"  best {study.best_value:.4f}"
          f"  {elapsed(optuna_start)}", flush=True)

study.optimize(lgb_objective, n_trials=OPTUNA_TRIALS,
               callbacks=[print_callback], show_progress_bar=False)

best_lgb_params = study.best_params
best_lgb_params.update({'n_estimators': 10000, 'subsample_freq': 1,
                         'random_state': 42, 'verbose': -1})
print(f"\n  최적 파라미터: {study.best_params}")
print(f"  Optuna 소요: {elapsed(optuna_start)}")

# ============================================================
# 6. LightGBM 학습 (최적 파라미터 사용)
# ============================================================
section("LightGBM 학습 (Optuna 파라미터)")

lgb_fold_maes = []
lgb_start = time.time()

for fold, (tr_idx, val_idx) in enumerate(fold_indices, 1):
    X_tr, y_tr   = train.loc[tr_idx, feature_cols], y.iloc[tr_idx]
    X_val, y_val = train.loc[val_idx, feature_cols], y.iloc[val_idx]
    cb_p  = LGBProgressCallback(fold, N_FOLDS, best_lgb_params['n_estimators'])
    model = lgb.LGBMRegressor(**best_lgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='mae',
              callbacks=[lgb.early_stopping(400, verbose=False), lgb.log_evaluation(-1), cb_p])
    fold_pred        = np.expm1(model.predict(X_val))
    oof_lgb[val_idx] = fold_pred
    test_lgb        += np.expm1(model.predict(test[feature_cols])) / N_FOLDS
    fold_mae = mean_absolute_error(train.loc[val_idx, TARGET], fold_pred)
    lgb_fold_maes.append(fold_mae)
    print(f"\n  ✔ Fold {fold}  MAE {fold_mae:.4f}  best iter {model.best_iteration_:,}  {elapsed(lgb_start)}")

lgb_mae = mean_absolute_error(train[TARGET], oof_lgb)
print(f"\n  ▶ LightGBM OOF MAE : {lgb_mae:.4f}  ({elapsed(lgb_start)})")
print(f"    Fold별: {[f'{m:.4f}' for m in lgb_fold_maes]}")
print(f"    편차:   {np.std(lgb_fold_maes):.4f}")

# ============================================================
# 7. XGBoost 학습
# ============================================================
section("XGBoost 학습")

xgb_params = dict(
    n_estimators=10000, learning_rate=best_lgb_params.get('learning_rate', 0.005),
    max_depth=best_lgb_params.get('max_depth', 7),
    subsample=best_lgb_params.get('subsample', 0.75),
    colsample_bytree=best_lgb_params.get('colsample_bytree', 0.6),
    reg_alpha=best_lgb_params.get('reg_alpha', 0.1),
    reg_lambda=best_lgb_params.get('reg_lambda', 1.0),
    random_state=42, tree_method='hist',
    early_stopping_rounds=400, eval_metric='mae', verbosity=0,
)

xgb_fold_maes = []
xgb_start = time.time()

for fold, (tr_idx, val_idx) in enumerate(fold_indices, 1):
    X_tr, y_tr   = train.loc[tr_idx, feature_cols], y.iloc[tr_idx]
    X_val, y_val = train.loc[val_idx, feature_cols], y.iloc[val_idx]
    cb_p  = XGBProgressCallback(fold, N_FOLDS, 10000)
    model = xgb.XGBRegressor(**xgb_params, callbacks=[cb_p])
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    fold_pred        = np.expm1(model.predict(X_val))
    oof_xgb[val_idx] = fold_pred
    test_xgb        += np.expm1(model.predict(test[feature_cols])) / N_FOLDS
    fold_mae = mean_absolute_error(train.loc[val_idx, TARGET], fold_pred)
    xgb_fold_maes.append(fold_mae)
    print(f"\n  ✔ Fold {fold}  MAE {fold_mae:.4f}  best iter {model.best_iteration:,}  {elapsed(xgb_start)}")

xgb_mae = mean_absolute_error(train[TARGET], oof_xgb)
print(f"\n  ▶ XGBoost OOF MAE  : {xgb_mae:.4f}  ({elapsed(xgb_start)})")
print(f"    Fold별: {[f'{m:.4f}' for m in xgb_fold_maes]}")
print(f"    편차:   {np.std(xgb_fold_maes):.4f}")

# ============================================================
# 8. CatBoost 학습
# ============================================================
section("CatBoost 학습")

cat_params = dict(
    iterations=10000,
    learning_rate=best_lgb_params.get('learning_rate', 0.005),
    depth=min(best_lgb_params.get('max_depth', 8), 10),
    l2_leaf_reg=3.0, bootstrap_type='MVS',
    subsample=best_lgb_params.get('subsample', 0.75),
    colsample_bylevel=best_lgb_params.get('colsample_bytree', 0.6),
    loss_function='MAE', eval_metric='MAE',
    random_seed=42, task_type='CPU',
    early_stopping_rounds=400,
)

cat_fold_maes = []
cat_start = time.time()

for fold, (tr_idx, val_idx) in enumerate(fold_indices, 1):
    X_tr, y_tr   = train.loc[tr_idx, feature_cols], y.iloc[tr_idx]
    X_val, y_val = train.loc[val_idx, feature_cols], y.iloc[val_idx]
    print(f"\n  Fold {fold}/{N_FOLDS} 학습 중...")
    model = cb.CatBoostRegressor(**cat_params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
              verbose=2000, use_best_model=True)
    fold_pred        = np.expm1(model.predict(X_val))
    oof_cat[val_idx] = fold_pred
    test_cat        += np.expm1(model.predict(test[feature_cols])) / N_FOLDS
    fold_mae = mean_absolute_error(train.loc[val_idx, TARGET], fold_pred)
    cat_fold_maes.append(fold_mae)
    print(f"  ✔ Fold {fold}  MAE {fold_mae:.4f}  best iter {model.best_iteration_:,}  {elapsed(cat_start)}")

cat_mae = mean_absolute_error(train[TARGET], oof_cat)
print(f"\n  ▶ CatBoost OOF MAE : {cat_mae:.4f}  ({elapsed(cat_start)})")
print(f"    Fold별: {[f'{m:.4f}' for m in cat_fold_maes]}")
print(f"    편차:   {np.std(cat_fold_maes):.4f}")

# ============================================================
# 9. 앙상블 가중치 Grid Search
# ============================================================
section("앙상블 가중치 최적화")

best_mae_blend = float('inf')
best_w = (0.33, 0.33, 0.34)

step = 0.05
for wl in np.arange(0.1, 0.8, step):
    for wx in np.arange(0.1, 0.8, step):
        wc = round(1.0 - wl - wx, 4)
        if wc < 0.05: continue
        blend = wl*oof_lgb + wx*oof_xgb + wc*oof_cat
        m = mean_absolute_error(train[TARGET], blend)
        if m < best_mae_blend:
            best_mae_blend = m
            best_w = (wl, wx, wc)

wl_opt, wx_opt, wc_opt = best_w
print(f"  최적 가중치 → LGB {wl_opt:.2f} / XGB {wx_opt:.2f} / CAT {wc_opt:.2f}")

# ============================================================
# 10. 제출
# ============================================================
section("제출")

oof_final  = wl_opt*oof_lgb  + wx_opt*oof_xgb  + wc_opt*oof_cat
test_final = wl_opt*test_lgb + wx_opt*test_xgb + wc_opt*test_cat
final_mae  = mean_absolute_error(train[TARGET], oof_final)

print(f"  LightGBM {wl_opt*100:.1f}%  OOF MAE {lgb_mae:.4f}  편차 {np.std(lgb_fold_maes):.4f}")
print(f"  XGBoost  {wx_opt*100:.1f}%  OOF MAE {xgb_mae:.4f}  편차 {np.std(xgb_fold_maes):.4f}")
print(f"  CatBoost {wc_opt*100:.1f}%  OOF MAE {cat_mae:.4f}  편차 {np.std(cat_fold_maes):.4f}")
print(f"\n  ★ 앙상블 최종 OOF MAE : {final_mae:.4f}")
print(f"  ★ 전체 소요 시간       : {elapsed(lgb_start)}")

submission = pd.DataFrame({'ID': test['ID'], TARGET: test_final})
submission.to_csv(path + 'submission_v16b.csv', index=False)
print(f"\n  저장 완료 → submission_v16.csv")
