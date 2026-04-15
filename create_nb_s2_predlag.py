"""
Build 스마트 창고 출고 지연 예측_0415_preproc_s2_predlag_exp.ipynb from
스마트 창고 출고 지연 예측_0415_preproc_speedup_exp.ipynb

Stage2 changes:
- No true-target lag columns; skip cell 16 engineering when USE_S2_PRED_LAG_ONLY.
- After Stage1, add PRED_LAG_COLS from shifted Stage1 OOF / test predictions.
- Stage2 OOF: direct predict on validation fold (pred-lag features are OOF-safe
  under GroupKFold by scenario_id).
- REPLACE_STAGE2_IF_NOT_IMPROVING = False so S2 can contribute to blend even if
  slightly worse than S1 (revisit after metrics).
- Slightly lighter Stage2 hyperparameters + CatBoost quieter logging.
"""
import copy
import glob
import json
from pathlib import Path


def _to_source_lines(text: str):
    lines = text.split("\n")
    if not lines:
        return []
    out = [f"{line}\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def _append_to_cell(nb, idx: int, suffix: str):
    cur = "".join(nb["cells"][idx].get("source", []))
    nb["cells"][idx]["source"] = _to_source_lines(cur + suffix)


ADD_PRED_LAG_FN = '''
def add_pred_lag_features(df, pred_vec, gm):
    """Scenario-wise lags of Stage1 predictions (OOF or test). gm fills cold-start."""
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    p = np.asarray(pred_vec, dtype=np.float64)
    if len(p) != len(df):
        raise ValueError(f'add_pred_lag_features: len(pred_vec)={len(p)} != len(df)={len(df)}')
    df['_pred_for_lag'] = p
    g = df.groupby('scenario_id')['_pred_for_lag']
    df['pred_lag1'] = g.shift(1)
    df['pred_lag2'] = g.shift(2)
    df['pred_lag3'] = g.shift(3)
    s1, s2, s3 = g.shift(1), g.shift(2), g.shift(3)
    df['pred_diff1'] = (s1 - s2).fillna(0.0)
    df['pred_diff2'] = (s2 - s3).fillna(0.0)
    df['pred_roll3_mean'] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['pred_roll5_mean'] = g.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['pred_ewm3'] = g.transform(lambda x: x.shift(1).ewm(alpha=0.3, adjust=False).mean())
    df['pred_lag1_log'] = np.log1p(np.maximum(df['pred_lag1'].fillna(gm).astype(np.float64), 0.0))
    for c in ['pred_lag1', 'pred_lag2', 'pred_lag3', 'pred_roll3_mean', 'pred_roll5_mean', 'pred_ewm3']:
        df[c] = df[c].fillna(gm)
    df.drop(columns=['_pred_for_lag'], inplace=True)
    return df


'''

CELL_6_REPLACEMENT = """TARGET = 'avg_delay_minutes_next_30m'
ID_COLS = ['ID', 'layout_id', 'scenario_id']

# Full true-target lag column names (used only if USE_S2_PRED_LAG_ONLY == False)
TRUE_LAG_COLS_LEGACY = [
    'target_lag1', 'target_lag2', 'target_lag3', 'target_lag4', 'target_lag5',
    'target_roll3_mean', 'target_roll5_mean', 'target_roll10_mean',
    'target_ewm3', 'target_ewm5',
    'target_diff1', 'target_diff2',
    'target_lag1_log',
    'target_lag_max3', 'target_lag_min3', 'target_lag_std3',
]

TRUE_LAG_COLS = [] if USE_S2_PRED_LAG_ONLY else TRUE_LAG_COLS_LEGACY

PRED_LAG_COLS = [
    'pred_lag1', 'pred_lag2', 'pred_lag3',
    'pred_diff1', 'pred_diff2',
    'pred_roll3_mean', 'pred_roll5_mean',
    'pred_ewm3',
    'pred_lag1_log',
]
"""

CELL_16_REPLACEMENT = """section('True Target Lag Feature Engineering (train only)')
t0 = time.time()
if USE_S2_PRED_LAG_ONLY:
    print('▶ Skipped true-target lag features (Stage2 uses Stage1 pred-lag columns only)')
else:
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

    for col in TRUE_LAG_COLS_LEGACY:
        if col in train.columns:
            train[col] = train[col].fillna(global_mean)

    print(f"  target_lag1 corr with target: {train['target_lag1'].corr(train[TARGET]):.4f}")
    print(f"  target_lag2 corr with target: {train['target_lag2'].corr(train[TARGET]):.4f}")
    print(f"  target_ewm3 corr with target: {train['target_ewm3'].corr(train[TARGET]):.4f}")
    print(f"  TRUE_LAG_COLS_LEGACY count: {len(TRUE_LAG_COLS_LEGACY)}")

print(f"▶ True target lag block done ({elapsed(t0)})")
"""

STAGE1_APPEND = """

# --- Stage2 pred-lag features (from Stage1 OOF; scenario-safe under group CV) ---
if USE_S2_PRED_LAG_ONLY:
    train = add_pred_lag_features(train, oof_s1, global_mean)
    print(f"▶ Added train pred-lag cols: {PRED_LAG_COLS}  (n={len(PRED_LAG_COLS)})")
"""

CELL_29_REPLACEMENT = """feature_cols_s2 = feature_cols_s1 + (PRED_LAG_COLS if USE_S2_PRED_LAG_ONLY else TRUE_LAG_COLS)

lgb_params_s2 = dict(
    objective='regression_l1',
    n_estimators=12000,
    learning_rate=0.02,
    max_depth=-1,
    num_leaves=511,
    min_child_samples=80,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.65,
    reg_alpha=0.15,
    reg_lambda=4.0,
    random_state=SEED,
    verbose=-1,
)

xgb_params_s2 = dict(
    objective='reg:absoluteerror',
    n_estimators=8000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.65,
    colsample_bynode=0.5,
    reg_alpha=0.15,
    reg_lambda=2.5,
    random_state=SEED,
    tree_method='hist',
    eval_metric='mae',
    early_stopping_rounds=120,
    verbosity=0,
)

cat_params_s2 = dict(
    iterations=8000,
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=4.0,
    bootstrap_type='MVS',
    subsample=0.8,
    colsample_bylevel=0.65,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=SEED,
    task_type='CPU',
    early_stopping_rounds=120,
)

section('Stage 2 - Pred-lag stack (LGB + XGB + CAT), fold OOF')
t0 = time.time()

if USE_S2_PRED_LAG_ONLY and all(c in train.columns for c in PRED_LAG_COLS):
    pl = train['pred_lag1'].values
    print(f"  pred_lag1 stats: mean={np.nanmean(pl):.2f}  std={np.nanstd(pl):.2f}")
    print(f"  pred_lag1 corr with target: {np.corrcoef(np.nan_to_num(pl), y_raw)[0,1]:.4f}")
else:
    print(f"  target_lag1 stats: mean={train['target_lag1'].mean():.2f}  std={train['target_lag1'].std():.2f}")
    print(f"  target_lag1 corr with target: {np.corrcoef(train['target_lag1'], y_raw)[0,1]:.4f}")

if USE_S2_PRED_LAG_ONLY:
    print('  [Pred-lag S2] OOF = direct predict on each fold val (lags from Stage1 OOF).')
else:
    print('  [Method B] OOF는 slotwise sequential 예측으로 평가 → OOF-LB 갭 제거')

oof_s2_lgb = np.zeros(len(train))
oof_s2_xgb = np.zeros(len(train))
oof_s2_cat = np.zeros(len(train))
models_s2_lgb, models_s2_xgb, models_s2_cat = [], [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):
    print(f"\\n  --- Stage 2 Fold {fold} ---")

    X_tr = train.iloc[tr_idx][feature_cols_s2]
    y_tr = y_all[tr_idx]

    X_va_real = train.iloc[va_idx][feature_cols_s2]
    y_va_real = y_all[va_idx]

    m_lgb = lgb.LGBMRegressor(**lgb_params_s2)
    m_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va_real, y_va_real)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(-1)],
    )
    models_s2_lgb.append(m_lgb)

    try:
        m_xgb = xgb.XGBRegressor(**xgb_params_s2)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va_real, y_va_real)], verbose=False)
    except Exception:
        xgb_fb = dict(xgb_params_s2)
        xgb_fb['objective'] = 'reg:squarederror'
        m_xgb = xgb.XGBRegressor(**xgb_fb)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va_real, y_va_real)], verbose=False)
    models_s2_xgb.append(m_xgb)

    m_cat = cb.CatBoostRegressor(**cat_params_s2)
    m_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va_real, y_va_real),
        verbose=False,
        use_best_model=True,
    )
    models_s2_cat.append(m_cat)

    if USE_S2_PRED_LAG_ONLY:
        X_va = train.iloc[va_idx][feature_cols_s2]
        oof_s2_lgb[va_idx] = from_train_pred(m_lgb.predict(X_va))
        oof_s2_xgb[va_idx] = from_train_pred(m_xgb.predict(X_va))
        oof_s2_cat[va_idx] = from_train_pred(m_cat.predict(X_va))
    else:
        va_df = train.iloc[va_idx].sort_values(['scenario_id', 'timeslot'])
        lgb_seq = sequential_predict(va_df, [m_lgb], feature_cols_s2, global_mean, from_train_pred)
        for orig_pos, pred in zip(va_df.index, lgb_seq):
            oof_s2_lgb[orig_pos] = pred
        xgb_seq = sequential_predict(va_df, [m_xgb], feature_cols_s2, global_mean, from_train_pred)
        for orig_pos, pred in zip(va_df.index, xgb_seq):
            oof_s2_xgb[orig_pos] = pred
        cat_seq = sequential_predict(va_df, [m_cat], feature_cols_s2, global_mean, from_train_pred)
        for orig_pos, pred in zip(va_df.index, cat_seq):
            oof_s2_cat[orig_pos] = pred

    avg3 = (oof_s2_lgb[va_idx] + oof_s2_xgb[va_idx] + oof_s2_cat[va_idx]) / 3
    tag = 'fold val' if USE_S2_PRED_LAG_ONLY else 'sequential'
    print(f"    LGB MAE ({tag}): {mae(y_raw[va_idx], oof_s2_lgb[va_idx]):.6f}")
    print(f"    XGB MAE ({tag}): {mae(y_raw[va_idx], oof_s2_xgb[va_idx]):.6f}")
    print(f"    CAT MAE ({tag}): {mae(y_raw[va_idx], oof_s2_cat[va_idx]):.6f}")
    print(f"    AVG MAE ({tag}): {mae(y_raw[va_idx], avg3):.6f}")

mae_s2_lgb = mae(y_raw, oof_s2_lgb)
mae_s2_xgb = mae(y_raw, oof_s2_xgb)
mae_s2_cat = mae(y_raw, oof_s2_cat)
_oot = 'Pred-lag fold OOF' if USE_S2_PRED_LAG_ONLY else 'Method B'
print(f"\\n▶ Stage 2 OOF MAE ({_oot}) - LGB {mae_s2_lgb:.6f} | XGB {mae_s2_xgb:.6f} | CAT {mae_s2_cat:.6f}")


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

w_s2 = {m: 1.0 / (s2_model_maes[m] ** best_s2_p) for m in best_s2_models}
ws_s2 = sum(w_s2.values())
oof_s2_ens = sum(w_s2[m] * s2_oof_by[m] for m in best_s2_models) / ws_s2
s2_ens_mae = mae(y_raw, oof_s2_ens)
print(f"▶ Stage 2 Ensemble OOF MAE: {s2_ens_mae:.6f}")

print(f"\\n▶ Stage 1 OOF MAE: {s1_mae:.6f}")
print(f"▶ Stage 2 OOF MAE: {s2_ens_mae:.6f}")
print(f"▶ Improvement: {s1_mae - s2_ens_mae:.6f}")
oof_s2_used = oof_s2_ens
use_s1_instead_of_s2 = False
if REPLACE_STAGE2_IF_NOT_IMPROVING and s2_ens_mae >= s1_mae:
    print('▶ REPLACE: Stage2 OOF >= Stage1 → 블렌드/제출은 S2 대신 S1 OOF 사용')
    oof_s2_used = oof_s1.copy()
    use_s1_instead_of_s2 = True
"""


CELL_38_REPLACEMENT = """section('Predict test + submit (pred-lag S2: direct predict on test)')

test = test.sort_values(['scenario_id', 'timeslot', 'ID']).reset_index(drop=True)
if not USE_S2_PRED_LAG_ONLY:
    for col in TRUE_LAG_COLS:
        test[col] = global_mean

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

if USE_S2_PRED_LAG_ONLY:
    test = add_pred_lag_features(test, pred_s1_test, global_mean)
    print(f"  Stage 2 test: added pred-lag columns ({len(PRED_LAG_COLS)})")

print(f"  Stage 2 test prediction (direct, ensemble of {N_FOLDS} folds)...")
X_test_s2 = test[feature_cols_s2]
pred_s2_lgb_test = np.mean([from_train_pred(m.predict(X_test_s2)) for m in models_s2_lgb], axis=0)
pred_s2_xgb_test = np.mean([from_train_pred(m.predict(X_test_s2)) for m in models_s2_xgb], axis=0)
pred_s2_cat_test = np.mean([from_train_pred(m.predict(X_test_s2)) for m in models_s2_cat], axis=0)

pred_s2_test = sum(
    w_s2[m] * {'lgb': pred_s2_lgb_test, 'xgb': pred_s2_xgb_test, 'cat': pred_s2_cat_test}[m]
    for m in best_s2_models
) / ws_s2
print(f"  Stage 2 test pred: mean={pred_s2_test.mean():.2f}  std={pred_s2_test.std():.2f}")

pred_s2_test_blend = pred_s1_test if use_s1_instead_of_s2 else pred_s2_test

pred_blend_test = best_alpha * pred_s1_test + (1 - best_alpha) * pred_s2_test_blend

pred = np.maximum(pred_blend_test, CLIP_PRED_MIN)
pred_hi = float(np.percentile(y_raw, 100 * CLIP_PRED_MAX_Q))
pred = np.minimum(pred, pred_hi)

sub = pd.DataFrame({'ID': test['ID'], TARGET: pred})
save_path = os.path.join(project_root, 'submission_s2_predlag_exp.csv')
sub.to_csv(save_path, index=False)
print(f"▶ saved -> {save_path}")
print(f"\\n▶▶ DONE - FINAL OOF MAE: {best_final_mae:.6f}")
"""


def main():
    src = glob.glob("*0415_preproc_speedup_exp.ipynb")[0]
    with open(src, "r", encoding="utf-8") as f:
        nb = json.load(f)
    new_nb = copy.deepcopy(nb)

    # cell 2: flags
    c2 = "".join(new_nb["cells"][2]["source"])
    if "USE_S2_PRED_LAG_ONLY" not in c2:
        c2 = c2.replace(
            "REPLACE_STAGE2_IF_NOT_IMPROVING = True",
            "REPLACE_STAGE2_IF_NOT_IMPROVING = False  # allow S2 in blend; revisit after OOF/LB\nUSE_S2_PRED_LAG_ONLY = True  # Stage2 uses Stage1 pred-lag features only",
            1,
        )
        new_nb["cells"][2]["source"] = _to_source_lines(c2)

    new_nb["cells"][6]["source"] = _to_source_lines(CELL_6_REPLACEMENT)
    new_nb["cells"][8]["source"] = _to_source_lines(ADD_PRED_LAG_FN + "".join(new_nb["cells"][8]["source"]))
    new_nb["cells"][16]["source"] = _to_source_lines(CELL_16_REPLACEMENT)
    _append_to_cell(new_nb, 23, STAGE1_APPEND)
    new_nb["cells"][29]["source"] = _to_source_lines(CELL_29_REPLACEMENT)
    new_nb["cells"][38]["source"] = _to_source_lines(CELL_38_REPLACEMENT)

    # feature_cols_s1 must not exclude pred cols before they exist; TRUE_LAG empty OK
    dst = Path(src).with_name(
        Path(src).name.replace("_preproc_speedup_exp.ipynb", "_preproc_s2_predlag_exp.ipynb")
    )
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(new_nb, f, ensure_ascii=False, indent=1)
    print(dst)


if __name__ == "__main__":
    main()
