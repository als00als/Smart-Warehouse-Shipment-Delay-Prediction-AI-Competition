# -*- coding: utf-8 -*-
"""Build v4_fulllevers notebook from v3_refine."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def patch_preprocess(cell_src: str) -> str:
    if "SCENARIO_AGG_COLS = [" in cell_src:
        return cell_src
    anchor = (
        "SCEN_AGG_STATS = ('mean', 'max', 'min', 'std')\n\n\n"
        "def add_missing_indicators(df):\n"
    )
    helpers = (
        "SCENARIO_AGG_COLS = [\n"
        "    'congestion_score', 'order_inflow_15m', 'battery_mean', 'pack_utilization',\n"
        "    'avg_trip_distance', 'low_battery_ratio', 'max_zone_density', 'sku_concentration',\n"
        "    'robot_idle', 'outbound_truck_wait_min', 'order_per_station', 'robot_efficiency',\n"
        "    'order_pressure', 'risk_index', 'battery_risk', 'battery_cv',\n"
        "]\n\n\n"
        "def _scenario_agg_present(df):\n"
        "    return [c for c in SCENARIO_AGG_COLS if c in df.columns]\n\n\n"
        "def ensure_scenario_agg_placeholders(train_df, test_df):\n"
        "    if not USE_FOLD_SAFE_SCENARIO_AGG:\n"
        "        return\n"
        "    for d in (train_df, test_df):\n"
        "        present = _scenario_agg_present(d)\n"
        "        for c in present:\n"
        "            for st in SCEN_AGG_STATS:\n"
        "                col = f'{c}_scen_{st}'\n"
        "                if col not in d.columns:\n"
        "                    d[col] = np.float64(0.0)\n"
        "        for col in ['congestion_score', 'order_per_station', 'pack_utilization', 'avg_trip_distance']:\n"
        "            sm = f'{col}_scen_mean'\n"
        "            rel = f'{col}_rel_to_scen'\n"
        "            if col in d.columns and sm in d.columns and rel not in d.columns:\n"
        "                d[rel] = np.float64(0.0)\n"
        "        for col in ['congestion_score', 'order_per_station', 'pack_utilization']:\n"
        "            rk = f'{col}_scen_rank'\n"
        "            if col in d.columns and rk not in d.columns:\n"
        "                d[rk] = np.float64(0.0)\n\n\n"
        "def merge_scenario_agg_from_reference(ref_df, df):\n"
        "    present = [c for c in SCENARIO_AGG_COLS if c in ref_df.columns and c in df.columns]\n"
        "    if not present:\n"
        "        return df\n"
        "    g = ref_df.groupby('scenario_id')[present].agg(list(SCEN_AGG_STATS))\n"
        "    g.columns = [f'{a}_scen_{b}' for a, b in g.columns]\n"
        "    mp = g.reset_index().set_index('scenario_id')\n"
        "    for col in mp.columns:\n"
        "        base = col.rsplit('_scen_', 1)[0]\n"
        "        fill = float(ref_df[base].mean())\n"
        "        ser = df['scenario_id'].map(mp[col])\n"
        "        df[col] = ser.astype(np.float64).fillna(fill)\n"
        "    for col in ['congestion_score', 'order_per_station', 'pack_utilization', 'avg_trip_distance']:\n"
        "        sm = f'{col}_scen_mean'\n"
        "        if col in df.columns and sm in df.columns:\n"
        "            df[f'{col}_rel_to_scen'] = df[col] / (df[sm] + 1e-6)\n"
        "    for col in ['congestion_score', 'order_per_station', 'pack_utilization']:\n"
        "        if col in df.columns:\n"
        "            df[f'{col}_scen_rank'] = df.groupby('scenario_id')[col].rank(pct=True)\n"
        "    return df\n\n\n"
        "def _inject_fold_safe_scenario(tr_df, va_df):\n"
        "    if not USE_FOLD_SAFE_SCENARIO_AGG:\n"
        "        return tr_df, va_df\n"
        "    present = _scenario_agg_present(tr_df)\n"
        "    if not present:\n"
        "        return tr_df, va_df\n"
        "    g = tr_df.groupby('scenario_id')[present].agg(list(SCEN_AGG_STATS))\n"
        "    g.columns = [f'{a}_scen_{b}' for a, b in g.columns]\n"
        "    mp = g.reset_index().set_index('scenario_id')\n"
        "    for col in mp.columns:\n"
        "        base = col.rsplit('_scen_', 1)[0]\n"
        "        fill = float(tr_df[base].mean())\n"
        "        tr_df[col] = tr_df['scenario_id'].map(mp[col]).astype(np.float64).fillna(fill)\n"
        "        va_df[col] = va_df['scenario_id'].map(mp[col]).astype(np.float64).fillna(fill)\n"
        "    for col in ['congestion_score', 'order_per_station', 'pack_utilization', 'avg_trip_distance']:\n"
        "        sm = f'{col}_scen_mean'\n"
        "        if col in tr_df.columns and sm in tr_df.columns:\n"
        "            tr_df[f'{col}_rel_to_scen'] = tr_df[col] / (tr_df[sm] + 1e-6)\n"
        "            va_df[f'{col}_rel_to_scen'] = va_df[col] / (va_df[sm] + 1e-6)\n"
        "    comb = pd.concat([tr_df, va_df])\n"
        "    for col in ['congestion_score', 'order_per_station', 'pack_utilization']:\n"
        "        if col not in comb.columns:\n"
        "            continue\n"
        "        rk = comb.groupby('scenario_id')[col].rank(pct=True)\n"
        "        tr_df[f'{col}_scen_rank'] = rk.loc[tr_df.index].to_numpy()\n"
        "        va_df[f'{col}_scen_rank'] = rk.loc[va_df.index].to_numpy()\n"
        "    return tr_df, va_df\n\n\n"
    )
    if anchor not in cell_src:
        raise RuntimeError("preprocess anchor not found")
    cell_src = cell_src.replace(
        anchor,
        "SCEN_AGG_STATS = ('mean', 'max', 'min', 'std')\n\n\n" + helpers + "def add_missing_indicators(df):\n",
    )

    old_tail = (
        "    scen_agg_cols = [\n"
        "        'congestion_score', 'order_inflow_15m', 'battery_mean', 'pack_utilization',\n"
        "        'avg_trip_distance', 'low_battery_ratio', 'max_zone_density', 'sku_concentration',\n"
        "        'robot_idle', 'outbound_truck_wait_min', 'order_per_station', 'robot_efficiency',\n"
        "        'order_pressure', 'risk_index', 'battery_risk', 'battery_cv',\n"
        "    ]\n"
        "    present = [c for c in scen_agg_cols if c in df.columns]\n"
        "    if present:\n"
        "        stats = df.groupby('scenario_id')[present].agg(SCEN_AGG_STATS)\n"
        "        stats.columns = [f'{c}_scen_{s}' for c, s in stats.columns]\n"
        "        stats = stats.reset_index()\n"
        "        df = df.merge(stats, on='scenario_id', how='left')\n\n"
        "    for col in ['congestion_score', 'order_per_station', 'pack_utilization', 'avg_trip_distance']:\n"
        "        sm = f'{col}_scen_mean'\n"
        "        if col in df.columns and sm in df.columns:\n"
        "            df[f'{col}_rel_to_scen'] = df[col] / (df[sm] + 1e-6)\n\n"
        "    for col in ['congestion_score', 'order_per_station', 'pack_utilization']:\n"
        "        if col in df.columns:\n"
        "            df[f'{col}_scen_rank'] = df.groupby('scenario_id')[col].rank(pct=True)\n"
        "    return df\n"
    )
    new_tail = (
        "    present = [c for c in SCENARIO_AGG_COLS if c in df.columns]\n"
        "    if present and not USE_FOLD_SAFE_SCENARIO_AGG:\n"
        "        stats = df.groupby('scenario_id')[present].agg(SCEN_AGG_STATS)\n"
        "        stats.columns = [f'{c}_scen_{s}' for c, s in stats.columns]\n"
        "        stats = stats.reset_index()\n"
        "        df = df.merge(stats, on='scenario_id', how='left')\n\n"
        "    if not USE_FOLD_SAFE_SCENARIO_AGG:\n"
        "        for col in ['congestion_score', 'order_per_station', 'pack_utilization', 'avg_trip_distance']:\n"
        "            sm = f'{col}_scen_mean'\n"
        "            if col in df.columns and sm in df.columns:\n"
        "                df[f'{col}_rel_to_scen'] = df[col] / (df[sm] + 1e-6)\n\n"
        "        for col in ['congestion_score', 'order_per_station', 'pack_utilization']:\n"
        "            if col in df.columns:\n"
        "                df[f'{col}_scen_rank'] = df.groupby('scenario_id')[col].rank(pct=True)\n"
        "    return df\n"
    )
    if old_tail not in cell_src:
        raise RuntimeError("add_interaction tail not found")
    cell_src = cell_src.replace(old_tail, new_tail)

    prep_marker = "test = preprocess_all(test, layout)\n"
    prep_inj = (
        "test = preprocess_all(test, layout)\n"
        "if USE_FOLD_SAFE_SCENARIO_AGG:\n"
        "    ensure_scenario_agg_placeholders(train, test)\n"
        "    merge_scenario_agg_from_reference(train, test)\n"
        "    print('▶ fold-safe scenario: train-ref stats merged to test')\n"
    )
    if prep_marker in cell_src and "ensure_scenario_agg_placeholders" not in cell_src:
        cell_src = cell_src.replace(prep_marker, prep_inj)
    return cell_src


def patch_stage1_cell(old: str) -> str:
    tree = (
        "\nif TREE_TUNING_PROFILE == 'regularized_slight':\n"
        "    lgb_params_s1.update(num_leaves=1023, min_child_samples=80, subsample=0.70, colsample_bytree=0.45)\n"
        "    xgb_params_s1.update(max_depth=9, min_child_weight=6.0, subsample=0.70)\n"
        "    cat_params_s1.update(depth=9, l2_leaf_reg=6.0)\n"
        "    print('▶ TREE_TUNING_PROFILE=regularized_slight (Stage1 params adjusted; Stage2 patched later cell)')\n"
        "\n"
    )
    old = old.replace(
        "section('Stage 1 - Base model (LGB + XGB + CAT + ensemble)')\n",
        tree + "section('Stage 1 - Base model (LGB + XGB + CAT + ensemble)')\n",
    )

    old_fold = (
        "for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):\n"
        "    if USE_AUTOENCODER:\n"
        "        med_ae = train.iloc[tr_idx][ae_input_cols].median()\n"
        "        X_ae_tr = _ae_prepare_matrix(train.iloc[tr_idx], ae_input_cols, med_ae)\n"
        "        X_ae_va = _ae_prepare_matrix(train.iloc[va_idx], ae_input_cols, med_ae)\n"
        "        sc_ae, sd_ae, z_va = _ae_train_fold(X_ae_tr, X_ae_va, AE_DEVICE, SEED + fold)\n"
        "        train.loc[train.index[va_idx], AE_COLS] = z_va\n"
        "        ae_fold_artifacts.append((sc_ae, sd_ae))\n"
        "        print(f\"  [AE] fold {fold}: val embedding shape {z_va.shape}\")\n\n"
        "    X_tr = train.iloc[tr_idx][feature_cols_s1]\n"
        "    X_va = train.iloc[va_idx][feature_cols_s1]\n"
    )
    new_fold = (
        "for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):\n"
        "    tr_df = train.iloc[tr_idx].copy()\n"
        "    va_df = train.iloc[va_idx].copy()\n"
        "    if USE_FOLD_SAFE_SCENARIO_AGG:\n"
        "        tr_df, va_df = _inject_fold_safe_scenario(tr_df, va_df)\n"
        "    if USE_AUTOENCODER:\n"
        "        med_ae = tr_df[ae_input_cols].median()\n"
        "        X_ae_tr = _ae_prepare_matrix(tr_df, ae_input_cols, med_ae)\n"
        "        X_ae_va = _ae_prepare_matrix(va_df, ae_input_cols, med_ae)\n"
        "        sc_ae, sd_ae, z_va = _ae_train_fold(X_ae_tr, X_ae_va, AE_DEVICE, SEED + fold)\n"
        "        train.loc[train.index[va_idx], AE_COLS] = z_va\n"
        "        va_df[AE_COLS] = z_va\n"
        "        ae_fold_artifacts.append((sc_ae, sd_ae))\n"
        "        print(f\"  [AE] fold {fold}: val embedding shape {z_va.shape}\")\n\n"
        "    X_tr = tr_df[feature_cols_s1]\n"
        "    X_va = va_df[feature_cols_s1]\n"
    )
    if old_fold not in old:
        raise RuntimeError("stage1 fold block not found")
    old = old.replace(old_fold, new_fold)

    s1b_old = (
        "# Stage 1b: residual (OOF-safe) — base 예측 오차를 추가로 학습\n"
        "section('Stage 1b - Residual LGB (stack on Stage1 ensemble)')\n"
        "resid_params = dict(\n"
        "    objective='regression_l1',\n"
        "    n_estimators=8000,\n"
        "    learning_rate=0.02,\n"
        "    num_leaves=127,\n"
        "    min_child_samples=40,\n"
        "    subsample=0.8,\n"
        "    colsample_bytree=0.7,\n"
        "    reg_alpha=0.2,\n"
        "    reg_lambda=2.0,\n"
        "    random_state=SEED,\n"
        "    verbose=-1,\n"
        ")\n"
        "oof_s1_resid = np.zeros(len(train))\n"
        "models_s1_resid = []\n"
        "t1b = time.time()\n"
        "for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):\n"
        "    X_tr = train.iloc[tr_idx][feature_cols_s1]\n"
        "    X_va = train.iloc[va_idx][feature_cols_s1]\n"
        "    r_tr = y_raw[tr_idx] - oof_s1_pre[tr_idx]\n"
        "    r_va = y_raw[va_idx] - oof_s1_pre[va_idx]\n"
        "    sw_tr = sw_all[tr_idx]\n"
        "    mr = lgb.LGBMRegressor(**resid_params)\n"
        "    mr.fit(\n"
        "        X_tr, r_tr,\n"
        "        sample_weight=sw_tr,\n"
        "        eval_set=[(X_va, r_va)],\n"
        "        eval_metric='mae',\n"
        "        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],\n"
        "    )\n"
        "    oof_s1_resid[va_idx] = mr.predict(X_va)\n"
        "    models_s1_resid.append(mr)\n"
        "    print(f\"  S1b Fold {fold} residual MAE: {mae(r_va, oof_s1_resid[va_idx]):.6f}\")\n\n"
        "oof_s1 = oof_s1_pre + oof_s1_resid\n"
        "s1_mae = mae(y_raw, oof_s1)\n"
        "print(f\"\\n▶ Stage 1 after residual OOF MAE: {s1_mae:.6f}  ({elapsed(t1b)})\")\n"
    )
    s1b_new = (
        "# Stage 1b: residual — optional (respect USE_STAGE1_RESIDUAL)\n"
        "oof_s1_resid = np.zeros(len(train))\n"
        "models_s1_resid = []\n"
        "if USE_STAGE1_RESIDUAL:\n"
        "    section('Stage 1b - Residual LGB (stack on Stage1 ensemble)')\n"
        "    resid_params = dict(\n"
        "        objective='regression_l1',\n"
        "        n_estimators=8000,\n"
        "        learning_rate=0.02,\n"
        "        num_leaves=127,\n"
        "        min_child_samples=40,\n"
        "        subsample=0.8,\n"
        "        colsample_bytree=0.7,\n"
        "        reg_alpha=0.2,\n"
        "        reg_lambda=2.0,\n"
        "        random_state=SEED,\n"
        "        verbose=-1,\n"
        "    )\n"
        "    t1b = time.time()\n"
        "    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):\n"
        "        tr_df = train.iloc[tr_idx].copy()\n"
        "        va_df = train.iloc[va_idx].copy()\n"
        "        if USE_FOLD_SAFE_SCENARIO_AGG:\n"
        "            tr_df, va_df = _inject_fold_safe_scenario(tr_df, va_df)\n"
        "        X_tr = tr_df[feature_cols_s1]\n"
        "        X_va = va_df[feature_cols_s1]\n"
        "        r_tr = y_raw[tr_idx] - oof_s1_pre[tr_idx]\n"
        "        r_va = y_raw[va_idx] - oof_s1_pre[va_idx]\n"
        "        sw_tr = sw_all[tr_idx]\n"
        "        mr = lgb.LGBMRegressor(**resid_params)\n"
        "        mr.fit(\n"
        "            X_tr, r_tr,\n"
        "            sample_weight=sw_tr,\n"
        "            eval_set=[(X_va, r_va)],\n"
        "            eval_metric='mae',\n"
        "            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],\n"
        "        )\n"
        "        oof_s1_resid[va_idx] = mr.predict(X_va)\n"
        "        models_s1_resid.append(mr)\n"
        "        print(f\"  S1b Fold {fold} residual MAE: {mae(r_va, oof_s1_resid[va_idx]):.6f}\")\n\n"
        "    oof_s1 = oof_s1_pre + oof_s1_resid\n"
        "    s1_mae = mae(y_raw, oof_s1)\n"
        "    print(f\"\\n▶ Stage 1 after residual OOF MAE: {s1_mae:.6f}  ({elapsed(t1b)})\")\n"
        "else:\n"
        "    oof_s1 = np.asarray(oof_s1_pre, dtype=np.float64).copy()\n"
        "    s1_mae = float(s1_pre_mae)\n"
        "    print('\\n▶ Stage 1 residual: OFF  (oof_s1 = pre-ensemble only)')\n"
    )
    if s1b_old not in old:
        raise RuntimeError("stage1b block not found")
    old = old.replace(s1b_old, s1b_new)

    tail_old = (
        "pred_s1_pre = sum(w_s1[m] * {'lgb': p_lgb, 'xgb': p_xgb, 'cat': p_cat}[m] for m in best_s1_models) / ws_s1\n"
        "pred_resid_test = np.mean([m.predict(X_test_s1) for m in models_s1_resid], axis=0)\n"
        "pred_s1_test = pred_s1_pre + pred_resid_test\n"
        "print(f\"▶ Stage 1 test predictions ready (ensemble + residual)\")\n"
    )
    tail_new = (
        "pred_s1_pre = sum(w_s1[m] * {'lgb': p_lgb, 'xgb': p_xgb, 'cat': p_cat}[m] for m in best_s1_models) / ws_s1\n"
        "if USE_STAGE1_RESIDUAL and models_s1_resid:\n"
        "    pred_resid_test = np.mean([m.predict(X_test_s1) for m in models_s1_resid], axis=0)\n"
        "    pred_s1_test = pred_s1_pre + pred_resid_test\n"
        "    print(f\"▶ Stage 1 test predictions ready (ensemble + residual)\")\n"
        "else:\n"
        "    pred_resid_test = np.zeros(len(test))\n"
        "    pred_s1_test = pred_s1_pre\n"
        "    print(f\"▶ Stage 1 test predictions ready (ensemble only, residual OFF)\")\n"
    )
    if tail_old not in old:
        raise RuntimeError("pred_s1 tail not found")
    old = old.replace(tail_old, tail_new)
    return old


def patch_stage2_cell(old: str) -> str:
    tree2 = (
        "\nif TREE_TUNING_PROFILE == 'regularized_slight':\n"
        "    lgb_params_s2.update(num_leaves=383, min_child_samples=100, subsample=0.75)\n"
        "    xgb_params_s2.update(max_depth=5, min_child_weight=8.0)\n"
        "    cat_params_s2.update(depth=5, l2_leaf_reg=5.0)\n"
        "    print('▶ TREE_TUNING_PROFILE=regularized_slight (Stage2 params adjusted)')\n"
        "\n"
    )
    old = old.replace(
        "section('Stage 2 - Pred-lag stack (LGB + XGB + CAT), fold OOF')\n",
        tree2 + "section('Stage 2 - Pred-lag stack (LGB + XGB + CAT), fold OOF')\n",
    )
    hdr = (
        "for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):\n"
        "    print(f\"\\n  --- Stage 2 Fold {fold} ---\")\n\n"
        "    X_tr = train.iloc[tr_idx][feature_cols_s2]\n"
    )
    hdr_new = (
        "for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):\n"
        "    print(f\"\\n  --- Stage 2 Fold {fold} ---\")\n\n"
        "    tr_df = train.iloc[tr_idx].copy()\n"
        "    va_df = train.iloc[va_idx].copy()\n"
        "    if USE_FOLD_SAFE_SCENARIO_AGG:\n"
        "        tr_df, va_df = _inject_fold_safe_scenario(tr_df, va_df)\n"
        "    X_tr = tr_df[feature_cols_s2]\n"
    )
    if hdr not in old:
        raise RuntimeError("stage2 header not found")
    old = old.replace(hdr, hdr_new)
    old = old.replace(
        "    X_va_real = train.iloc[va_idx][feature_cols_s2]\n",
        "    X_va_real = va_df[feature_cols_s2]\n",
    )
    old = old.replace(
        "        X_va = train.iloc[va_idx][feature_cols_s2]\n",
        "        X_va = va_df[feature_cols_s2]\n",
    )
    old = old.replace(
        "        va_df = train.iloc[va_idx].sort_values(['scenario_id', 'timeslot'])\n",
        "        va_df = va_df.sort_values(['scenario_id', 'timeslot'])\n",
    )
    return old


def patch_submit_cell(old: str) -> str:
    block_old = (
        "pred_resid_test = np.mean([m.predict(X_test_s1) for m in models_s1_resid], axis=0)\n"
        "pred_s1_test = pred_s1_pre + pred_resid_test\n"
    )
    block_new = (
        "if USE_STAGE1_RESIDUAL and models_s1_resid:\n"
        "    pred_resid_test = np.mean([m.predict(X_test_s1) for m in models_s1_resid], axis=0)\n"
        "    pred_s1_test = pred_s1_pre + pred_resid_test\n"
        "else:\n"
        "    pred_s1_test = pred_s1_pre\n"
    )
    if block_old not in old:
        raise RuntimeError("submit cell residual block not found")
    old = old.replace(block_old, block_new)
    old = old.replace(
        "'submission_logtarget_s2_predlag_autoencoder_0416.csv'",
        "'submission_logtarget_s2_predlag_autoencoder_v4_fulllevers.csv'",
    )
    return old


def insert_meta_and_log_cells(nb: dict) -> None:
    """Insert code cell after joint grid cell: meta ridge + CSV log."""
    src = '''\
section('Optional: Ridge meta-blend on OOF (Stage1 + S2 base models)')
best_meta_mae = float("nan")
if ENABLE_META_RIDGE_OOF:
    Xo = np.column_stack([oof_s1, oof_s2_lgb, oof_s2_xgb, oof_s2_cat]).astype(np.float64)
    yv = y_raw.astype(np.float64)
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 41), fit_intercept=True)
    ridge.fit(Xo, yv)
    pred = ridge.predict(Xo)
    best_meta_mae = float(mae(yv, pred))
    print(f"▶ Ridge meta OOF MAE (same-sample fit — diagnostic only): {best_meta_mae:.6f}")
    print(f"   RidgeCV alpha={ridge.alpha_!r} | compare FINAL OOF MAE: {best_final_mae:.6f}")
else:
    print("▶ Ridge meta: skipped (ENABLE_META_RIDGE_OOF=False)")

if EXPERIMENT_LOG_CSV:
    import csv
    from datetime import datetime
    logp = os.path.join(project_root, EXPERIMENT_LOG_CSV)
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "final_mae": float(best_final_mae),
        "use_fold_safe": USE_FOLD_SAFE_SCENARIO_AGG,
        "s1_resid": USE_STAGE1_RESIDUAL,
        "tree_profile": TREE_TUNING_PROFILE,
        "run_profile": RUN_PROFILE,
        "meta_mae": best_meta_mae,
    }
    file_exists = os.path.isfile(logp)
    with open(logp, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"▶ Appended experiment row -> {logp}")
'''
    new_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": src.splitlines(keepends=True),
        "id": "v4-meta-log",
        "execution_count": None,
        "outputs": [],
    }
    # insert after cell that prints 'Skipped joint grid'
    insert_at = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        t = "".join(c.get("source", []))
        if "ENABLE_FINAL_JOINT_GRID_SEARCH" in t and "Joint-grid FINAL OOF MAE" in t:
            insert_at = i + 1
            break
    if insert_at is None:
        raise RuntimeError("could not find joint grid cell")
    nb["cells"].insert(insert_at, new_cell)


def main() -> None:
    src = next(ROOT.glob("*strategy_v3_refine.ipynb"))
    dst = ROOT / "스마트 창고 출고 지연 예측_autoencoder_gridsearch_strategy_v4_fulllevers.ipynb"
    shutil.copyfile(src, dst)

    nb = json.loads(dst.read_text(encoding="utf-8"))

    md_intro = (
        "### v4 — 성능 개선 레버 (`fulllevers`)\n\n"
        "v3 refine 파이프라인 + 아래 플래그. 기본값은 v3와 동일한 의도입니다.\n\n"
        "| 레버 | 플래그/설정 |\n"
        "|------|-------------|\n"
        "| 시나리오 집계 누수 완화 | `USE_FOLD_SAFE_SCENARIO_AGG` |\n"
        "| 트리 정규화 프리셋 | `TREE_TUNING_PROFILE` |\n"
        "| Stage1 잔차 | `USE_STAGE1_RESIDUAL` |\n"
        "| AE / 표본가중 / 그리드 | 기존과 동일 + `WD_SWEEP_INDEX` |\n"
        "| 타겟 인코딩 스무딩 | `TE_SMOOTHING` |\n"
        "| Ridge 메타 (OOF 진단) | `ENABLE_META_RIDGE_OOF` |\n"
        "| 실험 로그 CSV | `EXPERIMENT_LOG_CSV` |\n"
    )

    inserted = False
    out_cells = []
    for cell in nb["cells"]:
        out_cells.append(cell)
        if not inserted and cell.get("cell_type") == "markdown":
            if "### 실행 파라미터" in "".join(cell.get("source", [])):
                out_cells.insert(
                    len(out_cells) - 1,
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": md_intro.splitlines(keepends=True),
                        "id": "v4-md-intro",
                    },
                )
                inserted = True
    nb["cells"] = out_cells

    cfg_needle = 'AE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    cfg_add = (
        cfg_needle
        + "\n"
        + "# --- v4 ---\n"
        + "USE_FOLD_SAFE_SCENARIO_AGG = False\n"
        + "TE_SMOOTHING = 20\n"
        + "TREE_TUNING_PROFILE = 'baseline'\n"
        + "ENABLE_META_RIDGE_OOF = False\n"
        + "EXPERIMENT_LOG_CSV = ''\n"
    )

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell.get("source", []))
        if "import torch.nn as nn" in s and "RidgeCV" not in s:
            s = s.replace(
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import RidgeCV\n",
            )
            cell["source"] = s.splitlines(keepends=True)
        if cfg_needle in s and "USE_FOLD_SAFE_SCENARIO_AGG" not in s:
            s = s.replace(cfg_needle, cfg_add)
            cell["source"] = s.splitlines(keepends=True)

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell.get("source", []))
        if "def preprocess_all(df, layout_df):" in s:
            cell["source"] = patch_preprocess(s).splitlines(keepends=True)
            break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell.get("source", []))
        if "section('Stage 1 - Base model" in s and "feature_cols_s1_base" in s:
            cell["source"] = patch_stage1_cell(s).splitlines(keepends=True)
            break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell.get("source", []))
        if "section('Stage 2 - Pred-lag stack" in s and "lgb_params_s2" in s:
            cell["source"] = patch_stage2_cell(s).splitlines(keepends=True)
            break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell.get("source", []))
        if "section('Target Encoding')" in s and "SMOOTHING = 20" in s:
            s = s.replace("SMOOTHING = 20\n", "SMOOTHING = int(TE_SMOOTHING)\n")
            cell["source"] = s.splitlines(keepends=True)
            break

    insert_meta_and_log_cells(nb)

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell.get("source", []))
        if "section('Predict test + submit" in s:
            cell["source"] = patch_submit_cell(s).splitlines(keepends=True)
            break

    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", dst)


if __name__ == "__main__":
    main()
