"""
Create expanded experiment notebook from:
  스마트 창고 출고 지연 예측_0415_preproc_logtarget_s2_predlag_exp.ipynb

Output:
  스마트 창고 출고 지연 예측_0415_preproc_logtarget_s2_predlag_full_weight_sweep_exp.ipynb

Applied changes:
- Fold/seed expansion: full mode + more S1 seeds
- sample_weight for Stage1/Stage1-residual/Stage2 fits
- finer blend/clip sweep (alpha step + clip q grid)
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


def _must_replace(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Replacement target not found:\n{old[:120]}")
    return text.replace(old, new, 1)


def main():
    src_matches = glob.glob("*0415_preproc_logtarget_s2_predlag_exp.ipynb")
    if not src_matches:
        raise FileNotFoundError("Could not find *0415_preproc_logtarget_s2_predlag_exp.ipynb")
    src = src_matches[0]

    with open(src, "r", encoding="utf-8") as f:
        nb = json.load(f)
    new_nb = copy.deepcopy(nb)

    # -----------------------------
    # Cell 2: config expansion
    # -----------------------------
    c2 = "".join(new_nb["cells"][2].get("source", []))
    c2 = _must_replace(
        c2,
        "_V13_FULL = os.environ.get('JW_V13_FULL', '0').strip() == '1'",
        "_V13_FULL = True  # expanded folds/seeds experiment",
    )
    c2 = _must_replace(
        c2,
        "S1_LGB_SEEDS = [SEED, SEED + 100, SEED + 200] if _V13_FULL else [SEED]",
        "S1_LGB_SEEDS = [SEED, SEED + 100, SEED + 200, SEED + 300] if _V13_FULL else [SEED, SEED + 100]",
    )
    c2 = _must_replace(
        c2,
        "BLEND_ALPHA_STEP = 0.02 if _V13_FULL else 0.05",
        "BLEND_ALPHA_STEP = 0.01  # fine alpha sweep",
    )
    c2 = _must_replace(
        c2,
        "CLIP_PRED_MAX_Q = 0.995",
        "CLIP_PRED_MAX_Q = 0.995\nENABLE_CLIP_Q_SWEEP = True\nCLIP_Q_GRID = [0.992, 0.994, 0.995, 0.996, 0.998]\n\nENABLE_SAMPLE_WEIGHT = True\nSW_HIGH_Q = 0.90\nSW_HIGH_MULT = 1.25\nSW_TS_EDGE_MULT = 1.10",
    )
    new_nb["cells"][2]["source"] = _to_source_lines(c2)

    # -----------------------------
    # Cell 4: helper function
    # -----------------------------
    c4 = "".join(new_nb["cells"][4].get("source", []))
    c4 = _must_replace(
        c4,
        "def mae(y_true, y_pred):\n    return mean_absolute_error(y_true, y_pred)\n\n\n",
        """def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def _build_sample_weight(df: pd.DataFrame) -> np.ndarray:
    \"\"\"Simple robust weighting: high-target + edge timeslot emphasis.\"\"\"
    w = np.ones(len(df), dtype=np.float64)
    if not ENABLE_SAMPLE_WEIGHT:
        return w

    y = df[TARGET].to_numpy(dtype=np.float64)
    q_thr = float(np.quantile(y, SW_HIGH_Q))
    w[y >= q_thr] *= float(SW_HIGH_MULT)

    if 'timeslot' in df.columns:
        ts = df['timeslot'].to_numpy(dtype=np.int64)
        edge_mask = (ts <= 2) | (ts >= 22)
        w[edge_mask] *= float(SW_TS_EDGE_MULT)
    return w


""",
    )
    new_nb["cells"][4]["source"] = _to_source_lines(c4)

    # -----------------------------
    # Cell 23: Stage1 + residual sample weights
    # -----------------------------
    c23 = "".join(new_nb["cells"][23].get("source", []))
    c23 = _must_replace(
        c23,
        "groups = train['scenario_id'].values\n\n_scen_mean = train.groupby('scenario_id')[TARGET].transform('mean')",
        "groups = train['scenario_id'].values\nsw_all = _build_sample_weight(train)\nprint(f\"▶ sample_weight enabled={ENABLE_SAMPLE_WEIGHT} | mean={sw_all.mean():.4f} | max={sw_all.max():.3f}\")\n\n_scen_mean = train.groupby('scenario_id')[TARGET].transform('mean')",
    )
    c23 = _must_replace(
        c23,
        "    y_tr = y_all[tr_idx]\n    y_va = y_all[va_idx]\n\n    lgb_va_stack = []",
        "    y_tr = y_all[tr_idx]\n    y_va = y_all[va_idx]\n    sw_tr = sw_all[tr_idx]\n\n    lgb_va_stack = []",
    )
    c23 = _must_replace(
        c23,
        "        m_lgb.fit(\n            X_tr, y_tr,\n            eval_set=[(X_va, y_va)],",
        "        m_lgb.fit(\n            X_tr, y_tr,\n            sample_weight=sw_tr,\n            eval_set=[(X_va, y_va)],",
    )
    c23 = _must_replace(
        c23,
        "        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)",
        "        m_xgb.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False)",
    )
    c23 = _must_replace(
        c23,
        "        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)",
        "        m_xgb.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False)",
    )
    c23 = _must_replace(
        c23,
        "    m_cat.fit(\n        X_tr, y_tr,\n        eval_set=(X_va, y_va),",
        "    m_cat.fit(\n        X_tr, y_tr,\n        sample_weight=sw_tr,\n        eval_set=(X_va, y_va),",
    )
    c23 = _must_replace(
        c23,
        "    r_tr = y_raw[tr_idx] - oof_s1_pre[tr_idx]\n    r_va = y_raw[va_idx] - oof_s1_pre[va_idx]\n    mr = lgb.LGBMRegressor(**resid_params)",
        "    r_tr = y_raw[tr_idx] - oof_s1_pre[tr_idx]\n    r_va = y_raw[va_idx] - oof_s1_pre[va_idx]\n    sw_tr = sw_all[tr_idx]\n    mr = lgb.LGBMRegressor(**resid_params)",
    )
    c23 = _must_replace(
        c23,
        "    mr.fit(\n        X_tr, r_tr,\n        eval_set=[(X_va, r_va)],",
        "    mr.fit(\n        X_tr, r_tr,\n        sample_weight=sw_tr,\n        eval_set=[(X_va, r_va)],",
    )
    new_nb["cells"][23]["source"] = _to_source_lines(c23)

    # -----------------------------
    # Cell 29: Stage2 sample weights
    # -----------------------------
    c29 = "".join(new_nb["cells"][29].get("source", []))
    c29 = _must_replace(
        c29,
        "    X_va_real = train.iloc[va_idx][feature_cols_s2]\n    y_va_real = y_all[va_idx]\n\n    m_lgb = lgb.LGBMRegressor(**lgb_params_s2)",
        "    X_va_real = train.iloc[va_idx][feature_cols_s2]\n    y_va_real = y_all[va_idx]\n    sw_tr = sw_all[tr_idx]\n\n    m_lgb = lgb.LGBMRegressor(**lgb_params_s2)",
    )
    c29 = _must_replace(
        c29,
        "    m_lgb.fit(\n        X_tr, y_tr,\n        eval_set=[(X_va_real, y_va_real)],",
        "    m_lgb.fit(\n        X_tr, y_tr,\n        sample_weight=sw_tr,\n        eval_set=[(X_va_real, y_va_real)],",
    )
    c29 = _must_replace(
        c29,
        "        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va_real, y_va_real)], verbose=False)",
        "        m_xgb.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va_real, y_va_real)], verbose=False)",
    )
    c29 = _must_replace(
        c29,
        "        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va_real, y_va_real)], verbose=False)",
        "        m_xgb.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va_real, y_va_real)], verbose=False)",
    )
    c29 = _must_replace(
        c29,
        "    m_cat.fit(\n        X_tr, y_tr,\n        eval_set=(X_va_real, y_va_real),",
        "    m_cat.fit(\n        X_tr, y_tr,\n        sample_weight=sw_tr,\n        eval_set=(X_va_real, y_va_real),",
    )
    new_nb["cells"][29]["source"] = _to_source_lines(c29)

    # -----------------------------
    # Cell 31: blend + clip q sweep
    # -----------------------------
    c31 = "".join(new_nb["cells"][31].get("source", []))
    c31 = _must_replace(
        c31,
        "best_blend_mae = mae(y_raw, oof_s2_used)\n_alpha_grid = _grid_01(BLEND_ALPHA_STEP)",
        "best_blend_mae = mae(y_raw, oof_s2_used)\nbest_blend_oof = oof_s2_used.copy()\n_alpha_grid = _grid_01(BLEND_ALPHA_STEP)",
    )
    c31 = _must_replace(
        c31,
        "    if m_val < best_blend_mae:\n        best_blend_mae = float(m_val)\n        best_alpha = float(alpha)",
        "    if m_val < best_blend_mae:\n        best_blend_mae = float(m_val)\n        best_alpha = float(alpha)\n        best_blend_oof = blend_pred.copy()",
    )
    c31 = _must_replace(
        c31,
        "print(f\"\\n▶ Best blend: alpha={best_alpha:.2f} (S1 weight)  MAE={best_blend_mae:.6f}\")\n\n# ============================================================",
        """print(f"\\n▶ Best blend: alpha={best_alpha:.2f} (S1 weight)  MAE={best_blend_mae:.6f}")

best_clip_q = float(CLIP_PRED_MAX_Q)
best_clip_mae = best_blend_mae
if ENABLE_CLIP_Q_SWEEP:
    for q in CLIP_Q_GRID:
        q = float(q)
        hi = float(np.percentile(y_raw, 100 * q))
        clipped = np.minimum(np.maximum(best_blend_oof, CLIP_PRED_MIN), hi)
        m_val = mae(y_raw, clipped)
        print(f"  clip_q={q:.4f} MAE={m_val:.6f}")
        if m_val < best_clip_mae:
            best_clip_mae = float(m_val)
            best_clip_q = q
else:
    hi = float(np.percentile(y_raw, 100 * best_clip_q))
    best_clip_mae = mae(y_raw, np.minimum(np.maximum(best_blend_oof, CLIP_PRED_MIN), hi))
print(f"▶ Best clip q={best_clip_q:.4f} MAE={best_clip_mae:.6f}")

# ============================================================""",
    )
    c31 = _must_replace(
        c31,
        "best_final_mae = best_blend_mae",
        "best_final_mae = best_clip_mae",
    )
    new_nb["cells"][31]["source"] = _to_source_lines(c31)

    # -----------------------------
    # Cell 38: apply selected clip q in test
    # -----------------------------
    c38 = "".join(new_nb["cells"][38].get("source", []))
    c38 = _must_replace(
        c38,
        "pred_hi = float(np.percentile(y_raw, 100 * CLIP_PRED_MAX_Q))",
        "pred_hi = float(np.percentile(y_raw, 100 * best_clip_q))",
    )
    c38 = _must_replace(
        c38,
        "save_path = os.path.join(project_root, 'submission_logtarget_s2_predlag_exp.csv')",
        "save_path = os.path.join(project_root, 'submission_logtarget_s2_predlag_full_weight_sweep_exp.csv')",
    )
    new_nb["cells"][38]["source"] = _to_source_lines(c38)

    dst = Path(src).with_name(
        Path(src).name.replace(
            "_preproc_logtarget_s2_predlag_exp.ipynb",
            "_preproc_logtarget_s2_predlag_full_weight_sweep_exp.ipynb",
        )
    )
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(new_nb, f, ensure_ascii=False, indent=1)

    print(str(dst))


if __name__ == "__main__":
    main()
