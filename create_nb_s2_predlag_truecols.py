"""
Create a hybrid lag experiment notebook from:
  스마트 창고 출고 지연 예측_0415_preproc_s2_predlag_exp.ipynb

Goal:
- Keep the successful pred-lag Stage2 setup.
- Additionally enable TRUE_LAG_COLS_LEGACY-shaped columns, but computed from
  Stage1 OOF/test predictions (not from true target), to avoid train/test mismatch.
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


HYBRID_TRUE_LAG_FN = """

def add_stage1_pred_true_lag_features(df, pred_vec, gm):
    \"\"\"Build TRUE_LAG_COLS_LEGACY-like columns from Stage1 predictions.\"\"\"
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    p = np.asarray(pred_vec, dtype=np.float64)
    if len(p) != len(df):
        raise ValueError(
            f'add_stage1_pred_true_lag_features: len(pred_vec)={len(p)} != len(df)={len(df)}'
        )

    df['_p_for_true'] = p
    g = df.groupby('scenario_id')['_p_for_true']

    for lag in [1, 2, 3, 4, 5]:
        df[f'target_lag{lag}'] = g.shift(lag)

    for w in [3, 5, 10]:
        df[f'target_roll{w}_mean'] = g.transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean()
        )

    df['target_lag_max3'] = g.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).max()
    )
    df['target_lag_min3'] = g.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).min()
    )
    df['target_lag_std3'] = g.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0.0)
    ).fillna(0.0)

    df['target_ewm3'] = g.transform(
        lambda x: x.shift(1).ewm(alpha=0.3, adjust=False).mean()
    )
    df['target_ewm5'] = g.transform(
        lambda x: x.shift(1).ewm(alpha=0.5, adjust=False).mean()
    )

    df['target_diff1'] = (df['target_lag1'] - df['target_lag2']).fillna(0.0)
    df['target_diff2'] = (df['target_lag2'] - df['target_lag3']).fillna(0.0)
    df['target_lag1_log'] = np.log1p(np.maximum(df['target_lag1'].fillna(gm).astype(np.float64), 0.0))

    for col in TRUE_LAG_COLS_LEGACY:
        if col in df.columns:
            if 'std' in col:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(gm)

    df.drop(columns=['_p_for_true'], inplace=True)
    return df
"""


TRAIN_APPEND = """

if USE_TRUE_LAG_COLS_HYBRID:
    train = add_stage1_pred_true_lag_features(train, oof_s1, global_mean)
    print(f"▶ Added hybrid TRUE_LAG_COLS from Stage1 OOF (n={len(TRUE_LAG_COLS_LEGACY)})")
"""


TEST_SNIPPET_OLD = """if USE_S2_PRED_LAG_ONLY:
    test = add_pred_lag_features(test, pred_s1_test, global_mean)
    print(f"  Stage 2 test: added pred-lag columns ({len(PRED_LAG_COLS)})")
"""

TEST_SNIPPET_NEW = """if USE_S2_PRED_LAG_ONLY:
    test = add_pred_lag_features(test, pred_s1_test, global_mean)
    print(f"  Stage 2 test: added pred-lag columns ({len(PRED_LAG_COLS)})")
if USE_TRUE_LAG_COLS_HYBRID:
    test = add_stage1_pred_true_lag_features(test, pred_s1_test, global_mean)
    print(f"  Stage 2 test: added hybrid TRUE_LAG_COLS ({len(TRUE_LAG_COLS_LEGACY)})")
"""


def main():
    src_matches = glob.glob("*0415_preproc_s2_predlag_exp.ipynb")
    if not src_matches:
        raise FileNotFoundError("Could not find *0415_preproc_s2_predlag_exp.ipynb")
    src = src_matches[0]

    with open(src, "r", encoding="utf-8") as f:
        nb = json.load(f)
    new_nb = copy.deepcopy(nb)

    # Cell 2: add flag
    c2 = "".join(new_nb["cells"][2].get("source", []))
    if "USE_TRUE_LAG_COLS_HYBRID" not in c2:
        c2 += "\nUSE_TRUE_LAG_COLS_HYBRID = True  # use TRUE_LAG_COLS_LEGACY built from Stage1 predictions\n"
    new_nb["cells"][2]["source"] = _to_source_lines(c2)

    # Cell 6: TRUE_LAG_COLS definition
    c6 = "".join(new_nb["cells"][6].get("source", []))
    old_line = "TRUE_LAG_COLS = [] if USE_S2_PRED_LAG_ONLY else TRUE_LAG_COLS_LEGACY"
    new_line = (
        "TRUE_LAG_COLS = TRUE_LAG_COLS_LEGACY if USE_TRUE_LAG_COLS_HYBRID "
        "else ([] if USE_S2_PRED_LAG_ONLY else TRUE_LAG_COLS_LEGACY)"
    )
    if old_line in c6:
        c6 = c6.replace(old_line, new_line, 1)
    elif new_line not in c6:
        raise RuntimeError("Failed to patch TRUE_LAG_COLS line in cell 6")
    new_nb["cells"][6]["source"] = _to_source_lines(c6)

    # Cell 8: append hybrid helper
    c8 = "".join(new_nb["cells"][8].get("source", []))
    if "def add_stage1_pred_true_lag_features" not in c8:
        _append_to_cell(new_nb, 8, HYBRID_TRUE_LAG_FN)

    # Cell 23: append train-side hybrid creation after Stage1
    c23 = "".join(new_nb["cells"][23].get("source", []))
    if "Added hybrid TRUE_LAG_COLS from Stage1 OOF" not in c23:
        _append_to_cell(new_nb, 23, TRAIN_APPEND)

    # Cell 29: Stage2 feature set includes both pred-lag and hybrid true-lag cols
    c29 = "".join(new_nb["cells"][29].get("source", []))
    old_feat_line = "feature_cols_s2 = feature_cols_s1 + (PRED_LAG_COLS if USE_S2_PRED_LAG_ONLY else TRUE_LAG_COLS)"
    new_feat_line = (
        "feature_cols_s2 = feature_cols_s1 + "
        "(PRED_LAG_COLS if USE_S2_PRED_LAG_ONLY else []) + "
        "(TRUE_LAG_COLS if USE_TRUE_LAG_COLS_HYBRID else [])"
    )
    if old_feat_line in c29:
        c29 = c29.replace(old_feat_line, new_feat_line, 1)
    elif new_feat_line not in c29:
        raise RuntimeError("Failed to patch feature_cols_s2 line in cell 29")
    new_nb["cells"][29]["source"] = _to_source_lines(c29)

    # Cell 38: add test-side hybrid creation
    c38 = "".join(new_nb["cells"][38].get("source", []))
    if TEST_SNIPPET_OLD in c38:
        c38 = c38.replace(TEST_SNIPPET_OLD, TEST_SNIPPET_NEW, 1)
    elif TEST_SNIPPET_NEW not in c38:
        raise RuntimeError("Failed to patch test lag block in cell 38")
    new_nb["cells"][38]["source"] = _to_source_lines(c38)

    dst = Path(src).with_name(
        Path(src).name.replace("_preproc_s2_predlag_exp.ipynb", "_preproc_s2_predlag_truecols_exp.ipynb")
    )
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(new_nb, f, ensure_ascii=False, indent=1)

    print(str(dst))


if __name__ == "__main__":
    main()
