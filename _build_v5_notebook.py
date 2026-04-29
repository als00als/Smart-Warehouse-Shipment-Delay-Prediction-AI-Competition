# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SRC = None
DST = None


def j(cell):
    return ''.join(cell.get('source', []))


def set_src(cell, text: str):
    cell['source'] = text.splitlines(keepends=True)


def patch_config(nb):
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        s = j(cell)
        if 'EXPERIMENT_LOG_CSV' in s and 'ENABLE_TIMESLOT_ALPHA_SEARCH' not in s:
            s += (
                "\n# --- v5 advanced finalization ---\n"
                "ENABLE_TIMESLOT_ALPHA_SEARCH = True\n"
                "TIMESLOT_ALPHA_STEP = 0.02\n"
                "TIMESLOT_ALPHA_SMOOTH_TO_GLOBAL = 0.20  # 0: raw per-timeslot alpha, 1: fully global\n"
                "ENABLE_OOF_META_STACK = True\n"
                "META_ALPHAS = np.logspace(-4, 4, 41)\n"
                "AUTO_SELECT_FINAL_METHOD = True  # pick best OOF among joint/ts_alpha/meta\n"
                "FORCE_FINAL_METHOD = ''  # '', 'joint', 'timeslot_alpha', 'meta_ridge'\n"
                "SAVE_OOF_PRED_CSV = ''  # e.g. 'oof_v5.csv'\n"
            )
            set_src(cell, s)
            return
    raise RuntimeError('config cell not found')


def patch_intro(nb):
    for cell in nb['cells']:
        if cell.get('cell_type') != 'markdown':
            continue
        s = j(cell)
        if '### v4 ???깅뒫 媛쒖꽑 ?덈쾭' in s:
            s = s.replace('### v4 ???깅뒫 媛쒖꽑 ?덈쾭 (`fulllevers`)', '### v5 ???깅뒫 媛쒖꽑 ?덈쾭 (`all_methods`)')
            s += (
                "\n| 理쒖쥌 寃고빀(異붽?) | `ENABLE_TIMESLOT_ALPHA_SEARCH`, `ENABLE_OOF_META_STACK` |\n"
                "| 理쒖쥌 諛⑸쾿 ?좏깮 | `AUTO_SELECT_FINAL_METHOD`, `FORCE_FINAL_METHOD` |\n"
                "| OOF ???| `SAVE_OOF_PRED_CSV` |\n"
            )
            set_src(cell, s)
            return


def replace_meta_cell(nb):
    code = """section('Advanced finalization: timeslot alpha + OOF-safe meta stack')
FINAL_METHOD = 'joint'
final_oof_for_submit = best_blend_oof.copy()
final_methods_score = {'joint': float(best_final_mae)}
ts_alpha_map = {}
meta_fold_models = []

# --- (A) timeslot-wise alpha search ---
if ENABLE_TIMESLOT_ALPHA_SEARCH and 'timeslot' in train.columns:
    alpha_grid_ts = np.arange(0.0, 1.0 + 1e-12, TIMESLOT_ALPHA_STEP)
    ts_arr = train['timeslot'].to_numpy()
    ts_oof = np.zeros(len(train), dtype=np.float64)

    for ts in np.unique(ts_arr):
        idx = (ts_arr == ts)
        if idx.sum() == 0:
            continue
        y_ts = y_raw[idx]
        s1_ts = oof_s1[idx]
        s2_ts = oof_s2_used[idx]
        a_best, m_best = float(best_alpha), float('inf')
        for a in alpha_grid_ts:
            pred_ts = a * s1_ts + (1.0 - a) * s2_ts
            m = float(np.mean(np.abs(y_ts - pred_ts)))
            if m < m_best:
                m_best = m
                a_best = float(a)
        a_smoothed = (1.0 - TIMESLOT_ALPHA_SMOOTH_TO_GLOBAL) * a_best + TIMESLOT_ALPHA_SMOOTH_TO_GLOBAL * float(best_alpha)
        ts_alpha_map[int(ts)] = float(a_smoothed)
        ts_oof[idx] = a_smoothed * s1_ts + (1.0 - a_smoothed) * s2_ts

    hi_ts = float(np.percentile(y_raw, 100 * best_clip_q))
    ts_oof = np.minimum(np.maximum(ts_oof, CLIP_PRED_MIN), hi_ts)
    mae_ts = float(mae(y_raw, ts_oof))
    final_methods_score['timeslot_alpha'] = mae_ts
    print(f"??timeslot alpha OOF MAE: {mae_ts:.6f} | n_timeslots={len(ts_alpha_map)}")
else:
    print('??timeslot alpha: skipped')

# --- (B) OOF-safe meta stack (RidgeCV) ---
if ENABLE_OOF_META_STACK and len(train) == len(oof_s1):
    ts = train['timeslot'].to_numpy(dtype=np.float64) if 'timeslot' in train.columns else np.zeros(len(train), dtype=np.float64)
    ang = 2.0 * np.pi * ts / 24.0
    X_meta = np.column_stack([
        oof_s1,
        oof_s2_lgb,
        oof_s2_xgb,
        oof_s2_cat,
        oof_s2_used,
        np.sin(ang),
        np.cos(ang),
    ]).astype(np.float64)

    oof_meta = np.zeros(len(train), dtype=np.float64)
    meta_fold_models = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):
        m = RidgeCV(alphas=META_ALPHAS, fit_intercept=True)
        m.fit(X_meta[tr_idx], y_raw[tr_idx])
        oof_meta[va_idx] = m.predict(X_meta[va_idx])
        meta_fold_models.append(m)

    hi_meta = float(np.percentile(y_raw, 100 * best_clip_q))
    oof_meta = np.minimum(np.maximum(oof_meta, CLIP_PRED_MIN), hi_meta)
    mae_meta = float(mae(y_raw, oof_meta))
    final_methods_score['meta_ridge'] = mae_meta
    print(f"??meta ridge OOF MAE: {mae_meta:.6f} | folds={len(meta_fold_models)}")
else:
    print('??meta stack: skipped')

# --- method selection ---
if FORCE_FINAL_METHOD in ('joint', 'timeslot_alpha', 'meta_ridge'):
    FINAL_METHOD = FORCE_FINAL_METHOD
elif AUTO_SELECT_FINAL_METHOD:
    FINAL_METHOD = min(final_methods_score, key=final_methods_score.get)

if FINAL_METHOD == 'timeslot_alpha' and 'timeslot_alpha' in final_methods_score:
    ts_arr = train['timeslot'].to_numpy()
    out = np.zeros(len(train), dtype=np.float64)
    for ts, a in ts_alpha_map.items():
        idx = (ts_arr == ts)
        out[idx] = a * oof_s1[idx] + (1.0 - a) * oof_s2_used[idx]
    hi = float(np.percentile(y_raw, 100 * best_clip_q))
    final_oof_for_submit = np.minimum(np.maximum(out, CLIP_PRED_MIN), hi)
    best_final_mae = float(mae(y_raw, final_oof_for_submit))
    best_final_name = 'timeslot_alpha'
elif FINAL_METHOD == 'meta_ridge' and 'meta_ridge' in final_methods_score:
    # rebuild OOF from stored score by recomputing one pass (cheap)
    ts = train['timeslot'].to_numpy(dtype=np.float64) if 'timeslot' in train.columns else np.zeros(len(train), dtype=np.float64)
    ang = 2.0 * np.pi * ts / 24.0
    X_meta = np.column_stack([oof_s1, oof_s2_lgb, oof_s2_xgb, oof_s2_cat, oof_s2_used, np.sin(ang), np.cos(ang)]).astype(np.float64)
    oof_meta = np.zeros(len(train), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, kf_y, groups=groups), 1):
        m = RidgeCV(alphas=META_ALPHAS, fit_intercept=True)
        m.fit(X_meta[tr_idx], y_raw[tr_idx])
        oof_meta[va_idx] = m.predict(X_meta[va_idx])
    hi = float(np.percentile(y_raw, 100 * best_clip_q))
    final_oof_for_submit = np.minimum(np.maximum(oof_meta, CLIP_PRED_MIN), hi)
    best_final_mae = float(mae(y_raw, final_oof_for_submit))
    best_final_name = 'meta_ridge'
else:
    FINAL_METHOD = 'joint'
    final_oof_for_submit = best_blend_oof.copy()
    best_final_mae = float(mae(y_raw, final_oof_for_submit))
    best_final_name = 'joint_grid_search'

print(f"??FINAL method selected: {FINAL_METHOD} | OOF MAE={best_final_mae:.6f}")

if SAVE_OOF_PRED_CSV:
    oof_path = os.path.join(project_root, SAVE_OOF_PRED_CSV)
    pd.DataFrame({'ID': train['ID'], 'y_true': y_raw, 'oof_pred': final_oof_for_submit}).to_csv(oof_path, index=False)
    print(f"??saved OOF -> {oof_path}")

if EXPERIMENT_LOG_CSV:
    import csv
    from datetime import datetime
    logp = os.path.join(project_root, EXPERIMENT_LOG_CSV)
    row = {
        'ts': datetime.now().isoformat(timespec='seconds'),
        'final_mae': float(best_final_mae),
        'final_method': FINAL_METHOD,
        's1_mae': float(s1_mae),
        's2_mae': float(s2_ens_mae),
        'tree_profile': TREE_TUNING_PROFILE,
        'run_profile': RUN_PROFILE,
        'timeslot_alpha': ENABLE_TIMESLOT_ALPHA_SEARCH,
        'meta_stack': ENABLE_OOF_META_STACK,
    }
    exists = os.path.isfile(logp)
    with open(logp, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"??appended experiment row -> {logp}")
"""
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        s = j(cell)
        if "Optional: Ridge meta-blend on OOF" in s:
            set_src(cell, code)
            cell['id'] = 'v5-advanced-finalization'
            return
    raise RuntimeError('meta cell not found')


def patch_submit(nb):
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        s = j(cell)
        if "section('Predict test + submit" not in s:
            continue
        old = (
            "pred_blend_test = best_alpha * pred_s1_test + (1 - best_alpha) * pred_s2_test_blend\n\n"
            "pred = np.maximum(pred_blend_test, CLIP_PRED_MIN)\n"
            "pred_hi = float(np.percentile(y_raw, 100 * best_clip_q))\n"
            "pred = np.minimum(pred, pred_hi)\n"
        )
        new = (
            "pred_blend_test = best_alpha * pred_s1_test + (1 - best_alpha) * pred_s2_test_blend\n"
            "pred_final_base = pred_blend_test.copy()\n"
            "if FINAL_METHOD == 'timeslot_alpha' and len(ts_alpha_map) > 0 and 'timeslot' in test.columns:\n"
            "    alpha_ts = test['timeslot'].map(ts_alpha_map).fillna(float(best_alpha)).to_numpy(dtype=np.float64)\n"
            "    pred_final_base = alpha_ts * pred_s1_test + (1.0 - alpha_ts) * pred_s2_test_blend\n"
            "    print('  Final method: timeslot_alpha')\n"
            "elif FINAL_METHOD == 'meta_ridge' and len(meta_fold_models) > 0:\n"
            "    tsv = test['timeslot'].to_numpy(dtype=np.float64) if 'timeslot' in test.columns else np.zeros(len(test), dtype=np.float64)\n"
            "    ang = 2.0 * np.pi * tsv / 24.0\n"
            "    X_meta_test = np.column_stack([\n"
            "        pred_s1_test, pred_s2_lgb_test, pred_s2_xgb_test, pred_s2_cat_test, pred_s2_test_blend, np.sin(ang), np.cos(ang)\n"
            "    ]).astype(np.float64)\n"
            "    pred_final_base = np.mean([m.predict(X_meta_test) for m in meta_fold_models], axis=0)\n"
            "    print('  Final method: meta_ridge')\n"
            "else:\n"
            "    print('  Final method: joint')\n\n"
            "pred = np.maximum(pred_final_base, CLIP_PRED_MIN)\n"
            "pred_hi = float(np.percentile(y_raw, 100 * best_clip_q))\n"
            "pred = np.minimum(pred, pred_hi)\n"
        )
        if old not in s:
            raise RuntimeError('submit blend block not found')
        s = s.replace(old, new)
        s = s.replace('submission_logtarget_s2_predlag_autoencoder_v4_fulllevers.csv', 'submission_logtarget_s2_predlag_autoencoder_v5_all_methods.csv')
        set_src(cell, s)
        return
    raise RuntimeError('submit cell not found')


def main():
    src = next(ROOT.glob('*v4_fulllevers.ipynb'))
    dst_name = src.name.replace('v4_fulllevers', 'v5_all_methods')
    if dst_name == src.name:
        dst_name = f"{src.stem}_v5_all_methods.ipynb"
    dst = ROOT / dst_name
    shutil.copyfile(src, dst)
    nb = json.loads(dst.read_text(encoding='utf-8'))
    patch_intro(nb)
    patch_config(nb)
    replace_meta_cell(nb)
    patch_submit(nb)
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('Wrote', dst)


if __name__ == '__main__':
    main()
