"""
From:
  스마트 창고 출고 지연 예측_0415_preproc_logtarget_s2_predlag_full_weight_sweep_exp.ipynb

Output:
  스마트 창고 출고 지연 예측_0415_preproc_logtarget_s2_predlag_5fold_weight_presets_exp.ipynb

Changes:
- N_FOLDS=5, S1_LGB_SEEDS=[SEED] (Stage1 시간 단축)
- ENABLE_CLIP_Q_SWEEP=False (후처리 그리드 단축)
- sample_weight: SW_PRESETS 3종 + JW_WEIGHT_PRESET 환경변수 선택
- submission 파일명 분리
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
        raise RuntimeError(f"Replacement target not found:\n{old[:200]}")
    return text.replace(old, new, 1)


CELL2_NEW = """SEED = 42
USE_LOG_TARGET = True  # logtarget + s2_predlag one-shot test

# Stage1 시간 단축: 5-fold 고정 + S1 시드 1개 + clip q 스윕 OFF (아래 ENABLE_CLIP_Q_SWEEP)
_V13_FULL = False
N_FOLDS = 5
S1_LGB_SEEDS = [SEED]  # 빠른 실행; 필요 시 [SEED, SEED + 100] 등으로 확장

RUN_STRATIFIED_GROUP_CV = True
BLEND_ALPHA_STEP = 0.02  # 필요 시 0.01로 세밀하게
REPLACE_STAGE2_IF_NOT_IMPROVING = False  # allow S2 in blend; revisit after OOF/LB
USE_S2_PRED_LAG_ONLY = True  # Stage2 uses Stage1 pred-lag features only
TOP_IMPORTANCE_K = 30

# 요청 반영: 모델에 직접 영향 주는 top10만 사용
USE_TOP10_ONLY = True

# Stage1 과적합 완화: residual은 기본 OFF (필요 시 True)
USE_STAGE1_RESIDUAL = False

CLIP_PRED_MIN = 0.0
CLIP_PRED_MAX_Q = 0.995
ENABLE_CLIP_Q_SWEEP = False  # full sweep 대비 후처리 시간 단축
CLIP_Q_GRID = [0.995]  # 스윕 사용 시에만 참조

# sample_weight: 가중치 3조합 (실행 전 선택)
#   PowerShell 예: `$env:JW_WEIGHT_PRESET='1';` 후 커널 재시작·전체 실행
ENABLE_SAMPLE_WEIGHT = True
SW_PRESETS = {
    0: {'q': 0.90, 'high_mult': 1.25, 'ts_mult': 1.10},  # 기본(풀 스윕 실험과 동일 계열)
    1: {'q': 0.92, 'high_mult': 1.15, 'ts_mult': 1.05},  # tail 약화
    2: {'q': 0.88, 'high_mult': 1.35, 'ts_mult': 1.12},  # tail 강화
}
WEIGHT_PRESET_ID = int(os.environ.get('JW_WEIGHT_PRESET', '0'))
assert WEIGHT_PRESET_ID in SW_PRESETS, f'invalid JW_WEIGHT_PRESET={WEIGHT_PRESET_ID}'
_sw = SW_PRESETS[WEIGHT_PRESET_ID]
SW_HIGH_Q = _sw['q']
SW_HIGH_MULT = _sw['high_mult']
SW_TS_EDGE_MULT = _sw['ts_mult']

PREPROC_ONLY_EXPERIMENT = True
ENABLE_DOMAIN_CLIP = True
ENABLE_MISSING_MASK = True

# ratio feature/column name hints
RATIO_COL_KEYWORDS = ['ratio', 'utilization', 'efficiency', 'pct', 'percent', 'share']

# non-negative metric name hints
NONNEG_COL_KEYWORDS = [
    'count', 'cnt', 'num', 'minutes', 'min', 'hour', 'distance',
    'delay', 'inflow', 'order', 'robot', 'battery', 'density',
    'area', 'weight', 'pressure', 'risk', 'wait', 'trip'
]
"""


def main():
    src_matches = glob.glob("*0415_preproc_logtarget_s2_predlag_full_weight_sweep_exp.ipynb")
    if not src_matches:
        raise FileNotFoundError("Could not find *full_weight_sweep_exp.ipynb")
    src = src_matches[0]

    out_name = "스마트 창고 출고 지연 예측_0415_preproc_logtarget_s2_predlag_5fold_weight_presets_exp.ipynb"
    out_path = Path(src).parent / out_name

    with open(src, "r", encoding="utf-8") as f:
        nb = json.load(f)
    new_nb = copy.deepcopy(nb)

    old_c2 = "".join(new_nb["cells"][2].get("source", []))
    new_nb["cells"][2]["source"] = _to_source_lines(CELL2_NEW.strip() + "\n")

    c10 = "".join(new_nb["cells"][10].get("source", []))
    c10 = _must_replace(
        c10,
        'print(\n    f"▶ jw_v13: mode={\'FULL(JW_V13_FULL=1)\' if _V13_FULL else \'quick(default)\'}, "\n    f"N_FOLDS={N_FOLDS}, S1_LGB seeds={len(S1_LGB_SEEDS)}, blend_step={BLEND_ALPHA_STEP}, true-lag+MethodB"\n)',
        'print(\n    f"▶ jw_v13 (5fold+weight-presets): N_FOLDS={N_FOLDS}, "\n    f"S1_LGB seeds={len(S1_LGB_SEEDS)}, blend_step={BLEND_ALPHA_STEP}, "\n    f"weight_preset={WEIGHT_PRESET_ID} (q={SW_HIGH_Q}, hm={SW_HIGH_MULT}, ts={SW_TS_EDGE_MULT}), true-lag+MethodB"\n)',
    )
    new_nb["cells"][10]["source"] = _to_source_lines(c10)

    last = "".join(new_nb["cells"][-1].get("source", []))
    last = _must_replace(
        last,
        "submission_logtarget_s2_predlag_full_weight_sweep_exp.csv",
        "submission_logtarget_s2_predlag_5fold_weight_presets_exp.csv",
    )
    new_nb["cells"][-1]["source"] = _to_source_lines(last)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_nb, f, ensure_ascii=False, indent=1)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
