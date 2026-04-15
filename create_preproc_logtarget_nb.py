"""
Copy 스마트 창고 출고 지연 예측_0415_preproc_speedup_exp.ipynb and create
스마트 창고 출고 지연 예측_0415_preproc_logtarget_exp.ipynb with USE_LOG_TARGET=True.

Pipeline already uses to_train_target / from_train_pred and sequential_predict
appends raw-scale preds to lag history, so log-target is consistent with TRUE_LAG
features in raw delay space.
"""
import copy
import re
import glob
import json
from pathlib import Path


def main():
    src_candidates = glob.glob("*0415_preproc_speedup_exp.ipynb")
    if not src_candidates:
        raise FileNotFoundError("Could not find *0415_preproc_speedup_exp.ipynb")
    src = src_candidates[0]

    with open(src, "r", encoding="utf-8") as f:
        nb = json.load(f)
    new_nb = copy.deepcopy(nb)

    cell2_lines = new_nb["cells"][2].get("source", [])
    cell2 = "".join(cell2_lines)

    cell2, n = re.subn(
        r"^USE_LOG_TARGET\s*=\s*.*$",
        "USE_LOG_TARGET = True  # log1p train / expm1 at inference; MAE on raw delay",
        cell2,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(f"Expected to replace exactly 1 USE_LOG_TARGET line, got {n}")

    banner = "\n# --- Notebook: log1p target experiment (copy of preproc_speedup_exp) ---\n"
    if "log1p target experiment" not in cell2:
        cell2 = cell2 + banner

    lines = cell2.split("\n")
    new_nb["cells"][2]["source"] = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        new_nb["cells"][2]["source"].append(lines[-1])

    src_path = Path(src)
    dst = src_path.with_name(
        src_path.name.replace("_preproc_speedup_exp.ipynb", "_preproc_logtarget_exp.ipynb")
    )
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(new_nb, f, ensure_ascii=False, indent=1)

    print(str(dst))


if __name__ == "__main__":
    main()
