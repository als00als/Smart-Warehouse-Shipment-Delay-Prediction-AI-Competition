"""
Create TOP10-relaxed variant from:
  스마트 창고 출고 지연 예측_0415_preproc_logtarget_s2_predlag_exp.ipynb

Output:
  스마트 창고 출고 지연 예측_0415_preproc_logtarget_s2_predlag_top10_relaxed_exp.ipynb

Only change:
- USE_TOP10_ONLY = False
All other settings remain unchanged.
"""
import copy
import glob
import json
import re
from pathlib import Path


def _to_source_lines(text: str):
    lines = text.split("\n")
    if not lines:
        return []
    out = [f"{line}\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def main():
    src_matches = glob.glob("*0415_preproc_logtarget_s2_predlag_exp.ipynb")
    if not src_matches:
        raise FileNotFoundError("Could not find *0415_preproc_logtarget_s2_predlag_exp.ipynb")
    src = src_matches[0]

    with open(src, "r", encoding="utf-8") as f:
        nb = json.load(f)
    new_nb = copy.deepcopy(nb)

    c2 = "".join(new_nb["cells"][2].get("source", []))
    c2, n = re.subn(
        r"^USE_TOP10_ONLY\s*=\s*.*$",
        "USE_TOP10_ONLY = False  # TOP10 relaxed experiment",
        c2,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(f"Expected to replace exactly one USE_TOP10_ONLY line, got {n}")

    if "TOP10 relaxed experiment" not in c2:
        c2 += "\n# --- Notebook: TOP10 relaxed (other settings fixed) ---\n"

    new_nb["cells"][2]["source"] = _to_source_lines(c2)

    dst = Path(src).with_name(
        Path(src).name.replace(
            "_preproc_logtarget_s2_predlag_exp.ipynb",
            "_preproc_logtarget_s2_predlag_top10_relaxed_exp.ipynb",
        )
    )
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(new_nb, f, ensure_ascii=False, indent=1)

    print(str(dst))


if __name__ == "__main__":
    main()
