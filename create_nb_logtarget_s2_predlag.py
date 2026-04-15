"""
Create:
  스마트 창고 출고 지연 예측_0415_preproc_logtarget_s2_predlag_exp.ipynb
from:
  스마트 창고 출고 지연 예측_0415_preproc_s2_predlag_exp.ipynb

Changes:
- USE_LOG_TARGET = True
- Keep USE_S2_PRED_LAG_ONLY = True (already in source)
- Update submission filename to avoid overwrite
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
    src_matches = glob.glob("*0415_preproc_s2_predlag_exp.ipynb")
    if not src_matches:
        raise FileNotFoundError("Could not find *0415_preproc_s2_predlag_exp.ipynb")
    src = src_matches[0]

    with open(src, "r", encoding="utf-8") as f:
        nb = json.load(f)
    new_nb = copy.deepcopy(nb)

    # Cell 2: switch log target on
    c2 = "".join(new_nb["cells"][2].get("source", []))
    c2, n = re.subn(
        r"^USE_LOG_TARGET\s*=\s*.*$",
        "USE_LOG_TARGET = True  # logtarget + s2_predlag one-shot test",
        c2,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(f"Expected 1 USE_LOG_TARGET line replacement, got {n}")
    if "logtarget + s2_predlag one-shot test" not in c2:
        c2 += "\n# --- Notebook: logtarget + s2_predlag experiment ---\n"
    new_nb["cells"][2]["source"] = _to_source_lines(c2)

    # Cell 38: avoid overwriting other submissions
    c38 = "".join(new_nb["cells"][38].get("source", []))
    c38 = c38.replace(
        "submission_s2_predlag_exp.csv",
        "submission_logtarget_s2_predlag_exp.csv",
    )
    new_nb["cells"][38]["source"] = _to_source_lines(c38)

    dst = Path(src).with_name(
        Path(src).name.replace(
            "_preproc_s2_predlag_exp.ipynb",
            "_preproc_logtarget_s2_predlag_exp.ipynb",
        )
    )
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(new_nb, f, ensure_ascii=False, indent=1)

    print(str(dst))


if __name__ == "__main__":
    main()
