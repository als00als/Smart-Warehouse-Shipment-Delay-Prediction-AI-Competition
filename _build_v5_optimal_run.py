# -*- coding: utf-8 -*-
"""Copy v5_all_methods notebook and apply LB-friendly optimal defaults."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    src = next(ROOT.glob("*strategy_v5_all_methods.ipynb"))
    dst = ROOT / (
        src.name.replace("v5_all_methods", "v5_optimal_run")
        if "v5_all_methods" in src.name
        else f"{src.stem}_optimal_run.ipynb"
    )
    shutil.copyfile(src, dst)
    nb = json.loads(dst.read_text(encoding="utf-8"))

    preset_md = {
        "cell_type": "markdown",
        "metadata": {},
        "id": "v5-optimal-preset-md",
        "source": (
            "### `v5_optimal_run` 기본 프리셋 (바로 실행용)\n\n"
            "이전 대화에서 **10-fold + OOF 자동 선택(timeslot α / meta)** 은 OOF는 좋아졌으나 "
            "**LB가 크게 악화**된 케이스가 있어, 제출·일반화 우선으로 아래를 고정했습니다.\n\n"
            "| 항목 | 값 |\n"
            "|------|-----|\n"
            "| `RUN_PROFILE` | `'quick'` → **5-fold** |\n"
            "| `FORCE_FINAL_METHOD` | `'joint'` (joint grid 결과만 사용) |\n"
            "| `ENABLE_TIMESLOT_ALPHA_SEARCH` | `False` |\n"
            "| `ENABLE_OOF_META_STACK` | `False` |\n"
            "| `AUTO_SELECT_FINAL_METHOD` | `False` |\n\n"
            "더 강한 OOF를 원하면 `RUN_PROFILE='confirm'` 및 v5 플래그를 다시 켜서 실험하세요.\n"
        ).splitlines(keepends=True),
    }

    inserted = False
    new_cells = []
    for cell in nb["cells"]:
        new_cells.append(cell)
        if (
            not inserted
            and cell.get("cell_type") == "markdown"
            and cell.get("id") == "4c7c42f6"
            and "### 실행 파라미터" in "".join(cell.get("source", []))
        ):
            new_cells.insert(len(new_cells) - 1, preset_md)
            inserted = True
    nb["cells"] = new_cells

    cfg_repls = [
        ("RUN_PROFILE = 'confirm'  # 'quick' or 'confirm'\n", "RUN_PROFILE = 'quick'  # 'quick' or 'confirm'\n"),
        ("ENABLE_TIMESLOT_ALPHA_SEARCH = True\n", "ENABLE_TIMESLOT_ALPHA_SEARCH = False\n"),
        ("ENABLE_OOF_META_STACK = True\n", "ENABLE_OOF_META_STACK = False\n"),
        ("AUTO_SELECT_FINAL_METHOD = True  # pick best OOF among joint/ts_alpha/meta\n", "AUTO_SELECT_FINAL_METHOD = False  # optimal run: fixed joint\n"),
        (
            "FORCE_FINAL_METHOD = ''  # '', 'joint', 'timeslot_alpha', 'meta_ridge'\n",
            "FORCE_FINAL_METHOD = 'joint'  # '', 'joint', 'timeslot_alpha', 'meta_ridge'\n",
        ),
    ]
    sub_old = "submission_logtarget_s2_predlag_autoencoder_v5_all_methods.csv"
    sub_new = "submission_logtarget_s2_predlag_autoencoder_v5_optimal_run.csv"

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell.get("source", []))
        if "SEED = 42" in s and "ENABLE_TIMESLOT_ALPHA_SEARCH" in s:
            for old, new in cfg_repls:
                if old in s:
                    s = s.replace(old, new)
            cell["source"] = s.splitlines(keepends=True)
            break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell.get("source", []))
        if sub_old in s:
            cell["source"] = s.replace(sub_old, sub_new).splitlines(keepends=True)

    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", dst)


if __name__ == "__main__":
    main()
