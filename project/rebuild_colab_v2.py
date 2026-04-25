#!/usr/bin/env python3
"""Apply v2 cell pack to colab_training.ipynb. Reads snippets from colab_notebook_rebuild/."""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REBUILD = HERE / "colab_notebook_rebuild"
NB = HERE / "colab_training.ipynb"


def load_txt(name: str) -> str:
    return (REBUILD / name).read_text()


def to_src_lines(s: str) -> list:
    if not s.endswith("\n"):
        s += "\n"
    return [s]


def code_cell(s: str, template: dict) -> dict:
    c = copy.deepcopy(template)
    c["cell_type"] = "code"
    c["source"] = to_src_lines(s)
    c["outputs"] = []
    c["execution_count"] = None
    return c


def main() -> None:
    with NB.open() as f:
        nb = json.load(f)
    orig = nb["cells"]

    c5, c7, c10, c12, c13 = orig[5], orig[7], orig[10], orig[12], orig[13]
    c15 = orig[15]
    c21, c23, c25, c27 = orig[21], orig[23], orig[25], orig[27]
    c32, c34, c36 = orig[32], orig[34], orig[36]

    out: list = []
    i, n = 0, len(orig)
    while i < n:
        if i == 5:
            out.append(code_cell(load_txt("f1_repo.txt"), c5))
        elif i == 7:
            out.append(code_cell(load_txt("f2_verify.txt"), c7))
        elif i == 10:
            out.append(code_cell(load_txt("f3_tasks.txt"), c10))
        elif i == 12:
            out.append(code_cell(load_txt("f4_dataset.txt"), c12))
        elif i == 13:
            i += 1
            continue
        elif i == 15:
            out.append(code_cell(load_txt("f5a_heuristic.txt"), c15))
            out.append(code_cell(load_txt("f5b_cisandbox.txt"), c15))
        elif 16 <= i <= 19:
            i += 1
            continue
        elif i == 21:
            out.append(code_cell(load_txt("f6_hidden.txt"), c21))
        elif i == 23:
            out.append(code_cell(load_txt("f7_model.txt"), c23))
        elif i == 25:
            out.append(code_cell(load_txt("f8_reward.txt"), c25))
        elif i == 27:
            out.append(code_cell(load_txt("f9_grpo.txt"), c27))
        elif i == 32:
            out.append(code_cell(load_txt("f10a_eval.txt"), c32))
        elif i == 34:
            out.append(code_cell(load_txt("f10b_trace.txt"), c34))
        elif i == 36:
            out.append(code_cell(load_txt("f11_save.txt"), c36))
        else:
            out.append(copy.deepcopy(orig[i]))
        i += 1

    out.append(code_cell(load_txt("verify.txt"), c36))
    nb["cells"] = out
    with NB.open("w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Wrote", NB, f"({len(out)} cells)")


if __name__ == "__main__":
    main()
