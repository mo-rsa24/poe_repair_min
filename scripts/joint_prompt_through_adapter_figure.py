#!/usr/bin/env python
"""The paper's joint-prompt-through-the-adapter figure, laid out two rows by four columns.

Top row: the joint prompt on the base model. Bottom row: the same prompt and noise through the
adapter (v58-03-self at step 50,000). Columns are the four pairs, all at seed 9.

    python scripts/joint_prompt_through_adapter_figure.py <tiles_dir> <out.png>

`tiles_dir` holds `<pair>/seed09/{mono,monolora_v58self_w00}.png` from the ablation_grid tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_study_grids import ADAPTER_NOTE, grid  # noqa: E402

PAIRS = [("a_cat__x__a_dog", "Cat and dog"), ("a_tiger__x__a_dog", "Tiger and dog"),
         ("a_chess_board__x__an_hourglass", "Chess board and hourglass"),
         ("a_picnic_basket__x__a_watermelon", "Picnic basket and watermelon")]
SEED = 9


def main():
    tiles, out = Path(sys.argv[1]), Path(sys.argv[2])

    def row(key: str):
        return lambda slug: tiles / slug / f"seed{SEED:02d}" / f"{key}.png"

    grid([("Joint prompt, base model", row("mono")),
          ("Joint prompt, through the adapter", row("monolora_v58self_w00"))],
         [(slug, [name]) for slug, name in PAIRS], out, "", "",
         note=ADAPTER_NOTE, vertical_labels=True)


if __name__ == "__main__":
    main()
