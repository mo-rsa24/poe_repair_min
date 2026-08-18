#!/usr/bin/env python
"""D3b: what the correction depends on, one factor changed at a time.

D2 changes the starting noise and finds the correction shares nothing. D3
changes the pair of animals and finds the same. Neither says which of the two
the correction is actually a function of, because neither holds one fixed
while moving the other. This does.

Two things can differ between any two cached runs: the starting noise, and the
prompt. Cells with the same seed start from IDENTICAL noise whatever the pair
(checked: cosine +1.0000), so all four combinations are already on disk:

    same noise, same prompt        the ceiling: a rerun of one cell
    different noise, same prompt   one pair, two seeds
    same noise, different prompt   two pairs, one seed
    both different                 two pairs, two seeds

Each bar is the direction agreement between two runs' corrections at matched
steps, split into the first 3 steps and steps 10 to 49, because the answer is
completely different in the two halves of the run.

The ceiling row is one measurement rather than a population: the same cell
rerun from its own starting noise. It does not reach 1.0 because fp16
nondeterminism compounds through the sampler (drift 0.24% at step 10, 17.1% at
step 49), and every other row is read against it rather than against 1.0.

    python scripts/make_d3b.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import load_cell  # noqa: E402

PAIRS = (
    "a_leopard__x__a_jaguar", "a_frog__x__a_toad", "an_eagle__x__a_hawk",
    "a_seal__x__a_walrus", "a_goose__x__a_swan", "a_cow__x__a_buffalo",
    "a_cat__x__a_dog", "an_elephant__x__a_penguin",
)
SEEDS = (9, 10, 11, 12, 13)
OUT_DIR = Path("paper/iclr/figures")
FIG_NAME = "D3b-what-the-correction-depends-on"
EARLY, LATE = "#1f77b4", "#d62728"
# From scripts/rt_noise_interpolation.py's zero-distance run, which reruns one
# cell from its own starting noise and compares against the cache.
CEILING = {"early": 0.996, "late": 0.427}


def main() -> int:
    runs = {}
    for p in PAIRS:
        for s in SEEDS:
            try:
                runs[(p, s)] = load_cell(p, s).r_t().float().flatten(1)
            except Exception:
                continue
    if len(runs) < 10:
        raise SystemExit(f"only {len(runs)} cached runs found, need the pool")

    buckets = {"different noise,\nsame prompt": [],
               "same noise,\ndifferent prompt": [],
               "different noise,\ndifferent prompt": []}
    for (p1, s1), (p2, s2) in combinations(sorted(runs), 2):
        c = torch.nn.functional.cosine_similarity(runs[(p1, s1)],
                                                  runs[(p2, s2)], dim=1)
        v = (float(c[:3].median()), float(c[10:].median()))
        if s1 != s2 and p1 == p2:
            buckets["different noise,\nsame prompt"].append(v)
        elif s1 == s2 and p1 != p2:
            buckets["same noise,\ndifferent prompt"].append(v)
        elif s1 != s2 and p1 != p2:
            buckets["different noise,\ndifferent prompt"].append(v)

    labels = ["same noise,\nsame prompt"] + list(buckets)
    early = [CEILING["early"]] + [float(np.median(np.array(v)[:, 0]))
                                  for v in buckets.values()]
    late = [CEILING["late"]] + [float(np.median(np.array(v)[:, 1]))
                                for v in buckets.values()]
    counts = ["1 rerun"] + [f"{len(v)} pairs of runs" for v in buckets.values()]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    })
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    y = np.arange(len(labels))[::-1]
    h = 0.32
    ax.barh(y + h / 2, early, height=h, color=EARLY, label="first 3 steps")
    ax.barh(y - h / 2, late, height=h, color=LATE, label="steps 10 to 49")
    for yi, e, l, n in zip(y, early, late, counts):
        ax.text(max(e, l) + 0.02, yi, f"{e:+.2f} / {l:+.2f}   {n}",
                va="center", fontsize=6.5, color="0.35")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("direction agreement of the two corrections (cosine)")
    ax.axvline(0, color="0.8", lw=0.6)
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=300)
    plt.close(fig)

    (OUT_DIR / f"{FIG_NAME}.json").write_text(json.dumps({
        "pairs": PAIRS, "seeds": SEEDS, "n_runs": len(runs),
        "rows": [{"condition": lab.replace("\n", " "), "first3": e,
                  "late": l, "n": n}
                 for lab, e, l, n in zip(labels, early, late, counts)],
        "ceiling_note": "the first row is one rerun of a single cell from its "
                        "own starting noise; it falls short of 1.0 because "
                        "fp16 nondeterminism compounds through the sampler",
        "caption_cap": "same-seed cells share identical starting noise only at "
                       "step 0; their trajectories separate as the prompts "
                       "differ, so the 'same noise' rows stop being "
                       "same-state after the first steps. That is why the "
                       "early and late columns are reported separately and "
                       "the claim is made on the early one.",
    }, indent=2))
    print(f"wrote {OUT_DIR / FIG_NAME}.png and .pdf")
    for lab, e, l, n in zip(labels, early, late, counts):
        print(f"  {lab.replace(chr(10),' '):40} {e:+.3f} / {l:+.3f}  ({n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
