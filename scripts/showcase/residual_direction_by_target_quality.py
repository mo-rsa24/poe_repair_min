"""Does a training cell whose joint-prompt render shows the wrong picture carry a wrong r_t?

The `break` on claim 7 of the "improving the pooled LoRA run" idea walk. Claim 7 wants to drop
cells whose `mono.png` shows one concept twice. But the adapter is never trained on the picture:
it is trained to predict

    Delta_t = gs * (eps_j_raw - eps_a_raw - eps_b_raw + eps_uncond)

at each cached state (`poe_repair/training_cache.py:13`), which is a property of the score fields
under the joint prompt, not of where one deterministic trajectory happened to land. If a bad
endpoint is a sampling accident rather than a corrupt target, the corrections in bad cells point
the same way as those in good cells and dropping them throws away usable data.

The read is a cosine between per-step corrections, with two baselines that bound it:

    ceiling  same pair, good cell against good cell
    floor    different pair, to show what an unrelated correction looks like

If bad-against-good sits at the ceiling, the targets are fine and claim 7 dies as a training
change. If it sits at the floor, the endpoint marks a corrupt target and cleaning is justified.

Ground truth for good and bad is by eye, recorded in `run-05-target-quality-instrument.md`, because
the automated read's false-negative rate on similar pairs is exactly what made it an upper bound.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/train")
OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality")
GS = 7.5

# Eye labels, lion x tiger, from the contact sheet in run 5. Seeds 1, 3, 4 and 6 show a lion
# beside a tiger; seed 5 is a single blended face; seeds 2, 7 and 8 carry a mane with tiger
# stripes on the same head and are not called either way.
GOOD = [1, 3, 4, 6]
BAD = [5]
AMBIGUOUS = [2, 7, 8]
PAIR = "a_lion__x__a_tiger"
OTHER_PAIR = "a_horse__x__a_zebra"

WINDOWS = {"early": range(0, 5), "commit": range(5, 25), "late": range(25, 50)}


def delta(pair: str, seed: int, step: int) -> torch.Tensor:
    d = torch.load(CACHE / pair / f"seed_{seed}" / "residuals" / f"step_{step:03d}.pt",
                   map_location="cpu", weights_only=False)
    return (GS * (d["eps_j_raw"] - d["eps_a_raw"] - d["eps_b_raw"] + d["eps_uncond"])).flatten()


def cos_over_steps(a: tuple[str, int], b: tuple[str, int]) -> dict[str, float]:
    out = {}
    for name, steps in WINDOWS.items():
        vals = [float(torch.nn.functional.cosine_similarity(
            delta(*a, s), delta(*b, s), dim=0)) for s in steps]
        out[name] = sum(vals) / len(vals)
    return out


def mean_of(pairs_of_cells) -> dict[str, float]:
    rows = [cos_over_steps(a, b) for a, b in pairs_of_cells]
    return {w: round(sum(r[w] for r in rows) / len(rows), 3) for w in WINDOWS} | {"n": len(rows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    res["ceiling: good vs good, same pair"] = mean_of(
        [((PAIR, i), (PAIR, j)) for i, j in itertools.combinations(GOOD, 2)])
    res["test: bad vs good, same pair"] = mean_of(
        [((PAIR, b), (PAIR, g)) for b in BAD for g in GOOD])
    res["ambiguous vs good, same pair"] = mean_of(
        [((PAIR, a), (PAIR, g)) for a in AMBIGUOUS for g in GOOD])
    res["floor: different pair, same seeds"] = mean_of(
        [((PAIR, g), (OTHER_PAIR, g)) for g in GOOD])

    print(f"{'comparison':<38} {'n':>3} {'early':>7} {'commit':>7} {'late':>7}")
    for k, v in res.items():
        print(f"{k:<38} {v['n']:>3} {v['early']:>7.3f} {v['commit']:>7.3f} {v['late']:>7.3f}")

    (OUT / "residual_direction_by_target_quality.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'residual_direction_by_target_quality.json'}")


def _entry() -> None:
    main()
    run_lean()


# ---------------------------------------------------------------------------
# The cross-cell cosine above is confounded and its result is recorded as a null:
# the same-pair good-to-good ceiling (0.004 commit, 0.001 late) sits on top of the
# different-pair floor (0.007, 0.002), because corrections at different states are
# orthogonal whatever the target shows. A good cell and a bad cell never share a
# state, so no cross-cell cosine can separate them.
#
# The question asked inside one cell instead: does the joint prompt's own score
# lean toward one of the two concepts? Both expert branches are cached at the same
# state as the joint branch, so this needs no comparison against another cell and
# no extra forward pass.
#
#     lean_t = cos(eps_j - eps_uncond, eps_a - eps_uncond)
#            - cos(eps_j - eps_uncond, eps_b - eps_uncond)
#
# Zero means the joint prompt sits evenly between its two concepts at this state.
# A large magnitude means it has committed to one of them, which is what a target
# rendering the same animal twice should look like.
# ---------------------------------------------------------------------------

def lean(pair: str, seed: int, step: int) -> float:
    d = torch.load(CACHE / pair / f"seed_{seed}" / "residuals" / f"step_{step:03d}.pt",
                   map_location="cpu", weights_only=False)
    j = (d["eps_j_raw"] - d["eps_uncond"]).flatten()
    a = (d["eps_a_raw"] - d["eps_uncond"]).flatten()
    b = (d["eps_b_raw"] - d["eps_uncond"]).flatten()
    cs = torch.nn.functional.cosine_similarity
    return float(cs(j, a, dim=0) - cs(j, b, dim=0))


def lean_profile(pair: str, seed: int) -> dict[str, float]:
    return {w: round(sum(lean(pair, seed, s) for s in steps) / len(list(steps)), 3)
            for w, steps in WINDOWS.items()}


def run_lean() -> None:
    labels = {**{s: "good" for s in GOOD}, **{s: "BAD" for s in BAD},
              **{s: "ambig" for s in AMBIGUOUS}}
    print(f"\n{PAIR}: does the joint score lean to one expert? (+ = toward 'a lion')")
    print(f"{'seed':>5} {'eye':>6} {'early':>7} {'commit':>7} {'late':>7}")
    rows = {}
    for seed in sorted(labels):
        p = lean_profile(PAIR, seed)
        rows[seed] = {"eye": labels[seed], **p}
        print(f"{seed:>5} {labels[seed]:>6} {p['early']:>7.3f} {p['commit']:>7.3f} {p['late']:>7.3f}")
    (OUT / "joint_lean_lion_tiger.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT / 'joint_lean_lion_tiger.json'}")


if __name__ == "__main__":
    _entry()
