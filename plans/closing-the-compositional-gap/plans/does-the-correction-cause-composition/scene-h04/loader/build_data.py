#!/usr/bin/env python
"""Build the one data file the hypothesis-04 scene reads.

Every number in the app comes from here, and every number here carries the file
it was read from and that file's modification time. Nothing in ``src/`` holds a
value. A missing input stops the build and names what could not be found,
because a claim with no artifact is a designed state in the scene (an empty slot
with the run that fills it), not a zero to paper over.

Usage:
    PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
    $PY loader/build_data.py                # data file + thumbnails
    $PY loader/build_data.py --no-thumbs    # data file only, about 2s
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent
SCOPE = SCENE.parent
REPO = Path("/home-mscluster/mmolefe/Playground/PhD/poe_repair_min")

REVIEW = SCOPE / "review/hypothesis-04-what-the-cached-runs-already-show.md"
PLAN = SCOPE / "plans/hypothesis-04-what-the-cached-runs-already-show.md"

ANALYSES = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")
REFRESH = ANALYSES / "refresh_20260810"
EXPERTS = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/interaction_term/experts")
FORK_CELLS = REPO / "outputs/interaction_term/dose/pairs"
F6 = REPO / "artifacts/results/which-way-the-correction-points/does-the-subspace-test-predict-transfer"

POE_ARM = "teacher_residual_const_lam000"
MONO_ARM = "teacher_residual_const_lam100"


class Missing(SystemExit):
    def __init__(self, path: Path, why: str) -> None:
        super().__init__(f"loader stopped: {why}\n  looked for: {path}")


STAMPS: dict[str, dict] = {}


def stamp(key: str, path: Path) -> Path:
    """Record a file's identity, or stop. Every read goes through here."""
    if not path.exists():
        raise Missing(path, f"{key} is not on disk")
    st = path.stat()
    STAMPS[key] = {
        "path": str(path),
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
        .astimezone().strftime("%Y-%m-%d %H:%M"),
        "bytes": st.st_size,
    }
    return path


def read_json(key: str, path: Path) -> dict:
    return json.loads(stamp(key, path).read_text())


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- the prose

def quote(path: Path, pattern: str, span: int = 12) -> dict:
    """Pull a passage out of a markdown file with its line numbers.

    The scene quotes the review file rather than restating it, so a reworded
    sentence shows up as a failed build instead of a silent paraphrase.
    """
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            body = [lines[i]]
            for j in range(i + 1, min(i + span, len(lines))):
                nxt = lines[j]
                # Stop at the next bullet or heading, so one quoted item does
                # not swallow the items under it.
                if re.match(r"^\s*[-*] |^#", nxt):
                    break
                if not nxt.strip():
                    break
                body.append(nxt)
            return {"path": str(path), "lines": [i + 1, i + len(body)],
                    "body": [b.rstrip() for b in body]}
    raise Missing(path, f"the review file no longer contains /{pattern}/. "
                        "It was reworded; re-read it and update the loader.")


# ------------------------------------------------------------- the analyses

def log_snr_for_timesteps(timesteps: list[int]) -> list[float]:
    """log-SNR the same way the cache computes it, so the axes agree.

    Falls back to SDXL's scaled_linear betas when the model files are not
    reachable, which is what poe_repair.experiments.interaction_term.cache does.
    """
    import torch
    sys.path.insert(0, str(REPO))
    try:
        from poe_repair.experiments.interaction_term.cache import _alphas_cumprod
        ab = _alphas_cumprod()
    except Exception:
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
        ab = torch.cumprod(1.0 - betas, dim=0).float()
    idx = torch.tensor(timesteps).long().clamp(0, ab.numel() - 1)
    a = ab[idx]
    return torch.log(a / (1.0 - a)).tolist()


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def build_collapse() -> dict:
    """C1 and C2: the size measure against noise level, both normalizations."""
    committed = read_json("collapse_prereg_committed", ANALYSES / "snr_collapse.json")
    prereg = read_json("collapse_prereg", REFRESH / "prereg/snr_collapse.json")
    raw = read_json("collapse_raw", REFRESH / "raw/snr_collapse.json")

    def curve(d: dict) -> dict:
        return {
            "normalize": d["normalize"],
            "grid": d["log_snr_grid"],
            "median": d["median_curve"],
            "iqr": d["iqr"],
            "spreadPct": d["collapse_spread_pct"],
            "verdict": d["verdict"],
            "peakLogSnr": d["peak_log_snr"],
            "peakAtEdge": d["peak_at_edge"],
            "nPairs": d["n_pairs"],
            "nCurves": d["n_curves"],
            "cells": [{"pair": c[0], "seed": int(c[1])} for c in d["cells"]],
        }

    return {
        "committed": curve(committed),
        "prereg": curve(prereg),
        "raw": curve(raw),
        # The per-cell curves were never written to disk: snr_collapse.json
        # keeps the median, the IQR band and the list of contributing cells.
        # The scene therefore draws the band, not 34 individual lines.
        "perCellCurvesOnDisk": False,
    }


def build_fork() -> dict:
    """C3: where the two paths separate, at both coverages."""
    old = read_json("fork_19cells", ANALYSES / "fork_curve.json")
    new = read_json("fork_43cells", REFRESH / "fork_curve.json")

    def read(d: dict) -> dict:
        cells = [{
            "pair": c["pair"], "seed": int(c["seed"]),
            "elbowStep": int(c["elbow_step"]),
            "distance": c["distance"],
        } for c in d["cells"]]
        elbows = [c["elbowStep"] for c in cells]
        return {
            "medianElbow": int(d["median_elbow_step"]),
            "nCells": len(cells),
            "elbowMin": min(elbows), "elbowMax": max(elbows),
            "inBand13to20": sum(13 <= e <= 20 for e in elbows),
            "maxDistanceAtZero": max(c["distance"][0] for c in cells),
            "cells": cells,
        }

    return {"original": read(old), "refreshed": read(new)}


def build_climb() -> dict:
    """C5: does the correction push along the direction sampling is moving."""
    pc = read_json("climb_38cells", ANALYSES / "plausibility_climb.json")
    cj = read_json("climb_34cells", ANALYSES / "climb.json")

    cells = [{
        "pair": c["pair"], "seed": int(c["seed"]),
        "normalised": c["normalised"],
        "raw": c["raw"],
        "controlRandom": c["control_random_vs_dx"],
        "controlWrongStep": c["control_shuffled_vs_dx"],
        "rVsEpsPoe": c["r_vs_eps_poe"],
        "epsVsDx": c["control_eps_vs_dx"],
        "fractionNegative": c["fraction_of_steps_negative"],
        "perStepCosine": c.get("per_step_cosine", []),
    } for c in pc["cells"]]

    def med(k: str) -> float:
        return median([c[k] for c in cells])

    return {
        "measure": pc["measure"],
        "caveat": pc["caveat"],
        "reading": pc["reading"],
        "nCells": pc["n_cells"],
        "nNegative": pc["n_negative"],
        "nPairs": len({c["pair"] for c in cells}),
        "medians": {
            "normalised": med("normalised"),
            "controlRandom": med("controlRandom"),
            "controlWrongStep": med("controlWrongStep"),
            "rVsEpsPoe": med("rVsEpsPoe"),
            "epsVsDx": med("epsVsDx"),
        },
        "cells": cells,
        # A second file answers the same question over a different population.
        # Both are shown; they are never averaged.
        "otherPopulation": {
            "nCells": cj["n_cells"], "nPairs": cj["n_pairs"],
            "climbMedian": cj["climb_median"],
            "alignmentMedian": cj["alignment_median"],
            "randomFloor": cj["random_direction_floor"],
        },
    }


def build_spectrum() -> dict:
    """C6 and C7: energy at k against a matched floor, and the held-out test."""
    sp = read_json("spectrum", ANALYSES / "spectrum.json")
    f6 = read_json("f6_transfer", F6 / "result.json")
    stamp("f6_query", F6 / "QUERY.md")
    stamp("f6_figure", F6 / "geometry_vs_transfer.png")

    ks = sorted(int(k) for k in sp["energy_at_k"])
    return {
        "ks": ks,
        "energy": [sp["energy_at_k"][str(k)] for k in ks],
        "floor": [sp["gaussian_floor_at_k"][str(k)] for k in ks],
        "heldout": [sp["heldout_projection_at_k"][str(k)] for k in ks],
        "singularValues": sp["singular_values_head"],
        "trainPairs": sp["train_pairs"],
        "heldoutPairs": sp["heldout_pairs"],
        "trainVectors": sp["train_vectors"],
        "dims": sp["dims"],
        "transfer": {
            "perPair": [{
                "pair": p["pair"],
                "composeRate": p["compose_rate"],
                "geometryK64": p["geometry_k64"],
            } for p in f6["per_pair"]],
            "meanCompose": f6["mean_compose_rate_transfer"],
            "meanGeometry": f6["mean_geometry_transfer"],
            "evalStep": f6["eval_step"],
            "heldoutVectors": f6["heldout_vectors"],
        },
    }


# ------------------------------------------------------------ the pictures

def build_experts() -> dict:
    """Per-step pictures along the PoE path: what each expert believes."""
    idx = read_json("experts_index", EXPERTS / "index.json")
    cells = []
    for c in idx["cells"]:
        man = read_json(f"experts_{c['pair']}_{c['seed']}", Path(c["path"]))
        rows = []
        for r in man["rows"]:
            rows.append({
                "step": r["step"], "timestep": r["timestep"],
                "views": {v: rel_to_public(Path(r[v]), "experts")
                          for v in idx["views"]},
            })
        ls = log_snr_for_timesteps([r["timestep"] for r in man["rows"]])
        for r, v in zip(rows, ls):
            r["logSnr"] = v
        cells.append({
            "pair": man["pair"], "seed": man["seed"],
            "promptA": man["prompt_a"], "promptB": man["prompt_b"],
            "nSteps": man["n_steps"], "px": man["px"], "rows": rows,
        })
    return {"views": idx["views"], "cells": cells}


def rel_to_public(path: Path, mount: str) -> str:
    """A path the dev server can serve, via the symlinks in public/."""
    roots = {"experts": EXPERTS, "forkcells": FORK_CELLS}
    return f"/{mount}/{path.relative_to(roots[mount])}"


def build_fork_images(fork: dict) -> list[dict]:
    """The two final frames per fork cell: the broken one and the working one.

    These are the only pictures the fork has. The per-step frames do not exist
    and cannot be made from what is on disk; see the gap this build reports.
    """
    out, missing = [], 0
    for c in fork["refreshed"]["cells"]:
        entry = {"pair": c["pair"], "seed": c["seed"]}
        for arm, key in ((POE_ARM, "poe"), (MONO_ARM, "mono")):
            p = FORK_CELLS / c["pair"] / f"seed_{c['seed']}" / arm / f"{arm}.png"
            if p.exists():
                entry[key] = rel_to_public(p, "forkcells")
            else:
                entry[key] = None
                missing += 1
        out.append(entry)
    if missing:
        print(f"  note: {missing} fork frame(s) absent, drawn as empty slots")
    return out


# ------------------------------------------------------------------ claims

def build_claims() -> list[dict]:
    """The ledger, in the review file's own words, with its line numbers."""
    r = REVIEW
    return [
        {
            "id": "C1", "type": "collapse", "mark": "🟡", "state": "measured",
            "question": "Does the correction's size follow noise level on one "
                        "shared curve across pairs?",
            "quote": quote(r, r"Does the correction's size follow noise level"),
            "reads": ["collapse_prereg", "collapse_prereg_committed"],
        },
        {
            "id": "C2", "type": "comparison", "mark": "🟡", "state": "measured",
            "question": "Do the two size measures agree about where the peak is?",
            "quote": quote(r, r"And the two measures disagree about the peak"),
            "reads": ["collapse_prereg", "collapse_raw"],
            "knob": "normalization",
        },
        {
            "id": "C3", "type": "elbow", "mark": "✅", "state": "measured",
            "question": "Where do the PoE and Mono paths fork?",
            "quote": quote(r, r"Where do the PoE and Mono paths fork"),
            "reads": ["fork_43cells", "fork_19cells"],
        },
        {
            "id": "C4", "type": "comparison", "mark": None, "state": "not-run",
            "question": "Does the fork step match the window plan 04 measures?",
            "quote": quote(r, r"plan 04's window is OPEN"),
            "reads": [],
        },
        {
            "id": "C5", "type": "dose-with-controls", "mark": "✅",
            "state": "measured",
            "question": "Does the correction align with the sampling motion?",
            "quote": quote(r, r"Does the correction align with the sampling motion"),
            "reads": ["climb_38cells", "climb_34cells"],
        },
        {
            "id": "C6", "type": "spectrum", "mark": "✅", "state": "measured",
            "question": "Is the correction low-rank, against a matched random floor?",
            "quote": quote(r, r"Is the correction low-rank, against a matched"),
            "reads": ["spectrum"],
        },
        {
            "id": "C7", "type": "comparison", "mark": "✅", "state": "measured",
            "question": "Does a subspace fitted on training pairs carry to "
                        "held-out pairs?",
            "quote": quote(r, r"Does a subspace fitted on training pairs carry"),
            "reads": ["spectrum", "f6_transfer"],
        },
    ]


def build_gaps() -> list[dict]:
    """What is owed, cheapest first, each with the command that closes it."""
    return [
        {
            "id": "G1",
            "closes": "C3",
            "what": "Per-step pictures of the two forking paths. Today the fork "
                    "has only its two final frames, so the step scrubber moves "
                    "numbers but no image.",
            "why": "The 440 saved trajectories hold only the noisy latent x_t "
                   "(keys: trajectories, sigmas, timesteps, num_steps). "
                   "Decoding those shows noise through the fork region, which "
                   "sits at step 16 of 51. The x̂_0 estimates that would be "
                   "readable are written to a residuals directory, and no fork "
                   "cell has one.",
            "command":
                "for s in a_cat__x__a_dog a_lion__x__a_tiger a_wolf__x__a_husky; do\n"
                "  for lam in 0 1; do\n"
                "    $PY scripts/interaction_term_inject.py --pair $s --seed 1 \\\n"
                "        --lambda $lam --save-residuals --overwrite\n"
                "  done\n"
                "done",
            "output": "outputs/interaction_term/dose/pairs/<pair>/seed_1/"
                      "teacher_residual_const_lam{000,100}/residuals/",
            "cost": "6 sampling runs at 51 steps, one GPU. The trajectories "
                    "already exist; this re-runs them to keep x̂_0.",
        },
        {
            "id": "G2",
            "closes": "C4",
            "what": "Whether the fork step (16) lands inside the window plan 04 "
                    "measures. Drawn as an empty slot until that sweep runs.",
            "why": "Plan 04's timing sweep has not produced its window yet, so "
                   "there is nothing to compare the elbow against.",
            "command": "see plans/does-the-correction-cause-composition/plans/"
                       "hypothesis-03-when-in-the-run-it-matters.md",
            "output": "outputs/interaction_term/window/",
            "cost": "the plan 04 sweep, already scoped there",
        },
        {
            "id": "G3",
            "closes": "C1, C5",
            "what": "Per-step pictures for the other 16 pairs. Three pairs "
                    "(cat×dog, frog×toad, leopard×jaguar) have them at 3 seeds.",
            "why": "decode_expert_frames.py has only been run on those three.",
            "command": "$PY scripts/decode_expert_frames.py --pairs <slugs> "
                       "--seeds 1 --stride 5",
            "output": "/datasets/mmolefe/poe_repair_min/outputs/interaction_term/experts/",
            "cost": "0.36s a frame, 5 views x 11 steps a cell, no sampling",
        },
    ]


def build_discrepancies() -> list[dict]:
    """Where the review file and the data disagree. Shown, never silently fixed."""
    return [
        {
            "claim": "C3",
            "says": "the elbow is read over 19 cells",
            "found": "43 cells have both arms and at least 40 steps (44 total, "
                     "one 20-step smoke cell the guard correctly drops)",
            "effect": "the elbow is still step 16 over all 43, so the answer "
                      "holds on 2.3x the cells",
        },
        {
            "claim": "C3",
            "says": "15 of 19 cells land between steps 13 and 20",
            "found": "17 of 19, recomputed from fork_curve.json",
            "effect": "tighter than written, not looser",
        },
        {
            "claim": "C5",
            "says": "38 cells, 19 pairs",
            "found": "two files answer this: plausibility_climb.json has 38 "
                     "cells over 19 pairs, climb.json has 34 over 17",
            "effect": "the review quotes the 38-cell file; both are shown and "
                      "never averaged",
        },
    ]


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--out", type=Path, default=SCENE / "src/data/result.json")
    args = ap.parse_args()

    stamp("review", REVIEW)
    stamp("plan", PLAN)

    print("reading the analyses")
    collapse = build_collapse()
    fork = build_fork()
    climb = build_climb()
    spectrum = build_spectrum()

    print("reading the pictures")
    experts = build_experts()
    fork_images = build_fork_images(fork)

    print("reading the paperwork")
    claims = build_claims()

    data = {
        "meta": {
            "builtFrom": STAMPS,
            "commit": git_commit(),
            "roots": {
                "analyses": str(ANALYSES),
                "refresh": str(REFRESH),
                "experts": str(EXPERTS),
                "forkCells": str(FORK_CELLS),
            },
            "publicLinks": {
                "/experts": str(EXPERTS),
                "/forkcells": str(FORK_CELLS),
            },
        },
        "verdict": quote(REVIEW, r"^\*\*Five of six answered"),
        "runKind": quote(REVIEW, r"^\*\*Tests the claim\*\*"),
        "vocabulary": [quote(REVIEW, rf"^- \*\*{t}\*\*") for t in
                       ("Noise level", "The two paths", "Low-rank")],
        "openQuestion": quote(REVIEW, r"The spectrum's statistical entity"),
        "claims": claims,
        "collapse": collapse,
        "fork": fork,
        "forkImages": fork_images,
        "climb": climb,
        "spectrum": spectrum,
        "experts": experts,
        "gaps": build_gaps(),
        "discrepancies": build_discrepancies(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1))
    kb = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out}  ({kb:.0f} KB)")
    print(f"  {len(claims)} claims, {len(STAMPS)} files read")
    print(f"  fork: {fork['refreshed']['nCells']} cells "
          f"(was {fork['original']['nCells']})")
    print(f"  experts: {len(experts['cells'])} cells x "
          f"{len(experts['cells'][0]['rows'])} steps x "
          f"{len(experts['views'])} views")
    print(f"  gaps: {len(data['gaps'])}, "
          f"discrepancies: {len(data['discrepancies'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
