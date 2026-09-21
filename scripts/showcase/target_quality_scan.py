"""Score every cached joint-prompt target (`mono.png`) two ways.

Claim 7 of the "improving the pooled LoRA run" idea walk. Each training cell's
target is the joint-prompt prediction on that seed's trajectory, and `mono.png`
is the picture of that target. Where the picture does not show both concepts,
the correction the adapter is fitted toward at all 50 steps of that cell points
at the wrong thing.

Two reads, because the compose scorer alone cannot answer this:

  1. GENERIC, the shipped compose read: `count_instances()` queries the term
     "animal" and counts distinct boxes. A target showing two cats returns 2 and
     passes. That is exactly the failure this claim is about, so the generic read
     is recorded as the baseline, not as the answer.

  2. PER-NAME, REJECTED at validation: GroundingDINO run once per animal word.
     It is inverted on the case that matters. On the two-cat target of cat x dog
     seed 14 it returns "a dog" at 0.787, above the 0.615 it gives the real dog in
     seed 9. An open-vocabulary detector will put a confident box for any animal
     word on any animal, so detection was never the missing piece.

  3. FORCED CHOICE, the read that works: detect boxes with the generic "animal"
     query, crop each one, and make CLIP choose between the pair's two names. A
     target is good when both names win at least one box. Discrimination between
     two candidates is the question; open-vocabulary detection is not.

The compose scorer keeps its generic query and is NOT changed here. Its own
docstring records why: per-query reads under-count real composes on the hard
pairs, a measurement bias. Target quality is a different question and gets its
own instrument.

The bar was set by `--validate` against eight cat x dog targets labelled by eye,
before any sweep ran. The forced-choice read scores 7 of 8. Its one error is seed
15, a border collie called "a cat" at 0.53, a coin flip at the decision boundary.
That error direction is a false negative: it calls a good target bad, so the read
can only drop usable cells, never keep broken ones. Both other reads score 0 of 3
on the bad targets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from PIL import Image

from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
    MIN_BOX_FRACTION,
    count_instances,
)
from poe_repair.experiments.residual_between_mono_and_poe import metrics as vmetrics

CACHE = Path("/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache")
OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality")

# Images whose answer is already known, from run 2's render notes and the
# report finding. Used by --validate to set the per-name bar before any sweep
# result is believed.
KNOWN = [
    ("heldout", "a_cat__x__a_dog", "seed_14", "one animal (report: never composes at any rank)"),
    ("heldout", "a_cat__x__a_dog", "seed_15", "two cats (run 2 render note)"),
    ("heldout", "a_cat__x__a_dog", "seed_9",  "a cat and a dog (run 2 render note)"),
    ("heldout", "a_cat__x__a_dog", "seed_10", "a cat and a dog (run 2 render note)"),
    ("heldout", "a_cat__x__a_dog", "seed_11", "a dog with two cats (run 2 render note)"),
    ("heldout", "a_cat__x__a_dog", "seed_12", "a cat and a dog (run 2 render note)"),
    ("heldout", "a_cat__x__a_dog", "seed_13", "a cat and a dog (run 2 render note)"),
    ("heldout", "a_cat__x__a_dog", "seed_16", "a cat and a dog (run 2 render note)"),
]


def queries_from_slug(slug: str) -> tuple[str, str]:
    """`a_cat__x__a_dog` -> ("a cat", "a dog")."""
    a, b = slug.split("__x__")
    return a.replace("_", " "), b.replace("_", " ")


def forced_choice_read(path: Path, qa: str, qb: str, device) -> dict:
    """Crop each generically-detected animal and make CLIP pick which of the two names it is."""
    cache = vmetrics._get_clip(device)
    img = Image.open(path).convert("RGB")
    floor = MIN_BOX_FRACTION * max(img.size)
    crops = []
    for d in vmetrics.detect_boxes(path, ["animal"], device=device):
        x0, y0, x1, y1 = d["box"]
        if max(x1 - x0, y1 - y0) < floor:
            continue
        crops.append(img.crop((max(0, x0), max(0, y0), x1, y1)))
    if not crops:
        return {"n_boxes": 0, "picks": [], "both_present": False}
    inp = cache.processor(text=[qa, qb], images=crops, return_tensors="pt",
                          padding=True).to(cache.device)
    with torch.no_grad():
        probs = cache.model(**inp).logits_per_image.softmax(dim=1)
    picks = [(qa if p[0] > p[1] else qb, round(float(p.max()), 3)) for p in probs]
    names = {n for n, _ in picks}
    return {"n_boxes": len(crops), "picks": picks,
            "both_present": bool(qa in names and qb in names)}


def per_name_read(path: Path, qa: str, qb: str, device) -> dict:
    """REJECTED at validation, kept as the recorded baseline. See the module docstring."""
    img_side = max(Image.open(path).size)
    floor = MIN_BOX_FRACTION * img_side
    out = {}
    for key, q in (("a", qa), ("b", qb)):
        dets = vmetrics.detect_boxes(path, [q], device=device)
        kept = [
            d for d in dets
            if max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) >= floor
        ]
        out[f"n_{key}"] = len(kept)
        out[f"conf_{key}"] = float(max((d["confidence"] for d in kept), default=0.0))
        out[f"confs_{key}"] = sorted((round(float(d["confidence"]), 3) for d in kept), reverse=True)
    return out


def score_one(split: str, pair: str, seed: str, device) -> dict | None:
    path = CACHE / split / pair / seed / "mono.png"
    if not path.exists():
        return None
    qa, qb = queries_from_slug(pair)
    n_generic, _ = count_instances(path, device=device)
    row = {"split": split, "pair": pair, "seed": seed, "query_a": qa, "query_b": qb,
           "n_generic": int(n_generic)}
    row.update(per_name_read(path, qa, qb, device))
    row.update(forced_choice_read(path, qa, qb, device))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="score only the images whose answer is already known, and print them")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.validate:
        rows = []
        print(f"{'seed':>8}  {'generic':>7}  {'n_a':>3} {'conf_a':>6}  {'n_b':>3} {'conf_b':>6}  {'both':>5}   known answer")
        for split, pair, seed, known in KNOWN:
            r = score_one(split, pair, seed, device)
            if r is None:
                print(f"{seed:>8}  MISSING")
                continue
            r["known"] = known
            rows.append(r)
            print(f"{seed:>8}  {r['n_generic']:>7}  {r['n_a']:>3} {r['conf_a']:>6.3f}  "
                  f"{r['n_b']:>3} {r['conf_b']:>6.3f}  {str(r['both_present']):>5}   {known}")
        (args.out / "validation.json").write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out / 'validation.json'}")
        return

    rows = []
    for split in ("train", "heldout"):
        for pair_dir in sorted((CACHE / split).iterdir()):
            if not pair_dir.is_dir():
                continue
            for seed_dir in sorted(pair_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                r = score_one(split, pair_dir.name, seed_dir.name, device)
                if r is not None:
                    rows.append(r)
            print(f"{split}/{pair_dir.name}: {len([r for r in rows if r['pair'] == pair_dir.name])} cells",
                  flush=True)

    (args.out / "target_quality.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out / 'target_quality.json'} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
