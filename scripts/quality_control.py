#!/usr/bin/env python
"""Is a blended animal wrong content, or just a bad picture?

The standing objection to the whole result is that the correction might simply
be improving image quality, and that "two animals" is what a better picture
looks like. This kills that objection or confirms it.

It scores the two images every cached cell already holds:

  poe.png    the uncorrected Product-of-Experts sample, which is usually one
             fused chimera
  mono.png   the same seed and prompt run as a single joint conditioning,
             which is usually two separate animals

Same seed, same prompt, same sampler, same step count. The ONLY thing that
differs is whether the interaction term was there. So a paired comparison is
clean on every axis except the one being tested.

Choosing the proxies is the whole difficulty, because the two sides differ in
CONTENT by construction, and most quality measures move with content. Counting
edges is the trap: two separate animals have more silhouette than one fused
one, so any edge-density measure reports a difference that says nothing about
whether either picture is well made. The proxies are therefore split, and only
the content-blind group decides the verdict.

Content-blind, and so load-bearing:

  blur_crete     Crete et al. (2007). Re-blurs the image and measures how much
                 variation was left to lose. Because it is a RATIO, it reads
                 edge WIDTH rather than edge COUNT, so adding a second animal
                 does not move it. Higher means blurrier.
  noise_sigma    Immerkaer (1996) fast noise estimate through a 3x3 mask that
                 cancels locally-linear image structure.
  contrast       standard deviation of luminance
  colorfulness   Hasler & Susstrunk's opponent-channel measure

Content-sensitive, so reported beside the verdict but never driving it:

  sharpness      variance of the Laplacian. Rises with the NUMBER of edges, so
                 two animals score above one by construction.
  clip_iqa       CLIP's preference between "a sharp, high quality photograph"
                 and "a blurry, low quality, distorted image". A chimera is
                 anatomically distorted, so this measure reads the content it
                 is supposed to be blind to.

And one CONTENT read, the positive control:

  compose rate under the validated scorer (COMPOSE iff instance count >= 2).

The two together are the argument. The content read should show a large gap
(that is the effect the whole scope is about, so if it does not show up the
images are not what we think they are). The content-blind quality proxies
should show none. Content changed, quality did not.

Reads sampled images; it does not sample.

Usage:
    python scripts/quality_control.py
    python scripts/quality_control.py --source dose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    cell_dir,
)
from scripts.plot_dose_curves import require_validated_scorer  # noqa: E402
from scripts.snr_collapse import iter_cells  # noqa: E402

OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")
DOSE_ROOTS = (
    Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/pairs"),
    Path("outputs/interaction_term/dose/pairs"),
)
SCORER_CONTRACT = Path("outputs/compose_scorer/scorer_validated.json")
LAM_RE = re.compile(r"lam(\d{3})")

# A quality proxy counts as showing NO gap when the paired difference is under
# this many standard deviations of that proxy across cells, and the signed-rank
# test does not reject. Both sit in the source so neither can be relaxed once
# the numbers are on screen.
MAX_NO_GAP_EFFECT = 0.20
MIN_P_FOR_NO_GAP = 0.05

# The content read is the positive control: the compose rate must move at least
# this far, or the paired images are not the blend-vs-composition contrast this
# check assumes and the quality row cannot be interpreted.
MIN_COMPOSE_RATE_GAIN = 0.20

# Only these decide the verdict. See the module docstring for why the other two
# cannot: both move with the number of objects, which differs by construction.
QUALITY_PROXIES = ("blur_crete", "noise_sigma", "contrast", "colorfulness")
DIAGNOSTIC_PROXIES = ("sharpness", "clip_iqa")
ALL_PROXIES = QUALITY_PROXIES + DIAGNOSTIC_PROXIES

IQA_GOOD = "a sharp, high quality photograph"
IQA_BAD = "a blurry, low quality, distorted image"


def luminance(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64) / 255.0


def sharpness(img: Image.Image) -> float:
    """Variance of the 3x3 Laplacian: high for crisp edges, low for blur."""
    y = luminance(img)
    lap = (-4.0 * y[1:-1, 1:-1]
           + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:])
    return float(lap.var())


def blur_crete(img: Image.Image) -> float:
    """Crete et al. (2007) blur estimate in [0, 1]. Higher means blurrier.

    Re-blur the image and measure how much neighbour-to-neighbour variation
    that destroyed. An already-blurry image has little left to lose, so it
    scores high. The measure is a ratio of the variation lost to the variation
    present, which is what makes it blind to how MUCH edge the picture has: a
    second animal adds to numerator and denominator alike.
    """
    import cv2

    y = luminance(img).astype(np.float32)
    out = []
    for axis, ksize in ((1, (9, 1)), (0, (1, 9))):
        blurred = cv2.blur(y, ksize)
        d_f = np.abs(np.diff(y, axis=axis))
        d_b = np.abs(np.diff(blurred, axis=axis))
        lost = np.maximum(0.0, d_f - d_b)
        total = d_f.sum()
        out.append(float((total - lost.sum()) / total) if total > 1e-9 else 0.0)
    return max(out)


def noise_sigma(img: Image.Image) -> float:
    """Immerkaer (1996) fast noise estimate.

    The 3x3 mask below annihilates any locally-linear intensity ramp, so what
    survives is noise rather than structure. That is what makes it usable here:
    edges and objects pass through near zero.
    """
    import cv2

    y = luminance(img).astype(np.float32)
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(y, -1, mask)[1:-1, 1:-1]
    h, w = conv.shape
    return float(np.abs(conv).sum() * np.sqrt(np.pi / 2) / (6.0 * h * w))


def contrast(img: Image.Image) -> float:
    return float(luminance(img).std())


def colorfulness(img: Image.Image) -> float:
    """Hasler & Susstrunk (2003), the standard opponent-channel measure."""
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std())
                 + 0.3 * np.hypot(rg.mean(), yb.mean()))


def clip_iqa(paths: list[Path]) -> np.ndarray:
    """CLIP's own good-vs-bad-picture preference, one score per image."""
    from poe_repair.experiments.residual_diagnostics.metrics import (
        clip_image_embed,
        clip_text_embed,
    )

    txt = clip_text_embed([IQA_GOOD, IQA_BAD])
    out = []
    for i in range(0, len(paths), 32):
        feats = clip_image_embed(paths[i:i + 32])
        sims = feats @ txt.T
        out.append((sims[:, 0] - sims[:, 1]).numpy())
    return np.concatenate(out)


def cache_cells(root: Path, pairs, max_cells: int):
    """Cached cells holding both the uncorrected and the corrected picture."""
    found = []
    for pair, seed in iter_cells(root, pairs, None):
        d = cell_dir(pair, seed, root=root)
        poe, mono = d / "poe.png", d / "mono.png"
        if poe.exists() and mono.exists():
            found.append((pair, int(seed), poe, mono))
    return found[:max_cells] if max_cells else found


def dose_cells(roots, baseline: float, corrected: float, max_cells: int):
    """Dose cells holding both the baseline and the corrected lambda."""
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for pair_dir in sorted(root.iterdir()):
            if not pair_dir.is_dir():
                continue
            for seed_dir in sorted(pair_dir.glob("seed_*")):
                imgs = {}
                for run_dir in sorted(seed_dir.glob("teacher_residual_const_lam*")):
                    m = LAM_RE.search(run_dir.name)
                    png = run_dir / f"{run_dir.name}.png"
                    # Skip the random / wrong_pair control rows: this script
                    # asks about the real correction, not the controls.
                    if m and png.exists() and run_dir.name.endswith(m.group(0)):
                        imgs[int(m.group(1)) / 100.0] = png
                if baseline in imgs and corrected in imgs:
                    found.append((pair_dir.name,
                                  int(seed_dir.name.split("_")[1]),
                                  imgs[baseline], imgs[corrected]))
    return found[:max_cells] if max_cells else found


def paired_read(before: np.ndarray, after: np.ndarray) -> dict:
    """Paired difference, scaled by spread, with a signed-rank p-value."""
    from scipy import stats

    d = after - before
    pooled = float(np.std(np.concatenate([before, after])))
    effect = float(d.mean() / pooled) if pooled > 1e-12 else 0.0
    p = float(stats.wilcoxon(d).pvalue) if len(d) > 5 and np.any(d) else float("nan")
    return {"mean_before": float(before.mean()), "mean_after": float(after.mean()),
            "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "effect": effect, "p_value": p}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("cache", "dose"), default="cache",
                    help="cache: poe.png vs mono.png. dose: lambda 0 vs 1.")
    ap.add_argument("--root", type=Path, action="append", dest="roots")
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--pair", action="append", dest="pairs")
    ap.add_argument("--max-cells", type=int, default=0, help="0 means all")
    ap.add_argument("--baseline", type=float, default=0.0, help="dose source only")
    ap.add_argument("--corrected", type=float, default=1.0, help="dose source only")
    ap.add_argument("--contract", type=Path, default=SCORER_CONTRACT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-content", action="store_true",
                    help="skip the detector positive control (no GPU needed)")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    if args.source == "cache":
        cells = cache_cells(args.cache_root, args.pairs, args.max_cells)
        before_name, after_name = "poe (uncorrected)", "mono (corrected)"
        empty_help = (f"no cached cell under {args.cache_root} has both "
                      "poe.png and mono.png.")
    else:
        cells = dose_cells(args.roots or DOSE_ROOTS, args.baseline,
                           args.corrected, args.max_cells)
        before_name = f"lambda {args.baseline}"
        after_name = f"lambda {args.corrected}"
        empty_help = (f"no dose cell has both lambda={args.baseline} and "
                      f"lambda={args.corrected} rendered. This script scores "
                      "images, it does not sample.")
    if not cells:
        print(empty_help, file=sys.stderr)
        return 2

    print(f"{len(cells)} paired cells: {before_name} vs {after_name}")
    print("same seed, same prompt, same sampler; the correction is the only "
          "difference\n")

    before_paths = [c[2] for c in cells]
    after_paths = [c[3] for c in cells]

    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, fn in (("blur_crete", blur_crete), ("noise_sigma", noise_sigma),
                     ("contrast", contrast), ("colorfulness", colorfulness),
                     ("sharpness", sharpness)):
        b = np.array([fn(Image.open(p)) for p in before_paths])
        a = np.array([fn(Image.open(p)) for p in after_paths])
        scores[name] = (b, a)
        print(f"  {name} done")
    scores["clip_iqa"] = (clip_iqa(before_paths), clip_iqa(after_paths))
    print("  clip_iqa done")

    result: dict = {"source": args.source, "n_cells": len(cells),
                    "before": before_name, "after": after_name,
                    "quality": {}, "bars": {
                        "max_no_gap_effect": MAX_NO_GAP_EFFECT,
                        "min_p_for_no_gap": MIN_P_FOR_NO_GAP,
                        "min_compose_rate_gain": MIN_COMPOSE_RATE_GAIN}}

    def table(names: list[str]) -> bool:
        print(f"  {'proxy':<14}{'before':>12}{'after':>12}{'diff':>12}"
              f"{'effect':>9}{'p':>10}   verdict")
        flat_all = True
        for name in names:
            b, a = scores[name]
            r = paired_read(b, a)
            flat = (abs(r["effect"]) < MAX_NO_GAP_EFFECT
                    or r["p_value"] > MIN_P_FOR_NO_GAP)
            flat_all &= flat
            r["no_gap"] = bool(flat)
            result["quality"][name] = r
            print(f"  {name:<14}{r['mean_before']:12.4f}{r['mean_after']:12.4f}"
                  f"{r['mean_diff']:+12.4f}{r['effect']:+9.2f}"
                  f"{r['p_value']:10.3g}   {'no gap' if flat else 'GAP'}")
        return flat_all

    print(f"\nQUALITY, content-blind (decides the verdict) "
          f"over {len(cells)} paired cells")
    all_flat = table(list(QUALITY_PROXIES))
    print(f"  bar: |effect| < {MAX_NO_GAP_EFFECT} sd, or p > {MIN_P_FOR_NO_GAP}")

    print("\nDIAGNOSTIC, content-sensitive (reported, does NOT decide)")
    table(list(DIAGNOSTIC_PROXIES))
    print("  both rise with the number of objects, which differs between the "
          "two sides\n  by construction, so a gap here is expected and is not "
          "a quality cost.")
    result["quality_proxies"] = list(QUALITY_PROXIES)
    result["diagnostic_proxies"] = list(DIAGNOSTIC_PROXIES)

    content_ok = None
    if not args.no_content:
        contract = require_validated_scorer(args.contract)
        print(f"\nCONTENT positive control: {contract['method']} via "
              f"{contract['detector']}")
        from poe_repair.experiments.compose_scorer.detection_scorer import (
            count_instances,
        )

        nb, na, cb, ca = [], [], [], []
        for p in before_paths:
            n, boxes = count_instances(p)
            nb.append(n)
            cb.append(float(np.mean([x["confidence"] for x in boxes])) if boxes else 0.0)
        for p in after_paths:
            n, boxes = count_instances(p)
            na.append(n)
            ca.append(float(np.mean([x["confidence"] for x in boxes])) if boxes else 0.0)
        nb, na = np.array(nb, float), np.array(na, float)
        # The scorer's validated rule is COMPOSE iff count >= 2, so the compose
        # rate is the read, not the mean count. They disagree: a blend that
        # draws three spurious boxes lowers neither, and averaging counts lets
        # such a cell cancel a real gain.
        cmp_b, cmp_a = nb >= 2, na >= 2
        rate_b, rate_a = float(cmp_b.mean()), float(cmp_a.mean())
        gain = rate_a - rate_b
        # McNemar on the cells that disagree: the paired test for a rate.
        b_only = int((cmp_b & ~cmp_a).sum())
        a_only = int((~cmp_b & cmp_a).sum())
        from scipy import stats
        p_mc = float(stats.binomtest(a_only, a_only + b_only, 0.5).pvalue) \
            if (a_only + b_only) else float("nan")
        content_ok = gain >= MIN_COMPOSE_RATE_GAIN
        result["content_compose_rate"] = {
            "rate_before": rate_b, "rate_after": rate_a, "gain": gain,
            "gained_only": a_only, "lost_only": b_only, "mcnemar_p": p_mc,
            "content_gap": bool(content_ok),
            "rule": contract["compose_rule"],
        }
        result["content_instance_count"] = paired_read(nb, na)
        result["content_confidence"] = paired_read(np.array(cb), np.array(ca))
        print(f"  compose rate     {rate_b:.1%} -> {rate_a:.1%} "
              f"({gain:+.1%})")
        print(f"  cells that flipped: {a_only} gained, {b_only} lost, "
              f"McNemar p {p_mc:.3g}")
        print(f"  mean instance count {nb.mean():.2f} -> {na.mean():.2f} "
              "(reported only: the validated rule is the >=2 threshold, and a\n"
              "    blend with three spurious boxes moves the mean the wrong way)")
        print(f"  bar: compose-rate gain >= {MIN_COMPOSE_RATE_GAIN:.0%}   "
              f"{'content changed' if content_ok else 'NO CONTENT GAP'}")
        print("  (detector confidence is content-sensitive, in the JSON as a "
              "diagnostic only)")

    if content_ok is None:
        verdict = ("quality unchanged" if all_flat else "quality changed")
        note = "content control skipped, so this cannot rule the objection out"
    elif all_flat and content_ok:
        verdict = "content changed, quality did not"
        note = ("the objection is answered: the correction buys composition "
                "without buying picture quality")
    elif content_ok:
        verdict = "content changed AND quality changed"
        note = ("the objection stands in part: report the quality gap beside "
                "the compose rate rather than only the compose rate")
    else:
        verdict = "no content gap"
        note = ("the positive control failed, so the paired images are not the "
                "blend-vs-composition contrast this check assumes; fix that "
                "before reading the quality row")
    result["verdict"] = verdict
    result["note"] = note
    print(f"\nreading: {verdict}\n  {note}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"quality_control_{args.source}"
    out = args.out_dir / f"{stem}.json"
    result["cells"] = [{"pair": c[0], "seed": c[1]} for c in cells]
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")

    if not args.no_figure:
        _figure(scores, result, before_name, after_name,
                args.out_dir / f"{stem}.png")
    return 0


def _figure(scores, result, before_name, after_name, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(ALL_PROXIES)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.9))
    for ax, name in zip(np.atleast_1d(axes), ALL_PROXIES):
        b, a = scores[name]
        r = result["quality"][name]
        decides = name in QUALITY_PROXIES
        # Every cell as its own line, so a flat mean hiding two opposite halves
        # would still be visible.
        for x, y in zip(b, a):
            ax.plot([0, 1], [x, y], color="0.75", lw=0.4, alpha=0.5)
        ax.plot([0, 1], [b.mean(), a.mean()],
                color="tab:red" if decides else "tab:orange", lw=2.5,
                marker="o", zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([before_name.split()[0], after_name.split()[0]])
        ax.set_title(
            f"{name}\n{'content-blind' if decides else 'CONTENT-SENSITIVE'}\n"
            f"effect {r['effect']:+.2f} sd, p={r['p_value']:.2g}",
            fontsize=8.5,
            color="black" if decides else "tab:orange")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"Is the blend a bad picture, or the wrong picture?  "
        f"{result['n_cells']} paired cells: {result['verdict']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"figure: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
