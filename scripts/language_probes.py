#!/usr/bin/env python
"""Can the text alone predict which pairs PoE will fail on?

Two probes, both reading the prompt embeddings the cache already stores, so
neither needs the UNet.

  L1  additivity gap. How far is the joint prompt's embedding from the sum of
      the two solo embeddings (minus the unconditional)? That sum is what PoE
      implicitly assumes the joint means. A large gap says the assumption is
      wrong for this pair before a single image is sampled. The gap is then
      scattered against how big a correction that pair actually needs.

  L3  shared binding direction. Subtract the two solo prompts from the joint
      and ask whether what is left points the same way for every pair. One
      shared direction is the language-space twin of the low-rank claim
      about r_t.

Both probes run over FOUR views of the prompt, not one. SDXL encodes a prompt
with two text encoders and keeps two forms of the result:

  pooled_bigG   (1280,)    the pooled projection, from text_encoder_2 only
  seq_clipL     (77, 768)  text_encoder 1's penultimate hidden states
  seq_bigG      (77, 1280) text_encoder_2's penultimate hidden states
  seq_both      (77, 2048) the two concatenated: what cross-attention consumes

The concatenation order is fixed by ``poe_repair/_sdxl/runtime.py``, which
builds ``cat([enc_1.hidden_states[-2], enc_2.hidden_states[-2]], dim=-1)``, so
the CLIP-L half is the first 768 channels and the bigG half is the last 1280.

**What the sequence views can and cannot tell you.** The pooled view is a
genuine vector-space addition: one vector per prompt, and adding them is
exactly the operation PoE's assumption implies. The sequence views are
position-wise, and position k does not hold the same word across the three
prompts ("a cat" has padding at position 4, "a cat and a dog" has "a" there).
So a sequence gap mixes a real semantic difference with a pure alignment
artefact. Both are reported because the sequence form is the one cross-
attention actually reads, but a large sequence gap is weaker evidence than a
large pooled gap. Sequence views are additionally reported over content
positions only (through the longest of the three prompts' end-of-text token),
since padding positions would otherwise dominate the norm.

Cache-only, no GPU.

Usage:
    python scripts/language_probes.py --probe l1 --probe l3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.config import DEFAULT_MODEL_ID  # noqa: E402
from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    cell_dir,
    load_cell,
)
from poe_repair.experiments.interaction_term.pool import load_pool  # noqa: E402
from scripts.composition_scatter import (  # noqa: E402
    PREREG,
    measure_for_cell,
    read_prereg,
)
from scripts.snr_collapse import iter_cells  # noqa: E402

OUT_DIR = paths.resolve(paths.CACHE_ANALYSES)

# Where the CLIP-L half of the 2048-wide sequence embedding ends. Fixed by the
# concatenation in poe_repair/_sdxl/runtime.py, not a tunable.
CLIP_L_WIDTH = 768

# A correlation is called real only above this magnitude AND below this p. Both
# sit in the source so neither can be relaxed after seeing the scatter.
MIN_ABS_RHO = 0.30
MAX_P = 0.05

# L3 calls a direction shared only if it beats the same-shape random floor by
# this ratio. A small sample of high-dimensional vectors concentrates on one
# direction by chance, so the raw share means nothing on its own.
MIN_SHARE_OVER_FLOOR = 2.0

VIEWS = ("pooled_bigG", "seq_clipL", "seq_bigG", "seq_both")


def load_embeddings(pair: str, seed: int, root: Path) -> dict:
    """The cached prompt embeddings for one cell."""
    p = cell_dir(pair, seed, root=root) / "embeddings.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    return torch.load(p, map_location="cpu", weights_only=True)


def read_meta(pair: str, seed: int, root: Path) -> dict:
    return json.loads((cell_dir(pair, seed, root=root) / "meta.json").read_text())


def content_length(meta: dict, tokenizer) -> int:
    """Positions through the longest of this cell's three prompts' EOT token.

    Padding positions carry content-dependent values under a causal text
    encoder, so they do not cancel in the residual. Masking to the union of the
    three prompts' real tokens keeps the gap about the words.
    """
    a, b = meta["pair"]
    texts = [a, b, meta["joint_prompt"]]
    return max(len(tokenizer(t).input_ids) for t in texts)


def views_for(emb: dict, branch: str, n_tok: int) -> dict[str, torch.Tensor]:
    """The four views of one branch's embedding, each flattened to a vector."""
    seq = emb[f"seq_{branch}"].float()[0]          # (77, 2048)
    pooled = emb[f"pool_{branch}"].float().flatten()
    return {
        "pooled_bigG": pooled,
        "seq_clipL": seq[:n_tok, :CLIP_L_WIDTH].flatten(),
        "seq_bigG": seq[:n_tok, CLIP_L_WIDTH:].flatten(),
        "seq_both": seq[:n_tok, :].flatten(),
    }


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12)


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Rank correlation and its p-value, without a scipy dependency at import."""
    from scipy import stats

    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="append", dest="probes",
                    choices=("l1", "l3"), help="repeatable; default both")
    ap.add_argument("--pair", action="append", dest="pairs")
    ap.add_argument("--pool", nargs="?", const=str(paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "pair_pool.yaml"),
                    help="restrict to one experiment's declared pairs")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="0 means every cached pair")
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--prereg", type=Path, default=PREREG)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    probes = args.probes or ["l1", "l3"]

    measure = read_prereg(args.prereg)
    print(f"committed correction-size measure: {measure}\n")

    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(DEFAULT_MODEL_ID, subfolder="tokenizer")

    # One seed per pair: the embeddings depend on the prompt, not the seed.
    pairs = args.pairs
    if args.pool:
        pool = load_pool(args.pool)
        print(pool.summary())
        pairs = pool.train + pool.heldout()
    cells = list(iter_cells(args.cache_root, pairs, 1))
    if args.max_pairs:
        cells = cells[: args.max_pairs]
    if not cells:
        print("no cached cells matched", file=sys.stderr)
        return 2

    # One token window for every pair. Cross-pair reads (the L3 cosine matrix,
    # the L1 scatter) need a common dimension, and a per-pair window would also
    # give each pair a different denominator, making the gaps incomparable.
    # The window is the longest content length over every prompt in the set.
    n_tok = 0
    for pair, seed in cells:
        try:
            n_tok = max(n_tok, content_length(
                read_meta(pair, seed, args.cache_root), tokenizer))
        except FileNotFoundError:
            continue
    print(f"token window: positions 0..{n_tok - 1}, the longest content length "
          f"over all {len(cells)} pairs (of 77 cached)\n")

    rows: list[dict] = []
    residuals: dict[str, list[np.ndarray]] = {v: [] for v in VIEWS}
    kept: list[str] = []
    scales: dict[str, list[float]] = {v: [] for v in VIEWS}
    # Kept so L3 can rebuild the residual with mismatched solos as a control.
    banked: list[dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []

    for pair, seed in cells:
        try:
            emb = load_embeddings(pair, seed, args.cache_root)
        except FileNotFoundError:
            continue
        need = tuple(f"{p}_{b}" for p in ("pool", "seq")
                     for b in ("a", "b", "j", "uncond"))
        missing = [k for k in need if k not in emb]
        if missing:
            print(f"embeddings.pt for {pair} lacks {missing}; "
                  f"found {sorted(emb)}", file=sys.stderr)
            return 2

        va = views_for(emb, "a", n_tok)
        vb = views_for(emb, "b", n_tok)
        vj = views_for(emb, "j", n_tok)
        ve = views_for(emb, "uncond", n_tok)

        row = {"pair": pair, "seed": int(seed),
               "correction_size": measure_for_cell(
                   load_cell(pair, seed, root=args.cache_root), measure)}
        for v in VIEWS:
            scales[v].append(float(vj[v].norm()))
            # L1: what PoE implicitly assumes the joint prompt means.
            assumed = va[v] + vb[v] - ve[v]
            gap = float((vj[v] - assumed).norm() / vj[v].norm().clamp_min(1e-12))
            row[f"gap_{v}"] = gap
            # L3: what is left of the joint once the two solos are subtracted,
            # with scale divided out on both sides so only direction remains.
            b_dir = unit(vj[v]) - unit(va[v] + vb[v])
            residuals[v].append(unit(b_dir).numpy())
        rows.append(row)
        kept.append(pair)
        banked.append({v: (va[v], vb[v], vj[v]) for v in VIEWS})
        print(f"  {pair:<42} " + "  ".join(
            f"{v} {row['gap_' + v]:.3f}" for v in VIEWS)
            + f"   corr {row['correction_size']:.3f}")

    if not rows:
        print("no cell had usable embeddings", file=sys.stderr)
        return 2

    result: dict = {"measure": measure, "n_pairs": len(rows),
                    "n_content_tokens": n_tok, "views": list(VIEWS),
                    "view_median_norm": {v: float(np.median(scales[v]))
                                         for v in VIEWS},
                    "cells": rows}
    print(f"\n{len(rows)} pairs, one seed each\n")
    # The two encoder halves live at different scales, so the concatenated view
    # is not the average of the two: whichever half carries more norm dominates
    # both the residual and the denominator. Print it rather than let the
    # reader wonder why seq_both sits where it does.
    print("median ||e_J|| per view (why seq_both is not the average of the halves)")
    for v in VIEWS:
        print(f"  {v:<12} {np.median(scales[v]):9.2f}")
    print()

    if "l1" in probes:
        print("L1 additivity gap, and whether it predicts correction size")
        y = np.array([r["correction_size"] for r in rows])
        result["l1"] = {}
        for v in VIEWS:
            g = np.array([r[f"gap_{v}"] for r in rows])
            rho, p = spearman(g, y)
            real = abs(rho) >= MIN_ABS_RHO and p <= MAX_P
            print(f"  {v:<12} gap median {np.median(g):.3f} "
                  f"[{g.min():.3f}, {g.max():.3f}]   "
                  f"rho vs {measure} {rho:+.3f} (p={p:.3g}) "
                  f"{'PREDICTS' if real else 'no'}")
            result["l1"][v] = {"median_gap": float(np.median(g)),
                               "min_gap": float(g.min()), "max_gap": float(g.max()),
                               "spearman_rho": rho, "p_value": p,
                               "predicts": bool(real)}
        print(f"  bar: |rho| >= {MIN_ABS_RHO} and p <= {MAX_P}")
        print("  a null here is a finding: it would put the binding information "
              "in how\n  the model processes the prompt jointly, not in the "
              "prompt's embedding.")

    if "l3" in probes and len(rows) > 2:
        print("\nL3 shared binding direction")
        rng = np.random.default_rng(0)
        n = len(rows)
        # Mismatched-solos control. Every prompt here has the same shape
        # ("a X and a Y" against "a X" and "a Y"), so a direction shared across
        # pairs could just be the shape: the "and", the extra length, the
        # anisotropy every CLIP text embedding has. Rebuilding the residual
        # with partner B taken from a DIFFERENT pair keeps all of that and
        # destroys only the binding. If the control shares a direction just as
        # strongly, the effect is the prompt template, not the binding.
        perm = (np.arange(n) + 1 + rng.integers(0, n - 1)) % n   # no fixed point
        result["l3"] = {}
        for v in VIEWS:
            R = np.stack(residuals[v])
            M = np.stack([
                unit(unit(banked[i][v][2])
                     - unit(banked[i][v][0] + banked[perm[i]][v][1])).numpy()
                for i in range(n)
            ])
            # Raw prompt anisotropy: how aligned the joint embeddings already
            # are before any subtraction. The reference the residual must beat.
            J = np.stack([unit(banked[i][v][2]).numpy() for i in range(n)])
            G = rng.normal(size=R.shape)
            G = G / np.linalg.norm(G, axis=1, keepdims=True)

            def read(X):
                c = X @ X.T
                off = c[~np.eye(len(X), dtype=bool)]
                s = np.linalg.svd(X - X.mean(0, keepdims=True), compute_uv=False)
                return (float(off.mean()),
                        float(s[0] ** 2 / max((s ** 2).sum(), 1e-12)))

            cos_r, share = read(R)
            cos_m, share_m = read(M)
            cos_j, share_j = read(J)
            _, floor = read(G)
            ratio = share / max(floor, 1e-12)
            over_mismatch = share / max(share_m, 1e-12)
            # Both bars: beat the random floor AND beat the mismatched control.
            shared = (ratio >= MIN_SHARE_OVER_FLOOR
                      and over_mismatch >= MIN_SHARE_OVER_FLOOR)
            print(f"  {v}")
            print(f"    real pairs     cosine {cos_r:+.3f}   "
                  f"top direction {share:.1%}")
            print(f"    mismatched B   cosine {cos_m:+.3f}   "
                  f"top direction {share_m:.1%}   ({over_mismatch:.2f}x real)")
            print(f"    joint prompts  cosine {cos_j:+.3f}   "
                  f"top direction {share_j:.1%}   (anisotropy before subtraction)")
            print(f"    random floor                       "
                  f"top direction {floor:.1%}   ({ratio:.1f}x real)")
            print(f"    -> {'SHARED' if shared else 'not shared beyond the template'}")
            result["l3"][v] = {
                "mean_pairwise_cosine": cos_r,
                "top_direction_share": share,
                "mismatched_mean_pairwise_cosine": cos_m,
                "mismatched_top_direction_share": share_m,
                "joint_anisotropy_cosine": cos_j,
                "joint_anisotropy_share": share_j,
                "random_floor": floor,
                "ratio_over_floor": float(ratio),
                "ratio_over_mismatched": float(over_mismatch),
                "shared": bool(shared),
            }
        print(f"  bar: top-direction share >= {MIN_SHARE_OVER_FLOOR}x the random "
              f"floor AND >= {MIN_SHARE_OVER_FLOOR}x the mismatched-solos control")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "language_probes.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")

    if not args.no_figure:
        _figures(rows, residuals, kept, result, measure, args.out_dir, probes)
    return 0


def _figures(rows, residuals, kept, result, measure, out_dir, probes) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "l1" in probes:
        y = np.array([r["correction_size"] for r in rows])
        fig, axes = plt.subplots(1, len(VIEWS), figsize=(4 * len(VIEWS), 3.6),
                                 sharey=True)
        for ax, v in zip(np.atleast_1d(axes), VIEWS):
            g = np.array([r[f"gap_{v}"] for r in rows])
            st = result["l1"][v]
            ax.scatter(g, y, s=22, alpha=0.75, color="tab:blue",
                       edgecolor="white", linewidth=0.5)
            ax.set_title(f"{v}\nrho {st['spearman_rho']:+.2f} "
                         f"(p={st['p_value']:.2g})", fontsize=9)
            ax.set_xlabel("additivity gap")
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel(f"correction size\n({measure})")
        fig.suptitle(
            f"L1: does the prompt's own arithmetic predict how big a "
            f"correction the pair needs?  ({len(rows)} pairs, 1 seed each)",
            fontsize=11)
        fig.tight_layout()
        p = out_dir / "language_probe_l1_additivity.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"figure: {p}")

    if "l3" in probes and len(rows) > 2:
        fig, axes = plt.subplots(2, len(VIEWS), figsize=(4 * len(VIEWS), 7.2))
        for col, v in enumerate(VIEWS):
            R = np.stack(residuals[v])
            C = R @ R.T
            ax = axes[0, col]
            im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"{v}\nmean off-diagonal "
                         f"{result['l3'][v]['mean_pairwise_cosine']:+.2f}",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)

            s = np.linalg.svd(R - R.mean(0, keepdims=True), compute_uv=False)
            e = s ** 2 / (s ** 2).sum()
            ax = axes[1, col]
            ax.bar(np.arange(1, min(11, len(e) + 1)), e[:10], color="tab:purple",
                   label="real pairs")
            ax.axhline(result["l3"][v]["mismatched_top_direction_share"],
                       ls="-", color="tab:red",
                       label="mismatched-B control (top direction)")
            ax.axhline(result["l3"][v]["random_floor"], ls="--", color="0.4",
                       label="random floor")
            ax.set_xlabel("singular direction")
            ax.set_ylabel("share of energy" if col == 0 else "")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle(
            "L3: subtract the two solo prompts from the joint. Does every pair "
            f"leave the same thing behind?  ({len(rows)} pairs)", fontsize=11)
        fig.tight_layout()
        p = out_dir / "language_probe_l3_binding.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"figure: {p}")


if __name__ == "__main__":
    raise SystemExit(main())
