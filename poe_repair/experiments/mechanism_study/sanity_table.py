"""12×50 attention-mass sanity table (mechanism study, plan 01, task 2).

Loads the captured ``.pt`` files for all seeds of one regime and prints a
seed×timestep table of attention mass on a chosen concept's token, plus a
matching heatmap PNG. This is the checkpoint that confirms the capture
pipeline behaves before any new wiring is trusted.

"Attention mass" is the spatial sum of a step's per-token attention map
(``map.sum()``), i.e. total cross-attention probability the token receives
across the latent grid at that step.

Usage::

    python -m poe_repair.experiments.mechanism_study.sanity_table \
        --regime plain_poe --token cat_branch_poe

    # any regime dir written by capture_attention.py works
    python -m poe_repair.experiments.mechanism_study.sanity_table \
        --root /datasets/.../attn_mechanism/plain_poe/a_cat__x__a_dog \
        --token dog_branch_poe
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from poe_repair import paths

DEFAULT_ATTN_ROOT = paths.resolve(paths.ATTENTION_MECHANISM)

_STEP_RE = re.compile(r"step_(\d+)_token_(.+)\.pt$")


def _seed_dirs(root: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for d in sorted(root.glob("seed_*")):
        m = re.match(r"seed_(\d+)$", d.name)
        if m and (d / "attn_maps").is_dir():
            out.append((int(m.group(1)), d / "attn_maps"))
    return out


def _load_mass_table(
    root: Path, token_key: str,
) -> tuple[list[int], int, dict[int, dict[int, float]]]:
    """Return (seeds, n_steps, mass[seed][step]) for one token key."""
    seed_dirs = _seed_dirs(root)
    if not seed_dirs:
        raise SystemExit(f"no seed_*/attn_maps dirs under {root}")
    mass: dict[int, dict[int, float]] = {}
    max_step = -1
    for seed, adir in seed_dirs:
        per_step: dict[int, float] = {}
        for f in sorted(adir.glob(f"step_*_token_{token_key}.pt")):
            m = _STEP_RE.search(f.name)
            if not m:
                continue
            step = int(m.group(1))
            sd = torch.load(f, weights_only=False)
            per_step[step] = float(sd["map"].sum())
            max_step = max(max_step, step)
        mass[seed] = per_step
    return [s for s, _ in seed_dirs], max_step + 1, mass


def _print_table(
    seeds: list[int], n_steps: int, mass: dict[int, dict[int, float]],
    token_key: str, regime: str, stride: int,
) -> None:
    cols = list(range(0, n_steps, stride))
    if cols and cols[-1] != n_steps - 1:
        cols.append(n_steps - 1)
    print(f"\nAttention mass on token '{token_key}'  (regime={regime})")
    print(f"seed × timestep  —  {len(seeds)} seeds × {n_steps} steps "
          f"(showing every {stride}th step)\n")
    header = "seed |" + "".join(f"  t{c:<3d}" for c in cols)
    print(header)
    print("-" * len(header))
    for s in seeds:
        row = f"{s:>4d} |"
        for c in cols:
            v = mass[s].get(c)
            row += f" {v:6.3f}" if v is not None else "   n/a"
        print(row)
    # Per-seed summary: mean mass across steps.
    print()
    for s in seeds:
        vals = [mass[s][t] for t in range(n_steps) if t in mass[s]]
        if vals:
            print(f"  seed {s:>2d}: mean={sum(vals)/len(vals):.4f}  "
                  f"n_steps={len(vals)}  min={min(vals):.4f}  max={max(vals):.4f}")


def _save_heatmap(
    seeds: list[int], n_steps: int, mass: dict[int, dict[int, float]],
    token_key: str, regime: str, out_png: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"[sanity_table] matplotlib unavailable, skipping heatmap: {exc}")
        return
    grid = np.full((len(seeds), n_steps), np.nan, dtype=float)
    for i, s in enumerate(seeds):
        for t, v in mass[s].items():
            grid[i, t] = v
    fig, ax = plt.subplots(figsize=(12, 0.4 * len(seeds) + 1.5))
    im = ax.imshow(grid, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([f"seed {s}" for s in seeds])
    ax.set_xlabel("timestep (denoising step index)")
    ax.set_title(f"Attention mass on '{token_key}'  ·  regime={regime}")
    fig.colorbar(im, ax=ax, label="Σ attention map")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[sanity_table] heatmap → {out_png}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sanity_table")
    ap.add_argument("--regime", default="plain_poe",
                    help="subdir under attn_mechanism/ (ignored if --root set)")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--root", default=None,
                    help="explicit <...>/<slug> dir holding seed_*/attn_maps")
    ap.add_argument("--token", default="cat_branch_poe",
                    help="save_key to tabulate, e.g. cat_branch_poe / dog_branch_poe")
    ap.add_argument("--stride", type=int, default=5,
                    help="print every Nth timestep column")
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args(argv)

    root = (
        Path(args.root) if args.root
        else DEFAULT_ATTN_ROOT / args.regime / args.pair_slug
    )
    seeds, n_steps, mass = _load_mass_table(root, args.token)
    _print_table(seeds, n_steps, mass, args.token, args.regime, args.stride)
    out_png = (
        Path(args.out_png) if args.out_png
        else root / f"sanity_table_{args.token}.png"
    )
    _save_heatmap(seeds, n_steps, mass, args.token, args.regime, out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
