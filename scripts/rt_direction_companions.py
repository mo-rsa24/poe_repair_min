#!/usr/bin/env python
"""D1 and D3: the two companions of the no-reuse figure (D2).

All three share one template: x is the denoising step, y is direction
agreement (cosine) between two correction vectors, thumbnails float in the
empty axis space with a coloured border naming their run, and the vertical
position of a thumbnail carries no data.

    D1  within ONE run, step t vs t+1. The curve hugs +1: as a run unfolds,
        its correction turns smoothly rather than jumping. Shown on
        eagle x hawk seed 9, the typical case. The caption must say the
        smoothness varies by pair (cat x dog alternates sign mid-run,
        median -0.32; the per-pair values are in
        outputs/interaction_term/direction_wall/direction_wall.json).

    D3  two DIFFERENT pairs at matched steps. Flat on zero: no direction is
        shared across pairs at any step. The unsurprising sibling of D2,
        which shows the same flat zero for the SAME pair rerun.

Frames are each run's own estimate of where it is heading (Tweedie from the
cached x_t and the PoE prediction that steered the path, VAE-decoded), cached
under outputs/interaction_term/direction_wall/frames/.

    CUDA_VISIBLE_DEVICES=0 python scripts/rt_direction_companions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import load_cell  # noqa: E402
from scripts.rt_reuse_figure import FRAME_DIR, x0_estimate  # noqa: E402

SHOW_STEPS = (10, 20, 30, 40)
OUT_DIR = Path("paper/iclr/figures")
BLUE, ORANGE, INK = "#1f77b4", "#ff7f0e", "#333333"

D1_PAIR, D1_SEED = "an_eagle__x__a_hawk", 9
D3_A = ("a_cat__x__a_dog", 9)
D3_B = ("an_eagle__x__a_hawk", 9)


def ensure_frames(cells) -> dict[tuple[str, int, int], Path]:
    """Decode any missing (pair, seed, step) frames, reusing cached ones."""
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    want = {(p, s, k): FRAME_DIR / f"{p}_seed{s}_step{k:03d}.png"
            for p, s in cells for k in SHOW_STEPS}
    todo = {key: f for key, f in want.items() if not f.exists()}
    if todo:
        from diffusers import AutoencoderKL
        from poe_repair.config import RunConfig
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = AutoencoderKL.from_pretrained(
            RunConfig().model_id, subfolder="vae",
            torch_dtype=torch.float32).to(device).eval()
        loaded = {}
        for (p, s, k), path in todo.items():
            cell = loaded.setdefault((p, s), load_cell(p, s))
            lat = x0_estimate(cell, k).to(device) / vae.config.scaling_factor
            with torch.no_grad():
                img = vae.decode(lat).sample[0]
            arr = ((img / 2 + 0.5).clamp(0, 1) * 255).byte().cpu()
            Image.fromarray(arr.permute(1, 2, 0).numpy()).resize((256, 256)).save(path)
            print(f"decoded {p} seed {s} step {k}")
        del vae
    return want


def base_axes(plt):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.axhline(0.0, color="0.75", lw=0.6, zorder=1)
    ax.set_xlim(-1, 50)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("denoising step")
    ax.set_ylabel("direction agreement (cosine)")
    ax.text(0.0, -0.19, "noise", transform=ax.transAxes, fontsize=7, color="0.4")
    ax.text(1.0, -0.19, "image", transform=ax.transAxes, fontsize=7, color="0.4",
            ha="right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fig, ax


def add_thumb(ax, png: Path, cx_step: float, cy_frac: float, colour: str,
              w: float = 0.16):
    """cy_frac is an AXES FRACTION (0 bottom, 1 top), not a data value."""
    ia = ax.inset_axes([cx_step / 50 - w / 2, cy_frac - w * 5.5 / 3.4 / 2,
                        w, w * 5.5 / 3.4])
    import matplotlib.pyplot as plt
    ia.imshow(plt.imread(png))
    ia.set_xticks([]); ia.set_yticks([])
    for sp in ia.spines.values():
        sp.set_linewidth(1.4); sp.set_color(colour)


def step_marker(ax, k: int):
    ax.plot([k, k], [-0.14, 0.14], color="0.55", lw=0.6, zorder=2)
    ax.annotate(f"{k}", (k, 0), ha="center", va="center", fontsize=6,
                color="0.45",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    })
    frames = ensure_frames([(D1_PAIR, D1_SEED), D3_A, D3_B])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- D1: within one run, adjacent steps ----
    r = load_cell(D1_PAIR, D1_SEED).r_t().float().flatten(1)
    c1 = torch.nn.functional.cosine_similarity(r[:-1], r[1:], dim=1).numpy()
    fig, ax = base_axes(plt)
    ax.plot(np.arange(len(c1)) + 0.5, c1, color="0.2", lw=1.4, zorder=3)
    for k in SHOW_STEPS:
        add_thumb(ax, frames[(D1_PAIR, D1_SEED, k)], k, 0.28, INK)
        step_marker(ax, k)
    ax.text(0.01, 0.06, "one uncorrected run: eagle x hawk, seed 9",
            transform=ax.transAxes, fontsize=7.5, color=INK, va="bottom")
    ax.text(0.99, 0.80, "agreement between consecutive steps' corrections",
            transform=ax.transAxes, fontsize=7, color="0.35", ha="right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"direction-agreement-between-consecutive-steps.{ext}", dpi=300)
    plt.close(fig)
    (OUT_DIR / "direction-agreement-between-consecutive-steps.json").write_text(json.dumps({
        "pair": D1_PAIR, "seed": D1_SEED, "shown_steps": SHOW_STEPS,
        "cosine_median": float(np.median(c1)),
        "caption_cap": "smoothness varies by pair: same-run adjacent-step "
                       "medians range -0.32 (cat x dog) to +0.97 over the 8 "
                       "dose pairs x 2 seeds; see "
                       "outputs/interaction_term/direction_wall/direction_wall.json",
    }, indent=2))

    # ---- D1b: the counterexample. Cat x dog's correction alternates sign ----
    rb = load_cell("a_cat__x__a_dog", 9).r_t().float().flatten(1)
    c1b = torch.nn.functional.cosine_similarity(rb[:-1], rb[1:], dim=1).numpy()
    # No thumbnails: the curve fills the whole panel, and this run's frames
    # already appear in D2's top row.
    fig, ax = base_axes(plt)
    ax.plot(np.arange(len(c1b)) + 0.5, c1b, color="0.2", lw=1.4, zorder=3)
    ax.text(0.01, 0.985, "one uncorrected run: cat x dog, seed 9 "
            "(its frames are D2's top row)",
            transform=ax.transAxes, fontsize=7.5, color=BLUE, va="top")
    ax.text(0.99, 0.03, "agreement between consecutive steps' corrections",
            transform=ax.transAxes, fontsize=7, color="0.35", ha="right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"direction-agreement-between-consecutive-steps-when-it-alternates.{ext}", dpi=300)
    plt.close(fig)
    (OUT_DIR / "direction-agreement-between-consecutive-steps-when-it-alternates.json").write_text(json.dumps({
        "pair": "a_cat__x__a_dog", "seed": 9, "shown_steps": SHOW_STEPS,
        "cosine_median": float(np.median(c1b)),
        "mid_run_median_steps_16_32": float(np.median(c1b[16:33])),
        "role": "the real counterexample to D1: this pair's correction "
                "alternates sign mid-run, and per F2 it still composes at "
                "full dose, so temporal coherence is not necessary for the "
                "injected correction to work",
    }, indent=2))

    # ---- D1c: the floor. The same computation on random vectors ----
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(r.shape, generator=g)
    c1c = torch.nn.functional.cosine_similarity(rnd[:-1], rnd[1:], dim=1).numpy()
    fig, ax = base_axes(plt)
    ax.plot(np.arange(len(c1c)) + 0.5, c1c, color="0.2", lw=1.4, zorder=3)
    ax.text(0.01, 0.985, "random norm-matched vectors, no run behind this line",
            transform=ax.transAxes, fontsize=7.5, color=INK, va="top")
    ax.text(0.99, 0.9, "what \"no signal\" looks like on these axes",
            transform=ax.transAxes, fontsize=7, color="0.35", ha="right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"direction-agreement-for-random-vectors.{ext}", dpi=300)
    plt.close(fig)
    (OUT_DIR / "direction-agreement-for-random-vectors.json").write_text(json.dumps({
        "generator_seed": 0, "shape": list(r.shape),
        "cosine_median": float(np.median(c1c)),
        "role": "falsification floor: if real corrections looked like this, "
                "D1's claim would be dead wholesale",
    }, indent=2))

    # ---- D3: two different pairs, matched steps ----
    a = load_cell(*D3_A).r_t().float().flatten(1)
    b = load_cell(*D3_B).r_t().float().flatten(1)
    c3 = torch.nn.functional.cosine_similarity(a, b, dim=1).numpy()
    fig, ax = base_axes(plt)
    ax.plot(np.arange(len(c3)), c3, color="0.2", lw=1.4, zorder=3)
    for k in SHOW_STEPS:
        add_thumb(ax, frames[(*D3_A, k)], k, 0.76, BLUE)
        add_thumb(ax, frames[(*D3_B, k)], k, 0.24, ORANGE)
        step_marker(ax, k)
    ax.text(0.01, 0.985, "uncorrected cat x dog, seed 9",
            transform=ax.transAxes, fontsize=7.5, color=BLUE, va="top")
    ax.text(0.01, 0.015, "uncorrected eagle x hawk, seed 9",
            transform=ax.transAxes, fontsize=7.5, color=ORANGE, va="bottom")
    ax.text(0.99, 0.53, "direction agreement of the two pairs' corrections",
            transform=ax.transAxes, fontsize=7, color="0.35", ha="right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"direction-agreement-between-two-pairs.{ext}", dpi=300)
    plt.close(fig)
    (OUT_DIR / "direction-agreement-between-two-pairs.json").write_text(json.dumps({
        "pair_a": D3_A, "pair_b": D3_B, "shown_steps": SHOW_STEPS,
        "cosine_median": float(np.median(c3)),
        "cosine_range": [float(c3.min()), float(c3.max())],
    }, indent=2))

    print(f"D1 adjacent-step median {np.median(c1):+.3f}")
    print(f"D3 cross-pair median {np.median(c3):+.3f}")
    print(f"wrote {OUT_DIR}/direction-agreement-between-consecutive-steps.* and "
          f"direction-agreement-between-two-pairs.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
