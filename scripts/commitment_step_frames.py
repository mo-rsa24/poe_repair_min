#!/usr/bin/env python
"""EXP-01's qualitative half: what the commitment step actually looks like.

Two rows, the fastest-committing pair and the slowest-committing pair from
`docs/evidence/EXP01-commitment-step/result.json`. Three columns per row, all
the model's own running estimate x0(t) at three steps of the SAME run:

    step 0            the model's first full-image guess, before any denoising
    commitment step   that pair's own individual commitment step (nearest to
                       its registered median; see SEED_CHOICE below)
    final step        the finished picture

All three are Tweedie estimates decoded through the VAE, not raw latents, so
"step 0" is not noise: it is the model's earliest guess, which is why it looks
soft and generic rather than like static.

Seed choice, decided by nearest-to-median before decoding either one:
  a_cat__x__a_dog        seed 2,  individual step 36 (exact match to median 36)
  a_dolphin__x__a_porpoise seed 4, individual step 20 (median is 18; seed 7 at
                          step 16 ties on distance and was not chosen)

Designed by /pair-figure, spec in EXPERIMENTS.md under EXP-01.

    CUDA_VISIBLE_DEVICES=1 python scripts/commitment_step_frames.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    _alphas_cumprod,
    load_cell,
)

OUT_DIR = Path("docs/evidence/EXP01-commitment-step")
FIG_NAME = "commitment-step-frames"
FRAME_CACHE = OUT_DIR / "frames"

ROWS = [
    # pair, seed, commitment_step, pair_median, colour
    ("a_cat__x__a_dog", 2, 36, 36.0, "#1f77b4"),
    ("a_dolphin__x__a_porpoise", 4, 20, 18.0, "#1f77b4"),
]


def x0_estimate(pair: str, seed: int) -> tuple[torch.Tensor, list[int]]:
    """[T,1,4,128,128] Tweedie estimate at every step, and the timesteps."""
    cell = load_cell(pair, seed, root=CACHE_ROOT)
    abar = _alphas_cumprod()[cell.timesteps.long()].view(-1, 1, 1, 1, 1)
    eps = cell.eps_poe()
    x0 = (cell.x_t - (1 - abar).sqrt() * eps) / abar.sqrt()
    return x0, [int(t) for t in cell.timesteps]


def decode(vae, x0_latent: torch.Tensor, device) -> "Image.Image":
    from PIL import Image
    with torch.no_grad():
        img = vae.decode((x0_latent / vae.config.scaling_factor).to(device)).sample[0]
    arr = ((img / 2 + 0.5).clamp(0, 1) * 255).byte().cpu()
    return Image.fromarray(arr.permute(1, 2, 0).numpy()).resize((256, 256))


def frame_path(pair: str, seed: int, tag: str) -> Path:
    return FRAME_CACHE / f"{pair}_seed{seed}_{tag}.png"


def main() -> int:
    from diffusers import AutoencoderKL
    from poe_repair.config import RunConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("no GPU visible; this decode needs one. Aborting rather than "
              "running an hours-long CPU VAE decode.")
        return 2

    FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(RunConfig().model_id, subfolder="vae",
                                        torch_dtype=torch.float32).to(device).eval()

    built = []
    for pair, seed, commit_step, median, colour in ROWS:
        x0, timesteps = x0_estimate(pair, seed)
        n_steps = x0.shape[0]
        final_idx = n_steps - 1
        commit_idx = min(commit_step, final_idx)

        cols = [("start", 0), ("commit", commit_idx), ("final", final_idx)]
        paths = {}
        for tag, idx in cols:
            p = frame_path(pair, seed, tag)
            if not p.exists():
                img = decode(vae, x0[idx], device)
                img.save(p)
            paths[tag] = str(p)

        built.append({
            "pair": pair, "seed": seed, "colour": colour,
            "commitment_step_individual": commit_step,
            "commitment_step_pair_median": median,
            "n_steps": n_steps,
            "frames": paths,
        })
        print(f"{pair} seed {seed}: start=step0 commit=step{commit_idx} "
              f"final=step{final_idx} -> {list(paths.values())}")

    del vae
    torch.cuda.empty_cache()

    # ---- assemble the 2x3 strip ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.6))
    col_titles = ["start (step 0, first guess)", "commitment step", "final (step 49)"]
    for r, row in enumerate(built):
        for c, tag in enumerate(("start", "commit", "final")):
            ax = axes[r][c]
            ax.imshow(plt.imread(row["frames"][tag]))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(1.8 if tag == "commit" else 0.8)
                sp.set_color(row["colour"] if tag == "commit" else "#999999")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8, family="serif")
        pretty = row["pair"].replace("a_", "").replace("an_", "").replace("_", " ").replace("  x  ", " x ")
        label = (f"{pretty}\nmedian step {row['commitment_step_pair_median']:.0f}, "
                 f"seed {row['seed']} step {row['commitment_step_individual']}")
        axes[r][0].set_ylabel(label, fontsize=7, family="serif")

    fig.text(0.5, 0.965,
              "the running estimate at commitment already looks like the final picture",
              ha="center", fontsize=9.5, family="serif")
    fig.text(0.5, 0.015,
              "latent decode of the model's own running estimate, VAE-decoded for display. "
              "perceptual validation of the 0.90 threshold is still owed.",
              ha="center", fontsize=6.5, family="serif", color="#777777")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=220)
    plt.close(fig)

    (OUT_DIR / f"{FIG_NAME}.json").write_text(json.dumps({
        "rows": built,
        "columns": ["start: x0 estimate at step 0", "commitment: x0 estimate at "
                    "that run's own commitment step", "final: x0 estimate at "
                    "the last step"],
        "seed_choice": "nearest individual commitment step to the pair's "
                        "registered median, ties broken by lower seed number",
        "caveat": "all three columns are Tweedie estimates decoded through the "
                  "VAE for display; the 0.90 cosine threshold that defines "
                  "commitment was computed in latent space, not on these "
                  "decoded pixels, and the perceptual check against it is "
                  "still owed (docs/evidence/EXP01-commitment-step/QUERY.md)",
    }, indent=2))
    print(f"\nwrote {OUT_DIR / FIG_NAME}.png and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
