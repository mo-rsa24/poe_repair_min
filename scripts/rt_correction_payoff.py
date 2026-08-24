#!/usr/bin/env python
"""What the right correction buys on the very runs D1-D3 showed failing.

Three rows, one per run (cat x dog seed 9, cat x dog seed 13, eagle x hawk
seed 9). Left block: the uncorrected PoE run's own estimate of where it is
heading at steps 10, 20, 30, 40 (dashed borders, the same frames D1-D3 use).
Right block: the same starting noise with the pair's own correction on at
full dose (solid borders). Every run separates into two animals, and the two
cat x dog runs separate into DIFFERENT scenes, which is D2's claim made
positively: both runs are fixable, but each needed its own fix.

Caption caveat, not optional: in the corrected seed-13 row the two animals
read as a tan dog and a white puppy; the detector's count>=2 rule calls it
composed, but the caption may not call that cell "a cat and a dog".

Frame provenance, recorded in the sidecar:
- uncorrected frames: Tweedie from the cached x_t and the PoE prediction
  (built by scripts/rt_reuse_figure.py / rt_direction_companions.py).
- corrected seed 9: the cross run's saved clean estimates (x0_estimates).
- corrected seed 13 and eagle x hawk: solved exactly from consecutive saved
  latents of the lambda=1 dose runs (a DDIM step is linear in the clean image
  and the noise, so two consecutive latents pin the clean image down), then
  VAE-decoded. Solved frames are cached beside the uncorrected ones.

    CUDA_VISIBLE_DEVICES=0 python scripts/rt_correction_payoff.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.experiments.interaction_term.cache import _alphas_cumprod  # noqa: E402
from scripts.rt_reuse_figure import FRAME_DIR  # noqa: E402

# F4 measured that the correction only composes when it arrives in the first
# ten steps, so the frames are placed inside that window and just past it.
# Starting the strip at step 10, as an even spread over the run would, shows
# four frames of the aftermath and none of the moment the paths separate.
STEPS = (2, 6, 10, 20)
OUT_DIR = Path("paper/iclr/figures")
FIG_NAME = "samples-per-step-with-the-correction-on-and-off"
CROSS9_FRAMES = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cross/pairs/"
    "a_cat__x__a_dog/seed_9/call__rall/frames")
DOSE = Path("outputs/interaction_term/dose/pairs")

ROWS = [("a_cat__x__a_dog", 9, "#1f77b4", "cat x dog, seed 9"),
        ("a_cat__x__a_dog", 13, "#2ca02c", "cat x dog, seed 13"),
        ("an_eagle__x__a_hawk", 9, "#ff7f0e", "eagle x hawk, seed 9")]


def solved_corrected_frame(pair: str, seed: int, k: int) -> Path:
    """The lambda=1 run's clean estimate at step k, solved from its latents."""
    out = FRAME_DIR / f"{pair}_seed{seed}_step{k:03d}_corrected.png"
    if out.exists():
        return out

    from diffusers import AutoencoderKL
    from poe_repair.config import RunConfig
    from PIL import Image

    traj_file = (DOSE / pair / f"seed_{seed}" / "teacher_residual_const_lam100"
                 / "latent_trajectory.pt")
    if not traj_file.exists():
        raise SystemExit(
            f"no lambda=1 trajectory for {pair} seed {seed}. Generate it:\n"
            f"  python scripts/interaction_term_inject.py --pair {pair} "
            f"--seed {seed} --lambda 1.0 --exp-name interaction_term/dose")
    blob = torch.load(traj_file, map_location="cpu", weights_only=True)
    traj = blob["trajectories"].float()
    ts = [int(t) for t in blob["timesteps"]]
    ab = _alphas_cumprod()
    a1, b1 = ab[ts[k]].sqrt(), (1 - ab[ts[k]]).sqrt()
    if k + 1 < len(ts):
        a2, b2 = ab[ts[k + 1]].sqrt(), (1 - ab[ts[k + 1]]).sqrt()
    else:
        a2, b2 = torch.tensor(1.0), torch.tensor(0.0)
    x0 = (b2 * traj[k] - b1 * traj[k + 1]) / (a1 * b2 - a2 * b1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(RunConfig().model_id, subfolder="vae",
                                        torch_dtype=torch.float32).to(device).eval()
    with torch.no_grad():
        img = vae.decode((x0 / vae.config.scaling_factor).to(device)).sample[0]
    arr = ((img / 2 + 0.5).clamp(0, 1) * 255).byte().cpu()
    Image.fromarray(arr.permute(1, 2, 0).numpy()).resize((256, 256)).save(out)
    return out


def corrected_frame(pair: str, seed: int, k: int) -> Path:
    if (pair, seed) == ("a_cat__x__a_dog", 9):
        return CROSS9_FRAMES / f"step_{k:03d}.png"
    return solved_corrected_frame(pair, seed, k)


def uncorrected_frame(pair: str, seed: int, k: int) -> Path:
    p = FRAME_DIR / f"{pair}_seed{seed}_step{k:03d}.png"
    if not p.exists():
        raise SystemExit(
            f"missing uncorrected frame {p}. Build it first:\n"
            "  python scripts/rt_reuse_figure.py && "
            "python scripts/rt_direction_companions.py")
    return p


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 8, figsize=(11, 4.6))
    for r, (pair, seed, colour, label) in enumerate(ROWS):
        for c, k in enumerate(STEPS):
            for block, get in ((0, uncorrected_frame), (4, corrected_frame)):
                ax = axes[r][c + block]
                ax.imshow(plt.imread(get(pair, seed, k)))
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(1.6)
                    sp.set_color(colour)
                    if block == 0:
                        sp.set_linestyle((0, (3, 2)))
                if r == 0:
                    ax.set_title(f"step {k}", fontsize=8)
        axes[r][0].set_ylabel(label, fontsize=8, color=colour)

    fig.text(0.30, 0.965,
             "correction OFF (the runs shown in D1-D3, dashed borders)",
             ha="center", fontsize=10)
    fig.text(0.72, 0.965, "correction ON, full dose, same starting noise",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{FIG_NAME}.{ext}", dpi=200)
    plt.close(fig)

    (OUT_DIR / f"{FIG_NAME}.json").write_text(json.dumps({
        "rows": [{"pair": p, "seed": s} for p, s, _, _ in ROWS],
        "shown_steps": STEPS,
        "uncorrected": "Tweedie x0 from cached x_t + PoE prediction (the PoE "
                       "path's own estimate), frames shared with D1-D3",
        "corrected": "cat x dog seed 9 from the cross run's saved "
                     "x0_estimates; the other two solved from consecutive "
                     "lambda=1 dose-run latents (DDIM step linear in x0 and "
                     "eps) and VAE-decoded fp32",
        "step_choice": "steps sit inside F4's composing window (0-10) and just "
                       "past it. An even spread over the run starts at step 10, "
                       "by which point both sides have already committed",
        "what_is_visible": "at step 2 neither side has committed to anything; "
                           "by step 6 the uncorrected run is already one animal "
                           "and the corrected run is already two. Three cells, "
                           "so this locates the commitment for illustration and "
                           "F4's 9-window sweep is what measures it",
        "caption_caveat": "corrected seed 13 reads as a tan dog and a white "
                          "puppy; count>=2 calls it composed, but the caption "
                          "may not call that cell a cat and a dog",
    }, indent=2))
    print(f"wrote {OUT_DIR / FIG_NAME}.png and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
