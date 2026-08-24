#!/usr/bin/env python
"""Can a caption name the right animal from a noise prediction, mid-run?

The kill switch for reading the correction in language space. Before asking
what words the correction adds, ask the easy question first: take a prediction
that was conditioned on one prompt and one prompt only, turn it into a picture,
and see whether a caption scorer says that prompt back.

At step k the model's guess at the finished picture is

    x̂_0 = (x_t − sqrt(1 − ᾱ_t) · ε) / sqrt(ᾱ_t)

which is ``tweedie_mean`` in ``poe_repair/_sdxl/metrics.py``. Decoded through
the VAE it is a real image, blurry at high noise and sharp by the end, and a
CLIP scorer will accept it. Nothing is sampled: every tensor comes from the
cache.

Four predictions are decoded per step index:

  a      guided ε̃_A, conditioned on the first prompt alone
  b      guided ε̃_B, conditioned on the second prompt alone
  j      guided ε̃_J, conditioned on the joint prompt
  poe    ε̃_A + ε̃_B − ε_∅, what the product of the two experts predicts

Each decoded image is scored against four captions built from the cell's own
prompts rather than its folder name: the two solos, the joint, and a blend.

**The pass, written down before the run.** On the ``a`` image the first solo
caption outranks the second, and mirrored on the ``b`` image, at step 25 and
step 40 of 50. That is the readback surviving contact with a noise prediction.
A failure at every step index, while the ceiling rows below still separate,
kills the x̂_0 route for this purpose.

**The ceiling rows.** CLIP similarities compare captions within one image and
never across images, so a flat mid-run row means nothing on its own. The cell's
finished ``mono.png`` and ``poe.png`` are scored the same way and printed
alongside, as what the scorer can do when the picture is definitely there.

**The confound.** If the ``a`` and ``j`` images score almost identically, x̂_0 at
that step is dominated by ``x_t`` rather than by ε, and the readback is reading
the shared trajectory instead of the prediction. The mean per-pixel L2 distance
between the decoded images is reported beside the scores so that is visible
rather than inferred.

Reads the cache and decodes; it does not sample.

Usage:
    python scripts/xhat0_readback.py --pair a_cat__x__a_dog --seed 1 --steps 10 25 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair._sdxl.metrics import guided_eps, poe_eps, tweedie_mean  # noqa: E402
from poe_repair._sdxl.sdipc_utils import decode_latents_to_tensor  # noqa: E402
from poe_repair.config import DEFAULT_MODEL_ID  # noqa: E402
from poe_repair.experiments.interaction_term.cache import (  # noqa: E402
    CACHE_ROOT,
    _alphas_cumprod,
    cell_dir,
    load_cell,
)
from poe_repair.experiments.residual_between_mono_and_poe.metrics import (  # noqa: E402
    clip_image_text_similarities,
)

OUT_ROOT = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/interaction_term/xhat0_readback"
)

BRANCHES = ("a", "b", "j", "poe")


def caption_bank(meta: dict) -> dict[str, str]:
    """Four captions built from the cell's own prompts.

    ``a``/``b`` are the two solo prompts verbatim, ``joint`` is the joint
    prompt verbatim, ``blend`` is the hybrid wording assembled from the two
    animal nouns with any leading article removed.
    """
    prompt_a, prompt_b = meta["pair"]
    noun_a = prompt_a.split(" ", 1)[-1] if prompt_a.startswith("a ") else prompt_a
    noun_b = prompt_b.split(" ", 1)[-1] if prompt_b.startswith("a ") else prompt_b
    return {
        "a": prompt_a,
        "b": prompt_b,
        "joint": meta["joint_prompt"],
        "blend": f"one creature that is part {noun_a} and part {noun_b}",
    }


def load_vae(device: torch.device):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        DEFAULT_MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
        use_safetensors=True,
    ).to(device).eval()
    vae.enable_tiling()
    return vae


def write_image(img: torch.Tensor, path: Path) -> Path:
    """``img`` is (1,3,H,W) in [0,1]."""
    arr = (img.squeeze(0).clamp(0, 1) * 255).round().to(torch.uint8)
    Image.fromarray(arr.permute(1, 2, 0).cpu().numpy()).save(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 25, 40])
    ap.add_argument("--root", type=Path, default=CACHE_ROOT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_ROOT / f"{args.pair}__seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cell = load_cell(args.pair, args.seed, root=args.root)
    captions = caption_bank(cell.meta)
    texts = [captions[k] for k in ("a", "b", "joint", "blend")]
    w = cell.guidance_scale
    alphas = _alphas_cumprod()

    vae = load_vae(device)

    rows: list[dict] = []
    for k in args.steps:
        if k >= cell.n_steps:
            print(f"step {k} beyond the cached {cell.n_steps}, skipped")
            continue
        t = int(cell.timesteps[k].item())
        alpha_bar = alphas[t].to(device=device, dtype=torch.float32)
        x_t = cell.x_t[k].to(device=device, dtype=torch.float32)

        eps_a = guided_eps(cell.eps_a_raw[k], cell.eps_uncond[k], w).to(device).float()
        eps_b = guided_eps(cell.eps_b_raw[k], cell.eps_uncond[k], w).to(device).float()
        eps_j = guided_eps(cell.eps_j_raw[k], cell.eps_uncond[k], w).to(device).float()
        eps_p = poe_eps(eps_a, eps_b, cell.eps_uncond[k].to(device).float())

        images: dict[str, torch.Tensor] = {}
        paths: dict[str, Path] = {}
        for name, eps in zip(BRANCHES, (eps_a, eps_b, eps_j, eps_p)):
            x0 = tweedie_mean(x_t, alpha_bar, eps)
            with torch.no_grad():
                img = decode_latents_to_tensor(vae, x0).float().cpu()
            images[name] = img
            paths[name] = write_image(img, out_dir / f"step{k:02d}_{name}.png")

        sims = clip_image_text_similarities([paths[n] for n in BRANCHES], texts)
        # the confound: how far apart are the decoded images at all
        dists = {
            f"{p}_vs_{q}": float((images[p] - images[q]).squeeze(0).norm(dim=0).mean())
            for p, q in (("a", "b"), ("a", "j"), ("j", "poe"))
        }
        rows.append({
            "step_index": k,
            "timestep": t,
            "alpha_bar": float(alpha_bar),
            "similarities": {
                branch: {texts[i]: sims[texts[i]][bi] for i in range(len(texts))}
                for bi, branch in enumerate(BRANCHES)
            },
            "image_distances": dists,
        })

    # ceiling rows: the cell's own finished images, scored the same way
    cdir = cell_dir(args.pair, args.seed, root=args.root)
    finished = [p for p in (cdir / "mono.png", cdir / "poe.png") if p.exists()]
    ceiling = {}
    if finished:
        csims = clip_image_text_similarities(finished, texts)
        ceiling = {
            p.stem: {t: csims[t][i] for t in texts} for i, p in enumerate(finished)
        }

    payload = {
        "pair": args.pair,
        "seed": args.seed,
        "split": cell.split,
        "captions": captions,
        "rows": rows,
        "ceiling": ceiling,
        "pass_criterion": (
            "on the 'a' image the first solo caption outranks the second, and "
            "mirrored on the 'b' image, at step 25 and step 40"
        ),
    }
    (out_dir / "readback.json").write_text(json.dumps(payload, indent=2))

    # readable table
    print(f"\ncaptions: {json.dumps(captions)}\n")
    for row in rows:
        print(f"step {row['step_index']:>2}  t={row['timestep']}  "
              f"alpha_bar={row['alpha_bar']:.4f}")
        for branch in BRANCHES:
            scores = row["similarities"][branch]
            best = max(scores, key=scores.get)
            line = "  ".join(f"{v:.3f}" for v in scores.values())
            print(f"    {branch:<4} {line}   best: {best!r}")
        print(f"    image distances: {row['image_distances']}")
    if ceiling:
        print("\nceiling, the finished images:")
        for name, scores in ceiling.items():
            best = max(scores, key=scores.get)
            line = "  ".join(f"{v:.3f}" for v in scores.values())
            print(f"    {name:<6} {line}   best: {best!r}")
    print(f"\nwrote {out_dir / 'readback.json'}")


if __name__ == "__main__":
    main()
