"""Qualitative viewer for captured cross-attention (mechanism study, plan 01).

The sanity table (``sanity_table.py``) gives the *number* — total attention
mass per (seed, timestep). This gives the *picture*: the actual 32×32 spatial
attention maps, so you can SEE where cat-attention and dog-attention land, and
whether one concept's map collapses (the visual signature of PoE dropping it).

Two figures per seed:
  1. timestep strip — cat map and dog map side by side at a few denoising
     steps, on a shared color scale. Read left→right to watch attention
     localize (or not).
  2. overlay (optional, --overlay) — the late-step cat/dog maps blended over
     the decoded RGB sample, so attention is grounded on the actual image.
     Requires the sample PNG; pass --decode to render it from the pinned
     latent on the GPU if it isn't on disk.

Usage::

    # cheap, no GPU — just the attention maps
    python -m poe_repair.experiments.mechanism_study.view_attention \
        --regime plain_poe --seed 1

    # grounded overlay — decodes the sample from the pinned latent (GPU)
    python -m poe_repair.experiments.mechanism_study.view_attention \
        --regime plain_poe --seed 1 --overlay --decode
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from poe_repair import paths

DEFAULT_ATTN_ROOT = paths.resolve(paths.ATTENTION_MECHANISM)
_STEP_RE = re.compile(r"step_(\d+)_token_(.+)\.pt$")


def _load_maps(attn_dir: Path, token_key: str) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for f in sorted(attn_dir.glob(f"step_*_token_{token_key}.pt")):
        m = _STEP_RE.search(f.name)
        if not m:
            continue
        sd = torch.load(f, weights_only=False)
        out[int(m.group(1))] = sd["map"].float()
    return out


def _pick_steps(available: list[int], k: int) -> list[int]:
    if len(available) <= k:
        return available
    idx = [round(i * (len(available) - 1) / (k - 1)) for i in range(k)]
    return [available[i] for i in sorted(set(idx))]


def _decode_sample(seed: int, pair_slug: str, prompts: tuple[str, str, str],
                   euler_sigma: float) -> "object":
    """Decode the plain-PoE sample from the pinned init latent (deterministic)."""
    from poe_repair.experiments.one_pair_one_seed.main import encode_all_prompts
    from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents
    from poe_repair.methods._sampling import run_lora_residual_inject
    from poe_repair.runtime import (
        infer_device, infer_dtype, load_ddim_scheduler, load_sdxl_models,
    )
    from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath

    device = infer_device(None)
    dtype = infer_dtype("float16", device)
    models = load_sdxl_models(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device=device, dtype=dtype,
    )
    scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")

    class _P:
        prompt_a, prompt_b, joint_prompt = prompts

    class _C:
        cell = _P()

    emb = encode_all_prompts(_C(), models, device, dtype)
    cell = CellPath.from_root(pair_slug, int(seed), cache_root=DEFAULT_CACHE_ROOT)
    init = load_pinned_init_latents(
        cell, device=device, dtype=dtype, euler_init_noise_sigma=euler_sigma,
    )
    out = run_lora_residual_inject(
        init_latents=init, models=models, scheduler=scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_j=emb["seq_j"], pool_j=emb["pool_j"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=7.5, num_inference_steps=50,
        height=1024, width=1024, euler_init_noise_sigma=euler_sigma,
        device=device, dtype=dtype, lambda_value=0.0,
    )
    return out.image  # (1,3,H,W) or (3,H,W) cpu tensor in [0,1] / [0,255]


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ap = argparse.ArgumentParser(prog="view_attention")
    ap.add_argument("--regime", default="plain_poe")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--root", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tokens", default="cat_branch_poe,dog_branch_poe",
                    help="comma-separated save_keys to show as rows")
    ap.add_argument("--n-steps", type=int, default=6,
                    help="how many timestep columns to show")
    ap.add_argument("--overlay", action="store_true",
                    help="also blend late-step maps over the decoded sample")
    ap.add_argument("--decode", action="store_true",
                    help="decode the sample from the pinned latent (GPU) if "
                         "no PNG is supplied")
    ap.add_argument("--sample-png", default=None,
                    help="path to an existing decoded sample PNG for overlay")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    args = ap.parse_args(argv)

    root = (
        Path(args.root) if args.root
        else DEFAULT_ATTN_ROOT / args.regime / args.pair_slug
    )
    attn_dir = root / f"seed_{args.seed}" / "attn_maps"
    if not attn_dir.is_dir():
        raise SystemExit(f"no attn_maps under {attn_dir}")
    tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else (root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    maps_by_tok = {t: _load_maps(attn_dir, t) for t in tokens}
    steps_all = sorted(set().union(*[set(m) for m in maps_by_tok.values()]))
    if not steps_all:
        raise SystemExit(f"no maps found for tokens {tokens} under {attn_dir}")
    steps = _pick_steps(steps_all, args.n_steps)

    # Shared color scale across the whole figure so columns are comparable.
    vmax = max(
        float(maps_by_tok[t][s].max())
        for t in tokens for s in steps if s in maps_by_tok[t]
    )

    # --- Figure 1: timestep strip ---
    fig, axes = plt.subplots(
        len(tokens), len(steps),
        figsize=(1.9 * len(steps) + 0.5, 1.9 * len(tokens) + 0.6),
        squeeze=False,
    )
    for r, t in enumerate(tokens):
        for c, s in enumerate(steps):
            ax = axes[r][c]
            mp = maps_by_tok[t].get(s)
            if mp is not None:
                ax.imshow(mp.numpy(), cmap="magma", vmin=0, vmax=vmax,
                          interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t{s}", fontsize=9)
            if c == 0:
                ax.set_ylabel(t.replace("_branch_poe", ""), fontsize=10)
    fig.suptitle(
        f"cross-attention · {args.regime} · seed {args.seed} · "
        f"shared vmax={vmax:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    strip_png = out_dir / f"attn_strip_seed_{args.seed}.png"
    fig.savefig(strip_png, dpi=130)
    plt.close(fig)
    print(f"[view_attention] strip → {strip_png}")

    # --- Figure 2: overlay on decoded sample ---
    if args.overlay:
        rgb = None
        if args.sample_png:
            rgb = np.asarray(plt.imread(args.sample_png))[..., :3]
        elif args.decode:
            img = _decode_sample(
                args.seed, args.pair_slug,
                (args.prompt_a, args.prompt_b, args.joint_prompt),
                float(args.euler_sigma),
            )
            arr = img.squeeze(0) if img.dim() == 4 else img
            arr = arr.permute(1, 2, 0).float().numpy()
            if arr.max() > 1.5:
                arr = arr / 255.0
            rgb = np.clip(arr, 0, 1)
        if rgb is None:
            print("[view_attention] overlay requested but no sample "
                  "(pass --sample-png or --decode); skipping overlay")
        else:
            H, W = rgb.shape[:2]
            late = steps[-1]
            fig, axes = plt.subplots(
                1, len(tokens) + 1, figsize=(4.0 * (len(tokens) + 1), 4.2),
            )
            axes[0].imshow(rgb); axes[0].set_title("decoded sample")
            axes[0].set_xticks([]); axes[0].set_yticks([])
            for i, t in enumerate(tokens, start=1):
                mp = maps_by_tok[t].get(late)
                axes[i].imshow(rgb)
                if mp is not None:
                    up = torch.nn.functional.interpolate(
                        mp[None, None], size=(H, W), mode="bilinear",
                        align_corners=False,
                    )[0, 0].numpy()
                    axes[i].imshow(up, cmap="magma", alpha=0.55,
                                   vmin=0, vmax=vmax)
                axes[i].set_title(f"{t.replace('_branch_poe','')} attn @ t{late}")
                axes[i].set_xticks([]); axes[i].set_yticks([])
            fig.suptitle(
                f"{args.regime} · seed {args.seed} · attention grounded on sample",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            ov_png = out_dir / f"attn_overlay_seed_{args.seed}.png"
            fig.savefig(ov_png, dpi=130)
            plt.close(fig)
            print(f"[view_attention] overlay → {ov_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
