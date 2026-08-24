"""Training-checkpoint attention sweep (mechanism study, follow-up).

Loads SDXL once, then for each LoRA training checkpoint: attaches those
weights, runs the λ=1 capture on the chosen seed(s), and decodes the sample.
Answers "does the LoRA learn to separate cat from dog as training proceeds?"
by giving one attention capture + one decoded image per checkpoint.

Output per checkpoint:
    <out-root>/step_<N>/seed_<S>/attn_maps/step_XXX_token_*.pt
    <out-root>/step_<N>/seed_<S>/sample.png

Usage::

    python -m poe_repair.experiments.mechanism_study.train_sweep \
        --ckpt-dir <.../checkpoints> \
        --steps 12500,15000,20000,30000,45000,62500,80000,100000 \
        --seeds 9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from poe_repair.experiments.one_pair_one_seed.main import encode_all_prompts
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents
from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
from poe_repair.experiments.mechanism_study.capture_attention import (
    CAT_DOG_TOKEN_INDICES, _maybe_attach_lora,
)
from poe_repair.methods._sampling import (
    run_lora_residual_inject, write_decoded_image,
)
from poe_repair.runtime import (
    ensure_dir, infer_device, infer_dtype, load_ddim_scheduler,
    load_sdxl_models, write_json,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath
from poe_repair import paths

DEFAULT_OUT = paths.resolve(paths.ATTENTION_MECHANISM) / "lora_train_sweep"


def _parse_ints(arg: str) -> list[int]:
    out: list[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_sweep")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--steps", required=True,
                    help="comma-separated training steps, e.g. 12500,20000,...")
    ap.add_argument("--seeds", default="9")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--attn-resolution", type=int, default=32)
    args = ap.parse_args(argv)

    steps = _parse_ints(args.steps)
    seeds = _parse_ints(args.seeds)
    ckpt_dir = Path(args.ckpt_dir)
    out_root = (
        Path(args.out_root) if args.out_root
        else DEFAULT_OUT / args.pair_slug
    )

    device = infer_device(None)
    dtype = infer_dtype("float16", device)
    models = load_sdxl_models(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device=device, dtype=dtype,
    )
    scheduler = load_ddim_scheduler("stabilityai/stable-diffusion-xl-base-1.0")
    unet = models["unet"]

    class _P:
        prompt_a, prompt_b, joint_prompt = (
            args.prompt_a, args.prompt_b, args.joint_prompt
        )

    class _C:
        cell = _P()

    emb = encode_all_prompts(_C(), models, device, dtype)

    # pin init latents per seed once
    inits = {}
    for s in seeds:
        cell = CellPath.from_root(
            args.pair_slug, int(s), cache_root=DEFAULT_CACHE_ROOT
        )
        inits[s] = load_pinned_init_latents(
            cell, device=device, dtype=dtype,
            euler_init_noise_sigma=float(args.euler_sigma),
        )

    # Attach the LoRA shell ONCE from the first present checkpoint (this reads
    # rank/alpha/targets and loads its weights). Later checkpoints just copy
    # new weights into the same adapter via load_lora_state — no re-attach,
    # which diffusers 0.29 can't cleanly undo.
    present = [s for s in steps if (ckpt_dir / f"lora_step_{s:06d}.pt").exists()]
    if not present:
        raise SystemExit(f"no checkpoints found in {ckpt_dir} for {steps}")
    adapter_name = _maybe_attach_lora(
        unet, str(ckpt_dir / f"lora_step_{present[0]:06d}.pt")
    )

    manifest = []
    for step in steps:
        ckpt = ckpt_dir / f"lora_step_{step:06d}.pt"
        if not ckpt.exists():
            print(f"[train_sweep] SKIP missing {ckpt}", flush=True)
            continue
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        lora_trainer.load_lora_state(unet, sd["lora_state"])
        unet.eval()
        for s in seeds:
            seed_dir = ensure_dir(out_root / f"step_{step}" / f"seed_{s}")
            attn_dir = seed_dir / "attn_maps"
            out = run_lora_residual_inject(
                init_latents=inits[s], models=models, scheduler=scheduler,
                seq_a=emb["seq_a"], pool_a=emb["pool_a"],
                seq_b=emb["seq_b"], pool_b=emb["pool_b"],
                seq_j=emb["seq_j"], pool_j=emb["pool_j"],
                seq_e=emb["seq_e"], pool_e=emb["pool_e"],
                guidance_scale=7.5, num_inference_steps=50,
                height=1024, width=1024,
                euler_init_noise_sigma=float(args.euler_sigma),
                device=device, dtype=dtype, lambda_value=1.0,
                lora_adapter_name=adapter_name,
                attn_capture_dir=attn_dir,
                attn_token_indices=CAT_DOG_TOKEN_INDICES,
                attn_resolution=int(args.attn_resolution),
                attn_capture_lora=True,
            )
            write_decoded_image(out.image, seed_dir / "sample.png")
            n = len(list(attn_dir.glob("step_*_token_*.pt")))
            dsum = float(sum(out.extras["delta_norm_per_step"]))
            manifest.append({"step": step, "seed": s, "n_attn": n,
                             "delta_sum": round(dsum, 2)})
            print(f"[train_sweep] step={step} seed={s} n={n} Δ={dsum:.1f}",
                  flush=True)

    write_json(out_root / "sweep_manifest.json", {
        "pair_slug": args.pair_slug,
        "ckpt_dir": str(ckpt_dir),
        "steps": steps, "seeds": seeds,
        "token_indices": CAT_DOG_TOKEN_INDICES,
        "records": manifest,
    })
    print(f"[train_sweep] done — {len(manifest)} captures → {out_root}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
