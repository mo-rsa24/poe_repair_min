"""Capture cross-attention on the LoRA inference path (mechanism study, plan 01).

Runs ``run_lora_residual_inject`` on cat×dog's 12 seeds with cross-attention
capture wired around the 3-branch forward, and dumps per-token spatial
attention maps to disk in the same schema as the ``veracity_attn`` cache.

Two regimes, selected by ``--lambda`` + ``--capture``:

  λ=0, --capture frozen  → plain PoE (adapter never invoked). Task 1.
       Output: outputs/attn_mechanism/plain_poe/<slug>/seed_<N>/attn_maps/
  λ=1, --capture lora    → LoRA-corrected forward. Task 4. Needs --checkpoint.
       Output: outputs/attn_mechanism/lora_lambda1/<slug>/seed_<N>/attn_maps/

Branch mapping on the LoRA path is the 3-branch (A, B, ∅) cat order:
``0=A, 1=B, 2=∅`` — there is no J branch here. The token spec reuses the
solo-subject index from the teacher_residual ``_branch_poe`` variants
(cat→branch 0 tok 2, dog→branch 1 tok 2).

Usage::

    # Task 1 — plain PoE, no checkpoint needed
    python -m poe_repair.experiments.mechanism_study.capture_attention \
        --lambda 0 --capture frozen --seeds 1-12

    # Task 4 — LoRA-corrected, needs the trained checkpoint
    python -m poe_repair.experiments.mechanism_study.capture_attention \
        --lambda 1 --capture lora --checkpoint <lora_step_NNNNNN.pt> --seeds 1-12
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    run_lora_residual_inject,
    write_decoded_image,
)
from poe_repair.runtime import (
    ensure_dir,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    write_json,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath


log = logging.getLogger(__name__)

# Plan 01 pins outputs to the shared /datasets tree (alongside the existing
# veracity_attn cache), not the repo-local outputs/ dir.
DEFAULT_ATTN_ROOT = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism"
)

# Token spec for cat×dog on the 3-branch (A, B, ∅) LoRA path. Reuses the
# solo-subject token index (2) and the marginal branch semantics from the
# teacher_residual ``_branch_poe`` cache. The J branch has no analog here.
CAT_DOG_TOKEN_INDICES = {
    "cat_branch_poe": {"branch_index": 0, "token_index": 2},
    "dog_branch_poe": {"branch_index": 1, "token_index": 2},
}


def _parse_seeds(arg: str) -> list[int]:
    """'1-12' or '1,2,3' or a mix."""
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


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="capture_attention")
    ap.add_argument("--lambda", dest="lambda_value", type=float, required=True,
                    help="0 = plain frozen PoE; 1 = LoRA-corrected")
    ap.add_argument("--capture", choices=("frozen", "lora"), required=True,
                    help="frozen = capture adapter-OFF forward (plain PoE); "
                         "lora = capture adapter-ON forward (needs λ!=0)")
    ap.add_argument("--seeds", default="1-12",
                    help="e.g. '1-12' or '1,4,9'")
    ap.add_argument("--checkpoint", default=None,
                    help="lora_step_NNNNNN.pt — required only for --capture lora")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--out-root", default=None,
                    help="defaults to outputs/attn_mechanism/<regime>/<slug>/")
    ap.add_argument("--regime-name", default=None,
                    help="subdir under attn_mechanism/; defaults to "
                         "plain_poe (frozen) or lora_lambda1 (lora)")
    ap.add_argument("--attn-resolution", type=int, default=32)
    ap.add_argument("--renorm-tokens", type=int, default=None,
                    help="if set, softmax-renormalize each token map over the "
                         "first N real words (AAE-style share, not raw prob). "
                         "For solo 'a cat'/'a dog' prompts use 4.")
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--model-id",
                    default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32",
                             "bfloat16", "bf16"))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--write-image", action="store_true",
                    help="also decode + save the final PNG per seed (slower)")
    return ap


def _maybe_attach_lora(unet, checkpoint: str):
    """Attach the trained LoRA and load its weights. Only needed for --capture lora."""
    from peft import LoraConfig

    from poe_repair.experiments.lora import trainer as lora_trainer

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    rank = int(cfg_dict.get("lora", {}).get("rank", 8))
    alpha = int(cfg_dict.get("lora", {}).get("alpha", rank))
    target_modules = tuple(
        cfg_dict.get("lora", {}).get(
            "target_modules", ("attn2.to_q", "attn2.to_k", "attn2.to_v")
        )
    )
    adapter_name = str(cfg_dict.get("lora", {}).get("adapter_name", "lora"))
    log.info("ckpt config: rank=%d alpha=%d targets=%s", rank, alpha, target_modules)
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=list(target_modules), init_lora_weights=True,
    )
    unet.add_adapter(lora_cfg, adapter_name=adapter_name)
    with torch.no_grad():
        for name, prm in unet.named_parameters():
            prm.requires_grad_(False)
            if "lora_" in name:
                prm.data = prm.data.to(torch.float32)
    lora_trainer.load_lora_state(unet, ckpt["lora_state"])
    log.info("loaded %d LoRA tensors from %s", len(ckpt["lora_state"]), checkpoint)
    return adapter_name


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("MECH_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = build_argparser().parse_args(argv)

    capture_lora = (p.capture == "lora")
    if capture_lora and float(p.lambda_value) == 0.0:
        raise SystemExit(
            "--capture lora requires --lambda != 0 (the adapter-ON forward "
            "never runs at λ=0)"
        )
    if capture_lora and not p.checkpoint:
        raise SystemExit("--capture lora requires --checkpoint")

    regime = p.regime_name or ("lora_lambda1" if capture_lora else "plain_poe")
    out_root = (
        Path(p.out_root) if p.out_root
        else DEFAULT_ATTN_ROOT / regime / p.pair_slug
    )
    seeds = _parse_seeds(p.seeds)

    device = infer_device(p.device)
    dtype = infer_dtype(p.dtype, device)
    log.info("device=%s dtype=%s regime=%s λ=%s capture=%s seeds=%s",
             device, dtype, regime, p.lambda_value, p.capture, seeds)

    models = load_sdxl_models(model_id=p.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(p.model_id)
    unet = models["unet"]

    adapter_name = "lora"
    if capture_lora:
        adapter_name = _maybe_attach_lora(unet, p.checkpoint)

    class _PromptShim:
        prompt_a = p.prompt_a
        prompt_b = p.prompt_b
        joint_prompt = p.joint_prompt

    class _CfgShim:
        cell = _PromptShim()

    embeddings = encode_all_prompts(_CfgShim(), models, device, dtype)
    cache_root = Path(p.cache_root) if p.cache_root else DEFAULT_CACHE_ROOT

    manifest: list[dict] = []
    for s in seeds:
        seed_dir = ensure_dir(out_root / f"seed_{int(s)}")
        attn_dir = seed_dir / "attn_maps"
        cell = CellPath.from_root(p.pair_slug, int(s), cache_root=cache_root)
        init = load_pinned_init_latents(
            cell, device=device, dtype=dtype,
            euler_init_noise_sigma=float(p.euler_sigma),
        )
        log.info("seed=%d → %s", int(s), attn_dir)
        unet.eval()
        out = run_lora_residual_inject(
            init_latents=init, models=models, scheduler=scheduler,
            seq_a=embeddings["seq_a"], pool_a=embeddings["pool_a"],
            seq_b=embeddings["seq_b"], pool_b=embeddings["pool_b"],
            seq_j=embeddings["seq_j"], pool_j=embeddings["pool_j"],
            seq_e=embeddings["seq_e"], pool_e=embeddings["pool_e"],
            guidance_scale=float(p.guidance_scale),
            num_inference_steps=int(p.num_inference_steps),
            height=int(p.height), width=int(p.width),
            euler_init_noise_sigma=float(p.euler_sigma),
            device=device, dtype=dtype,
            lambda_value=float(p.lambda_value),
            lora_adapter_name=adapter_name,
            attn_capture_dir=attn_dir,
            attn_token_indices=CAT_DOG_TOKEN_INDICES,
            attn_resolution=int(p.attn_resolution),
            attn_capture_lora=capture_lora,
            attn_renorm_tokens=p.renorm_tokens,
        )
        n_files = len(list(attn_dir.glob("step_*_token_*.pt")))
        if p.write_image:
            write_decoded_image(out.image, seed_dir / f"sample_seed_{int(s)}.png")
        manifest.append({
            "seed": int(s),
            "attn_dir": str(attn_dir),
            "n_attn_files": n_files,
            "delta_norm_sum": float(sum(out.extras["delta_norm_per_step"])),
        })
        log.info("seed=%d wrote %d attn files (Δ_sum=%.4g)",
                 int(s), n_files, manifest[-1]["delta_norm_sum"])

    write_json(out_root / "capture_manifest.json", {
        "regime": regime,
        "pair_slug": p.pair_slug,
        "lambda_value": float(p.lambda_value),
        "capture": p.capture,
        "checkpoint": (None if not p.checkpoint else str(p.checkpoint)),
        "prompt_a": p.prompt_a,
        "prompt_b": p.prompt_b,
        "joint_prompt": p.joint_prompt,
        "token_indices": CAT_DOG_TOKEN_INDICES,
        "attn_resolution": int(p.attn_resolution),
        "num_inference_steps": int(p.num_inference_steps),
        "seeds": seeds,
        "samples": manifest,
    })
    log.info("done — %d seeds, manifest at %s",
             len(seeds), out_root / "capture_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
