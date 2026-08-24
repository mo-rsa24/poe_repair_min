"""Export a real end-to-end trace of "a cat and a dog" for the picture-speak artifact.

DRAFT FOR REVIEW / GPU run. Runs the real LoRA residual-inject pipeline once on the
cat×dog marginals with the trained adapter, and dumps a single compact JSON that
makes the artifact's "Watch it run" tab and the VAE thumbnails REAL instead of
schematic. Inference-only (no training), but still a GPU job: run it through the
usual preflight, not on a login node.

What the JSON carries (all small, normalised, ready to inline as the page's
`TRACE` and `VAE_ACT` globals):

  prompt        the joint prompt string
  tokens, ids   real CLIP BPE tokens + integer IDs (transformers tokenizer)
  embed         per-token embedding preview (each 2048-vec pooled to 8 bins, 0..1)
  steps         the denoising steps sampled (e.g. 0,7,…,49)
  latents       per sampled step: 6×6 channel-mean latent slice, 0..1 (noise→clean)
  delta_norm    per sampled step: ‖Δ̂‖ (the LoRA residual magnitude)
  attn          one mid step's cat-token and dog-token 8×8 cross-attention maps
  vae_act       per VAE block: 6×6 activation slice, 0..1 (for the featureThumbs)
  image         the REAL decoded 128×128 cat+dog, as a data:image/png;base64 URI

Usage::

    python -m poe_repair.experiments.mechanism_study.export_trace \
        --checkpoint artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/\
taskB__k04_ep2000_resumed__wandb-pueuo7bl/checkpoints/lora_step_062500.pt \
        --seed 9 --out scratchpad/trace.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from poe_repair.experiments.one_pair_one_seed.main import encode_all_prompts
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents
from poe_repair.experiments.mechanism_study.capture_attention import (
    CAT_DOG_TOKEN_INDICES,
    _maybe_attach_lora,
)
from poe_repair.experiments.mechanism_study.export_vae_activations import (
    _module_map,
    _slice6,
)
from poe_repair.methods._sampling import run_lora_residual_inject, write_decoded_image
from poe_repair.runtime import (
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    write_json,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath

log = logging.getLogger(__name__)


def _grid6(x: torch.Tensor) -> list[float]:
    """[.,C,H,W] or [C,H,W] → 36 floats: channel-mean, 6×6 pool, min-max normalised."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    a = x.detach().float().abs().mean(dim=1, keepdim=True)
    a = F.adaptive_avg_pool2d(a, (6, 6)).flatten()
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return [round(float(v), 4) for v in a.tolist()]


def _map8(pt_path: Path) -> list[float]:
    """A saved attention map .pt → 64 floats (8×8), min-max normalised."""
    m = torch.load(pt_path, weights_only=False)["map"].float()
    m = F.adaptive_avg_pool2d(m.unsqueeze(0).unsqueeze(0), (8, 8)).flatten()
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return [round(float(v), 4) for v in m.tolist()]


def main(argv=None) -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(prog="export_trace")
    ap.add_argument("--checkpoint", required=True, help="trained lora_step_*.pt")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--out", default="scratchpad/trace.json")
    ap.add_argument("--pair-slug", default="a_cat__x__a_dog")
    ap.add_argument("--prompt-a", default="a cat")
    ap.add_argument("--prompt-b", default="a dog")
    ap.add_argument("--joint-prompt", default="a cat and a dog")
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--euler-sigma", type=float, default=1.0)
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = infer_device(args.device)
    dtype = infer_dtype("float16", device)
    models = load_sdxl_models(model_id=args.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(args.model_id)
    unet = models["unet"]
    adapter_name = _maybe_attach_lora(unet, args.checkpoint)

    # ---- 1. tokens + ids (real CLIP BPE tokenizer) -------------------------
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    enc = tok(args.joint_prompt)
    ids = list(enc["input_ids"])
    tokens = tok.convert_ids_to_tokens(ids)
    log.info("tokens: %s", tokens)

    # ---- 2. embeddings + preview -------------------------------------------
    class _P: prompt_a = args.prompt_a; prompt_b = args.prompt_b; joint_prompt = args.joint_prompt
    class _C: cell = _P()
    emb = encode_all_prompts(_C(), models, device, dtype)
    seq_j = emb["seq_j"][0].detach().float().cpu()            # [77, 2048]
    n_tok = len(ids)
    embed_preview = []
    for i in range(min(n_tok, seq_j.shape[0])):
        v = seq_j[i].reshape(8, -1).mean(dim=1)               # 2048 → 8 bins
        v = (v - v.min()) / (v.max() - v.min() + 1e-6)
        embed_preview.append([round(float(x), 3) for x in v.tolist()])

    # ---- 3. run the real LoRA pipeline, capturing latents + attention ------
    cell = CellPath.from_root(args.pair_slug, int(args.seed), cache_root=DEFAULT_CACHE_ROOT)
    init = load_pinned_init_latents(cell, device=device, dtype=dtype,
                                    euler_init_noise_sigma=float(args.euler_sigma))
    steps = list(range(0, args.num_inference_steps, max(1, args.num_inference_steps // 8)))
    if steps[-1] != args.num_inference_steps - 1:
        steps.append(args.num_inference_steps - 1)
    attn_dir = Path(tempfile.mkdtemp(prefix="trace_attn_"))
    unet.eval()
    out = run_lora_residual_inject(
        init_latents=init, models=models, scheduler=scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_j=emb["seq_j"], pool_j=emb["pool_j"], seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=float(args.guidance_scale),
        num_inference_steps=int(args.num_inference_steps),
        height=1024, width=1024, euler_init_noise_sigma=float(args.euler_sigma),
        device=device, dtype=dtype, lambda_value=1.0, lora_adapter_name=adapter_name,
        record_delta_at_steps=steps,
        attn_capture_dir=attn_dir, attn_token_indices=CAT_DOG_TOKEN_INDICES,
        attn_resolution=32, attn_capture_lora=True,
    )
    where = out.extras["where_applied_cache"]
    dnorm = out.extras["delta_norm_per_step"]
    latents = [_grid6(where[s]["x_t"]) for s in steps if s in where]
    delta_norm = [round(float(dnorm[s]), 3) for s in steps if s < len(dnorm)]

    # ---- 4. one mid-step cat/dog attention map -----------------------------
    mid = steps[len(steps) // 2]
    attn = {}
    for key in ("cat_branch_poe", "dog_branch_poe"):
        p = attn_dir / f"step_{mid:03d}_token_{key}.pt"
        if p.exists():
            attn[key.split("_")[0]] = _map8(p)
    attn["step"] = mid

    # ---- 5. VAE per-block activations on the generated image ----------------
    vae = models["vae"]
    tmp_png = attn_dir / "gen.png"
    write_decoded_image(out.image, tmp_png)
    pil = Image.open(tmp_png).convert("RGB")
    x = (torch.from_numpy(np.asarray(pil.resize((1024, 1024)))).permute(2, 0, 1)
         .unsqueeze(0).float().div(127.5).sub(1.0).to(device=device, dtype=vae.dtype))
    vae_act = {"input": _slice6(x)}
    handles, mm = [], _module_map(vae)
    for name, mod in mm.items():
        handles.append(mod.register_forward_hook(
            lambda _m, _i, o, n=name: vae_act.__setitem__(
                n, _slice6(o[0] if isinstance(o, (tuple, list)) else o))))
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample()
        vae_act["latent"] = _slice6(z * float(getattr(vae.config, "scaling_factor", 0.13025)))
        _ = vae.decode(z).sample
    for h in handles:
        h.remove()

    # ---- 6. the real decoded image (128px base64) --------------------------
    small = pil.resize((128, 128))
    buf = io.BytesIO(); small.save(buf, format="PNG")
    image_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    write_json(Path(args.out), {
        "prompt": args.joint_prompt, "seed": int(args.seed),
        "tokens": tokens, "ids": ids, "embed": embed_preview,
        "steps": steps, "latents": latents, "delta_norm": delta_norm,
        "attn": attn, "vae_act": vae_act, "image": image_uri,
    })
    log.info("wrote trace to %s (%d steps, %d tokens, image %d bytes b64)",
             args.out, len(steps), len(tokens), len(image_uri))
    log.info("paste this JSON back and it inlines as the artifact's TRACE + VAE_ACT globals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
