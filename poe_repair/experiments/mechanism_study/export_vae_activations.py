"""Export real VAE per-block activation slices for the picture-speak explainer.

DRAFT FOR REVIEW / GPU run. Runs the real SDXL AutoencoderKL on one image,
hooks every encoder/decoder block, and dumps a tiny 6x6 channel-mean slice per
block (normalised 0..1) to JSON. That JSON is what makes the "activation slice"
thumbnails in the artifact real instead of schematic: paste it back and it is
inlined as the page's VAE_ACT global.

This is inference-only (no training), but it is still a GPU job: run it through
the usual preflight, not on a login node.

Usage::

    python -m poe_repair.experiments.mechanism_study.export_vae_activations \
        --image artifacts/results/can-lora-learn-a-residual-that-corrects-poe/results-by-pair/a_cat__x__a_dog/seed_42/results/... .png \
        --out scratchpad/vae_act.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from poe_repair.runtime import infer_device, infer_dtype, load_sdxl_models

log = logging.getLogger(__name__)

# Maps artifact block-id -> the submodule whose OUTPUT we capture.
def _module_map(vae):
    enc, dec = vae.encoder, vae.decoder
    m = {
        "conv_in": enc.conv_in,
        "down0": enc.down_blocks[0], "down1": enc.down_blocks[1],
        "down2": enc.down_blocks[2], "down3": enc.down_blocks[3],
        "mid_e": enc.mid_block, "conv_out_e": enc.conv_out,
        "conv_in_d": dec.conv_in, "mid_d": dec.mid_block,
        "up0": dec.up_blocks[0], "up1": dec.up_blocks[1],
        "up2": dec.up_blocks[2], "up3": dec.up_blocks[3],
        "image": dec.conv_out,
    }
    return m


def _slice6(x: torch.Tensor) -> list[float]:
    """[B,C,H,W] -> 36 floats: channel-mean, 6x6 adaptive-pool, min-max normalised."""
    a = x.detach().float().abs().mean(dim=1, keepdim=True)          # [B,1,H,W]
    a = F.adaptive_avg_pool2d(a, (6, 6)).flatten()                  # [36]
    lo, hi = a.min(), a.max()
    a = (a - lo) / (hi - lo + 1e-6)
    return [round(float(v), 4) for v in a.tolist()]


def main(argv=None) -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(prog="export_vae_activations")
    ap.add_argument("--image", required=True, help="a 1024x1024-ish RGB image (a dog, ideally)")
    ap.add_argument("--out", default="scratchpad/vae_act.json")
    ap.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    from PIL import Image
    device = infer_device(args.device)
    dtype = infer_dtype("float32", device)  # VAE force_upcast → fp32 anyway
    models = load_sdxl_models(model_id=args.model_id, device=device, dtype=dtype)
    vae = models["vae"]

    img = Image.open(args.image).convert("RGB").resize((1024, 1024))
    x = torch.from_numpy(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(1024, 1024, 3)).numpy()
    ).permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1.0).to(device=device, dtype=vae.dtype)

    acts: dict[str, list[float]] = {"input": _slice6(x)}
    handles = []
    mm = _module_map(vae)

    def mk(name):
        def hook(_m, _in, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            acts[name] = _slice6(t)
        return hook

    for name, mod in mm.items():
        handles.append(mod.register_forward_hook(mk(name)))

    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        z = posterior.sample()
        acts["latent"] = _slice6(z * float(getattr(vae.config, "scaling_factor", 0.13025)))
        _ = vae.decode(z).sample
    for h in handles:
        h.remove()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(acts, indent=0))
    log.info("wrote %d block slices to %s", len(acts), out)
    log.info("paste this JSON back to inline it as the artifact's VAE_ACT global")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
