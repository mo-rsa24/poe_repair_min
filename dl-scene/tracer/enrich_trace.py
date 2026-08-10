"""Merge the shape trace with real source snippets into dl-scene/trace.json."""
import inspect, json, sys
from pathlib import Path

import torch.nn as nn
import diffusers
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import Encoder, Decoder, DiagonalGaussianDistribution
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D, UpDecoderBlock2D, UNetMidBlock2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D

TRACE = sys.argv[1]
OUT = sys.argv[2]
REPO = Path("/home-mscluster/mmolefe/Playground/PhD/poe_repair_min")

def snippet(obj, max_lines=80):
    try:
        src, start = inspect.getsourcelines(obj)
        f = inspect.getsourcefile(obj)
        src = src[:max_lines]
        return {"file": f, "start": start, "lines": [l.rstrip("\n") for l in src]}
    except Exception as e:
        return None

CLASSES = {
    "AutoencoderKL": AutoencoderKL.forward,
    "AutoencoderKL.encode": AutoencoderKL.encode,
    "AutoencoderKL.decode": AutoencoderKL.decode,
    "Encoder": Encoder.forward,
    "Decoder": Decoder.forward,
    "DiagonalGaussianDistribution": DiagonalGaussianDistribution,
    "DownEncoderBlock2D": DownEncoderBlock2D.forward,
    "UpDecoderBlock2D": UpDecoderBlock2D.forward,
    "UNetMidBlock2D": UNetMidBlock2D.forward,
    "ResnetBlock2D": ResnetBlock2D.forward,
    "Attention": Attention.forward,
    "AttnProcessor2_0": AttnProcessor2_0.__call__,
    "Downsample2D": Downsample2D.forward,
    "Upsample2D": Upsample2D.forward,
    "Conv2d": nn.Conv2d.forward,
    "GroupNorm": nn.GroupNorm.forward,
    "SiLU": nn.SiLU.forward,
    "Linear": nn.Linear.forward,
    "Dropout": nn.Dropout.forward,
}
class_src = {k: snippet(v) for k, v in CLASSES.items()}

def repo_snippet(rel, a, b, label):
    lines = (REPO / rel).read_text().splitlines()
    return {"file": rel, "start": a, "label": label, "lines": lines[a - 1 : b]}

call_sites = [
    repo_snippet("poe_repair/_sdxl/runtime.py", 110, 132, "load: fp16 ask, fp32 forced (force_upcast)"),
    repo_snippet("poe_repair/_sdxl/runtime.py", 214, 217, "decode_latents: every sampled image exits here"),
    repo_snippet("poe_repair/_sdxl/sdipc_utils.py", 133, 145, "decode_latents_to_tensor: latents / scaling_factor, then vae.decode"),
    repo_snippet("poe_repair/methods/_sampling.py", 180, 188, "sampling loop tail: latents to image"),
    repo_snippet("poe_repair/experiments/mechanism_study/export_vae_activations.py", 80, 96, "activation taps: fills this app's real-data slot"),
]

with open(TRACE) as f:
    trace = json.load(f)

trace["class_src"] = class_src
trace["call_sites"] = call_sites
trace["versions"] = {"diffusers": diffusers.__version__}
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(trace, f)
print("wrote", OUT, "classes with src:", sum(1 for v in class_src.values() if v), "/", len(class_src))
