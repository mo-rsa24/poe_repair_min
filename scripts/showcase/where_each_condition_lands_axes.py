#!/usr/bin/env python
"""Task 2 of plan 06 (scope 05): label the cloud-axes figure's axes with decoded pictures.

The eigenfaces move from the X post (context/sources.md entry 4): walk along an axis of an
embedding space and decode each point through a representation autoencoder (RAE, entry 3), so
the reader sees what the direction means instead of reading a legend.

The catch this script is built around: the RAE decoder for DINOv2-B reads the 16 x 16 grid of
768-d patch tokens, not the single 384-d class token the endpoint figure's plane is built on. So
the walk is rebuilt in the decoder's own space: every render is encoded to its token grid, the
same two axes are constructed there (x: cat centroid to dog centroid; y: solo midpoint to joint
centroid, orthogonalised), and points along each axis are decoded. Whether that space orders the
48 renders the same way as the class-token plane is checked and written to the sidecar.

Steps:
  1. reconstruct one chimera render and one joint render, save beside the originals
     (does the decoder keep a fused animal, or clean it into one?)
  2. build the token-space axes on the 48 endpoint renders, compare both-ness ordering with the
     class-token sidecar (Spearman)
  3. decode 7 points along each axis (mean +- 2 sd of the 48 projections), save the strips
  4. paste the strips on the cloud-axes figure, every frame captioned "decoder reconstruction"

Encoder: torch hub dinov2_vitb14_reg (DINOv2-B/14 with 4 registers), last-layer patch tokens
with a non-affine layer norm, which is what RAE's Dinov2withNorm(normalize=True) computes.
Decoder: RAE ViT-XL decoder for dinov2/wReg_base (nyu-visionx/RAE-collections,
decoders/dinov2/wReg_base/ViTXL_n08/model.pt) with its ImageNet normalization stats.

Writes to artifacts/results/where-does-each-condition-land/axes/:
    reconstruction_check.png, axis-x-which-animal.png, axis-y-both-ness.png,
    cat-x-dog-in-dino-space-cloud-axes-with-axis-pictures.png, axes.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/showcase"))
from rae_vendor import GeneralDecoder, ViTMAEConfig  # noqa: E402

RESULTS = REPO_ROOT / "artifacts/results/where-does-each-condition-land"
OUT = RESULTS / "axes"
SIDECAR = RESULTS / "cat-x-dog-in-dino-space.json"
RAE_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/rae")
DECODER_PT = RAE_ROOT / "decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
STATS_PT = RAE_ROOT / "stats/dinov2/wReg_base/imagenet1k/stat.pt"
DECODER_CFG = REPO_ROOT / "scripts/showcase/rae_vendor/ViTXL_config.json"

SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
CLOUDS = ("solo_a", "solo_b", "joint")
ENC_SIZE = 224          # 224 / 14 = 16 patches a side -> 256 tokens
N_UNUSED = 5            # 1 class token + 4 registers
WALK_T = (-2.0, -4 / 3, -2 / 3, 0.0, 2 / 3, 4 / 3, 2.0)   # in sd of the 48 projections
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class RAEDecoderOnly:
    """Encoder from torch hub, decoder vendored from RAE, normalization from RAE's stats."""

    def __init__(self, device: torch.device):
        self.device = device
        self.enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", trust_repo=True)
        self.enc.eval().to(device)
        raw = json.loads(DECODER_CFG.read_text())
        raw["patch_size"] = 16          # the shipped config says "SHOULD BE RELOADED"
        cfg = ViTMAEConfig.from_dict(raw)
        cfg.hidden_size = 768
        cfg.patch_size = 16
        cfg.image_size = 16 * 16
        self.dec = GeneralDecoder(cfg, num_patches=256)
        sd = torch.load(DECODER_PT, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        res = self.dec.load_state_dict(sd, strict=False)
        print(f"[rae] decoder loaded: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
        if res.missing_keys:
            print("  missing:", res.missing_keys[:8])
        self.dec.eval().to(device)
        stats = torch.load(STATS_PT, map_location="cpu")
        # the shipped ImageNet stats carry var only (mean is None); RAE treats a missing mean as 0
        self.mean = stats["mean"].to(device) if stats.get("mean") is not None else torch.zeros(1, device=device)
        self.var = stats["var"].to(device)     # (768, 16, 16)
        self.eps = 1e-5

    @torch.no_grad()
    def encode(self, img01: torch.Tensor) -> torch.Tensor:
        """img01: (B, 3, H, W) in [0, 1] -> normalized latent (B, 768, 16, 16)."""
        x = F.interpolate(img01, size=(ENC_SIZE, ENC_SIZE), mode="bicubic", align_corners=False)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        feats = self.enc.forward_features(x.to(self.device))
        tokens = F.layer_norm(feats["x_prenorm"][:, N_UNUSED:], (768,))   # non-affine norm, as RAE
        b, n, c = tokens.shape
        z = tokens.transpose(1, 2).reshape(b, c, 16, 16)
        return (z - self.mean) / torch.sqrt(self.var + self.eps)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """normalized latent (B, 768, 16, 16) -> image (B, 3, 256, 256) in [0, 1]."""
        z = z * torch.sqrt(self.var + self.eps) + self.mean
        b, c, h, w = z.shape
        tokens = z.reshape(b, c, h * w).transpose(1, 2)
        out = self.dec(tokens, drop_cls_token=False).logits
        x = self.dec.unpatchify(out)
        x = x * IMAGENET_STD.to(x.device) + IMAGENET_MEAN.to(x.device)
        return x.clamp(0, 1)


def load01(p: Path) -> torch.Tensor:
    im = Image.open(p).convert("RGB")
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]


def to_pil(x01: torch.Tensor, size: int = 256) -> Image.Image:
    arr = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr).resize((size, size), Image.LANCZOS)


def labelled_row(images: list[Image.Image], labels: list[str], title: str, size: int = 256) -> Image.Image:
    pad, lab = 8, 22
    W = len(images) * (size + pad) + pad
    H = lab + size + lab + pad
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((pad, 4), title, fill="black")
    for i, (im, lb) in enumerate(zip(images, labels)):
        x = pad + i * (size + pad)
        canvas.paste(im.resize((size, size)), (x, lab))
        d.text((x, lab + size + 4), lb, fill="black")
    return canvas


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rae] device {device}")
    rae = RAEDecoderOnly(device)
    side = json.loads(SIDECAR.read_text())
    pts = side["points"]
    idx = {(r["condition"], r["seed"]): r for r in pts}

    # ---- 1. reconstruction check ----------------------------------------------
    checks = [("poe", 9, "PoE seed 9 (a cat with dog ears)"), ("poe", 10, "PoE seed 10 (one fused face)"),
              ("joint", 9, "joint prompt seed 9"), ("lora_1.2", 15, "PoE + 1.2 x correction seed 15")]
    ims, labels = [], []
    recon_err = {}
    for cond, seed, lb in checks:
        img = load01(Path(idx[(cond, seed)]["png"]))
        rec = rae.decode(rae.encode(img.to(device)))
        ref = F.interpolate(img.to(device), size=(256, 256), mode="bicubic", align_corners=False).clamp(0, 1)
        err = float((rec - ref).abs().mean().item() * 255)
        recon_err[f"{cond}_seed{seed}"] = round(err, 2)
        ims += [to_pil(ref), to_pil(rec)]
        labels += [lb, f"reconstruction, mean |diff| {err:.1f}/255"]
    labelled_row(ims, labels, "original | RAE reconstruction (DINOv2-B tokens -> ViT-XL decoder), 256 px", 224).save(
        OUT / "reconstruction_check.png")
    print("[rae] reconstruction mean abs error /255:", recon_err)

    # ---- 2. token-space axes on the 48 endpoint renders ----------------------
    Z = {}
    for r in pts:
        Z[(r["condition"], r["seed"])] = rae.encode(load01(Path(r["png"])).to(device))[0]  # (768,16,16)
    flat = {k: v.flatten() for k, v in Z.items()}
    cent = {c: torch.stack([flat[(c, s)] for s in SEEDS]).mean(0) for c in CLOUDS}
    origin = 0.5 * (cent["solo_a"] + cent["solo_b"])
    u1 = cent["solo_b"] - cent["solo_a"]; u1 = u1 / u1.norm()
    u2 = cent["joint"] - origin; u2 = u2 - (u2 @ u1) * u1; u2 = u2 / u2.norm()
    proj = {k: (float((v - origin) @ u1), float((v - origin) @ u2)) for k, v in flat.items()}
    both_tok = {c: [proj[(c, s)][1] for s in SEEDS] for c in ("solo_a", "solo_b", "joint", "poe", "lora_1.0", "lora_1.2")}
    both_cls = side["cloud_axes"]["both_ness_by_condition"]
    # Spearman over all 48 between token-space and class-token both-ness
    conds = ["solo_a", "solo_b", "joint", "poe", "lora_1.0", "lora_1.2"]
    a = np.array([both_tok[c][i] for c in conds for i in range(8)])
    b = np.array([both_cls[c][i] for c in conds for i in range(8)])
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    spearman = float(np.corrcoef(ra, rb)[0, 1])
    print(f"[rae] both-ness means, token space: "
          f"{ {c: round(float(np.mean(v)), 3) for c, v in both_tok.items()} }; spearman vs class-token plane {spearman:.3f}")

    # ---- 3. decode walks along each axis --------------------------------------
    x_all = np.array([proj[k][0] for k in proj]); y_all = np.array([proj[k][1] for k in proj])
    sx, sy = float(x_all.std()), float(y_all.std())
    mx, my = float(x_all.mean()), float(y_all.mean())
    strips = {}
    for name, u, s0, m0, other_u, other_m in (
        ("axis-x-which-animal", u1, sx, mx, u2, my),
        ("axis-y-both-ness", u2, sy, my, u1, mx),
    ):
        ims, labels = [], []
        for t in WALK_T:
            z = origin + (m0 + t * s0) * u + other_m * other_u
            rec = rae.decode(z.view(1, 768, 16, 16))
            ims.append(to_pil(rec, 192))
            labels.append(f"{t:+.2f} sd")
        title = ("x: cat centroid (left) to dog centroid (right), decoder reconstructions"
                 if name.startswith("axis-x") else
                 "y: solo midpoint (low) toward the joint-prompt centroid (high), decoder reconstructions")
        strip = labelled_row(ims, labels, title, 192)
        strip.save(OUT / f"{name}.png")
        strips[name] = strip

    # ---- 4. paste strips on the cloud-axes figure -----------------------------
    base = Image.open(RESULTS / "cat-x-dog-in-dino-space-cloud-axes.png").convert("RGB")
    xs = strips["axis-x-which-animal"]; ys = strips["axis-y-both-ness"].rotate(90, expand=True)
    xs = xs.resize((base.width, int(xs.height * base.width / xs.width)))
    ys = ys.resize((int(ys.width * base.height / ys.height), base.height))
    W = ys.width + base.width; H = base.height + xs.height + 30
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(ys, (0, 0)); canvas.paste(base, (ys.width, 0)); canvas.paste(xs, (ys.width, base.height))
    ImageDraw.Draw(canvas).text((ys.width + 8, base.height + xs.height + 8),
                                "Axis pictures are RAE decoder reconstructions of points along each axis in DINOv2-B token space, "
                                "not SDXL outputs; the plane itself is the class-token plane.", fill="black")
    canvas.save(OUT / "cat-x-dog-in-dino-space-cloud-axes-with-axis-pictures.png")

    (OUT / "axes.json").write_text(json.dumps({
        "encoder": "torch hub dinov2_vitb14_reg, last-layer patch tokens, non-affine layer norm (RAE Dinov2withNorm, normalize=True)",
        "decoder": str(DECODER_PT), "stats": str(STATS_PT), "encoder_input_px": ENC_SIZE, "decoder_output_px": 256,
        "reconstruction_mean_abs_error_over_255": recon_err,
        "token_space_axes": {
            "x": "unit vector from cat-alone centroid to dog-alone centroid, in flattened 768x16x16 token space",
            "y": "unit vector from the solo midpoint to the joint centroid, orthogonalised against x",
            "walk_sd_multiples": list(WALK_T), "x_mean_sd": [mx, sx], "y_mean_sd": [my, sy],
        },
        "both_ness_by_condition_token_space": {c: [round(v, 4) for v in vs] for c, vs in both_tok.items()},
        "both_ness_means_token_space": {c: round(float(np.mean(v)), 4) for c, v in both_tok.items()},
        "spearman_token_vs_class_token_both_ness_over_48": round(spearman, 4),
    }, indent=2))
    print(f"[rae] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
