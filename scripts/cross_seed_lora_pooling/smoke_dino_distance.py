"""Plan-13 validation gate: does DINOv2 distance reorder the failure case?

Loads three terminal images on cat × dog seed 42:

    case1  : (λ=0.50, epoch 0900)  — single cat, no dog       (semantically bad)
    case2  : (λ=1.00, epoch 1600)  — both animals co-located  (semantically good)
    mono   : joint-CFG ceiling                                 (target)

Embeds each through DINOv2 ViT-S/14 (CLS, L2-normalised), prints the cosine
distance to mono, and reports PASS / FAIL.

Pass criterion (mandatory before building the semantic MDS pipeline):

    d_DINO(case1, mono)  >  d_DINO(case2, mono)

Run::

    python scripts/cross_seed_lora_pooling/smoke_dino_distance.py

Optionally override image paths with CLI flags. The default paths point at
the existing probe PNGs and the cached mono thumbnail.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from poe_repair import paths

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

log = logging.getLogger("smoke_dino_distance")


DEFAULT_RESULTS = paths.resolve(paths.ONE_PAIR_ONE_SEED) / "a_cat__x__a_dog/seed_42/run__local"
DEFAULT_CASE1 = DEFAULT_RESULTS / "probes/epoch_0900/lambda_0.50/decoded.png"
DEFAULT_CASE2 = DEFAULT_RESULTS / "probes/epoch_1600/lambda_1.00/decoded.png"
DEFAULT_MONO = DEFAULT_RESULTS / "mds_cache/static/mono.png"


def _load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def _dino_embed(images: list[Image.Image], *, device: str) -> np.ndarray:
    """Return ``(N, D)`` L2-normalised CLS embeddings under DINOv2 ViT-S/14."""
    import torch

    log.info("loading DINOv2 ViT-S/14 ...")
    # torch.hub gives us the standard preprocessing pipeline + 384-d CLS.
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True,
    ).to(device=device, dtype=torch.float32)
    model.eval()

    # Standard DINOv2 preprocessing: resize short side to 224, centre-crop 224,
    # ImageNet mean/std normalisation.
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tensors = torch.stack([preprocess(img) for img in images], dim=0).to(device)
    with torch.no_grad():
        feats = model(tensors)  # (N, 384) — DINOv2 returns CLS by default
    feats = feats.float().cpu().numpy()
    norms = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-8)
    return feats / norms


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="smoke_dino_distance")
    ap.add_argument("--case1", default=str(DEFAULT_CASE1),
                    help="path to the semantically BAD terminal image "
                         "(default: λ=0.50, epoch 0900)")
    ap.add_argument("--case2", default=str(DEFAULT_CASE2),
                    help="path to the semantically GOOD terminal image "
                         "(default: λ=1.00, epoch 1600)")
    ap.add_argument("--mono", default=str(DEFAULT_MONO),
                    help="path to the mono (joint-CFG) terminal image "
                         "(default: cached mds_cache/static/mono.png)")
    ap.add_argument("--device", default=None,
                    help="torch device. Default: cuda if available, else cpu.")
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_argparser().parse_args(argv)

    paths = {
        "case1 (bad: ep 0900, λ=0.50)": Path(args.case1),
        "case2 (good: ep 1600, λ=1.00)": Path(args.case2),
        "mono (ceiling)": Path(args.mono),
    }
    for name, p in paths.items():
        if not p.is_file():
            log.error("missing image for %s: %s", name, p)
            return 2
        log.info("loaded %s <- %s", name, p)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    images = [_load_image(p) for p in paths.values()]
    feats = _dino_embed(images, device=device)
    case1, case2, mono = feats[0], feats[1], feats[2]

    d_case1_mono = cosine_distance(case1, mono)
    d_case2_mono = cosine_distance(case2, mono)
    d_case1_case2 = cosine_distance(case1, case2)

    print()
    print(f"  d_DINO(case1, mono)  = {d_case1_mono:.4f}   (bad → mono)")
    print(f"  d_DINO(case2, mono)  = {d_case2_mono:.4f}   (good → mono)")
    print(f"  d_DINO(case1, case2) = {d_case1_case2:.4f}")
    print()
    print(f"  Δ (case1 − case2)    = {d_case1_mono - d_case2_mono:+.4f}")
    print()

    passed = d_case1_mono > d_case2_mono
    if passed:
        print(
            "  PASS — DINOv2 places the semantically GOOD sample closer to "
            "mono than the BAD one.\n"
            "  Plan-13 semantic MDS pipeline is worth building."
        )
        return 0
    print(
        "  FAIL — DINOv2 cosine on terminal images does not reorder the "
        "failure case.\n"
        "  Stop. Escalate to an object-detector co-occurrence score "
        "(OWLv2 / Grounding DINO) in a follow-up plan; demote MDS to "
        "'what a path looks like' with no convergence reading."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
