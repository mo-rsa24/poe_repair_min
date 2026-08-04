"""The 3-anchor compose/blend scorer (compose-scorer plan 02).

Given an output image and a pair's three reference anchors (A-alone, B-alone,
joint "A and B"), decide COMPOSE vs BLEND by a RELATIVE distance read:

    d_a    = dist(output, A_alone)
    d_b    = dist(output, B_alone)
    d_join = dist(output, joint)

    COMPOSE  iff  d_join < min(d_a, d_b) - margin          (nearest the joint)
    BLEND    otherwise (nearest one single-animal anchor)

The joint acts as the positive-anchor guard: a real compose is not merely far
from both monos, it actually lands near the joint. `margin` is the relative
threshold, determined empirically from the validation pairs (plan 02 task 3).

Two INDEPENDENT image-embedding spaces are used:
  - "dino"  : DINOv2 ViT-S/14 CLS (structure/instance-aware)
  - "clip"  : CLIP ViT-B/32 image features (semantic)

Why NOT the trajectory "MDS/latent space" the master plan first named: that space
is a per-step 2D basis over 65536-dim epsilon/latent residual tensors and cannot
embed a rendered PNG; a "semantic MDS" would just be a re-projection of DINOv2 and
so not independent. Excluded by design (pressure-tested). See scorer_validated.json.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

# Both spaces produce L2-normalised features, so cosine distance = 1 - dot.
SPACES = ("dino", "clip")


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return 1.0 - float(np.dot(a, b))


# --------------------------------------------------------------------------- #
# Embedders (lazy singletons so we load each backbone at most once).
# --------------------------------------------------------------------------- #
class _Embedders:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self._dino = None

    def _load_png_batch(self, paths: Sequence[Path]) -> torch.Tensor:
        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)  # (H,W,3)
            imgs.append(t.permute(2, 0, 1))
        return torch.stack(imgs, dim=0)  # (N,3,H,W) in [0,1]

    def dino(self, paths: Sequence[Path]) -> np.ndarray:
        from scripts.build_lora_inspector_mds_semantic import DinoEmbedder

        if self._dino is None:
            self._dino = DinoEmbedder(device=self.device)
        batch = self._load_png_batch(paths)
        return self._dino.embed_decoded_batch(batch)  # (N,384) L2-normed np

    def clip(self, paths: Sequence[Path]) -> np.ndarray:
        from poe_repair.experiments.residual_diagnostics.metrics import clip_image_embed

        feats = clip_image_embed(list(paths), device=self.device)  # (N,D) L2 torch
        return feats.numpy()


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #
@dataclass
class SpaceScore:
    space: str
    d_a: float
    d_b: float
    d_joint: float
    nearest: str          # "a_alone" | "b_alone" | "joint"
    label: str            # "compose" | "blend"
    compose_margin: float  # min(d_a,d_b) - d_joint  (positive ⇒ nearer joint)


@dataclass
class AnchorEmb:
    a_alone: np.ndarray
    b_alone: np.ndarray
    joint: np.ndarray


def _label_from_distances(d_a: float, d_b: float, d_joint: float, margin: float) -> tuple[str, str, float]:
    d_min_single = min(d_a, d_b)
    compose_margin = d_min_single - d_joint  # >0 ⇒ joint is nearer than both singles
    if compose_margin > margin:
        return "joint", "compose", compose_margin
    nearest = "a_alone" if d_a <= d_b else "b_alone"
    return nearest, "blend", compose_margin


def score_output(
    output_path: Path,
    anchor_paths: dict[str, Path],
    *,
    embedders: _Embedders,
    margin: float = 0.0,
    spaces: Sequence[str] = SPACES,
) -> dict[str, SpaceScore]:
    """Score one output against a pair's 3 anchors, in each space.

    anchor_paths: {"a_alone": ..., "b_alone": ..., "joint": ...}
    Returns {space: SpaceScore}.
    """
    out: dict[str, SpaceScore] = {}
    for space in spaces:
        embed = getattr(embedders, space)
        # Embed [output, a_alone, b_alone, joint] together for consistency.
        paths = [output_path, anchor_paths["a_alone"], anchor_paths["b_alone"], anchor_paths["joint"]]
        E = embed(paths)  # (4, D)
        e_out, e_a, e_b, e_j = E[0], E[1], E[2], E[3]
        d_a = _cosine_distance(e_out, e_a)
        d_b = _cosine_distance(e_out, e_b)
        d_j = _cosine_distance(e_out, e_j)
        nearest, label, cm = _label_from_distances(d_a, d_b, d_j, margin)
        out[space] = SpaceScore(
            space=space, d_a=d_a, d_b=d_b, d_joint=d_j,
            nearest=nearest, label=label, compose_margin=cm,
        )
    return out


def space_score_to_dict(s: SpaceScore) -> dict:
    return asdict(s)
