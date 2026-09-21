"""Contrastive data for the twist head, built from the training cache.

Positives are the joint prompt's finished latents (``mono.png`` of each cache
cell, VAE-encoded once and banked). Negatives are the PoE finished latents
(``poe.png``). Both are re-noised to the same random timestep with the forward
kernel ``x_t = sqrt(a_t) x0 + sqrt(1 - a_t) eps``, which is CDM's closed-form
re-noising trick and what keeps the timestep from being a giveaway: the two
classes share one timestep distribution by construction.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class BankCell:
    cell_dir: Path
    pair_slug: str
    seed: int
    split: str


def discover_bank_cells(
    cache_root: Path,
    split: str,
    *,
    pair_filter: list[str] | None = None,
    max_pairs: int | None = None,
    max_seeds_per_pair: int | None = None,
    exclude_pairs: set[str] | None = None,
) -> list[BankCell]:
    split_dir = Path(cache_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"split dir not found: {split_dir}")
    cells: list[BankCell] = []
    pair_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if pair_filter:
        keep = set(pair_filter)
        pair_dirs = [p for p in pair_dirs if p.name in keep]
    if exclude_pairs:
        # the heldout split repeats some training pairs; a validation bank must not see them
        pair_dirs = [p for p in pair_dirs if p.name not in exclude_pairs]
    if max_pairs is not None:
        pair_dirs = pair_dirs[:max_pairs]
    for pair_dir in pair_dirs:
        seed_dirs = sorted(
            (s for s in pair_dir.iterdir() if s.is_dir() and s.name.startswith("seed_")),
            key=lambda s: int(s.name.split("_", 1)[1]),
        )
        n = 0
        for sd in seed_dirs:
            need = ("mono.png", "poe.png", "embeddings.pt", "meta.json")
            if not all((sd / f).exists() for f in need):
                continue
            cells.append(BankCell(sd, pair_dir.name, int(sd.name.split("_", 1)[1]), split))
            n += 1
            if max_seeds_per_pair is not None and n >= max_seeds_per_pair:
                break
    if not cells:
        raise RuntimeError(f"no complete cells under {split_dir}")
    return cells


def _bank_key(cells: list[BankCell]) -> str:
    h = hashlib.md5("\n".join(f"{c.split}/{c.pair_slug}/{c.seed}" for c in cells).encode()).hexdigest()
    return h[:12]


@torch.no_grad()
def _encode_png(vae, path: Path, device: torch.device) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0
    x = (arr * 2.0 - 1.0)[None].to(device=device, dtype=vae.dtype)
    posterior = vae.encode(x).latent_dist
    z = posterior.mean
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    z = (z - shift) * vae.config.scaling_factor
    return z.to(torch.float16).cpu()


@torch.no_grad()
def build_or_load_bank(
    cells: list[BankCell],
    *,
    bank_dir: Path,
    vae=None,
    device: torch.device | None = None,
) -> dict:
    """Return ``{mono (N,4,h,w) fp16, poe (N,4,h,w) fp16, pool_j (N,1280) fp32,
    pair_slug list, seed list}``. Built with the VAE once per cell set and cached."""
    bank_dir = Path(bank_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)
    key = _bank_key(cells)
    path = bank_dir / f"bank_{cells[0].split}_{key}.pt"
    if path.exists():
        log.info("loading latent bank %s", path)
        return torch.load(path, map_location="cpu", weights_only=False)
    if vae is None or device is None:
        raise RuntimeError(f"bank {path} missing and no VAE given to build it")
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    monos, poes, pools, slugs, seeds = [], [], [], [], []
    for i, c in enumerate(cells):
        emb = torch.load(c.cell_dir / "embeddings.pt", map_location="cpu", weights_only=False)
        monos.append(_encode_png(vae, c.cell_dir / "mono.png", device))
        poes.append(_encode_png(vae, c.cell_dir / "poe.png", device))
        pools.append(emb["pool_j"].reshape(1, -1).float())
        slugs.append(c.pair_slug)
        seeds.append(c.seed)
        if (i + 1) % 10 == 0:
            log.info("bank: encoded %d/%d cells", i + 1, len(cells))
    bank = {
        "mono": torch.cat(monos), "poe": torch.cat(poes), "pool_j": torch.cat(pools),
        "pair_slug": slugs, "seed": seeds, "key": key,
    }
    torch.save(bank, path)
    (bank_dir / f"bank_{cells[0].split}_{key}.json").write_text(json.dumps(
        {"cells": [f"{c.split}/{c.pair_slug}/seed_{c.seed}" for c in cells]}, indent=2))
    log.info("wrote latent bank %s (%d cells)", path, len(cells))
    return bank


class ContrastiveSampler:
    """Yields balanced batches of re-noised positives (label 1) and negatives (label 0)."""

    def __init__(
        self,
        bank: dict,
        *,
        alphas_cumprod: torch.Tensor,
        batch_size: int,
        device: torch.device,
        seed: int = 0,
        t_min: int = 1,
        t_max: int = 999,
        hflip: bool = True,
    ):
        self.mono = bank["mono"]
        self.poe = bank["poe"]
        self.pool_j = bank["pool_j"]
        self.n = self.mono.shape[0]
        self.alphas_cumprod = alphas_cumprod.float().cpu()
        self.batch_size = batch_size
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.t_min, self.t_max = t_min, t_max
        self.hflip = hflip

    def _noise_to(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ab = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        eps = torch.randn(x0.shape, generator=self.gen)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps

    def next_batch(self) -> dict:
        B = self.batch_size
        pos_idx = torch.randint(0, self.n, (B,), generator=self.gen)
        neg_idx = torch.randint(0, self.n, (B,), generator=self.gen)
        t = torch.randint(self.t_min, self.t_max + 1, (2 * B,), generator=self.gen)
        x0 = torch.cat([self.mono[pos_idx], self.poe[neg_idx]]).float()
        if self.hflip:
            flip = torch.rand(2 * B, generator=self.gen) < 0.5
            x0[flip] = x0[flip].flip(-1)
        x_t = self._noise_to(x0, t)
        cond = torch.cat([self.pool_j[pos_idx], self.pool_j[neg_idx]])
        y = torch.cat([torch.ones(B), torch.zeros(B)])
        return {
            "x_t": x_t.to(self.device), "timestep": t.to(self.device),
            "cond": cond.to(self.device), "y": y.to(self.device),
        }
