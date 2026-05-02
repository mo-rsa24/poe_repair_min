"""Text-only training dataset for the embedding synthesizer.

Each training example is a tuple of three (prompt_embeds, pooled) triples:
- ``A`` from ``"a {noun}"``
- ``B`` likewise
- ``J`` from ``f"{A} and {B}"``

The synthesizer learns ``f(e_A, e_B, e_empty) -> e_J``. No images.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch.utils.data import Dataset

from poe_repair.config import joint_prompt
from poe_repair.embeddings.holdout_pairs import COLLISION_PAIRS
from poe_repair.embeddings.noun_list_base import BASE_NOUNS, SUPERCATEGORY_NOUNS
from poe_repair.runtime import encode_prompt_sdxl


def _ensure_article(noun: str) -> str:
    n = noun.strip()
    if n.lower().startswith(("a ", "an ", "the ")):
        return n
    first = n[:1].lower()
    return f"{'an' if first in 'aeiou' else 'a'} {n}"


def _build_pair_pool(
    *,
    n_random: int,
    n_holdout_oversample: int,
    n_supercategory_oversample: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pool: list[tuple[str, str]] = []

    # Random pairs from the base noun list.
    for _ in range(n_random):
        a, b = rng.sample(BASE_NOUNS, 2)
        pool.append((_ensure_article(a), _ensure_article(b)))

    # Oversample the held-out collision pairs (cat+dog and similar).
    if COLLISION_PAIRS and n_holdout_oversample > 0:
        for _ in range(n_holdout_oversample):
            pool.append(rng.choice(COLLISION_PAIRS))

    # Oversample G6-style same-supercategory collisions.
    if SUPERCATEGORY_NOUNS and n_supercategory_oversample > 0:
        cat_keys = list(SUPERCATEGORY_NOUNS.keys())
        for _ in range(n_supercategory_oversample):
            cat = rng.choice(cat_keys)
            choices = SUPERCATEGORY_NOUNS[cat]
            if len(choices) < 2:
                continue
            a, b = rng.sample(choices, 2)
            pool.append((_ensure_article(a), _ensure_article(b)))

    rng.shuffle(pool)
    return pool


@dataclass
class TrainingExample:
    prompt_a: str
    prompt_b: str
    prompt_joint: str
    seq_a: torch.Tensor   # (77, 2048)
    pool_a: torch.Tensor  # (1280,)
    seq_b: torch.Tensor
    pool_b: torch.Tensor
    seq_e: torch.Tensor   # empty prompt
    pool_e: torch.Tensor
    seq_j: torch.Tensor
    pool_j: torch.Tensor


class TextPairDataset(Dataset):
    """In-memory cached text-encoder dataset (cache_in_memory=False by default)."""

    def __init__(
        self,
        *,
        models: dict,
        device: torch.device,
        dtype: torch.dtype,
        n_pairs: int,
        n_holdout_oversample: int,
        n_supercategory_oversample: int,
        joint_template: str,
        seed: int,
        cache_in_memory: bool = False,
    ) -> None:
        self.models = models
        self.device = device
        self.dtype = dtype
        self.joint_template = joint_template
        self.cache_in_memory = cache_in_memory

        self.pairs = _build_pair_pool(
            n_random=n_pairs,
            n_holdout_oversample=n_holdout_oversample,
            n_supercategory_oversample=n_supercategory_oversample,
            seed=seed,
        )
        with torch.no_grad():
            seq_e, pool_e = encode_prompt_sdxl("", models=models, device=device, dtype=dtype)
            self.seq_e = seq_e.squeeze(0).cpu()
            self.pool_e = pool_e.squeeze(0).cpu()
        self._cache: dict[int, TrainingExample] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    @torch.no_grad()
    def _encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        seq, pool = encode_prompt_sdxl(prompt, models=self.models, device=self.device, dtype=self.dtype)
        return seq.squeeze(0).cpu(), pool.squeeze(0).cpu()

    def __getitem__(self, idx: int) -> TrainingExample:
        if idx in self._cache:
            return self._cache[idx]
        a, b = self.pairs[idx]
        prompt_j = joint_prompt(a, b, self.joint_template)
        seq_a, pool_a = self._encode(a)
        seq_b, pool_b = self._encode(b)
        seq_j, pool_j = self._encode(prompt_j)
        ex = TrainingExample(
            prompt_a=a,
            prompt_b=b,
            prompt_joint=prompt_j,
            seq_a=seq_a, pool_a=pool_a,
            seq_b=seq_b, pool_b=pool_b,
            seq_e=self.seq_e, pool_e=self.pool_e,
            seq_j=seq_j, pool_j=pool_j,
        )
        if self.cache_in_memory:
            self._cache[idx] = ex
        return ex


def collate_examples(batch: Sequence[TrainingExample]) -> dict[str, torch.Tensor]:
    return {
        "seq_a": torch.stack([b.seq_a for b in batch]),
        "pool_a": torch.stack([b.pool_a for b in batch]),
        "seq_b": torch.stack([b.seq_b for b in batch]),
        "pool_b": torch.stack([b.pool_b for b in batch]),
        "seq_e": torch.stack([b.seq_e for b in batch]),
        "pool_e": torch.stack([b.pool_e for b in batch]),
        "seq_j": torch.stack([b.seq_j for b in batch]),
        "pool_j": torch.stack([b.pool_j for b in batch]),
    }


def iter_minibatches(
    dataset: TextPairDataset,
    *,
    batch_size: int,
    seed: int = 0,
    shuffle: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    while True:
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            batch = [dataset[i] for i in indices[start : start + batch_size]]
            yield collate_examples(batch)
