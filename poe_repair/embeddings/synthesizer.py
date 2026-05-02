"""Embedding synthesizer architectures.

Three variants behind a common interface:

- ``LinearSynthesizer`` — token-wise linear regression on the concatenation
  of (e_A_tok, e_B_tok, e_empty_tok). Capacity floor; useful for diagnostics.
- ``ResidualMLPSynthesizer`` — token-wise MLP that predicts a *residual*
  added to the naive ``e_A + e_B - e_empty`` baseline. The default arch.
- ``GatedAttentionSynthesizer`` — a 1-layer multi-head cross-attention from
  the empty embedding (queries) into ``[e_A, e_B]`` (keys/values), with a
  gated residual.

All variants predict both the sequence (77, 2048) and pooled (1280) outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _concat_token_features(
    seq_a: torch.Tensor,
    seq_b: torch.Tensor,
    seq_e: torch.Tensor,
) -> torch.Tensor:
    """Concatenate three (B, T, D) sequences along the feature axis."""
    return torch.cat([seq_a, seq_b, seq_e], dim=-1)


def _pool_features(
    pool_a: torch.Tensor,
    pool_b: torch.Tensor,
    pool_e: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([pool_a, pool_b, pool_e], dim=-1)


@dataclass
class SynthesizerOutputs:
    seq: torch.Tensor   # (B, 77, 2048)
    pooled: torch.Tensor  # (B, 1280)


class LinearSynthesizer(nn.Module):
    """Token-wise linear regression."""

    def __init__(self, *, seq_dim: int = 2048, pooled_dim: int = 1280) -> None:
        super().__init__()
        self.seq_proj = nn.Linear(3 * seq_dim, seq_dim)
        self.pool_proj = nn.Linear(3 * pooled_dim, pooled_dim)

    def forward(
        self,
        *,
        seq_a: torch.Tensor,
        seq_b: torch.Tensor,
        seq_e: torch.Tensor,
        pool_a: torch.Tensor,
        pool_b: torch.Tensor,
        pool_e: torch.Tensor,
    ) -> SynthesizerOutputs:
        seq_x = _concat_token_features(seq_a, seq_b, seq_e)
        pool_x = _pool_features(pool_a, pool_b, pool_e)
        seq_y = self.seq_proj(seq_x)
        pool_y = self.pool_proj(pool_x)
        return SynthesizerOutputs(seq=seq_y, pooled=pool_y)


class ResidualMLPSynthesizer(nn.Module):
    """Predicts a residual added to the naive ``e_A + e_B - e_empty`` baseline.

    Token-wise MLP for the sequence; separate MLP for the pooled vector.
    """

    def __init__(
        self,
        *,
        seq_dim: int = 2048,
        pooled_dim: int = 1280,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_dim = seq_dim
        self.pooled_dim = pooled_dim
        self.seq_mlp = self._make_mlp(3 * seq_dim, seq_dim, hidden_dim, num_layers, dropout)
        self.pool_mlp = self._make_mlp(3 * pooled_dim, pooled_dim, hidden_dim, num_layers, dropout)

    @staticmethod
    def _make_mlp(in_dim: int, out_dim: int, hidden: int, depth: int, dropout: float) -> nn.Module:
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(max(depth - 1, 1)):
            layers += [nn.Linear(prev, hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(
        self,
        *,
        seq_a: torch.Tensor,
        seq_b: torch.Tensor,
        seq_e: torch.Tensor,
        pool_a: torch.Tensor,
        pool_b: torch.Tensor,
        pool_e: torch.Tensor,
    ) -> SynthesizerOutputs:
        seq_baseline = seq_a + seq_b - seq_e
        pool_baseline = pool_a + pool_b - pool_e
        seq_x = _concat_token_features(seq_a, seq_b, seq_e)
        pool_x = _pool_features(pool_a, pool_b, pool_e)
        seq_residual = self.seq_mlp(seq_x)
        pool_residual = self.pool_mlp(pool_x)
        return SynthesizerOutputs(seq=seq_baseline + seq_residual, pooled=pool_baseline + pool_residual)


class GatedAttentionSynthesizer(nn.Module):
    """Single-block cross-attention from e_empty into [e_A, e_B], with a gated residual."""

    def __init__(
        self,
        *,
        seq_dim: int = 2048,
        pooled_dim: int = 1280,
        n_heads: int = 8,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(seq_dim, num_heads=n_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(seq_dim)
        self.norm_kv = nn.LayerNorm(seq_dim)
        self.gate = nn.Linear(seq_dim, 1)
        self.ff = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_dim),
        )
        self.pool_mlp = nn.Sequential(
            nn.Linear(3 * pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pooled_dim),
        )

    def forward(
        self,
        *,
        seq_a: torch.Tensor,
        seq_b: torch.Tensor,
        seq_e: torch.Tensor,
        pool_a: torch.Tensor,
        pool_b: torch.Tensor,
        pool_e: torch.Tensor,
    ) -> SynthesizerOutputs:
        seq_baseline = seq_a + seq_b - seq_e
        kv = torch.cat([self.norm_kv(seq_a), self.norm_kv(seq_b)], dim=1)  # (B, 2T, D)
        q = self.norm_q(seq_e)
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        gate = torch.sigmoid(self.gate(attended))
        attn_residual = self.ff(attended) * gate
        seq_y = seq_baseline + attn_residual
        pool_x = _pool_features(pool_a, pool_b, pool_e)
        pool_baseline = pool_a + pool_b - pool_e
        pool_y = pool_baseline + self.pool_mlp(pool_x)
        return SynthesizerOutputs(seq=seq_y, pooled=pool_y)


SYNTH_REGISTRY = {
    "linear": LinearSynthesizer,
    "residual_mlp": ResidualMLPSynthesizer,
    "gated_attn": GatedAttentionSynthesizer,
}


def build_synthesizer(arch: str, **kwargs) -> nn.Module:
    if arch not in SYNTH_REGISTRY:
        raise ValueError(f"Unknown synthesizer arch '{arch}'. Available: {list(SYNTH_REGISTRY)}")
    cls = SYNTH_REGISTRY[arch]
    if cls is LinearSynthesizer:
        return cls(seq_dim=kwargs.get("seq_dim", 2048), pooled_dim=kwargs.get("pooled_dim", 1280))
    if cls is ResidualMLPSynthesizer:
        return cls(
            seq_dim=kwargs.get("seq_dim", 2048),
            pooled_dim=kwargs.get("pooled_dim", 1280),
            hidden_dim=kwargs.get("hidden_dim", 1024),
            num_layers=kwargs.get("num_layers", 3),
            dropout=kwargs.get("dropout", 0.0),
        )
    if cls is GatedAttentionSynthesizer:
        return cls(
            seq_dim=kwargs.get("seq_dim", 2048),
            pooled_dim=kwargs.get("pooled_dim", 1280),
            n_heads=kwargs.get("n_heads", 8),
            hidden_dim=kwargs.get("hidden_dim", 1024),
            dropout=kwargs.get("dropout", 0.0),
        )
    raise RuntimeError(f"Unhandled arch: {arch}")
