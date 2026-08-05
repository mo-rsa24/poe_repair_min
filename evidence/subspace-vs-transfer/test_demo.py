"""Proof the checks in demo.py can fail.

The headline is a disagreement between two measurements, which is exactly the
kind of result that could be an artifact of the comparison rather than a fact
about the data. These tests make each moving part fail on purpose.

    python -m pytest evidence/subspace-vs-transfer/test_demo.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo import captured_fraction  # noqa: E402

RESULT = Path("docs/evidence/subspace-vs-transfer/result.json")


def test_captured_fraction_is_one_inside_the_subspace():
    """A vector built from the basis must be fully captured, or the measure lies."""
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(64, 200))[0][:8].T.T[:8]
    basis = torch.linalg.svd(torch.randn(8, 200), full_matrices=False)[2]
    inside = torch.randn(5, 8) @ basis
    got = captured_fraction(inside, basis, torch.zeros(1, 200))
    assert got == pytest.approx(1.0, abs=1e-4), got


def test_captured_fraction_is_zero_outside_the_subspace():
    """The complementary case. Without this, the measure could return 1 always."""
    V = torch.linalg.svd(torch.randn(16, 200), full_matrices=False)[2]
    basis, orthogonal = V[:8], V[8:16]
    outside = torch.randn(5, 8) @ orthogonal
    got = captured_fraction(outside, basis, torch.zeros(1, 200))
    assert got == pytest.approx(0.0, abs=1e-4), got


def test_captured_fraction_respects_the_mean():
    """Centring is load-bearing: the wrong mean changes the answer.

    The demo subtracts the TRAIN mean from held-out data on purpose. This test
    fails if someone 'simplifies' that to each set's own mean.
    """
    V = torch.linalg.svd(torch.randn(16, 200), full_matrices=False)[2]
    basis = V[:8]
    x = torch.randn(5, 8) @ V[8:16] + 50.0        # far from the origin
    with_zero_mean = captured_fraction(x, basis, torch.zeros(1, 200))
    with_own_mean = captured_fraction(x, basis, x.mean(0, keepdim=True))
    assert abs(with_zero_mean - with_own_mean) > 1e-3


@pytest.mark.skipif(not RESULT.exists(), reason="run demo.py first")
def test_the_disagreement_is_not_a_units_mistake():
    """Both numbers must be fractions in [0,1], or comparing them is meaningless."""
    r = json.loads(RESULT.read_text())
    for row in r["per_pair"]:
        assert 0.0 <= row["compose_rate"] <= 1.0, row
        assert 0.0 <= row["geometry_k64"] <= 1.0, row


@pytest.mark.skipif(not RESULT.exists(), reason="run demo.py first")
def test_train_and_heldout_pairs_are_disjoint():
    """If a pair were in both, the held-out reading would be inflated, not deflated.

    Worth asserting because a leak here would push the geometry number UP,
    which would weaken the reported disagreement rather than manufacture it.
    """
    r = json.loads(RESULT.read_text())
    assert not (set(r["train_pairs"]) & set(r["heldout_pairs"]))


@pytest.mark.skipif(not RESULT.exists(), reason="run demo.py first")
def test_geometry_would_be_high_if_the_subspace_did_contain_them():
    """The measure is capable of returning a high number on this data.

    The core risk in the finding is that captured_fraction always reads low for
    stacks of this shape, making the disagreement an artifact. It does not: on
    the training pairs the same measure reads 62.6% at k=64.
    """
    r = json.loads(RESULT.read_text())
    train_at_64 = r["geometry_at_k"]["64"]["train"]
    held_at_64 = r["geometry_at_k"]["64"]["heldout"]
    assert train_at_64 > 0.4, train_at_64
    assert held_at_64 < 0.2, held_at_64


@pytest.mark.skipif(not RESULT.exists(), reason="run demo.py first")
def test_the_headline_gap_is_large():
    """The claim itself, as an assertion that can go red if the data changes."""
    r = json.loads(RESULT.read_text())
    assert r["mean_compose_rate_transfer"] > 0.8
    assert r["mean_geometry_transfer"] < 0.2
