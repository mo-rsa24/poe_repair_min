"""The notebook's 2D toy datasets (cells 14 to 20), in NumPy.

Each sampler takes n and returns an (n, 2) float array. The notebook draws from NumPy's global
generator; these take an optional Generator so runs can be seeded.
"""
import numpy as np


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


def ring_means(n_comp=8, radius=0.5):
    ang = 2 * np.pi * np.linspace(0, (n_comp - 1) / n_comp, n_comp)
    return radius * np.c_[np.cos(ang), np.sin(ang)]


def _gmm(means, std, n, rng):
    comp = rng.integers(0, len(means), n)   # equal weights, as the notebook's multinomial draw
    return means[comp] + std * rng.standard_normal((n, 2))


def toy_gmm(n, std=0.075, radius=0.5, rng=None):
    """Ring of 8 Gaussians."""
    return _gmm(ring_means(8, radius), std, n, _rng(rng))


def toy_gmm_left(n, std=0.075, radius=0.5, rng=None):
    """Ring modes 0 to 3 (the upper-right half of the ring)."""
    return _gmm(ring_means(8, radius)[[0, 1, 2, 3]], std, n, _rng(rng))


def toy_gmm_right(n, std=0.075, radius=0.5, rng=None):
    """Ring modes 4 to 7 (the lower-left half of the ring)."""
    return _gmm(ring_means(8, radius)[[4, 5, 6, 7]], std, n, _rng(rng))


def toy_gauss(n, radius=0.5, rng=None):
    return _rng(rng).standard_normal((n, 2)) * radius


def toy_box(n, scale=1.0, rng=None):
    return _rng(rng).uniform(-scale, scale, (n, 2))


def bar(n, scale=0.2, rng=None):
    """Vertical bar: x1 in [-scale, scale], x2 in [-1, 1]."""
    d = _rng(rng).uniform(-1, 1, (n, 2)); d[:, 0] *= scale; return d


def bar_horizontal(n, scale=0.2, rng=None):
    d = _rng(rng).uniform(-1, 1, (n, 2)); d[:, 1] *= scale; return d


def right_bar(n, scale=0.1, rng=None):
    d = _rng(rng).uniform(-1, 1, (n, 2)); d[:, 0] = d[:, 0] * scale + 0.2; return d


def left_bar(n, scale=0.1, rng=None):
    d = _rng(rng).uniform(-1, 1, (n, 2)); d[:, 0] = d[:, 0] * scale - 0.2; return d


# the ten panels of datasets-1000-samples-each.png, in its order, with the notebook's arguments
PANELS = {
    "toy_gmm(std=.03)": lambda n, rng=None: toy_gmm(n, std=0.03, rng=rng),
    "toy_gauss()": toy_gauss,
    "toy_box(0.7)": lambda n, rng=None: toy_box(n, 0.7, rng=rng),
    "toy_box(0.3)": lambda n, rng=None: toy_box(n, 0.3, rng=rng),
    "bar(scale=0.2)": bar,
    "bar_horizontal()": bar_horizontal,
    "toy_gmm_left()": toy_gmm_left,
    "toy_gmm_right()": toy_gmm_right,
    "right_bar()": right_bar,
    "left_bar()": left_bar,
}
