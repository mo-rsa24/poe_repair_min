"""Shared aesthetic spec for the Phase 0 Veracity figure set.

A single styling module so all 4 main + appendix figures share fonts,
colors, gridline density, and line styles. Apply via
``apply_veracity_style()`` at the top of each figure function, before
any plotting.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# Color palette --------------------------------------------------------------

ACCENT = "#2C6FBB"            # Δ_t and the contested pair (foreground signal)
CONTROL_GRAY = "#888888"      # self-pair and disjoint controls (recessive)
SIGMA_T_REF = "#999999"       # σ_t and reference quantities (very recessive)
PEAK_BAND = "#2C6FBB22"       # faint accent tint for basin-commit window overlay
ANCHOR_FRAME = "#2C6FBB"      # frame for λ=0/λ=1 anchor panels in Fig 4

LAMBDA_CMAP = "viridis"       # ordinal sweeps (λ, CFG)
HEATMAP_EPS_CMAP = "magma"    # ‖Δ_t‖ ε-space heatmap
HEATMAP_X0_CMAP = "magma"     # x̂_0-space heatmap (same family for unity)

# Line styles for controls (referenced by figure functions) ------------------

CONTESTED_STYLE = dict(color=ACCENT, linestyle="-", linewidth=2.0, label="contested")
SELF_PAIR_STYLE = dict(
    color=CONTROL_GRAY, linestyle="--", linewidth=1.4, label="self-pair (control)",
)
DISJOINT_STYLE = dict(
    color=CONTROL_GRAY, linestyle=":", linewidth=1.4, label="disjoint (control)",
)
MONO_ANCHOR_STYLE = dict(
    color=ACCENT, linestyle="--", linewidth=1.6, label="along Mono trajectory",
)
POE_ANCHOR_STYLE = dict(
    color=ACCENT, linestyle="-", linewidth=1.8, label="along PoE trajectory",
)


# Layout -------------------------------------------------------------------

PANEL_INCHES = (4.0, 3.0)     # base unit; composites scale linearly
DPI = 200
GRID_ALPHA = 0.25
LEGEND_FONTSIZE = 8

# fp16 tolerance for the PMI identity check (Fig 1) --------------------------
FP16_PMI_TOLERANCE = 1e-3


def apply_veracity_style() -> None:
    """Apply consistent matplotlib rcParams across all veracity figures."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": DPI,
            "figure.dpi": 100,
            "axes.axisbelow": True,
        }
    )


def lambda_color(i: int, n: int):
    """Return a viridis color for the i-th of n ordinal series."""
    cmap = plt.get_cmap(LAMBDA_CMAP)
    return cmap(i / max(1, n - 1))
