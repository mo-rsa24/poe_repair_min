"""Figures for the sampler comparison. Labels sit on the plot; no legends."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "#e4e3df"
SIGMA = 0.15
CENTRES = {"TR": (0.5, 0.5), "BR": (0.5, -0.5), "TL": (-0.5, 0.5), "BL": (-0.5, -0.5)}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _stats(rows, name, key):
    v = [r[key] for r in rows if r["sampler"] == name]
    return np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0.0), v


def _blob_rings(ax):
    for q, c in CENTRES.items():
        ax.add_patch(plt.Circle(c, 3 * SIGMA, fill=False, ls="--", lw=0.8, color=INK2, alpha=0.6))
    ax.add_patch(plt.Circle((0, 0), 3 * SIGMA / math.sqrt(2), fill=False, ls=":", lw=0.8, color=INK2,
                            alpha=0.6))


def draw(out, rows, grid_mass, edges, gs, order):
    names = [n for n in order if n in ("exact samples", "reverse diffusion", "annealed ULA",
                                        "annealed MALA", "annealed UHA")]
    titles = {"exact samples": "exact samples (rejection)", "reverse diffusion": "reverse diffusion, summed score",
              "annealed ULA": "annealed ULA", "annealed MALA": "annealed MALA (fixed)",
              "annealed UHA": "annealed UHA"}

    # Figure 1: exact target beside each sampler's samples (seed 0), top-right share on each panel.
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.2))
    ax = axes[0, 0]
    ax.imshow(grid_mass.T, origin="lower", extent=[edges[0], edges[-1], edges[0], edges[-1]],
              cmap="Blues", interpolation="nearest")
    _blob_rings(ax)
    ax.set_title("exact composed target, p(x,A) p(x,B) / p(x)", color=INK, fontsize=10)
    ax.text(0.03, 0.04, f"top-right share {gs['in_TR_blob']:.3f}\norigin share {gs['at_origin']:.3f}\n"
                        f"centre (0.500, 0.500)",
            transform=ax.transAxes, color=INK, fontsize=9, va="bottom")
    ax.text(0.5, 0.5 + 3 * SIGMA + 0.05, "3σ ring", color=INK2, fontsize=8, ha="center")
    ax.text(0, -3 * SIGMA / math.sqrt(2) - 0.1, "origin disc", color=INK2, fontsize=8, ha="center")
    for ax, name in zip(axes.ravel()[1:], names):
        xs = np.load(out / f"samples_{name.replace(' ', '_')}_seed0.npy")
        ax.scatter(xs[:, 0], xs[:, 1], s=3, alpha=0.35, color=BLUE if name != "exact samples" else INK2,
                   linewidths=0)
        _blob_rings(ax)
        m, sd, v = _stats(rows, name, "in_TR_blob")
        mo, sdo, _ = _stats(rows, name, "at_origin")
        mt, sdt, _ = _stats(rows, name, "tv")
        r0 = [r for r in rows if r["sampler"] == name and r["seed"] == 0][0]
        ax.set_title(titles[name], color=INK, fontsize=10)
        ax.text(0.03, 0.04, f"top-right share {r0['in_TR_blob']:.3f} (seed 0)\n"
                            f"3 seeds: {m:.3f} ± {sd:.3f}\n"
                            f"origin share {mo:.3f} ± {sdo:.3f}\nTV {mt:.3f} ± {sdt:.3f}\n"
                            f"centre ({_stats(rows, name, 'mean_x1')[0]:.3f}, {_stats(rows, name, 'mean_x2')[0]:.3f})",
                transform=ax.transAxes, color=INK, fontsize=9, va="bottom",
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8))
    for ax in axes.ravel():
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x1  (A is x1 > 0)")
        ax.set_ylabel("x2  (B is x2 > 0)")
    fig.suptitle("Composed samples at the clean level, 2,000 per panel, seed 0", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "target-beside-samplers.png", dpi=150)
    plt.close(fig)

    # Figure 2: total variation per sampler, floor as a line.
    bars = [n for n in names if n != "exact samples"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    floor_m, floor_sd, _ = _stats(rows, "exact samples", "tv")
    for i, n in enumerate(bars):
        m, sd, v = _stats(rows, n, "tv")
        ax.bar(i, m, width=0.6, color=BLUE, zorder=2)
        ax.errorbar(i, m, yerr=sd, color=INK, capsize=4, lw=1, zorder=3)
        ax.scatter([i] * len(v), v, s=12, color=INK, zorder=4)
        ax.text(i, m + sd + 0.015, f"{m:.3f} ± {sd:.3f}", ha="center", color=INK, fontsize=9)
    ax.axhline(floor_m, color=ORANGE, lw=2, zorder=5)
    ax.annotate(f"exact samples, finite-sample floor {floor_m:.3f} ± {floor_sd:.3f}",
                xy=(len(bars) - 0.62, floor_m), xytext=(len(bars) - 0.62, floor_m + 0.058),
                ha="right", color=ORANGE, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1, relpos=(1, 0)))
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([n.replace("reverse diffusion", "reverse diffusion\nsummed score") for n in bars])
    ax.set_ylabel("total variation to exact grid density")
    ax.set_ylim(0, max(0.2, max(_stats(rows, n, "tv")[0] + _stats(rows, n, "tv")[1] for n in bars) * 1.18))
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("TV on a 60 × 60 grid over [−1.5, 1.5]², mean ± sd over 3 seeds, dots are seeds",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "tv-per-sampler.png", dpi=150)
    plt.close(fig)

    # Figure 3: MALA acceptance per level in the main run.
    mala = [r for r in rows if r["sampler"] == "annealed MALA"]
    acc = np.array([r["accept_per_level"] for r in mala])
    lv = np.arange(acc.shape[1])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.fill_between(lv, acc.min(0), acc.max(0), color=BLUE, alpha=0.2, linewidth=0)
    ax.plot(lv, acc.mean(0), color=BLUE, lw=2)
    ax.axhline(0.6, color=INK2, ls="--", lw=1)
    ax.text(2, 0.62, "tuning target 0.6", color=INK2, fontsize=9)
    ax.text(lv[-1], acc.mean(0)[-1] - 0.06, f"clean level {acc[:, -1].mean():.2f}", ha="right",
            color=INK, fontsize=9)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["level 99\n(most noise)", "74", "49", "24", "clean"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("MALA acceptance share")
    ax.set_xlabel("noise level visited, in sampling order")
    ax.grid(axis="y", color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("MALA acceptance per level, main run; line is the mean, band spans 3 seeds",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "mala-acceptance-per-level.png", dpi=150)
    plt.close(fig)
