"""Figures for the sampler-by-expert interaction run. Labels sit on the plot; no legends."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import exact_toy as T

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SIGMA = T.SIGMA
KIND_COLOUR = {"exact": BLUE, "trained": ORANGE}
SAMPLERS = ["reverse diffusion", "annealed MALA"]
SAMPLER_LABEL = {"reverse diffusion": "reverse diffusion\nsummed score", "annealed MALA": "annealed MALA\n(fixed, tuned to 0.6)"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _rings(ax):
    for c in T.MU.values():
        ax.add_patch(plt.Circle(c, 3 * SIGMA, fill=False, ls="--", lw=0.8, color=INK2, alpha=0.6))
    ax.add_patch(plt.Circle((0, 0), 3 * SIGMA / math.sqrt(2), fill=False, ls=":", lw=0.8, color=INK2, alpha=0.6))


def _vals(rows, kind, sampler, key):
    return np.array([x[key] for x in rows if x["expert"] == kind and x["sampler"] == sampler])


def draw(out, per_r, effects):
    for r, res in per_r.items():
        rows, floor = res["rows"], res["floor_tv"]
        key_tr = res["answer_key_row"]["in_TR_blob"]
        grid_mass, edges = T.clean_density_grid(r=r)
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.8))
        for i, kind in enumerate(["exact", "trained"]):
            for j, s in enumerate(SAMPLERS):
                ax = axes[i, j]
                ax.imshow(grid_mass.T, origin="lower", extent=[edges[0], edges[-1], edges[0], edges[-1]],
                          cmap="Greys", alpha=0.35, interpolation="nearest")
                xs = np.load(out / f"samples_r{r}_{kind}_{s.replace(' ', '_')}_seed0.npy")
                ax.scatter(xs[:, 0], xs[:, 1], s=3, alpha=0.35, color=KIND_COLOUR[kind], linewidths=0)
                _rings(ax)
                tv, tr = _vals(rows, kind, s, "tv"), _vals(rows, kind, s, "in_TR_blob")
                c1, c2 = _vals(rows, kind, s, "mean_x1").mean(), _vals(rows, kind, s, "mean_x2").mean()
                outside = float(np.mean(~(np.abs(xs) <= 1.5).all(1)))
                txt = (f"TV {tv.mean():.3f} ± {tv.std(ddof=1):.3f}  (floor {floor:.3f})\n"
                       f"top-right share {tr.mean():.3f} ± {tr.std(ddof=1):.3f}  (key {key_tr:.3f})\n"
                       f"centre ({c1:.3f}, {c2:.3f})")
                if outside > 0:
                    txt += f"\nseed 0 outside the box {outside:.3f}"
                ax.text(0.03, 0.03, txt, transform=ax.transAxes, color=INK, fontsize=9, va="bottom",
                        bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85))
                ax.set_title(f"{kind} experts: {s}", color=INK, fontsize=11)
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect("equal")
                ax.set_xlabel("x1 (A is x1 > 0)")
                ax.set_ylabel("x2 (B is x2 > 0)")
        axes[0, 0].text(0.5, 0.5 + 3 * SIGMA + 0.05, "3σ ring", color=INK2, fontsize=8, ha="center")
        fig.suptitle(f"Composed samples at r = {r}: seed 0 scatter, numbers over seeds 0 to 2",
                     color=INK, fontsize=12)
        fig.tight_layout()
        fig.savefig(out / f"samples-2x2-r{r}.png", dpi=150)
        plt.close(fig)

    # Interaction plot: TV above floor against sampler, one line per expert kind, one panel per r.
    fig, axes = plt.subplots(1, len(per_r), figsize=(5.4 * len(per_r), 4.8), squeeze=False)
    for ax, (r, res) in zip(axes[0], per_r.items()):
        ef = effects(res["rows"], res["floor_tv"])
        xpos = np.arange(len(SAMPLERS))
        for kind in ["exact", "trained"]:
            vals = np.array([ef["above"][(kind, s)] for s in SAMPLERS])     # (2 samplers, 3 seeds)
            m, sd = vals.mean(1), vals.std(1, ddof=1)
            col = KIND_COLOUR[kind]
            ax.errorbar(xpos, m, yerr=sd, color=col, lw=2, marker="o", capsize=4)
            for k in range(vals.shape[1]):
                ax.scatter(xpos + (k - 1) * 0.04, vals[:, k], s=12, color=col, alpha=0.5, zorder=3)
            ax.text(xpos[-1] + 0.08, m[-1], f"{kind} experts", color=col, va="center", fontsize=10,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=1))
        inter = ef["interaction"]
        ax.axhline(0, color=INK2, lw=0.8, ls=":")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.3 * (hi - lo))
        ax.text(0.3, 0.0, "exact-sample floor", color=INK2, fontsize=8, ha="center", va="bottom",
                transform=ax.get_yaxis_transform())
        ax.text(0.02, 0.97, f"interaction {inter.mean():+.3f} ± {inter.std(ddof=1):.3f}\n"
                            f"(trained sampler effect minus exact sampler effect)",
                transform=ax.transAxes, va="top", fontsize=9, color=INK)
        ax.set_xticks(xpos, [SAMPLER_LABEL[s] for s in SAMPLERS])
        ax.set_xlim(-0.3, 1.75)
        ax.set_ylabel("TV above the exact-sample floor")
        ax.set_title(f"r = {r}", color=INK)
    fig.tight_layout()
    fig.savefig(out / "interaction-tv-above-floor.png", dpi=150)
    plt.close(fig)
