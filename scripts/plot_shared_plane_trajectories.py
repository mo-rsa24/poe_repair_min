#!/usr/bin/env python
"""Two-panel figure: all 64 held-out PoE trajectories in the one shared PCA plane
(left), and the seed 9 bundle magnified with two PoE render insets (right).

Reads artifacts/drips/showcase-the-trained-adapter/manifold/manifold_data.json
(coordinates, variance shares, embedded 256px renders). Writes
paper/iclr/figures/held-out-trajectories-in-one-shared-plane.{pdf,png,json}.

Every number drawn on the figure is recomputed here from the coordinates; the
asserts fail the build if they drift from the sidecar's recorded values.
"""
import base64, io, itertools, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "artifacts/drips/showcase-the-trained-adapter/manifold/manifold_data.json"
OUT = ROOT / "paper/iclr/figures"
NAME = "held-out-trajectories-in-one-shared-plane"

d = json.loads(SRC.read_text())
pairs, seeds = d["pairs"], d["seeds"]
pts = {(c["pair"], c["seed"]): np.asarray(c["pts"]) for c in d["cells"]}
imgs = {(c["pair"], c["seed"]): c["poe"] for c in d["cells"]}

# ---- recompute the drawn numbers from the coordinates -----------------------
def mean_pairwise(a):
    return float(np.mean([np.linalg.norm(x - y)
                          for x, y in itertools.combinations(a, 2)]))

starts = {s: pts[(pairs[0], s)][0] for s in seeds}
start_spread = max(
    float(np.max(np.linalg.norm(
        np.array([pts[(p, s)][0] for p in pairs])
        - np.array([pts[(p, s)][0] for p in pairs]).mean(0), axis=1)))
    for s in seeds)
seed_sep = mean_pairwise([starts[s] for s in seeds])
end_pw = {s: mean_pairwise([pts[(p, s)][-1] for p in pairs]) for s in seeds}
end_cent = np.mean([
    float(np.mean(np.linalg.norm(
        np.array([pts[(p, s)][-1] for p in pairs])
        - np.array([pts[(p, s)][-1] for p in pairs]).mean(0), axis=1)))
    for s in seeds])

assert start_spread < 1e-9, start_spread
assert abs(seed_sep - d["insight"]["seed_sep"]) < 0.5, seed_sep
assert abs(end_cent - d["insight"]["end_spread"]) < 0.3, end_cent
print(f"seed_sep {seed_sep:.1f}  end_pw mean {np.mean(list(end_pw.values())):.1f}"
      f"  seed9 end_pw {end_pw[9]:.1f}  end_to_centre {end_cent:.2f}")

def show(pair):
    a, b = pair.split("__x__")
    strip = lambda t: t.removeprefix("a_").removeprefix("an_").replace("_", " ")
    return f"{strip(a)} × {strip(b)}"

# ---- style -------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
CMAP = LinearSegmentedColormap.from_list(
    "stepblue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
INK, MUT, GRAY = "0.25", "0.45", "0.82"
ZOOM_SEED = 9
WIN = (-25.0, 6.0, 2.0, 58.0)  # x0, x1, y0, y1 of the zoom window

def draw_thread(ax, xy, lw, alpha):
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(seg, cmap=CMAP, lw=lw, alpha=alpha, capstyle="round")
    lc.set_array(np.arange(len(seg)))
    ax.add_collection(lc)

fig = plt.figure(figsize=(7.2, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.16,
                      left=0.07, right=0.845, bottom=0.145, top=0.925)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ---- left: all 64 threads ----------------------------------------------------
for k, xy in pts.items():
    draw_thread(axL, xy, lw=0.65, alpha=0.55)
for s in seeds:
    x, y = starts[s]
    axL.scatter([x], [y], s=26, facecolors="none", edgecolors=INK,
                lw=0.9, zorder=5)
    dx, dy, ha = {9: (-4, 9, "right"), 11: (4, -8, "left"),
                  13: (4, -9, "left")}.get(s, (4, 5, "left"))
    axL.annotate(f"seed {s}", (x, y), (x + dx, y + dy), fontsize=6,
                 color=INK, ha=ha, zorder=6)

a, b = starts[11], starts[15]  # the seed pair whose distance is nearest the mean
axL.annotate("", xy=tuple(b), xytext=tuple(a),
             arrowprops=dict(arrowstyle="<->", color=MUT, lw=0.8,
                             shrinkA=4, shrinkB=4), zorder=4)
axL.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 10,
         f"starts, different seeds:\nmean {seed_sep:.1f} units apart",
         fontsize=6, color=MUT, ha="center", va="bottom")

axL.add_patch(Rectangle((WIN[0], WIN[2]), WIN[1] - WIN[0], WIN[3] - WIN[2],
                        fill=False, ec=MUT, lw=0.7, ls=(0, (3, 2)), zorder=4))
axL.autoscale()
axL.set_aspect("equal")
axL.set_xlabel("shared PC 1 (latent units)")
axL.set_ylabel("shared PC 2 (latent units)")
axL.set_title("all 64 held-out trajectories, one shared plane", fontsize=7.5)

cax = axL.inset_axes([0.05, 0.26, 0.30, 0.032])
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 49))
cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 25, 49])
cb.ax.tick_params(labelsize=6, width=0.6, length=2)
cb.set_label("denoising step (DDIM)", fontsize=6.5, labelpad=1.5)
cb.outline.set_linewidth(0.6)

# ---- right: the seed 9 bundle, magnified --------------------------------------
for s in seeds:  # intruding threads stay visible: hiding them would overclaim
    if s == ZOOM_SEED:
        continue
    for p in pairs:
        axR.plot(*pts[(p, s)].T, color=GRAY, lw=0.6, alpha=0.6, zorder=1)
for p in pairs:
    draw_thread(axR, pts[(p, ZOOM_SEED)], lw=1.3, alpha=0.95)

sx, sy = starts[ZOOM_SEED]
axR.scatter([sx], [sy], s=34, facecolors="none", edgecolors=INK, lw=1.0, zorder=5)
axR.annotate("shared start of the 8 pairs\n(spread 0.00 by construction)",
             (sx, sy), (sx + 0.6, sy + 0.6), fontsize=5.6, color=INK, va="center")

ends = {p: pts[(p, ZOOM_SEED)][-1] for p in pairs}
OFF = {  # hand-tuned label offsets (dx, dy, ha)
    "a_cat__x__a_dog":            (0.8, -1.6, "left"),
    "a_cow__x__a_buffalo":        (2.4, 2.2, "left"),
    "a_leopard__x__a_jaguar":     (2.6, -0.4, "left"),
    "a_frog__x__a_toad":          (-0.2, 1.7, "center"),
    "a_goose__x__a_swan":         (0.6, -2.4, "center"),
    "an_eagle__x__a_hawk":        (-1.4, -1.8, "right"),
    "an_elephant__x__a_penguin":  (-1.6, -2.2, "center"),
    "a_seal__x__a_walrus":        (-0.8, 1.7, "center"),
}
for p, (ex, ey) in ends.items():
    axR.scatter([ex], [ey], s=9, color="#1c5cab", zorder=6)
    dx, dy, ha = OFF[p]
    axR.annotate(show(p), (ex, ey), (ex + dx, ey + dy), fontsize=5.8,
                 color=INK, ha=ha, va="center", zorder=7)

exs = [e[0] for e in ends.values()]
by = 5.4
axR.annotate("", xy=(max(exs), by), xytext=(min(exs), by),
             arrowprops=dict(arrowstyle="<->", color=MUT, lw=0.8))
axR.text((max(exs) + min(exs)) / 2, by - 1.2,
         f"these 8 ends: mean {end_pw[ZOOM_SEED]:.1f} units apart (pairwise)",
         fontsize=6, color=MUT, ha="center", va="top")
axR.text(0.5, -0.105, "gray: threads of seeds 11 and 16 passing through this window",
         transform=axR.transAxes, fontsize=5.8, color="0.5", ha="center", va="top")

axR.set_xlim(WIN[0], WIN[1]); axR.set_ylim(WIN[2], WIN[3])
axR.set_aspect("equal")
axR.set_title(f"the seed {ZOOM_SEED} bundle, magnified", fontsize=7.5)

for corner in ((1, 1), (1, 0)):
    fig.add_artist(ConnectionPatch(
        xyA=(WIN[1], WIN[3] if corner[1] else WIN[2]), coordsA=axL.transData,
        xyB=(0, corner[1]), coordsB=axR.transAxes, color="0.7", lw=0.6))

# ---- the two PoE render insets ------------------------------------------------
INSETS = [("an_eagle__x__a_hawk", [1.07, 0.52, 0.34, 0.34]),
          ("a_cat__x__a_dog",     [1.07, 0.06, 0.34, 0.34])]
for p, box in INSETS:
    im = Image.open(io.BytesIO(base64.b64decode(imgs[(p, ZOOM_SEED)].split(",", 1)[1])))
    ia = axR.inset_axes(box)
    ia.imshow(im); ia.set_xticks([]); ia.set_yticks([])
    for sp in ia.spines.values():
        sp.set_linewidth(0.9); sp.set_color("0.35")
    ia.set_xlabel(f"PoE, {show(p)}", fontsize=5.8, color=INK, labelpad=2)
    fig.add_artist(ConnectionPatch(
        xyA=tuple(ends[p]), coordsA=axR.transData,
        xyB=(0, 0.5), coordsB=ia.transAxes, color="0.6", lw=0.7,
        shrinkA=2, shrinkB=1))

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / f"{NAME}.pdf")
fig.savefig(OUT / f"{NAME}.png", dpi=200)

sidecar = {
    "object": d["object"], "method": d["method"],
    "plane_variance_share_pct": d["var2"], "top10_variance_share_pct": d["var10"],
    "numbers_drawn": {
        "seed_start_separation_mean_pairwise": round(seed_sep, 2),
        "seed9_end_spread_mean_pairwise": round(end_pw[9], 2),
        "start_spread_within_seed": 0.0,
    },
    "numbers_recorded_not_drawn": {
        "end_spread_mean_pairwise_over_seeds":
            round(float(np.mean(list(end_pw.values()))), 2),
        "end_spread_mean_distance_to_seed_centroid": round(float(end_cent), 2),
        "end_spread_mean_pairwise_per_seed":
            {s: round(v, 2) for s, v in end_pw.items()},
    },
    "units": "latent units of the flattened mean-centred 4x128x128 x_t; "
             "the plane is an orthogonal projection, so in-plane distances "
             "never exceed true latent distances",
    "renders_in_insets": [f"PoE final render, {p}, seed {ZOOM_SEED}"
                          for p, _ in INSETS],
    "source_sidecar": str(SRC.relative_to(ROOT)),
    "cells": [{"pair": c["pair"], "seed": c["seed"], "pts": c["pts"]}
              for c in d["cells"]],
}
(OUT / f"{NAME}.json").write_text(json.dumps(sidecar))
print("wrote", OUT / f"{NAME}.pdf")
