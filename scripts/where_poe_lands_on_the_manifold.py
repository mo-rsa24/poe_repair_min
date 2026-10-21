#!/usr/bin/env python
"""Where each prompt lands on the manifold, and where the product lands instead.

A drawn surface with four real renders pinned to it. Height is how readily the model
produces an image at that point, over an illustrative two-dimensional plane. Three
peaks: a cat on its own, a dog on its own, and the two-animal scene the joint prompt
"a cat and a dog" reaches. Marked separately is the peak of the product of the two
single-animal densities, which is what plain PoE composition maximises. That peak lands
on the low saddle between the two single-animal mountains, and it is nowhere near the
two-animal one.

The surface is drawn; the four thumbnails are not. All four come from one cell,
a_cat__x__a_dog at seed 9, same starting latent, 50 steps, guidance 7.5, no adapter
attached, so the only difference between the four pictures is what the sampler was
conditioned on. The script prints every path it read.

Nothing about the surface is measured. The plane is not an embedding of anything, the
peaks are Gaussians placed by hand, and the original layouts use ripples and equal
peak heights. The surface layout uses a curved sheet and a lower joint peak. None
of these drawing choices makes a claim about how often each scene occurs. The one
geometric fact it honours is the one F1 honours: the product of the two equal-width
Gaussians is a Gaussian at their midpoint, so the product's peak is drawn where the
maths puts it.

Companion to scripts/what_the_product_misses.py (F1), which argues the same thing in
flat contour lines.

    python scripts/where_poe_lands_on_the_manifold.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.mplot3d import proj3d
import numpy as np

FIG_DIR = Path("paper/iclr/figures")
FIG_NAME = "where-the-product-lands-on-the-manifold"

# One cell, four conditions, one starting latent. Written by
# scripts/showcase/where_each_condition_lands.py; the manifest beside these frames
# records init_latents, steps, guidance and adapter_attached_before_references=false.
CELL_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/"
                 "where_each_condition_lands/pairs/a_cat__x__a_dog")
SEED = 9
FINAL_FRAME = "step_050.png"

# The drawn plane. Coordinates mean nothing; its extent is a layout choice below.
CAT_XY, DOG_XY = (-2.05, 1.45), (2.05, 1.45)
SIG = (0.95, 0.92)
JOINT_XY, JOINT_SIG = (0.05, -2.30), (1.40, 1.02)
PEAK_H = 1.00          # original layouts use equal nominal peak heights

WARM = "#b2402a"       # one concept on its own
GREEN = "#1f6b45"      # the reading the prompt asks for and the product never reaches
COOL = "#26527f"       # what the product asks for
BORDER = "#555555"
LABEL_PT = 8.0
CHIP_PT = 6.4

# Page geometry, in inches. ICLR 2027 is single column at 5.5in textwidth.
#
# Two layouts, because the two are read at different sizes. `spacious` is the original
# balance, which reads on screen. `print` is measured against a 5.5in column: the four
# renders carry the argument, and at 0.95in they print at 142px, where the fused
# animal's dog muzzle on a cat's frame stops being visible. So `print` grows every
# thumbnail, trims the surface's empty skirt, and coarsens the mesh, whose fine white
# lines on a coloured ground are where print moire shows up.
#
# Each condition entry: the frame directory, the words beside its thumbnail, its
# colour, which corner the leader leaves from, and whether its words go above or below.
# The joint render sits low on the page, where a word above it would land on the
# surface, so its words go underneath.
CONDITIONS = [
    ("solo_a", '"a cat"',                     WARM,  "s", "above"),
    ("poe",    "PoE: the product of the two", COOL,  "s", "above"),
    ("solo_b", '"a dog"',                     WARM,  "s", "above"),
    ("joint",  '"a cat and a dog"',           GREEN, "e", "below"),
]

LAYOUTS = {
    "spacious": dict(
        fig=(5.50, 4.35), thumb=0.95, plane=4.00, edge=(3.70, 4), mesh=(150, 2),
        axes=(-0.10, -0.11, 1.20, 1.13), elev=36.0, zbox=0.62,
        slots={"solo_a": (0.108, 0.775), "poe": (0.500, 0.805),
               "solo_b": (0.892, 0.775), "joint": (0.150, 0.215)},
    ),
    # Sized so the float and its caption fit on the page that introduces it. At the
    # `print` layout's 4.35in the float plus caption fills two thirds of an ICLR page,
    # and LaTeX pushes it past the paragraph that says "as shown in Figure 1". This one
    # is 3.30in: a lower camera and a flatter surface buy the height back without
    # shrinking the renders below the size at which the fused animal reads.
    "column": dict(
        fig=(5.50, 3.62), thumb=1.04, plane=3.40, edge=(3.15, 8), mesh=(132, 3),
        axes=(-0.03, -0.085, 1.06, 1.15), elev=27.0, zbox=0.50,
        slots={"solo_a": (0.140, 0.775), "poe": (0.520, 0.808),
               "solo_b": (0.860, 0.775), "joint": (0.108, 0.218)},
        border=1.7,
        label_side={"joint": "below"},
    ),
    "print": dict(
        fig=(5.50, 4.35), thumb=1.26, plane=3.40, edge=(3.15, 8), mesh=(132, 3),
        axes=(0.005, -0.055, 0.99, 0.95), elev=36.0, zbox=0.62,
        slots={"solo_a": (0.128, 0.755), "poe": (0.500, 0.790),
               "solo_b": (0.872, 0.755), "joint": (0.145, 0.183)},
    ),
    # An orthographic vector mesh gives this illustration its broad curved skirt.
    # The same projection places every thumbnail and the mathematical midpoint.
    "surface": dict(
        fig=(5.50, 3.09375), thumb=0.90, thumb_aspect=4.0 / 3.0,
        plane=5.40, edge=(3.30, 6), mesh=(161, 1),
        axes=(0.0, 0.0, 1.0, 1.0), elev=46.0, azim=-90.0, zbox=0.50,
        pinned=True, diamond=False,
        peaks=((-2.55, 1.05), (2.55, 1.05), (0.0, -2.30)),
        # The lower joint peak is a drawing choice, not a measured mass.
        joint_height=0.60,
        sheet_depth=8.83, rear_lip=0.28, sheet_taper=1.0,
        facet_edge=0.40, facet_light=(-0.45, 0.35, 0.82), facet_light_strength=0.12,
        projection_origin_y=1.05, projection_base=0.332,
        projection_span=0.998, projection_depth=0.055,
        projection_crown=0.150, projection_height=0.235,
        projection_elev=46.0,
        poe_drop=0.050, joint_drop=0.198,
        colours={"solo_a": "#f0160a", "solo_b": "#f0160a",
                 "poe": "#064dff", "joint": "#008337"},
        crop_y={"solo_a": 0.0, "solo_b": 0.20, "poe": 0.0, "joint": 0.65},
        border=1.35, corner=0.048, label_pt=12.3, joint_label_pt=11.6,
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "surface-blue-pink-red",
            [(0.00, "#1a5fe8"), (0.16, "#397cf2"),
             (0.32, "#87b7ff"), (0.45, "#d7e6ef"),
             (0.55, "#f9d5c6"), (0.68, "#ff9c86"),
             (0.82, "#ff6048"), (1.00, "#e8281c")], N=1024),
    ),
}
LAYOUT = LAYOUTS["spacious"]     # set per run by build()


def bump(xx, yy, cx, cy, sx, sy):
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)


def surface(xx, yy):
    """Three prompts as peaks, on an illustrative floor."""
    cat, dog, joint = peaks()
    if LAYOUT.get("pinned"):
        return PEAK_H * (bump(xx, yy, *cat, *SIG)
                         + bump(xx, yy, *dog, *SIG)
                         + LAYOUT["joint_height"]
                         * bump(xx, yy, *joint, *JOINT_SIG))

    e_scale, e_pow = LAYOUT["edge"]
    z = PEAK_H * (bump(xx, yy, *cat, *SIG)
                  + bump(xx, yy, *dog, *SIG)
                  + bump(xx, yy, *joint, *JOINT_SIG))
    # Small enough never to rival a peak, large enough that the eye reads one
    # continuous landscape instead of three balloons on a table.
    ripple = (0.075 * np.sin(0.95 * xx + 0.35) * np.cos(0.85 * yy - 0.20)
              + 0.045 * np.sin(1.55 * yy + 1.1) * np.cos(0.70 * xx + 0.6))
    ex, ey = xx, yy
    if LAYOUT.get("diamond"):          # the plane turned 45 degrees, a corner to the viewer
        ex, ey = (xx + yy) / np.sqrt(2), (yy - xx) / np.sqrt(2)
    edge = np.exp(-((ex / e_scale) ** e_pow + (ey / e_scale) ** e_pow))
    return (z + ripple) * edge


def peaks():
    """Cat, dog and joint peak centres in the current illustrative layout."""
    return LAYOUT.get("peaks", (CAT_XY, DOG_XY, JOINT_XY))


def height_at(x, y):
    return float(surface(np.array([[x]]), np.array([[y]]))[0, 0])


def pinned_projection(x, y, z, elev, azim):
    """Project the curved sheet to figure fractions and return camera depth.

    This explicit orthographic projection keeps the vector sheet's silhouette
    independent of mplot3d's axes margins. Elevation and azimuth still rotate the
    same surface, including its anchor points.
    """
    x, y, z = np.broadcast_arrays(np.asarray(x), np.asarray(y), np.asarray(z))
    reference = np.deg2rad(LAYOUT["projection_elev"])
    elevation = np.deg2rad(elev)
    yaw = np.deg2rad(azim + 90.0)
    dy = y - LAYOUT["projection_origin_y"]
    horizontal = np.cos(yaw) * x - np.sin(yaw) * dy
    away = np.sin(yaw) * x + np.cos(yaw) * dy
    crown = LAYOUT["projection_crown"] * (1.0 - (x / LAYOUT["plane"]) ** 2)
    vertical = crown + LAYOUT["projection_height"] * z
    scale = LAYOUT["projection_depth"] / np.sin(reference)
    world_height = vertical / (scale * np.cos(reference))
    fx = 0.5 + LAYOUT["projection_span"] * horizontal / (2.0 * LAYOUT["plane"])
    fy = (LAYOUT["projection_base"]
          + scale * (away * np.sin(elevation) + world_height * np.cos(elevation)))
    depth = away * np.cos(elevation) - world_height * np.sin(elevation)
    return fx, fy, depth


def to_figure_xy(fig, ax, x, y, z):
    """A point on the surface, in figure fractions, so a flat artist can reach it."""
    if LAYOUT.get("pinned"):
        fx, fy, _ = pinned_projection(x, y, z, ax.elev, ax.azim)
        return np.array([float(fx), float(fy)])
    px, py, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    disp = ax.transData.transform((px, py))
    return fig.transFigure.inverted().transform(disp)


def draw_surface(ax) -> None:
    if LAYOUT.get("pinned"):
        from matplotlib.collections import PolyCollection

        n, stride = LAYOUT["mesh"]
        samples = np.arange(0, n, stride)
        if samples[-1] != n - 1:
            samples = np.append(samples, n - 1)
        u = np.linspace(-1.0, 1.0, n)[samples]
        v = np.linspace(0.0, 1.0, n)[samples]
        uu, vv = np.meshgrid(u, v)
        xx = LAYOUT["plane"] * uu
        span = np.maximum(0.0, 1.0 - uu ** 2)
        # Keep the rear contour and centre depth. A parabolic front boundary
        # meets each rear endpoint with finite slope, giving two pointed tips.
        rear = LAYOUT["projection_origin_y"] + LAYOUT["rear_lip"] * span
        yy = rear - LAYOUT["sheet_depth"] * vv * span ** LAYOUT["sheet_taper"]
        zz = surface(xx, yy)
        fx, fy, _ = pinned_projection(xx, yy, zz, ax.elev, ax.azim)
        points = np.stack((fx, fy), axis=-1)
        vertices = np.stack((points[:-1, :-1], points[:-1, 1:],
                             points[1:, 1:], points[1:, :-1]), axis=2)
        vertices = vertices.reshape(-1, 4, 2)

        cx = (xx[:-1, :-1] + xx[:-1, 1:] + xx[1:, 1:] + xx[1:, :-1]) / 4.0
        cy = (yy[:-1, :-1] + yy[:-1, 1:] + yy[1:, 1:] + yy[1:, :-1]) / 4.0
        cz = surface(cx, cy)
        _, _, depth = pinned_projection(cx, cy, cz, ax.elev, ax.azim)
        order = np.argsort(-depth.ravel(), kind="stable")

        # Evaluate the smooth colour field at every facet centre. Each hill
        # reaches red at its own crest; colour has no quantitative mass scale.
        cat, dog, joint = peaks()
        colour_height = np.zeros_like(cx)
        dz_dx = np.zeros_like(cx)
        dz_dy = np.zeros_like(cy)
        hills = ((cat, SIG, 1.0), (dog, SIG, 1.0),
                 (joint, JOINT_SIG, LAYOUT["joint_height"]))
        for (hx, hy), (sx, sy), amplitude in hills:
            hill = bump(cx, cy, hx, hy, sx, sy)
            colour_height += hill
            dz_dx -= PEAK_H * amplitude * hill * (cx - hx) / sx ** 2
            dz_dy -= PEAK_H * amplitude * hill * (cy - hy) / sy ** 2
        colours = LAYOUT["cmap"](np.clip(colour_height.ravel(), 0.0, 1.0))

        # Normals include the curved floor and the drawn peak heights. Analytic
        # derivatives stay smooth even where the mesh collapses at its tips.
        reference = np.deg2rad(LAYOUT["projection_elev"])
        height_unit = (LAYOUT["projection_depth"] / np.sin(reference)
                       * np.cos(reference))
        crown_dx = -2.0 * LAYOUT["projection_crown"] * cx / LAYOUT["plane"] ** 2
        world_dx = (crown_dx + LAYOUT["projection_height"] * dz_dx) / height_unit
        world_dy = LAYOUT["projection_height"] * dz_dy / height_unit
        normals = np.stack((-world_dx, -world_dy, np.ones_like(cx)), axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        light = np.asarray(LAYOUT["facet_light"], dtype=float)
        light /= np.linalg.norm(light)
        illumination = np.sum(normals * light, axis=-1)
        # A horizontal patch retains the palette. Subtle shading separates
        # the near slope of the joint hill from the sheet behind it.
        brightness = 1.0 + LAYOUT["facet_light_strength"] * (illumination - light[2])
        colours[:, :3] = np.clip(colours[:, :3] * brightness.ravel()[:, None], 0.0, 1.0)

        # Opaque face-coloured strokes overlap adjacent fills, covering the
        # hairline gaps that PDF and SVG viewers can expose between polygons.
        sheet = PolyCollection(
            vertices[order], facecolors=colours[order], edgecolors="face",
            linewidths=LAYOUT["facet_edge"], antialiaseds=True, joinstyle="round",
            transform=ax.figure.transFigure, clip_on=False, zorder=1)
        sheet.set_rasterized(False)
        ax.figure.add_artist(sheet)
        ax.set_axis_off()
        return

    lim = (-LAYOUT["plane"], LAYOUT["plane"])
    n, stride = LAYOUT["mesh"]
    g = np.linspace(*lim, n)
    xx, yy = np.meshgrid(g, g)
    if LAYOUT.get("diamond"):
        xx, yy = (xx - yy) / np.sqrt(2), (xx + yy) / np.sqrt(2)
        lim = (lim[0] * np.sqrt(2), lim[1] * np.sqrt(2))
    ax.plot_surface(xx, yy, surface(xx, yy), rstride=stride, cstride=stride,
                    cmap=LAYOUT.get("cmap", "coolwarm"), linewidth=0.18, edgecolors=(1, 1, 1, 0.32),
                    antialiased=True, shade=False, vmin=-0.12, vmax=PEAK_H * 1.02)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_zlim(-0.28, 1.30)
    ax.set_box_aspect((1.0, 1.0, LAYOUT["zbox"]))
    ax.set_axis_off()


def anchors():
    """Where each condition sits on the surface, in data coordinates."""
    cat, dog, joint = peaks()
    sx, sy = (cat[0] + dog[0]) / 2, (cat[1] + dog[1]) / 2
    return {"solo_a": cat, "solo_b": dog, "joint": joint, "poe": (sx, sy)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="spacious",
                    help="spacious reads on screen; print is balanced for a 5.5in "
                         "column, with larger renders and a quieter mesh")
    ap.add_argument("--elev", type=float, default=None,
                    help="camera elevation; each layout carries its own default")
    ap.add_argument("--azim", type=float, default=None,
                    help="camera azimuth; -62 unless the layout carries its own")
    ap.add_argument("--cell-root", type=Path, default=CELL_ROOT,
                    help="the cell's render folder, for frames copied off the cluster")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="which cell's four renders to pin on")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds; writes one file each, suffixed "
                         "-seedNN, for picking a cell by eye. The joint render is "
                         "two dogs on seeds 10 and 13 and ambiguous on 14 and 15, "
                         "so those cells cannot carry this figure.")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--name", default=FIG_NAME)
    args = ap.parse_args()

    seeds = ([int(t) for t in args.seeds.split(",")] if args.seeds
             else [args.seed])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        suffix = f"-seed{seed:02d}" if args.seeds else ""
        if args.layout != "spacious":
            suffix += f"-{args.layout}"
        build(args, seed, args.out_dir / f"{args.name}{suffix}.png")
    return 0


def build(args, seed: int, out: Path) -> None:
    global LAYOUT
    LAYOUT = LAYOUTS[args.layout]
    fig_w, fig_h = LAYOUT["fig"]
    thumb = LAYOUT["thumb"]
    frames = {c: args.cell_root / c / f"seed_{seed}" / FINAL_FRAME
              for c, *_ in CONDITIONS}
    for cond, p in frames.items():
        if not p.exists():
            raise SystemExit(f"no render for {cond} at {p}")
        print(f"seed {seed:>2d}  {cond:8s} {p}")

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(list(LAYOUT["axes"]), projection="3d")
    if LAYOUT.get("pinned"):
        ax.set_axis_off()
    else:
        draw_surface(ax)
    elev = args.elev if args.elev is not None else LAYOUT["elev"]
    azim = args.azim if args.azim is not None else LAYOUT.get("azim", -62.0)
    ax.view_init(elev=elev, azim=azim)
    fig.canvas.draw()          # the projection is only real once the canvas exists

    if LAYOUT.get("pinned"):
        build_pinned(fig, ax, frames, out, elev, azim)
        return

    at = anchors()
    zoom = thumb * fig.dpi / 512.0      # renders are 512px square

    for cond, words, colour, side, where in CONDITIONS:
        slot = LAYOUT["slots"][cond]
        where = LAYOUT.get("label_side", {}).get(cond, where)
        x, y = at[cond]
        z = height_at(x, y)
        tip = to_figure_xy(fig, ax, x, y, z)

        # The mark on the surface itself, so the thumbnail is pinned to a place and
        # not merely floating near one.
        ax.plot([x], [y], [z], marker="X" if cond == "poe" else "o",
                markersize=7.5 if cond == "poe" else 5.0, linestyle="none",
                color=colour, markeredgecolor="white", markeredgewidth=0.9,
                zorder=10)

        img = OffsetImage(plt.imread(frames[cond]), zoom=zoom)
        # The image is drawn at `xybox` (its slot on the page) and the arrow runs from
        # it to `xy` (the mark on the surface), so the thumbnail is pinned to a place.
        box = AnnotationBbox(
            img, tip, xycoords="figure fraction",
            xybox=slot, boxcoords="figure fraction", box_alignment=(0.5, 0.5),
            frameon=True, pad=0.0,
            bboxprops=dict(edgecolor=colour, linewidth=0.9),
            arrowprops=dict(arrowstyle="-", linewidth=0.7, color=colour,
                            shrinkA=1.5, shrinkB=2.0,
                            relpos={"s": (0.5, 0.0), "w": (0.0, 0.5),
                                    "n": (0.5, 1.0), "e": (1.0, 0.5)}[side]))
        fig.add_artist(box)

        half = (thumb / fig_h) / 2
        fig.text(slot[0], slot[1] + (half + 0.012 if where == "above"
                                     else -half - 0.012), words,
                 color=colour, fontsize=LABEL_PT, family="serif",
                 ha="center", va="bottom" if where == "above" else "top")

    # The gap between where the product peaks and where the joint prompt lands is the
    # whole argument, so it is the only distance drawn on the surface.
    px, py = at["poe"]
    t = np.linspace(0, 1, 60)
    gx, gy = px + t * (JOINT_XY[0] - px), py + t * (JOINT_XY[1] - py)
    gz = np.array([height_at(a, b) for a, b in zip(gx, gy)]) + 0.045
    ax.plot(gx, gy, gz, linestyle=(0, (3, 2.5)), linewidth=1.0, color=BORDER, zorder=9)
    mid = to_figure_xy(fig, ax, gx[34], gy[34], gz[34])
    fig.text(mid[0] + 0.018, mid[1], "the product never gets here", color=BORDER,
             fontsize=CHIP_PT, family="serif", ha="left", va="center",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.4, alpha=0.80))

    fig.savefig(out, dpi=400, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"         wrote {out} and {out.with_suffix('.pdf')}, "
          f"elev {elev:.0f} azim {azim:.0f}")


def build_pinned(fig, ax, frames, out: Path, elev: float, azim: float) -> None:
    """The `surface` layout, with a curved vector sheet and cropped real renders."""
    from matplotlib import patheffects
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig_w, fig_h = LAYOUT["fig"]
    thumb = LAYOUT["thumb"]
    thumb_h = thumb / LAYOUT["thumb_aspect"]
    half_h = thumb_h / fig_h / 2.0
    draw_surface(ax)
    at = anchors()
    tip = {c: to_figure_xy(fig, ax, *at[c], height_at(*at[c])) for c in at}

    # Thumbnail bottoms touch the rear crests; the foreground image covers the
    # joint hill's near slope while leaving its rounded red cap visible.
    top = max(tip["solo_a"][1], tip["solo_b"][1]) + half_h
    centre = {"solo_a": (tip["solo_a"][0], top),
              "solo_b": (tip["solo_b"][0], top),
              "poe": (tip["poe"][0], top - LAYOUT["poe_drop"]),
              "joint": (tip["joint"][0], tip["joint"][1] - LAYOUT["joint_drop"])}

    for cond, words, _colour, _side, _where in CONDITIONS:
        colour = LAYOUT["colours"][cond]
        pixels = plt.imread(frames[cond])
        height, width = pixels.shape[:2]
        aspect = LAYOUT["thumb_aspect"]
        crop_h = min(height, int(round(width / aspect)))
        crop_w = min(width, int(round(crop_h * aspect)))
        left = (width - crop_w) // 2
        upper = int(round((height - crop_h) * LAYOUT["crop_y"][cond]))
        pixels = pixels[upper:upper + crop_h, left:left + crop_w]
        # OffsetImage's zoom is per point, independent of the export DPI.
        img = OffsetImage(pixels, zoom=thumb * 72.0 / pixels.shape[1])
        cx, cy = centre[cond]
        actual_h = thumb * pixels.shape[0] / pixels.shape[1]
        frame = FancyBboxPatch(
            (cx * fig_w - thumb / 2.0, cy * fig_h - actual_h / 2.0),
            thumb, actual_h,
            boxstyle=f"round,pad=0,rounding_size={LAYOUT['corner']}",
            transform=fig.dpi_scale_trans,
            facecolor="none", edgecolor=colour,
            linewidth=LAYOUT["border"], clip_on=False, zorder=21)
        box = AnnotationBbox(
            img, (cx, cy), xycoords="figure fraction",
            box_alignment=(0.5, 0.5), frameon=False, pad=0.0,
            annotation_clip=False)
        box.set_zorder(20)
        fig.add_artist(box)
        img.image.set_clip_path(frame)
        img.image.set_clip_on(True)
        fig.add_artist(frame)

        if cond == "poe":
            continue
        above = cond != "joint"
        words = words.replace('"', "\u201c", 1).replace('"', "\u201d", 1)
        label = fig.text(
            cx, cy + half_h + 0.014 if above else cy - half_h - 0.014,
            words, color=colour,
            fontsize=LAYOUT["label_pt"] if above else LAYOUT["joint_label_pt"],
            family="serif", ha="center", va="bottom" if above else "top",
            clip_on=False, zorder=23)
        if not above:
            # Layered white strokes stay vector in PDF and SVG.
            label.set_path_effects([
                patheffects.Stroke(linewidth=8.0, foreground=(1, 1, 1, 0.14)),
                patheffects.Stroke(linewidth=5.0, foreground=(1, 1, 1, 0.30)),
                patheffects.Stroke(linewidth=2.5, foreground=(1, 1, 1, 0.85)),
                patheffects.Normal(),
            ])

    # Project the actual midpoint once, then draw its marker above the mesh so
    # painter ordering cannot hide it behind a foreground facet.
    px, py = tip["poe"]
    marker = Line2D(
        [px], [py], transform=fig.transFigure, marker="X", markersize=10.0,
        linestyle="none", color=LAYOUT["colours"]["poe"],
        markeredgecolor="white", markeredgewidth=1.25,
        clip_on=False, zorder=24)
    fig.add_artist(marker)

    callout_x = px + 0.070
    callout_y = centre["poe"][1] - half_h - 0.041
    fig.text(
        callout_x, callout_y, "PoE lands here",
        color="#111111", fontsize=8.8, family="serif",
        ha="center", va="center", clip_on=False, zorder=25,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f5f7fb",
                  edgecolor="#97a7bf", linewidth=0.65, alpha=0.97))
    fig.add_artist(FancyArrowPatch(
        (callout_x - 0.034, callout_y - 0.027), (px + 0.012, py + 0.019),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7.0,
        linestyle=(0, (2.3, 1.5)), linewidth=0.9, color="#111111",
        connectionstyle="arc3,rad=-0.25", shrinkA=0.0, shrinkB=0.0,
        clip_on=False, zorder=23))

    fig.savefig(out, dpi=400, facecolor="white")
    # The surface and words stay vector; dpi only sets how finely the renders are kept.
    fig.savefig(out.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(out.with_suffix(".svg"), dpi=600, facecolor="white")
    plt.close(fig)
    print(f"         wrote {out} and its .pdf and .svg, elev {elev:.0f} azim {azim:.0f}")


if __name__ == "__main__":
    raise SystemExit(main())
