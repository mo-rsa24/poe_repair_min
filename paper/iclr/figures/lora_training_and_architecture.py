"""Render the compositional-adapter training and architecture figure.

Run from anywhere with:

    python paper/iclr/figures/lora_training_and_architecture.py

The output is written to ``lora-training-and-architecture.png`` beside this script, at 1800 x 900.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


WIDTH, HEIGHT = 1800, 900
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = Path(__file__).resolve().parent / "lora-training-and-architecture.png"

rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.unicode_minus": False,
    }
)

GREEN_BG = "#eef8ee"
GREEN_EDGE = "#4a9a52"
PEACH_BG = "#fff6e9"
ORANGE = "#ff6d00"
BLUE_FILL = "#e6f3ff"
BLUE_EDGE = "#1976d2"
LILAC = "#f2eafd"
PURPLE = "#7b3cc4"
GREY = "#f0f0f0"
GREY_EDGE = "#777777"
RED_BG = "#fff0f0"
RED_EDGE = "#e56a6a"
INK = "#161616"
SOFT_GREY = "#8f9ca8"


def render(output_path: Path = OUTPUT_PATH) -> None:
    """Render the full figure and save it to *output_path*."""
    fig = plt.figure(figsize=(18, 9), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")

    def rounded(x, y, w, h, fc="white", ec=INK, lw=1.25, radius=8, z=2):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.015,rounding_size={radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def text(
        x,
        y,
        value,
        size=14,
        weight="normal",
        style="normal",
        ha="center",
        va="center",
        color=INK,
        z=8,
        rotation=0,
    ):
        return ax.text(
            x,
            y,
            value,
            fontsize=size,
            fontweight=weight,
            fontstyle=style,
            ha=ha,
            va=va,
            color=color,
            zorder=z,
            rotation=rotation,
        )

    def line(points, color=INK, lw=1.25, linestyle="-", z=3):
        xs, ys = zip(*points)
        ax.plot(
            xs,
            ys,
            color=color,
            lw=lw,
            ls=linestyle,
            solid_capstyle="round",
            zorder=z,
        )

    def arrow(start, end, color=INK, lw=1.25, scale=10, z=5, style="-|>"):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def unet(x, y, w, h, chips=True, chip_font=15):
        points = [
            (x, y),
            (x + w * 0.50, y + h * 0.22),
            (x + w, y),
            (x + w, y + h),
            (x + w * 0.50, y + h * 0.78),
            (x, y + h),
        ]
        ax.add_patch(
            Polygon(
                points,
                closed=True,
                facecolor=BLUE_FILL,
                edgecolor=BLUE_EDGE,
                linewidth=1.7,
                zorder=3,
            )
        )
        if chips:
            chip_w, chip_h = w * 0.27, 38
            rounded(x + 8, y + h * 0.36, chip_w, chip_h, "#fff9ef", ORANGE, 1.3, 6, 5)
            rounded(
                x + w - chip_w - 8,
                y + h * 0.36,
                chip_w,
                chip_h,
                "#fff9ef",
                ORANGE,
                1.3,
                6,
                5,
            )
            text(x + 8 + chip_w / 2, y + h * 0.36 + chip_h / 2, r"$Q\;K\;V$", chip_font, style="italic")
            text(
                x + w - chip_w / 2 - 8,
                y + h * 0.36 + chip_h / 2,
                r"$Q\;K\;V$",
                chip_font,
                style="italic",
            )
            text(x + w / 2, y + h / 2, r"$\cdots$", 18)
        return points

    def math_box(x, y, w, h, value, fc=GREY, ec=GREY_EDGE, size=16, z=5):
        rounded(x, y, w, h, fc, ec, 1.1, 6, z)
        text(x + w / 2, y + h / 2, value, size=size, style="italic", z=z + 2)

    def condition(x, y, value, caption=None, prompt=None):
        rounded(x, y, 58, 42, LILAC, PURPLE, 1.25, 6, 4)
        text(x + 29, y + 21, value, 17, style="italic")
        if caption:
            text(x + 29, y + 58, caption, 12.5, style="italic")
        if prompt:
            text(x + 29, y + 55, prompt, 11.5)

    def circle_node(x, y, mark=None, radius=17):
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor="white",
                edgecolor=INK,
                linewidth=1.35,
                zorder=6,
            )
        )
        if mark:
            text(x, y - 1, mark, 20, z=8)

    def noise_grid(x, y, size=46):
        ax.add_patch(
            Rectangle(
                (x, y),
                size,
                size,
                facecolor="#d4d8db",
                edgecolor=GREY_EDGE,
                lw=1,
                zorder=4,
            )
        )
        for index in range(1, 3):
            line(
                [(x + index * size / 3, y), (x + index * size / 3, y + size)],
                GREY_EDGE,
                0.8,
                z=5,
            )
            line(
                [(x, y + index * size / 3), (x + size, y + index * size / 3)],
                GREY_EDGE,
                0.8,
                z=5,
            )

    # Major regions.
    rounded(5, 5, 1790, 475, GREEN_BG, GREEN_EDGE, 1.3, 12, 0)
    rounded(1470, 56, 310, 408, RED_BG, "#e8aaaa", 1.0, 7, 1)
    rounded(5, 495, 900, 398, PEACH_BG, ORANGE, 1.25, 10, 1)
    rounded(915, 495, 455, 398, PEACH_BG, ORANGE, 1.25, 10, 1)
    rounded(1380, 495, 415, 398, PEACH_BG, ORANGE, 1.25, 10, 1)

    text(25, 28, "Training the compositional adapter", 26, weight="bold", ha="left")
    text(25, 72, "1. Run once, before training", 18, weight="bold", ha="left")
    text(650, 72, "2. Every training step", 18, weight="bold", ha="left")
    text(1490, 72, "3. Loss", 18, weight="bold", ha="left")
    line([(630, 58), (630, 462)], INK, 1.15, z=2)

    # 1. Run once, before training.
    condition(48, 112, r"$c_{ab}$", "the joint prompt")
    condition(48, 208, r"$\varnothing$", "the empty prompt")
    text(285, 112, "SDXL denoising UNet", 15)
    unet(165, 138, 245, 166)
    text(287, 325, "frozen, no adapter", 12.5, style="italic")
    text(287, 346, "one batched pass per step", 12.5, style="italic")
    arrow((106, 133), (165, 177))
    arrow((106, 229), (165, 222))

    rounded(438, 132, 166, 182, "#edf8eb", GREEN_EDGE, 1.2, 7, 3)
    text(521, 151, "the cache", 14, weight="bold")
    math_box(450, 168, 142, 52, r"$\tilde{\varepsilon}_{\mathrm{joint}}$", size=16)
    math_box(450, 240, 142, 52, r"$\varepsilon_{\varnothing}$", size=17)
    arrow((410, 190), (438, 190))
    arrow((410, 268), (438, 268))

    noise_grid(62, 365, 48)
    text(48, 354, r"$x_t$", 18, style="italic")
    text(55, 426, "a noisy latent", 12.5, style="italic", ha="left")
    text(55, 447, "from the PoE sampling trajectory", 12.5, style="italic", ha="left")
    line([(110, 389), (520, 389), (520, 314)], INK, 1.25, z=4)
    arrow((520, 334), (520, 314))
    text(390, 374, r"$x_t$", 17, style="italic")
    text(500, 414, "50 DDIM steps, stored once", 12, style="italic")
    text(500, 433, "and read back", 12, style="italic")

    # 2. Every training step. Conditioning and latent paths are separated.
    condition(638, 115, r"$c_a$", prompt="“a cat”")
    condition(638, 194, r"$c_b$", prompt="“a dog”")
    condition(638, 273, r"$\varnothing$", prompt="empty")

    text(830, 96, "SDXL denoising UNet,", 13.5, color=ORANGE)
    text(830, 115, "adapter attached (trainable)", 13.5, color=ORANGE)
    unet(760, 138, 140, 166, chip_font=12)
    arrow((696, 136), (760, 174))
    arrow((696, 215), (760, 220))
    arrow((696, 294), (760, 266))

    line([(642, 342), (735, 342)], INK, 1.25, z=4)
    arrow((735, 342), (760, 286), lw=1.25, scale=10)
    text(715, 328, r"$x_t$", 17, style="italic")
    text(680, 360, "three passes,", 11.5, style="italic")
    text(680, 380, "same latent", 11.5, style="italic")
    text(680, 400, "and timestep", 11.5, style="italic")

    math_box(915, 124, 58, 42, r"$\varepsilon_a$", size=16)
    math_box(915, 204, 58, 42, r"$\varepsilon_b$", size=16)
    math_box(915, 264, 58, 42, r"$\varepsilon_{\varnothing}$", size=16)
    arrow((900, 145), (915, 145))
    arrow((900, 225), (915, 225))
    arrow((900, 285), (915, 285))

    # Plain guidance nodes; the adjacent boxes state the guidance formulae.
    circle_node(1005, 145, None, 15)
    circle_node(1005, 225, None, 15)
    arrow((973, 145), (990, 145))
    arrow((973, 225), (990, 225))
    line([(973, 285), (984, 285), (984, 161)], INK, 1.05, z=4)
    arrow((984, 161), (995, 154), lw=1.05, scale=8)
    line([(973, 285), (994, 285), (994, 241)], INK, 1.05, z=4)
    arrow((994, 241), (999, 239), lw=1.05, scale=8)

    math_box(
        1027,
        118,
        214,
        54,
        r"$\tilde{\varepsilon}_a=\varepsilon_{\varnothing}+w(\varepsilon_a-\varepsilon_{\varnothing})$",
        size=12.6,
    )
    math_box(
        1027,
        198,
        214,
        54,
        r"$\tilde{\varepsilon}_b=\varepsilon_{\varnothing}+w(\varepsilon_b-\varepsilon_{\varnothing})$",
        size=12.6,
    )
    arrow((1020, 145), (1027, 145))
    arrow((1020, 225), (1027, 225))

    circle_node(1280, 185, r"$+$", 17)
    circle_node(1340, 185, r"$-$", 17)
    arrow((1241, 145), (1268, 174), lw=1.1)
    arrow((1241, 225), (1268, 196), lw=1.1)
    arrow((1297, 185), (1323, 185), lw=1.2)
    line([(973, 285), (1010, 285), (1010, 270), (1246, 270)], INK, 1.05, z=3)
    line([(1246, 270), (1246, 160), (1323, 160)], INK, 1.05, z=3)
    arrow((1310, 160), (1323, 172), lw=1.05, scale=9)

    # The two loss operands sit side by side, each one short hop from the loss box.
    math_box(
        1240,
        296,
        210,
        48,
        r"$\tilde{\varepsilon}_{\mathrm{PoE}}=\tilde{\varepsilon}_a+\tilde{\varepsilon}_b-\varepsilon_{\varnothing}$",
        size=13.6,
    )
    arrow((1345, 202), (1345, 296), lw=1.2, scale=10)

    math_box(
        1240,
        366,
        210,
        48,
        r"$\tilde{\varepsilon}_{\mathrm{joint}}$",
        fc="#edf8eb",
        ec=GREEN_EDGE,
        size=16,
    )

    text(1030, 358, r"guidance weight $w=7.5$;", 11.5, style="italic")
    text(1030, 377, "the empty-prompt prediction", 11.5, style="italic")
    text(1030, 396, "is subtracted raw", 11.5, style="italic")

    # The cached target is read back along the foot of the region.
    line([(604, 260), (608, 260), (608, 470), (1200, 470), (1200, 390)], INK, 1.25, z=4)
    arrow((1200, 390), (1240, 390), lw=1.25, scale=10)
    text(1035, 456, "read back from the cache", 11.8, style="italic")

    # 3. Loss.
    rounded(1492, 109, 266, 119, "#ffe8e8", RED_EDGE, 1.2, 7, 3)
    text(
        1625,
        168,
        r"$\mathcal{L}=\Vert\tilde{\varepsilon}_{\mathrm{PoE}}-\tilde{\varepsilon}_{\mathrm{joint}}\Vert_{2}^{2}$",
        19,
        style="italic",
    )

    # Both loss operands terminate on the loss box, each by one short hop.
    line([(1450, 320), (1470, 320), (1470, 190)], INK, 1.1, z=4)
    arrow((1470, 190), (1492, 190), lw=1.1, scale=9)
    line([(1450, 390), (1480, 390), (1480, 214)], INK, 1.1, z=4)
    arrow((1480, 214), (1492, 214), lw=1.1, scale=9)

    arrow((1625, 228), (1625, 276), lw=1.4, scale=12)
    rounded(1510, 276, 230, 122, "#fff6f6", RED_EDGE, 1.2, 7, 3)
    text(1625, 322, "update adapter", 16)
    text(1625, 350, "(LoRA weights only)", 15)

    # Exactly three local orange dashed callout pairs.
    dash = (0, (5, 4))
    # 1: the only pair leaving the green region, straight down to the left border.
    line([(760, 304), (760, 495)], ORANGE, 1.45, dash, z=2.2)
    line([(900, 304), (900, 495)], ORANGE, 1.45, dash, z=2.2)
    # 2: Q K V in the left panel to the middle panel's top border.
    line([(700, 590), (915, 495)], ORANGE, 1.45, dash, z=2.2)
    line([(790, 590), (1005, 495)], ORANGE, 1.45, dash, z=2.2)
    # 3: W_Q in the middle panel to the right panel's top border.
    line([(1120, 550), (1380, 495)], ORANGE, 1.45, dash, z=2.2)
    line([(1180, 550), (1440, 495)], ORANGE, 1.45, dash, z=2.2)

    # Bottom left: SDXL overview.
    text(20, 518, "SDXL denoising UNet (overview)", 17.5, weight="bold", ha="left")
    noise_grid(45, 644, 42)
    text(65, 624, r"$x_t$", 16, style="italic")
    text(66, 703, "(noisy latent)", 11.5, style="italic")
    unet(120, 580, 680, 154, chips=False)
    arrow((88, 665), (120, 665))

    text(460, 545, "(skip connections)", 11.5)
    line([(205, 558), (700, 558)], SOFT_GREY, 1.2, (0, (3, 3)), z=4)
    arrow((687, 558), (700, 558), color=SOFT_GREY, lw=1.2, scale=8)
    line([(235, 571), (670, 571)], SOFT_GREY, 1.2, (0, (3, 3)), z=4)
    arrow((657, 571), (670, 571), color=SOFT_GREY, lw=1.2, scale=8)

    rounded(160, 625, 90, 40, "#fff9ef", ORANGE, 1.3, 6, 6)
    text(205, 645, r"$Q\;K\;V$", 15, style="italic")
    rounded(700, 590, 90, 40, "#fff9ef", ORANGE, 1.3, 6, 6)
    text(745, 610, r"$Q\;K\;V$", 15, style="italic")
    text(460, 655, r"$\cdots$", 18)

    arrow((800, 665), (820, 665))
    text(845, 625, r"$\hat{\varepsilon}$", 17, style="italic")
    noise_grid(825, 648, 40)
    text(845, 706, "(noise prediction)", 11.5, style="italic")

    rounded(419, 742, 82, 48, LILAC, PURPLE, 1.2, 6, 5)
    text(460, 766, r"$\tau_{\theta}$", 17, style="italic")
    line([(431, 742), (431, 704), (205, 704), (205, 665)], INK, 1.1, z=4)
    arrow((205, 685), (205, 665), lw=1.1, scale=8)
    line([(489, 742), (489, 690), (745, 690), (745, 630)], INK, 1.1, z=4)
    arrow((745, 650), (745, 630), lw=1.1, scale=8)
    text(455, 814, "the text prompt enters here and nowhere else", 12.7, style="italic")
    text(455, 858, "70 cross-attention blocks × 3 projections = 210 adapted layers", 14.2)

    # Bottom middle: W_Q, W_K, W_V stack, attention, then W_O with clear gaps.
    text(970, 518, "One cross-attention block", 17.5, weight="bold", ha="left")
    text(1075, 574, r"$x$", 17, style="italic")
    text(1075, 594, "(latent)", 11.2, style="italic")
    text(1060, 677, r"$c$", 17, style="italic")
    text(1015, 697, "(text embedding)", 11.2, style="italic")

    rounded(1120, 550, 60, 48, "#fff6e8", ORANGE, 1.25, 6, 6)
    text(1150, 566, r"$W_Q$", 15, style="italic")
    text(1150, 586, "(adapted)", 10.5, color=ORANGE)
    rounded(1120, 630, 60, 48, "#fff6e8", ORANGE, 1.25, 6, 6)
    text(1150, 646, r"$W_K$", 15, style="italic")
    text(1150, 666, "(adapted)", 10.5, color=ORANGE)
    rounded(1120, 710, 60, 48, "#fff6e8", ORANGE, 1.25, 6, 6)
    text(1150, 726, r"$W_V$", 15, style="italic")
    text(1150, 746, "(adapted)", 10.5, color=ORANGE)

    rounded(1190, 610, 122, 136, BLUE_FILL, BLUE_EDGE, 1.3, 7, 5)
    text(1251, 639, "multi-head attention", 9.6)
    text(
        1251,
        672,
        r"$\operatorname{softmax}(QK^{\top}/\sqrt{d_{\mathrm{head}}})$",
        7.6,
        style="italic",
    )
    text(1251, 715, "(frozen)", 10.4, color=BLUE_EDGE)
    rounded(1320, 641, 42, 58, "#e8f3ff", BLUE_EDGE, 1.25, 6, 5)
    text(1341, 659, r"$W_O$", 12.0, style="italic")
    text(1341, 683, "(frozen)", 8.2, color=BLUE_EDGE)

    arrow((1095, 574), (1120, 574))
    line([(1080, 677), (1100, 677), (1100, 654), (1120, 654)], INK, 1.1, z=4)
    arrow((1100, 654), (1120, 654), lw=1.1, scale=8)
    line([(1100, 677), (1100, 734), (1120, 734)], INK, 1.1, z=4)
    arrow((1100, 734), (1120, 734), lw=1.1, scale=8)
    arrow((1180, 574), (1195, 636), lw=1.1, scale=8)
    arrow((1180, 654), (1195, 675), lw=1.1, scale=8)
    arrow((1180, 734), (1195, 716), lw=1.1, scale=8)
    arrow((1312, 669), (1320, 669))
    arrow((1362, 669), (1368, 669))

    text(1142, 799, "three of the four projections are adapted;", 12.3, style="italic")
    text(
        1142,
        823,
        "the output projection, self-attention and feed-forward are not",
        11.3,
        style="italic",
    )

    # Bottom right: low-rank A and B remain visibly narrower than W.
    text(1410, 518, "One adapted projection (LoRA)", 17.5, weight="bold", ha="left")
    text(1410, 610, r"$x$", 17, style="italic")
    line([(1430, 610), (1450, 610), (1450, 569), (1475, 569)], INK, 1.2, z=4)
    arrow((1460, 569), (1475, 569), lw=1.2, scale=9)
    line([(1450, 610), (1450, 645), (1475, 645)], INK, 1.2, z=4)
    arrow((1460, 645), (1475, 645), lw=1.2, scale=9)

    rounded(1475, 542, 120, 56, "#e8f3ff", BLUE_EDGE, 1.25, 6, 5)
    text(1535, 558, r"$W$", 16, style="italic")
    text(1535, 583, "(frozen)", 11, color=BLUE_EDGE)
    rounded(1475, 619, 55, 54, "#fff6e8", ORANGE, 1.25, 6, 5)
    text(1502.5, 635, r"$A$", 16, style="italic")
    text(1502.5, 659, "(trained)", 9.8, color=ORANGE)
    rounded(1570, 619, 55, 54, "#fff6e8", ORANGE, 1.25, 6, 5)
    text(1597.5, 635, r"$B$", 16, style="italic")
    text(1597.5, 659, "(trained)", 9.8, color=ORANGE)
    arrow((1530, 646), (1570, 646))

    circle_node(1710, 609, r"$+$", 21)
    arrow((1595, 570), (1710, 570), lw=1.15)
    line([(1710, 570), (1710, 588)], INK, 1.15, z=4)
    arrow((1625, 646), (1710, 630), lw=1.15)
    arrow((1731, 609), (1760, 609), lw=1.2)
    text(1775, 609, r"$h$", 18, style="italic")

    text(1590, 710, r"$h=Wx+\frac{\alpha}{r}\,BAx$", 20, style="italic")
    text(1445, 758, r"$W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$", 11.5, style="italic")
    text(1590, 758, r"$A\in\mathbb{R}^{r\times d_{\mathrm{in}}}$", 11.5, style="italic")
    text(1725, 758, r"$B\in\mathbb{R}^{d_{\mathrm{out}}\times r}$", 11.5, style="italic")
    text(1590, 796, r"$r=8$ (single pair), $r=32$ (pooled)", 12.5, style="italic")
    text(
        1590,
        832,
        r"$d_{\mathrm{in}}$ is the latent width for $W_Q$, the text width for $W_K$ and $W_V$",
        9.6,
        style="italic",
    )
    text(1590, 862, r"$\alpha=r$, so the scale is 1", 12.0, style="italic")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    render()
