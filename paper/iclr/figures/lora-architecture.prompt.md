# Figure specification: the LoRA adapter's architecture

A single architecture figure for an ICLR paper. Wide canvas, roughly 2:1.

## The visual register, which matters more than anything else here

This must look like a figure from a diffusion-models paper, specifically like the system
figure in "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al.,
2022). That is the target and the whole style brief. Restrained, matte, compact, drawn
rather than designed.

What that means concretely:

- Thin uniform strokes throughout, roughly one pixel, dark grey or black. No thick borders.
- Flat matte fills in desaturated pastels: pale blue, pale green, pale peach, pale lilac,
  pale grey. Low saturation. Nothing is vivid and nothing is filled with a strong colour.
- Small type. The figure is information-dense and read close up, not from across a room.
- Mathematics set in serif italic, in the manner of Computer Modern: `x`, `W`, `\alpha`,
  `\varepsilon_\theta`, `\tau_\theta`.
- Encoders and decoders drawn as trapezoids, wide edge toward the data, narrow edge toward
  the latent.
- Regions grouped by thin-bordered rounded rectangles with a very light tint and a small
  plain label sitting inside the top edge.
- A zoom drawn as two dashed lines in a warm colour running from a region in the main
  diagram to an enlarged panel beside it.

What it must not look like: a slide, a poster, or a product diagram. So no icons of any
kind, no flames, no snowflakes, no locks, no emoji. No bold coloured banners. No chunky
rounded boxes. No drop shadows, glows, gradients, glossy fills or 3-D. No large display
type. Say "frozen" and "trained" in small italic words where those facts are needed.

## Layout: three regions on one row, left to right, each narrower than the last

The canvas divides into three thin-bordered regions of roughly equal height, joined by two
dashed peach guide lines that read as successive magnifications: the network, then one
attention block inside it, then one projection inside that.

## 1. Left region: where the adapter lives in the network

Label inside the top edge: "SDXL denoising UNet, frozen".

Draw the UNet as a pale blue descending trapezoid, a narrow waist, and a pale blue
ascending trapezoid. A small grid of coloured squares enters at the left labelled `x_t` and
another leaves at the right labelled `\hat{\varepsilon}`. Faint dashed horizontal lines
span the waist, labelled "skip connections" once in small italic.

Four small pale peach boxes sit along the inside of the U at successive depths, two
descending and two ascending, each holding "cross-attn" in small plain type. A pale lilac
trapezoid labelled `\tau_\theta` sits at the lower right of the region with thin arrows
reaching every one of them, and "the text prompt enters here and nowhere else" beneath it
in small italic.

Beneath the region, in small plain type: "70 cross-attention blocks".

## 2. Middle region: one attention block, opened

Two dashed peach lines run from one of the four peach boxes to the corners of this region.
Label inside its top edge: "one cross-attention block".

Inside, left to right with thin arrows: a pale grey box `x` (the latent) and a pale grey
box `c` (the text embedding) on the left; then three stacked pale peach boxes labelled
`W_Q`, `W_K`, `W_V`, each carrying a small tag reading "adapted" in small plain type; then
a pale grey box "softmax attention, frozen"; then a pale grey box `W_O` labelled "frozen".
`x` feeds `W_Q`; `c` feeds `W_K` and `W_V`.

Beneath the region, in small italic: "three of the four projections are adapted; the output
projection, the self-attention and the feed-forward layers are not".

## 3. Right region: one adapted projection, exactly

Two dashed peach lines run from the `W_Q` box to the corners of this region. Label inside
its top edge: "one adapted projection".

Inside, two parallel paths from a single input dot on the left to a small circled plus on
the right:

- the upper path passes through one wide pale grey rectangle labelled `W`, with "frozen" in
  small italic beneath it;
- the lower path passes through two visibly much thinner pale peach rectangles labelled `A`
  then `B`, with "trained" in small italic beneath them.

The circled plus is a plus and must never be drawn as a minus. One arrow leaves it to the
right labelled `h`.

Directly beneath, on one line in serif italic: `h = W x + \frac{\alpha}{r} B A x`.

Beneath that, three short lines of small plain type:

- `W \in \mathbb{R}^{d \times d}`, `A \in \mathbb{R}^{r \times d}`, `B \in \mathbb{R}^{d \times r}`
- "rank `r = 8` for a single pair, `r = 32` pooled"
- "`\alpha = r`, so the scale is 1"

## Mathematics

All mathematics is set as proper typeset mathematics, serif italic, in the style of LaTeX
output. Each item below is LaTeX source; render exactly what it produces, and never
transcribe the markup itself.

- `h = W x + \frac{\alpha}{r} B A x`
- `W \in \mathbb{R}^{d \times d}`, `A \in \mathbb{R}^{r \times d}`, `B \in \mathbb{R}^{d \times r}`
- the standalone symbols `x_t`, `\hat{\varepsilon}`, `\tau_\theta`, `x`, `c`, `W_Q`, `W_K`,
  `W_V`, `W_O`, `W`, `A`, `B`, `h`, `r`, `\alpha`

`\varepsilon` is the curly epsilon, not a Latin "e". Every subscript sits below the baseline
and contains no stray dot or hyphen. `\mathbb{R}` is the blackboard-bold capital R. The
fraction `\alpha / r` is set as a real fraction with the bar, not as the three characters.

## What must not appear

- No snowflake, lock, flame, or any other icon standing in for frozen or trained.
- No optimizer box, no "AdamW", no learning rate, no parameter counts, no channel counts.
- No framework or class names: no `to_q`, no `to_k`, no `to_v`, no `nn.Linear`, no PyTorch.
- No decoded images, no photographs, no cartoon animals.
- No gradient-flow arrows fanning across the figure, and no loss term: this figure is the
  architecture, not the training objective.
- No watermark, no logo, no URL, no attribution mark anywhere on the canvas.
- No "Step 1 / Step 2 / Step 3" titles. The two dashed callouts carry the order.

## Fidelity check the figure has to pass

1. The overall impression is a restrained academic diffusion-paper figure: thin strokes,
   pale flat fills, small serif math, no icons, no banners.
2. Three regions on one row, joined by two dashed callouts, each a magnification of one
   thing in the region to its left.
3. `A` and `B` are visibly much thinner than `W`, and the two paths meet at a plus.
4. Exactly three projections carry the "adapted" tag; `W_O`, the softmax block and the UNet
   itself are labelled frozen.
5. Every equation matches its LaTeX source, the fraction has a bar, and no subscript
   carries a stray dot.
6. Nothing overflows its box and no two text runs overlap.

Re-render if any check fails; every character of mathematics has to survive.
