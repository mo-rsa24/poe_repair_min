# Figure specification: training the composition adapter

A single methodology figure for an ICLR paper, 16:9.

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
- Mathematics set in serif italic, in the manner of Computer Modern: `x_t`, `z_T`,
  `\varepsilon_\theta`, `\tau_\theta`.
- Encoders and decoders drawn as trapezoids, wide edge toward the data, narrow edge toward
  the latent.
- Regions grouped by thin-bordered rounded rectangles with a very light tint and a small
  plain label sitting inside the top edge.
- A zoom drawn as two dashed lines in a warm colour running from a region in the main
  diagram down to an enlarged panel beneath it.

What it must not look like: a slide, a poster, or a product diagram. So no icons of any
kind, no flames, no snowflakes, no locks, no emoji. No bold coloured banners. No chunky
rounded boxes. No drop shadows, glows, gradients, glossy fills or 3-D. No large display
type. Say "frozen" and "trainable" in small italic words where those facts are needed.

## Layout: a main row, with one zoom beneath it

The upper two fifths of the canvas hold the main flow, left to right. The lower three
fifths hold a single enlarged panel, connected to the main flow by two dashed peach guide
lines that clearly read as a magnification.

## Upper part: the main flow

Left to right, joined by thin arrows.

**Prompts.** A small thin-bordered box labelled "one concept pair" in small plain type,
holding four rows, each a literal string, none blank:

- `"a cat"`
- `"a dog"`
- `"a cat and a dog"`
- `""` followed by (empty)

Beneath it in small italic: "the empty prompt is shared, for guidance".

**Text encoder.** A pale lilac trapezoid labelled `\tau_\theta`, taking the prompts in.

**The sampler.** A thin-bordered pale green region labelled "50-step DDIM, frozen" inside
its top edge. Inside it, a small pale blue U-shaped block labelled
`\varepsilon_\theta`, with a loop arrow around it annotated "repeat 50 times", and a small
grid of coloured squares to its left labelled `x_t`, in the manner of a latent tensor.

**Two predictions.** Leaving the green region, two thin-bordered boxes stacked vertically:

- pale blue, holding `\varepsilon_{\mathrm{PoE}} = \varepsilon_{\text{cat}} + \varepsilon_{\text{dog}} - \varepsilon_{\varnothing}`
- pale green, holding `\varepsilon_{\text{joint}}`

**The residual.** Both feed thin arrows into a small circle containing a minus sign. This
is a subtraction and must never be drawn as a plus. The joint prediction is the left
operand. One arrow leaves it into a pale peach box holding
`r_t = \varepsilon_{\text{joint}} - \varepsilon_{\mathrm{PoE}}`, with "the residual" in
small italic beneath. This box carries a slightly heavier border than its neighbours, which
is the only emphasis device used anywhere in the figure.

## Lower part: the zoom

Two dashed peach lines run from the `\varepsilon_\theta` block in the green region down to
the corners of a large thin-bordered pale peach region. Inside its top edge, the label
"Denoising UNet, adapter attached" in small plain type.

**The architecture.** Draw the UNet across the middle of this region: a pale blue
descending trapezoid on the left for the encoder, a narrow waist, a pale blue ascending
trapezoid on the right for the decoder. A small grid of coloured squares enters at the
left labelled `x_t` and another leaves at the right labelled `\hat{\varepsilon}`. Faint
dashed horizontal lines span the waist for skip connections, labelled "skip connections"
once in small italic.

**The attention blocks.** Four small pale yellow boxes sit along the inside of the U at
successive depths, two on the descending side and two on the ascending side. Each holds
three short rows reading `Q`, `K`, `V` in serif italic. Thin arrows from the text encoder
`\tau_\theta`, redrawn small at the lower right of this region, reach every one of them.

**The adapter.** Each `Q`, `K`, `V` row carries a small pale peach tag reading "LoRA" in
small plain type. No icon, just the tag.

**The low-rank inset.** In a corner of the peach region, a small detail: a pale grey
rectangle labelled `W` with a thin arrow through it, and in parallel a narrow path through
two small pale peach rectangles labelled `A` then `B`, both visibly much thinner than `W`.
The two paths meet at a small circled plus. Beside it, `h = Wx + BAx` in serif italic, and
"only A and B are trained" in small italic.

**The objective.** At the foot of the peach region, a short horizontal strip: a pale peach
box holding `\hat{r}_t` labelled "what the adapter supplies", a dashed connector, a pale
green box holding `r_t` labelled "the residual, cached", and both feeding a thin-bordered
box holding `\mathcal{L} = \lVert \hat{r}_t - r_t \rVert^2`. A single thin curved arrow
returns from the loss to the nearest LoRA tag, unlabelled.

## Mathematics

All mathematics is set as proper typeset mathematics, serif italic, in the style of LaTeX
output. Each item below is LaTeX source; render exactly what it produces, and never
transcribe the markup itself.

- `\varepsilon_{\mathrm{PoE}} = \varepsilon_{\text{cat}} + \varepsilon_{\text{dog}} - \varepsilon_{\varnothing}`
- `\varepsilon_{\text{joint}}`
- `r_t = \varepsilon_{\text{joint}} - \varepsilon_{\mathrm{PoE}}`
- `h = Wx + BAx`
- `\mathcal{L} = \lVert \hat{r}_t - r_t \rVert^2`
- the standalone symbols `x_t`, `\hat{\varepsilon}`, `\varepsilon_\theta`, `\tau_\theta`,
  `Q`, `K`, `V`, `W`, `A`, `B`

`\varepsilon` is the curly epsilon, not a Latin "e". `\varnothing` is the empty-set symbol
and must not read as a digit zero. Every subscript sits below the baseline and contains no
stray dot or hyphen. The hat on `\hat{r}_t` is present and centred, and is a hat, not a
bar. `\mathcal{L}` is a calligraphic capital L, not a script lowercase letter.

`\hat{r}_t` and `r_t` are different symbols and must be drawn differently, since the whole
point of the objective is that one approximates the other.

## What must not appear

- No cache or database cylinder, no disk icon, no optimizer box, no "AdamW", no learning
  rate, no rank number, no parameter counts.
- No diffusers class names, no channel counts, no `to_q` / `to_k` / `to_v` / `to_out`.
- No decoded images, no photographs, no cartoon animals. Prompts are text strings.
- No gradient-flow arrows fanning across the figure. One thin curved arrow is the whole of
  the backward path.
- No watermark, no logo, no URL, no attribution mark anywhere on the canvas.
- No step numbers and no "Step 1 / Step 2 / Step 3" titles. The flow carries the order.

## Fidelity check the figure has to pass

1. The overall impression is a restrained academic diffusion-paper figure: thin strokes,
   pale flat fills, small serif math, no icons, no banners.
2. The main flow sits above; one dashed callout leads to one enlarged panel below.
3. The node joining the two prediction boxes is a minus, never a plus.
4. Four prompts are listed, each holding its literal string, none blank.
5. The UNet is drawn as two trapezoids with a waist, with `Q K V` blocks inside it at four
   depths, every one reached by an arrow from `\tau_\theta`.
6. `A` and `B` are visibly thinner than `W`.
7. Every equation matches its LaTeX source. No subscript contains a stray dot.
   `\hat{r}_t` carries a hat and is visibly distinct from `r_t`.
8. Nothing overflows its box and no two text runs overlap.
