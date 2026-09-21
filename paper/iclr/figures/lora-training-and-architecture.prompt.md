# Figure specification: training the compositional adapter

A single method figure for an ICLR paper. Landscape, 2:1.

A reference image is supplied. It is a **previous draft of this same figure**, it is very close to
correct, and it is the style and layout target. Reproduce its visual register and its structure
faithfully: the pale green banner region on top holding three numbered sub-regions, the row of
three pale-peach bordered panels beneath it, the thin dark strokes, the rounded rectangles, the
pale blue UNet trapezoids with orange `Q K V` chips, the pale lilac conditioning boxes, the pale
grey prediction boxes, and the small serif italic mathematics.

Everything below is a correction to that draft. Where this specification and the reference
disagree, this specification wins, every time. The corrections fall into two groups: the dashed
callout lines, which are a drawing fault, and the mathematics and captions, which are factual
errors.

## Correction group 1: the dashed callout lines

In the reference, three dashed orange callouts leave from a cluster of points near the middle of
the top region and fan outward to the three panels below. They cross each other, they meet the
region border at shallow angles, and they travel across the whole canvas. That fan is the single
worst visual fault in the draft and fixing it is the main purpose of this render.

**Replace the fan with a cascade.** Each bottom panel is a magnification of the panel to its left,
not of the top region. So:

- The **first** callout, and the only one that leaves the top region, runs from the adapted UNet in
  sub-region 2 down to the top edge of the left-hand panel.
- The **second** callout runs from one `Q K V` chip **inside the left-hand panel** to the top edge
  of the middle panel.
- The **third** callout runs from the `W_Q` box **inside the middle panel** to the top edge of the
  right-hand panel.

**The rules every callout obeys.** Two straight dashed lines, in the reference's orange, of equal
dash length. They begin at the two upper corners of the source box and end at the two upper corners
of the destination panel. They are close to vertical, never shallow, and never longer than they
need to be. They do not cross each other, they do not cross any other callout, and neither line
passes over or under any box, arrow, label or caption on its way. No callout begins at a dashed
rectangle drawn around its source, and no callout begins in empty space.

Because the second and third callouts are short hops between adjacent panels, no dashed line
crosses the boundary between the top region and the panel row except the first one.

## Correction group 2: the mathematics

### Classifier-free guidance is missing and must appear

The reference combines raw predictions and writes the loss on raw predictions. The method guides
each single-concept prediction first, then combines, and the target is guided too. Every prediction
downstream of guidance carries a tilde, and the two empty-prompt terms never do, because the
empty-prompt prediction enters the combination raw.

In sub-region 2, after the three prediction boxes `\varepsilon_a`, `\varepsilon_b`,
`\varepsilon_{\varnothing}`, insert the guidance step before the combination:

- `\varepsilon_a` and `\varepsilon_{\varnothing}` meet at a small circled node producing a box
  holding `\tilde{\varepsilon}_a = \varepsilon_{\varnothing} + w(\varepsilon_a - \varepsilon_{\varnothing})`
- `\varepsilon_b` and `\varepsilon_{\varnothing}` meet at a second such node producing
  `\tilde{\varepsilon}_b = \varepsilon_{\varnothing} + w(\varepsilon_b - \varepsilon_{\varnothing})`

Then the combination as the reference already draws it, a circled plus followed by a circled minus,
but on the guided quantities:

`\tilde{\varepsilon}_{\mathrm{PoE}} = \tilde{\varepsilon}_a + \tilde{\varepsilon}_b - \varepsilon_{\varnothing}`

Beneath that box, in small italic: "guidance weight `w = 7.5`; the empty-prompt prediction is
subtracted raw".

The circled plus must read unambiguously as a plus and the circled minus as a minus.

### The loss carries tildes

The loss box in sub-region 3 reads

`\mathcal{L} = \lVert \tilde{\varepsilon}_{\mathrm{PoE}} - \tilde{\varepsilon}_{\text{joint}} \rVert_2^2`

and the cached target box is labelled `\tilde{\varepsilon}_{\text{joint}}`, not
`\varepsilon_{\text{joint}}`.

### The noisy latent does not come from a dataset

The reference captions `x_t` "a noisy latent (from dataset)". There is no dataset and no image data
anywhere in this method. `x_t` is seeded Gaussian noise carried forward along the product-of-experts
sampling trajectory. Replace the caption with, on two small italic lines:

- "a noisy latent"
- "from the PoE sampling trajectory"

The word "dataset" appears nowhere on the canvas.

### The cached target is read from the cache, not recomputed

In sub-region 2 the reference draws the `\tilde{\varepsilon}_{\text{joint}}` box fed by a line
branching off the `x_t` path, which reads as a fresh computation. It is read back from the cache.
Draw its incoming arrow starting at the cache box in sub-region 1 and label that arrow "read back".
No part of sub-region 2 computes the joint prediction.

### Sub-region 1 is a single forward pass

The reference draws `c_{ab}` and `\varnothing` entering the frozen UNet on separate arrows in a way
that suggests two passes. All prompts are evaluated in one batched forward pass. Keep both
conditioning boxes and both arrows, and add beneath the frozen UNet, in small italic, "one batched
pass per step".

### The matrix shapes must compose

The reference writes `W \in \mathbb{R}^{d \times d}`, `A \in \mathbb{R}^{r \times d}`,
`B \in \mathbb{R}^{d_c \times r}`. Those cannot all hold of one layer: `B` emits `d_c` while `W`
emits `d`, so the sum in `h` is undefined. Replace with

`W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}` , `A \in \mathbb{R}^{r \times d_{\text{in}}}` , `B \in \mathbb{R}^{d_{\text{out}} \times r}`

and add one small italic line beneath the existing rank line: "`d_{\text{in}}` is the latent width
for `W_Q`, the text width for `W_K` and `W_V`".

No bare `d`, `d_c` or `d_1` appears anywhere in that panel.

### The attention denominator is the per-head dimension

The middle panel reads `\operatorname{softmax}(QK^{\top}/\sqrt{d})`. It must read
`\operatorname{softmax}(QK^{\top}/\sqrt{d_{\text{head}}})`, and the box's first line says
"multi-head attention" rather than "Attention".

### One caption in the reference is garbled and must be reset

Beneath the `\tau_\theta` box in the left-hand panel, the reference renders the nonsense string
"three pasms, enters heret and nuwhere else". That line must read, in small serif italic:

"the text prompt enters here and nowhere else"

Regenerate that text properly as part of the figure. Do not patch it as a pixel overlay in a system
font; it must match its sibling captions in face, weight and size.

### One count is worth adding

Beneath the left-hand panel, where the reference reads "70 cross-attention blocks (LoRA on Q, K,
V)", extend it to "70 cross-attention blocks × 3 projections = 210 adapted layers".

## Correction group 3: the generated-outputs panel is removed

The reference carries a dashed sub-panel titled "Generated outputs" holding four pictures of cats
and dogs and the annotation "MSE(ε) high → MSE(ε) lower". **Delete that sub-panel entirely.** No
photograph, no rendered sample, no cartoon animal and no image of any kind appears anywhere on this
canvas, and neither does the MSE annotation. Let sub-region 3 use the freed space: the loss box and
the "update adapter (LoRA weights only)" box become larger and better spaced within the same pale
red region.

## Everything else the figure keeps

Keep, unchanged from the reference: the title "Training the compositional adapter"; the three
numbered sub-region headings "1. Run once, before training", "2. Every training step", "3. Loss";
the thin vertical rule between sub-regions 1 and 2; the cache box holding `\varepsilon_{\text{joint}}`
and `\varepsilon_{\varnothing}`; the caption "50 DDIM steps, stored once and read back"; the three
conditioning boxes `c_a`, `c_b`, `\varnothing` with their quoted prompts "a cat", "a dog", "empty";
the caption "three passes, same latent and timestep"; the orange label "SDXL denoising UNet, adapter
attached (trainable)"; the single backward arrow from the loss to "update adapter (LoRA weights
only)"; and all three bottom panels' internal contents apart from the corrections listed above.

## Mathematics

All mathematics is set as proper typeset mathematics, serif italic, in the style of LaTeX output.
Each item below is LaTeX source: render exactly what it produces and never transcribe the markup.

- `\tilde{\varepsilon}_a = \varepsilon_{\varnothing} + w(\varepsilon_a - \varepsilon_{\varnothing})`
- `\tilde{\varepsilon}_b = \varepsilon_{\varnothing} + w(\varepsilon_b - \varepsilon_{\varnothing})`
- `\tilde{\varepsilon}_{\mathrm{PoE}} = \tilde{\varepsilon}_a + \tilde{\varepsilon}_b - \varepsilon_{\varnothing}`
- `\mathcal{L} = \lVert \tilde{\varepsilon}_{\mathrm{PoE}} - \tilde{\varepsilon}_{\text{joint}} \rVert_2^2`
- `h = Wx + \frac{\alpha}{r} BAx`
- `W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}`, `A \in \mathbb{R}^{r \times d_{\text{in}}}`, `B \in \mathbb{R}^{d_{\text{out}} \times r}`
- `\operatorname{softmax}(QK^{\top}/\sqrt{d_{\text{head}}})`
- the standalone symbols `x_t`, `\hat{\varepsilon}`, `c_a`, `c_b`, `c_{ab}`, `\varnothing`,
  `\varepsilon_a`, `\varepsilon_b`, `\varepsilon_{\varnothing}`,
  `\tilde{\varepsilon}_{\text{joint}}`, `\tau_\theta`, `x`, `c`, `Q`, `K`, `V`, `W_Q`, `W_K`,
  `W_V`, `W_O`, `W`, `A`, `B`, `h`, `r`, `\alpha`, `w`

`\varepsilon` is the curly epsilon, never a Latin "e". `\varnothing` is the empty-set symbol and
must not read as a digit zero or a Greek phi. The tilde on `\tilde{\varepsilon}` is present,
centred, and visibly a tilde rather than a bar or a hat, since guided and unguided predictions are
different quantities and this figure separates them by that mark alone. `\mathbb{R}` is
blackboard-bold. The fraction `\alpha / r` is set with a real bar. Every subscript sits below the
baseline and carries no stray dot or hyphen.

## What must not appear

- No image of any kind: no photograph, no rendered sample, no cat, no dog, no "Generated outputs"
  panel, no MSE annotation beneath pictures.
- No word "dataset" anywhere.
- No word "teacher" and no word "student".
- No snowflake, lock, flame or any other icon standing for frozen or trainable. Small italic words
  only.
- No pixel space, no VAE encoder or decoder bar, no `z_t`, no `z_{t-1}`.
- No optimizer box, no "AdamW", no learning rate, no parameter counts, no channel counts.
- No framework names: no `to_q`, `to_k`, `to_v`, `attn2`, `nn.Linear`, no PyTorch, no diffusers.
- No bare `d`, `d_c` or `d_1` as a matrix dimension.
- No dashed rectangle drawn around a box as a callout source.
- No watermark, no logo, no URL, no attribution mark.

## Fidelity check the figure has to pass

1. There are exactly three dashed callouts. One leaves the top region into the left panel, one runs
   from inside the left panel into the middle panel, one runs from inside the middle panel into the
   right panel. No callout spans more than one step of that chain.
2. No dashed line crosses another dashed line, and no dashed line passes over any box, arrow,
   label or caption. Each is two straight, near-vertical lines from source corners to destination
   corners.
3. Guidance appears explicitly: two boxes holding the `\tilde{\varepsilon}_a` and
   `\tilde{\varepsilon}_b` definitions, and "guidance weight `w = 7.5`" on the canvas.
4. Every epsilon downstream of guidance carries a tilde, and both `\varepsilon_{\varnothing}` terms
   carry none. The loss reads
   `\mathcal{L} = \lVert \tilde{\varepsilon}_{\mathrm{PoE}} - \tilde{\varepsilon}_{\text{joint}} \rVert_2^2`.
5. A circled plus and a circled minus both appear in the combination, in that order.
6. `x_t` is captioned "a noisy latent" over "from the PoE sampling trajectory". The word "dataset"
   appears nowhere.
7. The arrow into `\tilde{\varepsilon}_{\text{joint}}` starts at the cache box and is labelled
   "read back". Nothing in sub-region 2 computes the joint prediction.
8. The three matrix shapes compose: `W` is `d_{\text{out}} \times d_{\text{in}}`, `A` is
   `r \times d_{\text{in}}`, `B` is `d_{\text{out}} \times r`. No bare `d`, `d_c` or `d_1` survives.
9. The attention denominator reads `\sqrt{d_{\text{head}}}` and its box says "multi-head attention".
10. The caption beneath `\tau_\theta` reads "the text prompt enters here and nowhere else", in the
    same face, weight and size as its sibling captions.
11. No image, photograph or animal appears anywhere, and the "Generated outputs" sub-panel is gone.
12. Every equation matches its LaTeX source, every tilde is a tilde, and no subscript carries a
    stray dot.
13. Nothing overflows its box and no two text runs overlap.

Re-render if any check fails. Every character of mathematics has to survive.
