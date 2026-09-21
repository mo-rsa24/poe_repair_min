# Figure specification: sampling with the compositional adapter

A single method figure for an ICLR paper. Landscape, 2:1. It is the companion to the training
figure, and a rendered copy of that figure is attached as the style reference. Match its visual
register exactly: pale green banner region with a bold title at top left, numbered sub-region
headings in bold, thin dark strokes, rounded rectangles, pale blue UNet trapezoids with orange
`Q K V` chips, pale lilac conditioning boxes, pale grey prediction boxes, small serif italic
captions, orange dashed callout lines.

## What this figure has to say

Sampling runs the same three-pass combination the adapter was trained on, with two things removed
and one added. Removed: the joint prompt and the cache, which exist only in training. Added: the
sampler loop that advances the latent, and a single decode at the end.

The reader should finish the figure able to say: at sampling time the model is only ever shown one
concept at a time, and the picture of both concepts comes out anyway.

## Layout: a green region on top, two panels beneath

### Top region, titled "Sampling with the adapter"

Three numbered sub-regions left to right, divided by thin vertical rules, exactly as the reference
figure divides its three.

**1. Start.** A small pale grey grid of squares labelled `x_T`, with two small italic lines
beneath: "pure Gaussian noise" and "one seed, one sample". To its right a thin arrow into
sub-region 2.

**2. One sampling step, repeated.** This is the widest sub-region and holds the loop.

Three pale lilac conditioning boxes stacked vertically on the left, each with its quoted prompt in
small plain type centred beneath the box:

- `c_a`, beneath it "a cat"
- `c_b`, beneath it "a dog"
- `\varnothing`, beneath it "empty"

A small plain label beneath the stack: "three passes, one latent, one timestep".

One pale blue UNet, drawn as in the reference: descending trapezoid, waist, ascending trapezoid,
with two orange-outlined `Q K V` chips inside and `\cdots` between them. Above it, in small orange
type on two lines: "SDXL denoising UNet," / "adapter attached (frozen at sampling)". The current
latent `x_t` enters it on its own clean arrow, visibly separate from the three conditioning arrows.

Three thin arrows leave the UNet into three pale grey boxes stacked in the same order:
`\varepsilon_a`, `\varepsilon_b`, `\varepsilon_{\varnothing}`.

Then the same arithmetic the training figure shows, drawn the same way:

- `\varepsilon_a` and `\varepsilon_{\varnothing}` meet at a small plain circled node, producing a
  box holding `\tilde{\varepsilon}_a = \varepsilon_{\varnothing} + w(\varepsilon_a - \varepsilon_{\varnothing})`
- `\varepsilon_b` and `\varepsilon_{\varnothing}` meet at a second such node, producing
  `\tilde{\varepsilon}_b = \varepsilon_{\varnothing} + w(\varepsilon_b - \varepsilon_{\varnothing})`
- those two and `\varepsilon_{\varnothing}` meet at a circled plus then a circled minus, producing
  one box holding `\tilde{\varepsilon}_{\mathrm{PoE}} = \tilde{\varepsilon}_a + \tilde{\varepsilon}_b - \varepsilon_{\varnothing}`

Beneath that box, in small italic: "guidance weight `w = 7.5`; the empty-prompt prediction is
subtracted raw".

**The DDIM step and the loop.** One arrow from `\tilde{\varepsilon}_{\mathrm{PoE}}` into a small
pale green box labelled "DDIM step", with `x_{t-1}` on its outgoing arrow. That arrow curves back
to the left and re-enters the UNet's latent input, forming a visible closed loop. Label the return
path, in small italic, "50 steps".

The loop is the structural point of the sub-region and must read as a loop at a glance.

**3. Decode.** After the loop closes, one arrow leaves at `x_0` into a pale lilac trapezoid
labelled "VAE decoder", wide edge toward the output. Out of it, one arrow into a thin-bordered box
captioned "the sampled image" in small plain type. Leave that box **empty**, with a small italic
caption beneath reading "a cat and a dog, from two single-concept prompts". Draw no picture inside
it.

Beneath sub-region 3, in small italic on two lines:

- "The joint prompt is never used."
- "No cache, no target, no gradients."

### Bottom panels: two, side by side

An orange dashed callout runs from the UNet in sub-region 2 straight down to the top border of the
**left** panel. A second orange dashed callout runs from a `Q K V` chip inside the left panel to the
top border of the **right** panel. Those are the only two callouts.

**Left panel, "Where the adapter sits".** The UNet drawn wide, with orange `Q K V` chips at two
depths, a pale lilac `\tau_\theta` box beneath reaching them with thin arrows, and beneath that, in
small italic, "the text prompt enters here and nowhere else". Beneath the panel: "70
cross-attention blocks × 3 projections = 210 adapted layers".

**Right panel, "What the adapter adds".** The low-rank path: an input `x` splitting into two paths
meeting at a circled plus whose output arrow is labelled `h`. Upper path through one wide pale blue
box `W`, tagged "frozen". Lower path through two **visibly much narrower** pale peach boxes `A`
then `B`, tagged "trained, now fixed". Beneath, in display mathematics:
`h = Wx + \frac{\alpha}{r} BAx`, and under that, in small italic, "the same weights for every
prompt pair; nothing is fitted at sampling time".

## The callout rules

Every callout is two straight dashed orange lines of equal dash length, running from the two upper
corners of its source box to the two upper corners of its destination panel. The two lines of a
pair are parallel. They are near vertical, never shallow. They do not cross each other and neither
passes over any box, arrow, label or caption. No callout begins at a dashed rectangle drawn around
its source, and none terminates inside a panel's interior: each ends on the destination panel's
top border.

## Mathematics

All mathematics is set as proper typeset mathematics, serif italic, in the style of LaTeX output.
Each item below is LaTeX source: render exactly what it produces and never transcribe the markup.

- `\tilde{\varepsilon}_a = \varepsilon_{\varnothing} + w(\varepsilon_a - \varepsilon_{\varnothing})`
- `\tilde{\varepsilon}_b = \varepsilon_{\varnothing} + w(\varepsilon_b - \varepsilon_{\varnothing})`
- `\tilde{\varepsilon}_{\mathrm{PoE}} = \tilde{\varepsilon}_a + \tilde{\varepsilon}_b - \varepsilon_{\varnothing}`
- `h = Wx + \frac{\alpha}{r} BAx`
- the standalone symbols `x_T`, `x_t`, `x_{t-1}`, `x_0`, `c_a`, `c_b`, `\varnothing`,
  `\varepsilon_a`, `\varepsilon_b`, `\varepsilon_{\varnothing}`, `\tau_\theta`, `x`, `Q`, `K`, `V`,
  `W`, `A`, `B`, `h`, `r`, `\alpha`, `w`

`\varepsilon` is the curly epsilon, never a Latin "e". `\varnothing` is the empty-set symbol and
must not read as a digit zero or a Greek phi. Every tilde is present, centred, and visibly a tilde
rather than a bar or a hat. The fraction `\alpha / r` is set with a real bar. Every subscript sits
below the baseline and carries no stray dot or hyphen.

## What must not appear

- No joint prompt, no `c_{ab}`, no "a cat and a dog" as a prompt string, no cache, no target, no
  loss, no `\mathcal{L}`, no backward arrow, no gradients, no word "training" except in the two
  captions that say what sampling does not do.
- No photograph, no rendered sample, no cat, no dog, no picture inside the output box.
- No word "dataset".
- No snowflake, lock, flame or any other icon standing for frozen or trained. Small italic words
  only.
- No optimizer, no learning rate, no parameter counts, no channel counts.
- No framework names: no `to_q`, `to_k`, `to_v`, `attn2`, `nn.Linear`, no PyTorch, no diffusers.
- No bare `d`, `d_c` or `d_1` as a matrix dimension.
- No watermark, no logo, no URL, no attribution mark.

## Fidelity check the figure has to pass

1. The sampling step is a visibly closed loop: the arrow out of the DDIM step returns to the UNet's
   latent input and is labelled "50 steps".
2. Exactly three conditioning boxes, each on its own arrow, and exactly three prediction boxes. The
   latent enters on a fourth, separate arrow.
3. A circled plus and a circled minus both appear, in that order, producing the
   `\tilde{\varepsilon}_{\mathrm{PoE}}` box. "guidance weight `w = 7.5`" is on the canvas.
4. Every epsilon downstream of guidance carries a tilde; both `\varepsilon_{\varnothing}` terms
   carry none.
5. The joint prompt appears nowhere, and no loss, target, cache or backward arrow appears anywhere.
6. The decode happens exactly once, after the loop, never inside it.
7. The output box is empty, with its caption beneath it and no picture inside.
8. Exactly two callouts, each two parallel near-vertical dashed lines, neither crossing the other
   nor passing over any content, each ending on a panel's top border.
9. `A` and `B` are visibly much narrower than `W`, and the two paths meet at a plus.
10. Every equation matches its LaTeX source and no subscript carries a stray dot.
11. Nothing overflows its box and no two text runs overlap.

Re-render if any check fails. Every character of mathematics has to survive.
