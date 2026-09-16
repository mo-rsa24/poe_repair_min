# 🎨 Scene map: seeing the correction loss

**what this is** the scoping walk for an interactive scene that teaches how the product-of-experts
corrector is trained, and lets the reader see and compare every variation of its loss.

**run under** `/drip-picture-speak --reference`, opened 2026-09-16.

**the reference** four latent-diffusion figures the reader supplied: the Rombach latent-diffusion
figure (green `Latent Space` box, blue `Denoising U-Net` trapezoid, four orange `Q/KV`
cross-attention blocks, the `×(T−1)` loop, grey dashed skip connections, the switch and concat
glyphs), its full version with `Pixel Space` and the `Conditioning` column feeding `τ_θ`, the
video-LDM variant, and the SDXL `Prompt → Base → 128×128 Latent → Refiner → 1024×1024 Image`
pipeline. The scene is a descendant of these, not a redrawing.

## Table of contents

- [Position](#position)
- [The chunks](#the-chunks)
- [Assumed known](#assumed-known)
- [Held](#held)
- [Sources](#sources)

## Position

**Compiled and delivered.** The plan is in [SCENE_PLAN.md](SCENE_PLAN.md) and all sixteen of its
claims are built. The walk is closed.

**The scene** https://claude.ai/code/artifact/ae96d471-01cd-4764-b79d-41da5053230c

**The handoff did not happen, deliberately.** `compile` normally hands the plan to `picture-speak`
for the build. By the time the plan existed the sketchpad already implemented fourteen of the
sixteen claims, grounded in measured data and smoke-tested, so the two remaining claims were built
here and the page promoted instead. Rebuilding would have discarded verified work to satisfy a
process step. Recorded so a later reader does not go looking for a picture-speak build that was
never owed.

Chunk 2 is the spine, layout settled, drill sketched, with chunk 4 folded into it as a mode of the same picture.
Chunks 1 and 3 settled. Chunk 5 settled in shape, open on the compare control.

**Sketchpad** https://claude.ai/code/artifact/ae96d471-01cd-4764-b79d-41da5053230c
(one frame: the two legs, a six-step reveal, hover binding between every symbol and its box).
Source at `sketchpad.html` beside this file.

## The chunks

| # | name | what you see / what you touch | treatment | state | grounding |
|---|---|---|---|---|---|
| 1 | What sampling left on disk | five named tensors per denoising step, drawn as stacked 4-channel slabs; the cell as 50 of them; the disk figure | isometric volume slab, provenance badge per number | settled | real shapes and sizes; a decoded `x_t` image needs export |
| 2 | Two legs, one loss | leg A a disk read, leg B a live forward pass through one U-Net that fans into three; guidance on both sides before the subtraction; every symbol bound to its box both ways | frame-walk with cross-highlighting, term highlight on MathML | **open, sketched** | real shapes from `trainer.py` |
| 3 | What moves and what cannot | the same diagram, coloured frozen against trainable; the 0.38% stated on the face | one hue per meaning, bound across every pane | settled | real: 210 modules, 9,912,320 of 2,577,376,004 params |
| 4 | The direction the loss cannot see | now a *mode* of chunk 2's picture, not a second drawing: add the same tensor to the cat branch and the empty branch, watch the loss sit still | term ablation plus a draggable handle; predict before reveal | settled | the displacement arrow ships now; real renders slot under it when a render job produces them |
| 5 | The nine variations, side by side | every variation's full loss with bars marking what came from cache; pick any two and see what differs | target-function table, pins, shared page state | settled in shape, open on the compare control | real equations and code lines |

## Assumed known

Pointed at, never redrawn.

- **Why the latent is 128×128×4.** [the SDXL VAE architecture map](../../scenes/sdxl-vae-architecture-map/README.md)
  already answers this from a real shape trace of `AutoencoderKL`. Chunk 1 links to it and states
  the result in one line.
- **What the cached trajectories contain, as curves.**
  [the cached-runs page](../../scenes/what-the-cached-runs-already-show/README.md) already drives
  those arrays. Chunk 1 covers only the per-step tensor inventory, which that page does not.

## The descent under V0a: the empty branch

`deeper` on the penalty variation. Three pieces, because the reader's question was three questions.

| # | name | what you see / what you touch | state |
|---|---|---|---|
| 5a | What the empty prompt actually is | the literal empty string tokenised: start-of-text, then end-of-text to fill 77 slots; the same 77 x 2048 shape as any other prompt; beside it, what it renders on its own | **sketched, 3 steps** |
| 5b | Why it has the biggest coefficient | the composition expanded to `7.5a + 7.5b - 14u`, with the three coefficients drawn to scale as bars; drag the dial and watch the empty branch's bar outgrow both others | proposed |
| 5c | What a penalty can and cannot do | three named settings on one log scale, not a slider: mu 0, mu 10, mu to infinity, with the drift and the fit error plotted together | **built** |

**Settled in 5a.** The empty prompt is not a zero vector and not a special token. It is `""` through
the same CLIP tokenisers, giving token ids `[49406, 49407, 49407, ...]`, one non-pad token padded to
77, then a full `77 x 2048` sequence. Verified against `cross_trace.json` in
[the SDXL VAE map](../../scenes/sdxl-vae-architecture-map/README.md).

**Rendered, and the prediction was wrong.** Three renders on mscluster109 device 0, seed 9, 50
steps, about 11 seconds each, by `scripts/showcase/render_empty_prompt.py`; output under
`/datasets/mmolefe/poe_repair_min/outputs/showcase/empty_prompt/`.

The empty prompt was predicted to give something washed out and subjectless. Subjectless is right.
Washed out is wrong. It returns **a finished painterly tondo**: a circle of limb-like and fur-like
forms in muted grey and ochre, with a painted border, that never resolves into an animal.

Two things fall out that no amount of describing would have produced.

**The control kills the obvious objection.** `"a cat"` at guidance 1 gives a completely clean cat.
So the swirl is not an artefact of low guidance. It is what no words looks like.

**The empty render is the scaffold.** Put `""` and `"a cat"` at guidance 1 side by side on the same
seed: same circle, same palette, same ochre clumps in the same corners. The word did not build a new
picture, it resolved the middle of the existing one. The difference between those two images *is*
the guidance direction, made visible. And `"a cat"` at 7.5 throws the scaffold away: border gone,
tondo flattened, colour saturated.

That turned one owed frame into three, and it now earns the coefficient argument directly: the
empty branch is the thing every render is built on, and it enters the composition at -14.

## A treatment decision, corrected

The attention detail was first built as a table of shapes in the side rail. That was rejected on
sight and the reason is worth keeping: **a list of tensor shapes is valuable and unpicturable**. It
records what is true without letting anyone imagine it.

Replaced by a full-screen click-through where the matrices are drawn and multiplied: the prompt
filling 3 of 77 slots, the latent unrolling into 4096 rows, the three projections with the adapter
marked on the weight matrices, the `Q x K-transpose` product with a row sweeping down both operands,
the softmax turning one row into shares that add to one, and the blend landing back on one patch of
the 64x64 grid.

The rule this sets for the rest of the build: **a panel that lists shapes has to draw them instead,
or it does not go in.**

## What 5c turned up, and it argues against the fine

Plotting the two measured arms together says something neither had said alone.

| | mu | fit error | drift, amplified | |
|---|---|---|---|---|
| V0 | 0 | 4.9e-4 | 3.30e-2 | measured |
| V0a | 10 | 8.9e-4 | 8.45e-4 | measured |
| V1 | infinity | ? | exactly 0 | **not run** |

The fine cut the drift by 39 times. It also **nearly doubled the fit error**, and at mu 10 the two
quantities have met: the drift that is left is the same size as the whole error it perturbs.

The reason is worth keeping, because it decides between V0a and V1. **A penalty adds a term to the
objective; a freeze changes the parameterisation.** The penalty is therefore optimising something
other than the fit, and pays for it. The freeze is not, and by the reachable-set argument it gives
up nothing.

So the soft version has a cost the hard version does not, and the hard version is the one nobody has
run.

## The descent under the plane frame

Four panels behind the loss node, all measured, none of it previously drawn.

| # | what it shows | where the numbers came from |
|---|---|---|
| 1 | the plane itself: u at the origin, a and b as arrows, a+b-u as the parallelogram corner, the joint's shadow, and the out-of-plane part as its own flat bar | `scripts/showcase/composition_plane_coords.py`, cache only, no GPU |
| 2 | the angle between the cat and dog directions over the run | derived in the browser from the same coordinates |
| 3 | the weights each expert would actually need, against the 7.5 PoE uses | same |
| 4 | where the trained adapter lands in the same frozen frame | `scripts/showcase/adapter_in_the_plane.py`, 312 forwards, under a minute |

**What the descent turned up.**

**The experts turn against each other.** The angle between `a-u` and `b-u`, median over eight
seeds, runs 46 degrees at step 0, 86 at step 3, 106 at step 7, 113 at step 10 and past 120 late.
Early they point partly the same way, so adding them compounds. After about step 3 they oppose,
so adding them cancels. Composition is doing something qualitatively different at step 2 than at
step 20.

**The right weights are 2 to 4, and they are not equal.** Solving `alpha(a-u) + beta(b-u)` = the
guided joint exactly, in the plane, gives median alpha 2.95 and beta 4.18 over steps 0 to 10,
against the 7.5 PoE applies to both. Early the dog branch needs three times the cat branch; by
step 10 that has inverted. This confirms the filed "1 to 3" result and adds the asymmetry, which
that finding does not mention.

**The adapter does exactly what those weights predict.** In the same frozen frame, at 30,000
steps, it pulls the composition in by a factor of two to three (`|a+b-u|` 2.02 to 0.76 at step 0,
11.26 to 3.86 at step 30) and roughly halves the distance to the target (1.20 to 0.74 early,
10.52 to 3.15 at step 30). Two independent reads agreeing: the correction is, in large part, "you
are pushing far too hard".

**And it leaves the plane to do the rest.** Its out-of-plane component grows from 1.14 at step 0
to 3.77 at step 10, which is the part no re-weighting could ever have supplied.

**One honest limit, on the face of the figure.** The plane is rebuilt at every state, so
coordinates are not comparable across steps and no single trajectory can be drawn through it.

## Held

Nothing yet.

## Settled in the walk

- **The two legs are asymmetric and the drawing must say so.** Leg A is a lookup of a prediction
  made once with no adapter; leg B is a forward pass with a gradient. The reader's own sketch had
  them as two equal paths, and the asymmetry is the correction.
- **The empty branch is read three times in leg B**, netting to a coefficient of `1 - 2w = -14`.
  Show the three reads first, because that is what the code does, then animate the collapse to
  `-(w-1)` because that is what makes chunk 4's point visible.
- **The mismatch is the story.** Leg A guides against the cached empty branch, leg B against the
  adapted one. Every variation in chunk 5 comes from that one difference.

## Sources

- [the objectives and variations](../../ideas/designing-the-correction-loss/maths/objectives-and-variations.tex),
  the spine: shapes, the nine variations, the code change and tradeoff for each
- [the guided and unguided reconciliation](../../ideas/designing-the-correction-loss/maths/guided-and-unguided-objectives.tex),
  the measured numbers
- [the four variants](../../ideas/designing-the-correction-loss/maths/correction-loss-variants.tex),
  why freeing the empty branch buys nothing
- `poe_repair/experiments/one_pair_one_seed/trainer.py:339-520`, the real forward pass and loss
- `poe_repair/_sdxl/metrics.py:12-17`, `guided_eps` and `poe_eps`
