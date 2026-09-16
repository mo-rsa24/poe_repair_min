# Scene plan: seeing the correction loss

Compiled from the `/drip-picture-speak --reference` walk in
[SCENE_MAP.md](SCENE_MAP.md). This is the reviewable proposal that
`picture-speak` builds from. The scoping sketchpad it came out of is at
https://claude.ai/code/artifact/ae96d471-01cd-4764-b79d-41da5053230c

**Scene** Walk one training step of the product-of-experts corrector with the real tensor shapes,
then switch between the nine ways its loss could be written and watch the same picture rewire.

**Reference** The four latent-diffusion figures the reader supplied: the Rombach figure's green
latent-space container, blue U-Net trapezoid, orange Q/KV cross-attention blocks and its arrow
vocabulary, plus the SDXL 128x128 latent strip. Carried through verbatim to the build.

**Source**
- [the objectives and variations](../../ideas/designing-the-correction-loss/maths/objectives-and-variations.tex), the spine: shapes, nine variations, code change and tradeoff each
- [the reconciliation](../../ideas/designing-the-correction-loss/maths/guided-and-unguided-objectives.tex) and [the four variants](../../ideas/designing-the-correction-loss/maths/correction-loss-variants.tex), for the measured numbers and the derivations
- `poe_repair/experiments/one_pair_one_seed/trainer.py:339-520`, the real forward pass and loss
- `cross_trace.json` in [the SDXL VAE map](../../scenes/sdxl-vae-architecture-map/README.md), the 70 cross-attention modules
- `composition_plane.json` and `adapter_in_plane.json` under `outputs/showcase/composition_plane/`
- the three renders under `outputs/showcase/empty_prompt/`

**Tier** 3. One main view plus three doors, each a stepped panel set.

| | what it is | steps |
|---|---|---|
| main | the two legs, nine variation pills | 6 |
| door A | inside one Q/KV block | 6 |
| door B | the empty prompt | 7 |
| door C | the plane PoE lives in | 4 |

## Claims and treatments

| # | claim | treatment | tag |
|---|---|---|---|
| 1 | Sampling left five tensors per step on disk, each (1,4,128,128); a cell of 50 steps is 250 MB | volume slabs with the real shapes, one panel | **not yet built** |
| 2 | The target leg is a disk read; the composition leg is a live forward pass | two-leg frame, the asymmetry drawn not captioned | grounded |
| 3 | Three prompts go through one U-Net in one batch and come back as three | the fan, with the real batch widths | grounded |
| 4 | Guidance lands on both sides before the subtraction, and leg A guides against the cached empty branch while leg B guides against the adapted one | the two dial boxes, differing by one symbol | grounded |
| 5 | 0.38% of the weights train, all on cross-attention | colour the whole diagram by what can move | **not yet built** |
| 6 | A prompt is 77 slots; the latent is 4096 rows; Q comes from the latent, K and V from the text | six drawn steps with the real traced shapes | grounded |
| 7 | The empty prompt is a well-formed empty sentence, and it draws a finished painting with no subject | the tokeniser strip, then the three real renders | grounded |
| 8 | The empty branch is the scaffold every render is built on, and it enters the composition at −14 | side-by-side renders, then coefficient bars on a live dial | grounded |
| 9 | A fine is a soft freeze; at mu 10 the drift falls 39x and the fit error nearly doubles | three named settings on one log scale | grounded |
| 10 | The dialled loss can lower itself by shifting the empty branch without improving the composition | two bars and one slider; one stays pinned while the other reaches zero | grounded |
| 11 | Nine variations differ by which tensors come from cache, and the picture rewires for each | pills driving one spec-driven drawing | grounded |
| 12 | Each variation's loss written in full and collapsed, with cancelling terms struck and added terms underlined | drawn underbraces over live token spans | grounded |
| 13 | PoE lives in a plane; the joint prediction does not | the plane face-on, out-of-plane as its own flat bar | grounded |
| 14 | The two experts turn from cooperating to opposing, 46° to 127° | one curve over steps | grounded |
| 15 | The weights each expert needs are 2 to 4, not 7.5, and they are asymmetric and swap | two curves against a 7.5 rule | grounded |
| 16 | The adapter pulls the composition in by 2 to 3x, halves the distance to the target, and leaves the plane to do the rest | before and after in the frozen frame | grounded |

## Interactions

**The one primary control** is the variation pill row: it drives the main drawing, the batch
widths, the wires and both loss forms at once.

**Cross-highlight pairs** every symbol in either loss form against its box in the diagram, both
ways, delegated so it survives every redraw.

**Secondary controls**: the six-step stepper shared across all nine variations, so the step index
holds when the variation changes; the guidance dial on claim 8; the gauge slider on claim 10; seed
pills and a step slider on claims 13 and 16.

## Guided walk

Six steps on the main view: the prompts, leg A as a lookup, leg B as one pass, the split into
three, guidance on both sides, the loss. It closes on claim 10, the move the two objectives
disagree about, because that is the argument for the one variation nobody has run.

## Grounding needed

Claims 1 and 5 are specified but unbuilt; both are drawable from numbers already in the source,
no new compute. Everything else is built and measured. Two things are owed and named on their own
faces: a render of the empty prompt under the adapter, and the per-concept distortion norms the
step-0 probe computes and discards.

## Stays out

- Any field of arrows over a data space. These are noise predictions at one state, not positions,
  and the Song-style figure would teach something false about what we have.
- Any single trajectory drawn through the plane. It is rebuilt at every state, so cross-step
  coordinates are not comparable and a path would be an artefact.
- V5's corpus. The data does not exist for the pairs that fail, and the figure says so rather than
  showing a placeholder corpus.
- The tilted, fake-3D version of the plane. The out-of-plane part is the quantity PoE cannot
  represent; drawing it as a height inside a projection invites the confusion the frame exists to
  remove.
