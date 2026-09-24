# 🔎 Proofread map: section 5, "Learning the Plurality Term"

The walk's persistent state. Read it first on every invocation, write it before every round ends.

## Position in the walk

| Field | Value |
|---|---|
| Selection | `paper/overleaf-iclr/iclr2027_conference.tex` lines 305 to 355 |
| Kind | manuscript section |
| Grain | paragraph |
| Cadence | sweep |
| Current chunk | 4 of 11 |
| Marks | 2 ok, 1 changed, 1 cut, 7 unwalked |
| Compiled | nothing written to the file yet |
| Owner | this session |

## Table of contents

- [Position in the walk](#position-in-the-walk)
- [Quick context: what this is](#quick-context-what-this-is)
- [The chunk map](#the-chunk-map)
- [Accepted, not yet written](#accepted-not-yet-written)
- [Originals held](#originals-held)
- [Figures](#figures)
- [Open questions on the text](#open-questions-on-the-text)
- [Routes and threads](#routes-and-threads)
- [Compile log](#compile-log)
- [Next step](#next-step)

## Quick context: what this is

Navigation: ⬅️ [Position](#position-in-the-walk) | 📋 [TOC](#table-of-contents) | [Next](#the-chunk-map) ➡️

**What is being proofread**

The method section of the ICLR submission: the section that trains an adapter to produce the
plurality correction from the two concept prompts alone, so a corrected run no longer needs the
joint prompt.

**Why this pass is happening**

It has to read well and be clear, and the two architecture diagrams it carries (Figures 6 and 7)
have problems that a caption cannot fix.

**What the text is trying to do**

Say what was trained, on what data, with what loss, and what it costs at inference, so a reader
can tell that the correction now comes out of the weights rather than out of the joint prompt.

## The chunk map

Navigation: ⬅️ [Quick context](#quick-context-what-this-is) | 📋 [TOC](#table-of-contents) | [Next](#accepted-not-yet-written) ➡️

| # | Kind | First words or file | Mark | Fault | The read |
|---|---|---|---|---|---|
| 1 | heading | `\section{Learning the Plurality Term}` | ok | | names what is under it |
| 2 | paragraph | "In the previous section we measured" | changed | misnamed the trained thing, stacked ideas | said "a model" where an adapter was trained; three closing sentences became one |
| 3 | paragraph | "The model learns its correction from" | cut | decoration, term before thing | carried one fact, now in chunk 4's seam; "residual" no longer precedes its definition |
| 4 | paragraph | "The adapter is a low-rank adaptation" | work | stacked ideas, rank conflict | 84-word sentence; rank 8 contradicts Figure 6 |
| 5 | paragraph | "We split a pool of nineteen animal pairs" | work | stacked ideas | two paragraphs of content in one |
| 6 | figure | `figures/training-the-compositional-adapter.png` | work | two jobs in one panel, dangling item, typo | see Figures |
| 7 | caption | "Training the compositional adapter." | work | caption past the picture | rank-$r$ where the picture names two ranks |
| 8 | paragraph | "At inference the correction is not injected" | work | stacked ideas | the λ dial arrives with no reason given |
| 9 | figure | `figures/sampling-with-the-adapter.png` | work | double-counted algebra, clipped text | see Figures |
| 10 | caption | "Sampling with the adapter." | ok | | matches the picture's arithmetic |
| 11 | paragraph | "On pairs the adapter never trained on" | work | claim past the evidence | "reproduces" is not what the report says |

## Accepted, not yet written

Navigation: ⬅️ [The chunk map](#the-chunk-map) | 📋 [TOC](#table-of-contents) | [Next](#originals-held) ➡️

| # | Control | The accepted text |
|---|---|---|
| 3 | `cut` | *(deleted)* Seam: chunk 4's opening gains "We train one adapter on every pair together, so it has to learn a correction rule that covers all of them. It is a low-rank adaptation (LoRA) of rank 8 on..." (was "The adapter is a low-rank adaptation (LoRA) of rank 8 on...") |
| 2 | `improve` + user reorder | In the previous section we measured the plurality term along composed runs and showed that adding a small amount of it back, early in the run, restores composition. In this section we learn it. We train an adapter inside the model so that its own composed prediction carries the correction at each step, from the current state and the two concept prompts alone. During training the joint prompt supplies the target the adapter learns from; at inference it is barred, and the corrected run does not need it. |

**What changed in chunk 2, and why.** "A model that produces the correction" became "an adapter
inside the model so that its own composed prediction carries the correction", because the old
phrasing builds a second network that emits a correction vector, and nothing adds anything at
inference. The closing three sentences became one, ordered training first and inference second at
the user's direction. 89 words before, 89 after.

## Originals held

Navigation: ⬅️ [Accepted](#accepted-not-yet-written) | 📋 [TOC](#table-of-contents) | [Next](#figures) ➡️

| # | `was:` |
|---|---|
| 3 | The model learns its correction from the residuals themselves. For each training pair we therefore store the residual at every denoising step, over several runs, so the training data covers whole trajectories rather than isolated steps. We train a single adapter on every pair together to learn the correction rule that restores the plurality lost during composition. |
| 2 | In the previous section we measured the plurality term along composed runs and showed that adding a small amount of it back, early in the run, restores composition. In this section we learn it. We train a model that produces the correction at each step, from the current state and the two concept prompts, so a corrected run no longer needs the joint prompt. The joint prompt is barred only at inference, so the term can still be measured in advance, and those measurements are what the model trains on. |

## Figures

Navigation: ⬅️ [Accepted](#accepted-not-yet-written) | 📋 [TOC](#table-of-contents) | [Next](#open-questions-on-the-text) ➡️

**Figure 6, `training-the-compositional-adapter.png`**

Read off the image, not the caption.

1. Panel 1 says "50 DDM steps". The sampler is DDIM, named as DDIM everywhere else in the paper.
2. Panel 1 caches $\varepsilon_{\emptyset}$ and nothing in the figure ever consumes it. Panel 2
   produces its own $\varepsilon_{\emptyset}$ from the adapted model, which is what
   `trainer.py:507` does by default.
3. Panel 2 draws a $+$ node and a $-$ node into a box that already states
   $\tilde\varepsilon_{\mathrm{PoE}} = \tilde\varepsilon_a + \tilde\varepsilon_b - \varepsilon_{\emptyset}$.
   The arithmetic is written twice, once as glyphs and once as algebra.
4. The prompt symbol is $C_{ab}$, $C_a$, $C_b$ here and $c_a$, $c_b$ in Figure 7.
5. The LoRA panel says $r = 8$ (single pair), $r = 32$ (pooled). This section trains the pooled
   adapter and the prose says rank 8.

**Figure 7, `sampling-with-the-adapter.png`**

1. The $+$ nodes sit before the guidance boxes, so guidance reads as an addition of
   $\varepsilon_a$ and $\varepsilon_{\emptyset}$ rather than as the formula the box states.
   Same double-counting as Figure 6 item 3, and here the $-$ node does it a third time.
2. The caption line under $x_t$ is clipped at the bottom edge: "three passes, one latent," and
   then a cut-off line.
3. Prompt symbols are lowercase here, uppercase in Figure 6.

## Open questions on the text

Navigation: ⬅️ [Figures](#figures) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

Nothing open. The rank question is settled below.

## Settled by checking

**The section reports rank 8, and the prose is right.** `config.json` of `phase1_r8_100k`, read on
the cluster at `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/`,
gives `lora.rank 8`, `lora.alpha 8`, `target_modules ["attn2.to_q", "attn2.to_k", "attn2.to_v"]`,
`optim AdamW lr 1e-4 grad_clip 1.0`, `sampler ddim 50 steps guidance 7.5 at 1024x1024 fp16`.
`paper/iclr/figures.md:40` names the same run as the source of the held-out sample figure, so the
section and its figure are the same adapter.

Every number in chunk 4 and chunk 5 therefore checks out, and `alpha = rank` means the figure's
"$\alpha = r$, so the scale is 1" is right too. The one thing that does not is Figure 6's LoRA
panel, which says the pooled adapter is $r = 32$.

## The run the section must describe

Navigation: ⬅️ [Figures](#figures) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

**`tiiz0c0c`, display name `v57-11b-x0-self-decay`, still running at epoch 996 of 1000.**
Config read from W&B on 2026-09-23. It is a different adapter from `phase1_r8_100k`, which is what
the prose and both figures currently describe.

| Thing | Section 5 and Figures 6 and 7 say | `tiiz0c0c` |
|---|---|---|
| rank, alpha | 8, 8 | 32, 32 |
| adapted layers | `attn2` Q, K, V only: 70 blocks x 3 = 210 | `attn2` and `attn1` Q, K, V: 420 |
| loss space | epsilon | x0 (`loss_eps` logged beside `loss_fit`) |
| extra loss terms | none | `orth_weight 1`, bucket weights (`loss_weight_mean 1.42`), `loss_null_anchor`, `loss_undialled` |
| learning rate | 1e-4 constant | 1e-4 cosine decay, at 4.87e-9 by step 49,849 |
| optimizer steps | 2,000 epochs x 50 = 100,000 | 1,000 x 50 = 50,000 |
| peak memory | 23 GB | 28.0 GB |
| the pool | 19 animal pairs, 11 train and 8 eval | `all_groups`; the tracking set holds `a_typewriter__x__a_cactus` alongside the animals |
| sampler | DDIM 50 steps, guidance 7.5, 1024x1024, fp16 | identical |

**What this costs each chunk.** Chunks 4, 5, 7 and the whole of Figure 6's bottom row are about an
adapter the paper is no longer training. The self-attention addition is the one that changes the
story rather than a number: Figure 6 says the prompt enters only through cross-attention and that
this is why the adapter sits there, and with `attn1` adapted the adapter also sits where the image
attends to itself.

**What it does not cost.** The sampler is unchanged, so Figure 7's green block is wrong only in the
ways round 1 found, not because of this run.

**Not settled.** This run is in flight and is variant 11b of a cohort. Whether it is the one the
paper reports is the user's call, and nothing in the section is rewritten until it is made.

## Which render is in the paper

Navigation: ⬅️ [The run](#the-run-the-section-must-describe) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

**The paper still holds the 2026-09-17 render.** `paper/iclr/figures/` and
`paper/overleaf-iclr/figures/` both hold md5 `ce095d9db716dcdc2dd2c52e2da0bf43` for
`training-the-compositional-adapter.png`. The two folders are identical; the 2026-09-21 mtime on
`paper/iclr/` is a copy, not a new render.

**A newer candidate exists outside the repo** and has been seen only as a screenshot. Against the
render in the paper it fixes two things and breaks one.

| | In the paper | The newer candidate |
|---|---|---|
| panel 1 step count | "50 DDM steps" | "50 DDIM steps", correct |
| the cached $\varepsilon_{\emptyset}$ | nothing says it is consumed | "read back from the cache" added, correct |
| the guidance boxes | $\tilde\varepsilon_a = \varepsilon_{\emptyset} + w(\varepsilon_a - \varepsilon_{\emptyset})$, correct | $\tilde\varepsilon_a = \varepsilon_a + w(\varepsilon_a - \varepsilon_{\emptyset})$, **wrong on both boxes** |
| the nodes before those boxes | circled $+$ | empty circles, no operator |

The candidate must not replace the file until the leading term is back to
$\varepsilon_{\emptyset}$. Classifier-free guidance is
$\varepsilon_{\emptyset} + w(\varepsilon_c - \varepsilon_{\emptyset})$, and the caption in the
manuscript still states it that way.

## Which panels are the training architecture

The bottom row of Figure 6, three orange panels. The green block above is the training procedure
and carries no architecture fact.

| Panel | What is wrong under `tiiz0c0c` |
|---|---|
| SDXL denoising UNet (overview) | "70 cross-attention blocks x 3 projections = 210 adapted layers" is 420; "the text prompt enters here and nowhere else" no longer explains where the adapter is |
| One cross-attention block | "the output projection, self-attention and feed-forward are not [adapted]" is false: 210 of the 420 modules are `attn1` |
| One adapted projection (LoRA) | "$r = 8$ (single pair), $r = 32$ (pooled)" is $r = 32$; the $d_{\mathrm{in}}$ line holds for cross-attention only |

Counted from `lora_step_050000.pt`: 840 tensors, 420 modules, 70 each for `attn1.to_q/k/v` and
`attn2.to_q/k/v`, `lora_A` shaped `(32, 640)`.

## Routes and threads

Nothing routed yet.

## Compile log

Nothing compiled.

## Next step

Chunk 4, "The adapter is a low-rank adaptation (LoRA) of rank 8...", which is wrong in five places under `tiiz0c0c`.
