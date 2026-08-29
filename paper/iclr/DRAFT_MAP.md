# ✍️ PoE repair (ICLR): the draft, piece by piece

## Recommended prompt (when a section lands)

```
/restyle <section> against <exemplar paper>
```
(For a section that is finished and needs filing rather than restyling: `/polish`.)

## Position in the manuscript

| Step | Section | State | Figures it owns |
|------|---------|-------|-----------------|
| — | [Abstract](#-abstract) | candidate 3 adopted into the tex | none |
| 1 | [Introduction](#1--introduction) | settled and compiled | F1 |
| 2 | [Background and Related Work](#2--background-and-related-work) | ¶1a and ¶1b compiled; machinery paragraphs (¶2 diffusion, ¶3 score, ¶4 guidance, ¶5 close) not walked | none |
| **3 (done)** | **[Missing Implications: The Plurality Term](#3--missing-implications-the-plurality-term)** | **six paragraphs placed and compiled; probability-space blocks in Appendix `app:poe-derivation`** | **F1b** |
| **4 (current)** | **[Restoring the Plurality Term Restores Composition](#4--restoring-the-plurality-term-restores-composition)** | **main text compiled (4.1, 4.7 the coverage grids, 4.4 with figure, 4.6); the 4.5 slot open on the size-curve check** | **`fig:window-extends`, `fig:window-starts-later`, then `fig:window-map` last in the tex; the strength grids staged in Extras; F3 pending; F5 demoted to prose** |
| 5 | [Learning the Plurality Term](#5--learning-the-plurality-term) | ¶5.1 to ¶5.5 compiled with F9 placed; ¶5.6 numbers await the rescore | F9 placed, F10 awaiting render; F8a, F8b owed to ¶5.6; F8 reserved; F6, F7, F7a staged in Extras |
| 6 | [Discussion](#6--discussion) | not walked, bare heading in the tex | none |
| 7 | [Conclusion](#7--conclusion) | not walked, bare heading in the tex | none |

## Table of contents

- [Position in the manuscript](#position-in-the-manuscript)
- [Quick context: where you are](#quick-context-where-you-are)
- [How this manuscript is written](#how-this-manuscript-is-written)
- [The spine, one line per section](#the-spine-one-line-per-section)
- [The draft](#the-draft)
- [Loose lines](#loose-lines)
- [Blocked pieces](#blocked-pieces)
- [Sessions](#sessions)
- [Open citations](#open-citations)
- [Figures this draft leans on](#figures-this-draft-leans-on)
- [Compile log](#compile-log)
- [Rejected wordings](#rejected-wordings)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-manuscript) | 📋 [TOC](#table-of-contents) | [Next](#how-this-manuscript-is-written) ➡️

**What is being written**

The ICLR submission at [iclr2027_conference.tex](iclr2027_conference.tex), on why product-of-experts
composition fails and what fixes it.

**Where the walk is**

Section 3, the plurality term, is compiled and in the tex. Six paragraphs placed, build passing,
no overfull lines. The walk is open at section 4, which has a heading and no prose.

**What the next compile will land**

Section 4's first paragraph, once the section is broken into paragraphs and walked. Nothing is
queued behind it.

## How this manuscript is written

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-spine-one-line-per-section) ➡️

Applies to every section, not only the one being drafted.

**Short sentences, one idea each.**

**No dash used as punctuation between clauses, and no colon inside a sentence.** Rewrite with a
comma, a full stop, or parentheses.

**Plain words wherever a plain word exists.** Technical terms are used when the field genuinely
uses them, and each one is defined the first time it appears.

**A tired reader follows every paragraph on one pass.** If a sentence would not be said that way
out loud, it is rewritten.

**A sentence claims no more than its evidence.** Where the evidence stops, the sentence stops,
and the boundary is stated rather than softened.

## The spine, one line per section

Navigation: ⬅️ [How this is written](#how-this-manuscript-is-written) | 📋 [TOC](#table-of-contents) | [Next](#the-draft) ➡️

Quoted from [SPINE.md](SPINE.md). A sentence arguing a different claim than its section's line is
flagged, not drafted.

**The paper's one-sentence claim.** Composing pretrained diffusion models by sampling from a
product of experts assumes the composed concepts are independent, that assumption fails when they
are not, and the failure is exactly a missing plurality term that a low-rank adapter can learn
and carry to concept pairs it never trained on.

**2. Background and Related Work.** Combining several of one network's own noise predictions
linearly is already standard practice, and the product-of-experts and classifier-free composition
methods that extend it build on the independence assumption without testing it.

**3. Missing Implications: The Plurality Term.** The gap between the joint distribution over a
correct composed scene and the product-of-experts approximation to it is a specific, definable
term, an interaction term that this setting narrows to plurality, and this plurality term is what
the paper measures and later corrects.

**4. Restoring the Plurality Term Restores Composition.** A small amount of the measured
plurality term restores composition when it arrives early in the denoising run, and the full
amount arriving late does not.

**5. Learning the Plurality Term.** A low-rank adapter on SDXL's cross-attention layers predicts
the plurality term from the two concept prompts alone, the correction it produces reaches concept
pairs the adapter never trained on, and it matches the oracle correction it imitates without ever
seeing the joint prompt.

**6. Discussion.** The correction's reach has a boundary, and this section states where the fix
holds and where the paper's own checks found it does not extend as cleanly as the headline result
suggests.

**7. Conclusion.** Product-of-experts composition fails from a missing, learnable, transferable
plurality term, not from an unfixable property of the sampling procedure.

**The split that makes sections 4 and 5 different.** Sections 3 and 4 use the correction computed
from the joint prompt, which the composed sampler cannot have, so it exists to measure and not to
run. Section 5 uses the learned correction, which never sees the joint prompt.

## The draft

Navigation: ⬅️ [The spine](#the-spine-one-line-per-section) | 📋 [TOC](#table-of-contents) | [Next](#loose-lines) ➡️

### ✍️ Abstract

State: candidate 3 is the abstract in the tex, adopted from the three candidate wordings. The
sentence checklist below records the earlier inline wording it replaced.

- [x] **1.1** A natural image typically depicts several distinct concepts arranged so that they
      form a single coherent scene, and text-to-image models generate compositional scenes from a
      description of their parts.
- [x] **1.2** Recent methods compose pretrained diffusion models at inference time by sampling
      from a product of experts, one expert per concept.
- [x] **1.3** Product-of-experts composition can fail catastrophically, producing a single blended
      chimera instead of a scene containing both concepts, and we trace this failure to a
      specific, correctable gap.
- [x] **1.4** This gap is the plurality term, the difference between the sample a model
      conditioned on the true joint prompt would produce and the sample the product of experts
      actually produces, and it is exactly what the independence assumption drops.
- [x] **1.5** We train a low-rank adapter on the cross-attention layers of Stable Diffusion XL to
      predict this plurality term at every denoising step without ever seeing the joint prompt,
      and add its prediction back in as a correction.
- [x] **1.6** We show the correction transfers to concept pairs it was never trained on, and that
      adding more of it raises the compose rate toward a scene that composes both concepts
      correctly.
- [ ] **1.7** assemble the six into one paragraph, then `compile`

The numbers are deliberately left out of 1.6. That phrasing is qualitative by choice, not waiting
on a review file.

▶ **Next: ¶5.6**, whose numbers 1.6 may later quote.

#### Candidate abstracts, three wordings of one agreed text

A supervisor-agreed abstract is now the source text. Three candidate wordings are owed, each in
its own abstract-only file built on the ICLR template, so the wordings can be read side by side
before one is chosen for the manuscript. The agreed text is the ceiling on every claim. A
candidate may reorder and reword it and may not add a claim it does not make.

**What the agreed text adds over sentences 1.1 to 1.6 above.** Three things. The reason to
compose at all, which is that the number of possible scenes grows combinatorially as concepts are
added. The name plurality for the missing ingredient, a term that appears nowhere else in this
repo and is minted here. The closing framing, a model specialised to an abstract property of a
scene rather than to a concept.

**One angle per candidate, so the three are genuinely different documents.** Candidate 1 keeps
the agreed argument order and rewrites for plainness. Candidate 2 opens on the failure, the
hybrid object that inference-time composition returns, and reaches the motivation afterwards.
Candidate 3 opens on the capability, introduces plurality mid-argument, and closes on the
general-correction claim; it is the adopted abstract, in the tex.

**Every candidate is written for two readers at once.** A reviewer with no diffusion background
follows the argument from the pictures it describes. A reviewer who works on composition finds
the mechanism named exactly, the residual between the joint-prompt prediction and the
product-of-experts prediction, at every denoising step. Neither reader is served by a second
register, so no candidate carries a plain half and a technical half.

| Candidate | Angle | File | State | Session |
|---|---|---|---|---|
| 1 | agreed order, example-first wording | `abstract_candidate_01.tex` | compiled, build passed | candidate-abstract-1 |
| 2 | literature-entry order, technical wording | `abstract_candidate_02.tex` | compiled, build passed | candidate-abstract-2 |
| 3 | research-program order, coins plurality mid-arc | `abstract_candidate_03.tex` | compiled, build passed | candidate-abstract-3 |

**Candidate 1, sentence by sentence.** Sketch revision 2, minted for the sentence walk. Revision 1
(the eleven lines this replaces) is kept in [Rejected wordings](#rejected-wordings). The new arc
shows the single-prompt case with an example, shows composition with an example, motivates
composition, then shows the failure mechanism (pixel-region competition) before naming plurality,
so the term arrives already earned rather than defined cold.

- [x] **c1.1** placed. A text-to-image diffusion model given a single prompt that names several
      concepts can render each one as a distinct object, so "a cat and a dog" yields a cat sitting
      beside a dog in one coherent scene. The concepts are arranged naturally rather than
      strangely, interacting as they would in the real world.
- [x] **c1.2** placed. The scene can also be built a different way, by combining several
      pretrained diffusion models at inference time, one expert per concept. Composing a camel
      expert with a forest expert, for instance, produces a high-fidelity image of a camel in a
      forest. (Success stated unhedged by design; c1.4 carries the contrasting failure example.)
- [x] **c1.3** placed. The appeal is generality, since the space of possible scenes grows
      combinatorially with each new concept, and composing specialists sidesteps training any
      single model over it while reaching pairs never seen together in training.
- [x] **c1.4** placed. But for some pairs, such as "a cat and a dog," composition fails, the two
      concepts compete for the same pixel regions and the model produces a single blended, hybrid
      object instead of two distinct objects sharing the scene.
- [x] ~~**c1.5**~~ cut. c1.4's "instead of two distinct objects sharing the scene" already states
      the lost ability, so a sentence restating it as an abstraction added nothing. The naming of
      plurality moved into c1.6, where it follows the narrowing instead of preceding the
      measurement.
- [x] **c1.6** placed. This lost information is measurable, at every denoising step it appears as
      the residual between the prediction of a model given the joint prompt and the prediction of
      the composed experts. In our setting, where competing concepts are expressed as a plain
      conjunction, the residual is narrowed to a single aspect of compositionality, and we choose
      to call this aspect plurality. (The residual is the broad quantity, carrying whatever a
      joint prompt binds that separate prompts cannot, spatial relations, attributes, counts; the
      conjunction-over-competing-pairs setting is what isolates the plurality slice, and the
      naming is stated as a choice for that reason.)
- [x] **c1.7** placed. We train a low-rank adapter on the cross-attention layers of Stable
      Diffusion XL to predict this residual at every denoising step, without ever being
      conditioned on the joint prompt.
- [x] ~~**c1.8**~~ merged into c1.9 at assembly. Its two claims (distinct objects restored, unseen
      pairs reached) duplicated the closer, so the two sentences became one.
- [x] **c1.9** placed, absorbing c1.8. Adding this predicted correction back into the composed
      sampler restores distinct objects on concept pairs the adapter was never trained on,
      without requiring the joint prompt at inference.
- [x] **c1.10** assembled and compiled to `abstract_candidate_01.tex`, build passed with
      tectonic, PDF 17.5 KiB, no warnings surfaced.

**Candidate 1, final compiled text.** As landed in `abstract_candidate_01.tex`.

> A text-to-image diffusion model given a single prompt that names several concepts can render
> each one as a distinct object, so "a cat and a dog" yields a cat sitting beside a dog in one
> coherent scene. The concepts are arranged naturally rather than strangely, interacting as they
> would in the real world. The scene can also be built a different way, by combining several
> pretrained diffusion models at inference time, one expert per concept. Composing a camel expert
> with a forest expert, for instance, produces a high-fidelity image of a camel in a forest. The
> appeal is generality, since the space of possible scenes grows combinatorially with each new
> concept, and composing specialists sidesteps training any single model over it while reaching
> pairs never seen together in training. But for some pairs, such as "a cat and a dog,"
> composition fails, the two concepts compete for the same pixel regions and the model produces a
> single blended, hybrid object instead of two distinct objects sharing the scene. This lost
> information is measurable, at every denoising step it appears as the residual between the
> prediction of a model given the joint prompt and the prediction of the composed experts. In our
> setting, where competing concepts are expressed as a plain conjunction, the residual is
> narrowed to a single aspect of compositionality, and we choose to call this aspect plurality.
> We train a low-rank adapter on the cross-attention layers of Stable Diffusion XL to predict
> this residual at every denoising step, without ever being conditioned on the joint prompt.
> Adding this predicted correction back into the composed sampler restores distinct objects on
> concept pairs the adapter was never trained on, without requiring the joint prompt at
> inference.

**Candidate 2, sentence by sentence.** The order is show-then-name. The reader watches a joint
prompt work, watches composition work, learns why anyone would want composition, watches
composition break on a pair that competes for the same pixels, and only then meets the word
plurality. Nothing is asserted about scaling, and nothing about composition's success is claimed
before the failure is admitted.

- [x] **c2.1** T2I diffusion models render complex multi-concept scenes with high fidelity from a single prompt, the joint prompt staged here as the reference case. Two sentences: capability, then the cat-and-dog reference case
- [x] **c2.2** known compositional weaknesses, fused with the scale remedy. "Such as" carries the list; no verdict word on the models
- [x] **c2.3** placed as two sentences, promise first then product of experts as the standard tool. The logical-conjunction gloss was cut by the user's edit
- [x] **c2.4** placed. Dependence stated then instantiated on both measured pairs (butterfly x meadow composes, cat x dog blends). Takes the pair contrast that move 6 was holding
- [x] **c2.5** placed and pinned: scoping concession, then the definition, then the name. "As its own object" chosen over "independently" to stay inside what the instance-count scorer supports
- [x] **c2.6** placed as two sentences: conflicting pairs defined (contention kept descriptive), then the inference. "The same text-to-image model" asserts the one-network identity for the first time, which is what licenses blaming the combination
- [x] **c2.7** placed as two sentences: the carrier clause, then the residual definition. "Measurable" cut by the user's edit, the definition itself carrying that claim
- [x] **c2.8** placed as three sentences: adapter (conditioned only on the concept prompts), restoration and transfer, the property-specialised close. Seam repair: the adapter sentence's "at every denoising step" cut as a doubling of the definition's; the joint-prompt echo at moves 6-7 was reviewed and kept. No strength clause while the dose-response decision stays open. "Inference-time composition" minted in the close
- [x] **c2.9** assembled and compiled to `abstract_candidate_02.tex`, tectonic build passed, PDF 17.8 KiB. Fourteen sentences, about 270 words, no numbers by ceiling

**The adopted framing.** The abstract enters through the compositionality literature rather than
through a worked example. Move 5, placing plurality inside the taxonomy of known compositional
failures, is what this ordering buys. The joint prompt is staged in move 1 as the capability
example, so it is already in the reader's hands when move 6 shows composition failing on a pair
the joint prompt renders, and when move 7 defines the residual against it. "Product of experts"
is named in moves 6 and 7; this is the technical candidate and the term is how an expert
recognises the method family.

**House-rule relaxation, recorded.** A colon may introduce a list or an instantiation when the
clause before it is complete, as in the placed move 4. The manuscript style block's blanket ban
on colons inside a sentence is relaxed to that extent, for all three candidates.

**Three wording bounds carried from the discussion.** "Not robust" may be claimed, "not
scalable" may not, since nothing measures scalability. "We localise one failure" may be claimed,
"we find the conditions for success" may not. "Contend for the same image region" is
descriptive of the observed blends, not a measured mechanism, and the sentence must not promote
it to a cause.

**One open decision: dose-response in or out.** "Restores plurality at a strength we control" is
F2's λ-sweep and would strengthen move 8, but the supervisor-agreed text never mentions it and
that text is the ceiling. In or out is the user's call with their supervisors. Until decided,
move 8 carries no strength clause.

**Two claims this candidate may not make.** Not that combining is the route that scales, which is
a conclusion with no experiment behind it. Not that composition reaches unseen combinations,
which awards the paper's own result two sentences before the failure is admitted. What recent
methods *aim* for may be reported; what they achieve may not.

**What an expert is, in this paper.** One pretrained network, evaluated on the same noisy latent
under each concept's prompt and under the unconditional condition, then combined. Not several
separately trained single-concept models. SDXL knows both concepts already, and the composition
withholds the joint prompt rather than the knowledge.

**The two examples, and where they come from.** The pair that composes is butterfly and flower
meadow, and the pair that fails is cat and dog. Both are measured, and F1b carries them side by
side with the residual size under each (median 0.065 of the prediction for butterfly × meadow,
0.224 for cat × dog). No other pair may be used as an example without a run behind it.

**What the opening block may not say.** Not "the number of possible scenes grows". Concepts
interact inside a single scene, so counting scenes is the wrong object. What grows is the number
of concept combinations a prompt may name. This is a rewording of the agreed text's "the space of
possible scenes grows rapidly", not a new claim, and it applies to all three candidates.

**The objection, and where it is answered.** Frontier text-to-image models already render complex
scenes they were never trained on, so "composition reaches unseen combinations" will not stand as
a reason on its own. The real distinction is between combinations that are rare in the training
data, which scale does reach, and combinations absent from it because the concepts have never
co-occurred in any image, which more data does not close. **That argument is not made in the
abstract**, because this paper runs no experiment on it. The abstract reports what recent methods
aim for. The introduction argues the distinction and carries the worked example.

**The motivating example stays out of the abstract.** Two medical findings that each have their
own imaging data but never appear together in one image is the case that makes the argument
concrete.
It belongs in the introduction, because this paper's experiments are animal pairs on SDXL and an
abstract naming a medical case would claim a domain the evidence does not cover.

**Candidate 2, placed so far.** All eight moves, the three read repairs and the seam repair
applied. This block is the authoritative assembled text.

> Text-to-image diffusion models render complex scenes with high visual fidelity. Prompted with
> "a cat and a dog", a single model returns one coherent scene with both animals in it, each
> rendered whole. This capability has largely come from scale, and models that miss compositional
> cases such as counting, attribute binding, and spatial relations are answered with larger
> models and more data again. Recent work has shown that pretrained diffusion models can be
> composed during inference, one expert per concept, to generate complex scenes from concept
> combinations rarely seen together in training. The standard tool is product of experts, which
> multiplies the distributions the experts place over images, so that a sample from the product
> satisfies both concepts at once. However, whether composition yields a coherent scene holding
> both concepts depends on the pair: the same rule that renders a butterfly over a flower meadow
> returns a blended hybrid on a cat and a dog. Composition can fail along several of these axes
> at once, but we study one ability in particular: rendering each named concept as its own
> object in the scene, which we call plurality. Plurality is lost on conflicting pairs, in which
> the concepts contend for the same region of the image. Since the same text-to-image model
> renders both animals coherently from the joint prompt, the failure lies in what the
> combination leaves out. The omitted signal is carried by the joint prompt alone. At every
> denoising step, the residual between the model's prediction under the joint prompt and its
> prediction under the product of experts isolates this missing signal. We train a low-rank
> adapter on the cross-attention layers of Stable Diffusion XL to predict this residual,
> conditioned only on the two concept prompts. Adding the prediction back restores plurality to
> the composition, and the same adapter carries to concept pairs it never trained on. We thus
> obtain a diffusion model specialised to an abstract property of a scene rather than to any
> concept in it, keeping the generalisation of inference-time composition without its loss of
> global coherence.

**Loose lines for candidate 2.**

- Sentence 2's A1 wording from the pre-adoption walk, kept loose because move 3 absorbs its
  job: "The same scene can instead be assembled without ever forming that joint prompt, by
  conditioning the pretrained model on each concept separately and combining the results during
  generation, a setting known as inference-time composition." The old sentence 1 loose line was
  retired when move 1 placed as two sentences carrying the same bones.



**Candidate 3, sentence by sentence.** Sketch revision 3, minted for the sentence walk on a
ten-beat spine agreed in discussion. The arc is research-program shaped rather than story
shaped: capability, composition with an example, the field's known compositional weaknesses,
the cost of the standard fix, inference-time composition as the alternative, then this paper's
slice. Plurality is named before the failure is shown, the reverse of revision 2's
show-then-name order. Revision 2's opening sentences are kept in
[Rejected wordings](#rejected-wordings) territory only if rejected during the walk; until then
this list supersedes them.

Constraints carried forward from revision 2, still binding. Nothing about training data may be
claimed. The only examples with a run behind them are butterfly with flower meadow composing
and cat with dog failing. The motivation is reported as what the approach aims for, never as an
achieved result.

New constraints from the spine discussion. **The language claim is out**: the
same-story-from-three-sides review found both language-side probes null, so "a large part of
this is encoded in language" contradicts the evidence and may not appear in any form.
**Plurality is defined visually** (every concept a prompt names appears as its own distinct
object), never as "preservation of independence", because independence is the assumption the
paper shows failing. **Plurality is coined here**: the abstract says plainly that we introduce
the term, rather than using it as if the field already had it. **Beats 3 and 4 rest on the
field's word**, not ours (counting, attribute binding, spatial relationships; larger models on
larger datasets); they are standard intro-citeable claims and are the only part of the abstract
not backed by this repo's runs.

- [x] **c3.1** Text-to-image diffusion models generate high-fidelity scenes in which many
      concepts appear together, each rendered faithfully and consistently with the physical
      world. Chosen for saying fidelity, many concepts in one scene, and physical faithfulness
      in the flattest register; "understands physics" was kept out because the claim is about
      what the images show. The physical-structure claim already has an open-citations row owed
      to section 2.2.
- [x] **c3.2** Given a single prompt such as "a cat and a dog", these models render both
      concepts in the same scene as distinct objects that interact naturally. The user's own
      construction, from A2's shape. "Interact naturally" is still owed a check against F1's
      joint-prompt panel before compile.
- [x] **c3.3** Yet composing concepts correctly remains challenging. Models often struggle with
      object count, attribute binding, and fail to preserve the intended spatial relationships.
      Two sentences by the user's own construction, the first carrying the turn, the second the
      field's standard weakness list. Each weakness is an intro-citeable field claim, with
      open-citations rows already owed to section 2.2.
- [x] **c3.4** Scaling models and data becomes increasingly expensive as the prompt space grows
      to include more concepts and their combinations. Sits inside the allowed rewording of the
      combinatorial-growth claim (combinations a prompt may name, never "the number of possible
      scenes").
- [x] **c3.5** Rather than asking the model for "a cat and a dog" in a single prompt, recent
      methods evaluate the model separately on "a cat" and "a dog," then combine the two
      predictions during inference, reusing the pretrained model rather than retraining it. The
      user's construction plus the payoff tail. Keeps the model singular per the
      what-an-expert-is constraint, and states the mechanism on the running example, so c3.8's
      residual can lean on this sentence.
- [x] **c3.6** In this paper we introduce the term plurality for one aspect of compositionality,
      the property that every concept a prompt names appears in the scene as its own distinct
      object. The coinage made explicit, the visual definition per the spine constraint, and the
      abstract's first "we".
- [x] **c3.7** However, composing conflicting concepts can collapse the scene into a single
      hybrid object rather than rendering both. We show that this failure reflects a loss of
      plurality when two concepts contend for the same image region. Two sentences, the first
      putting the failure on stage, the second the user's diagnosis verbatim. "Reflects" chosen
      as the evidential verb; the contention condition attaches to the diagnosis, not asserted
      as the collapse's cause, per the F2b constraint.
- [x] **c3.8** This loss is measurable, as the per-step residual between the model's prediction
      under the joint prompt and its combined prediction from the separate prompts. The
      technical hinge; leans on c3.5 for the two prediction routes, stays inside the agreed
      text's "per-step residual" phrasing.
- [x] **c3.9** A small fraction of this residual, added back at the right stage of generation,
      is enough to restore plurality, returning both objects to the scene. The
      small-amount-at-the-right-time result, chosen over the dose-response framing at the
      user's correction. Re-check against the hypothesis-02 (dose) and hypothesis-03 (timing)
      review files before compile, both partially open.
- [x] **c3.10** To make this correction available without the joint prompt, we train a low-rank
      adapter on the cross-attention layers of Stable Diffusion XL to predict the residual at
      every denoising step. Trimmed at assembly: the transfer tail moved wholly into c3.11 so
      the claim lands once.
- [x] **c3.11** We show that the adapter learns a general correction for plurality rather than
      the appearance of any specific concept, which allows it to transfer to unseen pairs. The
      user's construction with "therefore" replaced by "we show" after the assembly trim moved
      the transfer evidence out of c3.10. Closes on transfer as the property claim's
      consequence.
- [x] **c3.12** assembled and compiled to `abstract_candidate_03.tex`. Tectonic build passed,
      PDF 17.7 KiB. One check still owed before a candidate is chosen: "interact naturally"
      (c3.2) against F1's joint-prompt panel, and c3.9's small-amount-at-the-right-time claim
      against the hypothesis-02 and hypothesis-03 review files.

**No numbers appear in any candidate**, because the agreed text carries none. Nothing here waits
on a review file.

### 1. ✅ Introduction

**Terms this paper has, and the one it must not invent.** The introduction defines joint
prompting (also called Mono) and inference-time composition. Those two, plus product-of-experts
sampling, are the only names for the settings. "The composed setting" was used loosely during the
walk and is not a defined term, so it does not appear in the manuscript.

**Notation decided in the walk, inherited by every later paragraph.** The joint-prompt
prediction is written `eps_Mono(x_t, t)`, leaning on the introduction's own naming of joint
prompting as Mono, and never `eps_theta(x_t, t | c_1, c_2)`. The composed prediction is
`hat_eps_PoE(x_t, t)`. So `r_t = eps_Mono(x_t, t) - hat_eps_PoE(x_t, t)`. Writing the joint-prompt
prediction with `c_1, c_2` would claim it is the true joint score, and the difference between
those two is what the paper is about.

State: settled and compiled. Figure 1 placed. The composition rule in noise-prediction space is
the labelled equation `eq:poe-composition` at
[iclr2027_conference.tex:106](iclr2027_conference.tex#L106), which is not equation 1. The
unstarred align block above it consumes numbers 1 to 4, so prose refers to the rule with
`\eqref` and never with a hand-written number. Home notation set here: noise prediction epsilon_theta, the score relation stated inline,
and r_t named as the gap later sections define.

- [x] **2.1** scenes are compositional
- [x] **2.2** models scale, compositional reasoning stays brittle
- [x] **2.3** joint prompting against inference-time composition
- [x] **2.4** the composition rule, `eq:poe-composition`, and r_t named
- [x] **2.5** the concrete failure on cat times dog, with Figure 1
- [x] **2.6** why this matters
- [x] **2.7** contributions, three bullets

▶ **Next: section 4.** Section 3 is drafted after the reading register has rows.

### 2. 📚 Background and Related Work

State: not walked, a bare heading in the tex. One section now, not two. Its single claim, from
[SPINE.md](SPINE.md): combining several of one network's own noise predictions linearly is already
standard practice, and the composition methods that extend it build on the independence assumption
without testing it.

Its paragraphs are numbered **§2 ¶1** to **§2 ¶3**, not 2.1 to 2.3, because those labels belong to
the introduction's seven paragraphs.

Six paragraphs. ¶1 frames the section and ¶6 closes it on the gap, so the section reads as an
argument rather than a recap. ¶6 is the claim paragraph, not an extra.

- [x] **§2 ¶1a** what composing by product means, and the question it begs. Six sentences, all
      placed, not yet compiled. The arc: the target named (the product), what the
      product favors (images every expert accepts at once), what a prompt intends (distinct
      objects co-occurring in one scene), and the open question, whether the product encodes that
      co-occurrence. The question must stay a question; section 3 answers it, and Figure 1's
      caption already draws the same contrast. The question is ours, uncited by design: the RRR
      check (2026-08-27, quotes on 2302.11552's row) confirmed Du treats the product as the
      correct target and never asks it. One cite gap left: Hinton's PoE origin, unregistered,
      would sit on the product-semantics sentence.
- [x] **§2 ¶1b** the survey. Five sentences, all placed, not yet compiled. The opener with its
      three-key citep (CO3, FactorDiff, the distillation LoRA), direct addition (Liu), the
      mismatch proof and its Markov chain correction (Du), the sharper correctors (Zhang's
      annealed importance sampling, the Skreta pair's sequential Monte Carlo), and the seam, which
      names the independence assumption as the forward promise ¶5 pays. The gap sentence was cut
      as redundant with the opener's "keep the product as the target". SuperDiff also handles
      logical OR, a mixture; we cite its product side, per its register row.

  **Placed so far (¶1a):** "Recent methods compose pretrained diffusion models at inference by
  sampling from the product of the individual concept distributions. Conditioning a pretrained
  model on a single concept prompt concentrates its distribution on images showing that concept,
  and this conditional distribution is each expert's contribution to the product. This
  multiplication is the product-of-experts construction \citep{liu2022compositional,
  zhang2025product}, in which the product stays large only on images to
  which every expert assigns high probability, so a sample must satisfy both concepts at once.
  A prompt such as ``a cat and a dog'' asks for plurality, each concept appearing in the scene as
  its own distinct object, the way the two animals co-occur in captioned training images.
  Assigning high probability under both experts and appearing as two distinct objects in one scene
  are different demands, and the product enforces only the first. In this paper we focus on
  conflicting concept pairs, where composition loses plurality, and show that the loss is
  measurable as a per-step residual that can be learned."

  **¶1a is fully placed**, six sentences, awaiting ¶1b before compile.

  **Placed so far (¶1b):** "Existing methods keep the product as the target and improve how it is
  composed and sampled \citep{dutta2026steer, huang2026factordiff, wang2026testtime}.
  \citet{liu2022compositional} treat a prompt naming two concepts as a conjunction, one expert per
  concept, and add the experts' predictions directly at each denoising step.
  \citet{du2023reduce} show that reverse diffusion on the added predictions does not sample the
  product the addition aims at, and correct the mismatch with Markov chain sampling. Later methods
  sharpen the correction, with annealed importance sampling \citep{zhang2025product} and
  sequential Monte Carlo correctors \citep{skreta2024superdiff, skreta2025feynman}. The rest of
  this section lays out how the predictions these methods combine are produced, and where the
  assumption of independence between the composed concepts enters."

  **¶1b is fully placed**, five sentences (the gap sentence cut as redundant with the opener).

  **Placed so far (¶2):** "A diffusion model generates an image by starting from pure Gaussian
  noise and removing noise step by step. Between an image $x_0$ and pure noise $x_T$ sits a fixed
  sequence of increasingly noised versions, produced by adding Gaussian noise one step at a time
  \citep{sohldickstein2015deep, ho2020denoising}."

  The Du sentence is proof-backed (RRR section 4, eqs. 11-12, Figure 2, verbatim quotes on the
  register row), honouring the condition attached when the expert-definition sentence was placed.
  **Every key in both paragraphs now resolves**: the bib holds 26 entries with no duplicates, and
  the two entries whose author lists were unverified (`huang2026factordiff`, `wang2026testtime`)
  were corrected against the arXiv abs pages on 2026-08-27.

  The opener's citep carries the three registered methods the survey never details (CO3
  2509.25940, FactorDiff 2607.11758, the test-time distillation LoRA 2605.07078), so it passes the
  more-than-two rule without double-citing the four methods the next sentences name individually.
  This also settles 2605.07078's open placement. All three keys need bib entries before compile.

  The demands sentence is ours, uncited by design (the 2026-08-27 RRR check confirmed the surveyed
  line never asks it), and it stops short of saying which pairs make the demands come apart or
  what wins when they do; section 3 owns both.

  **Citation note on the placed sentence 3**: Hinton's PoE origin was dropped from the citep by
  decision rather than verified; the construction is attributed to its diffusion-era uses. If a
  reviewer asks for the origin, the verification run is still a single slot.

  The expert is defined by conditioning, never by training on one concept's images: SDXL is
  internet-trained, and "fit to one concept's distribution" would misdescribe every run. The
  grounding-in-training-data beat survives only in the captioned-scene sentence, where it is true.
  Condition attached by the author to placing sentence 2: ¶1b's Du sentence (why sampling from the
  product is the problem, RRR section 4 proof) must stay referenced and cited.

  **How this section is walked: narrative flow is the standing check.** Every sentence is judged
  against its neighbours and the six-row arc, not in isolation. Each `place` reprints the seam and
  says what the sentence hands the next one. Vocabulary is held to the abstract's and
  introduction's register (predictions, prompts, denoising steps, what the model is asked), never
  meta-language about the paper ("operations", "ingredients", "standard components").

  2605.07078, the LoRA-absorbs-PoE precedent, is placed: it sits in ¶1b's opener citep. Whether
  section 5 also discusses it as method precedent is that walk's call.

**Section 2 carries its equations.** Decided in the walk: the reader needs the material to follow
the paper, so ¶2 to ¶4 each state their standard result as an equation or align block, not as
prose with a citation. ¶2 the forward process and the noise-prediction objective, ¶3 the
noise-to-score identity in full (the introduction's line 96 states it inline and compressed; this
is where it is stated properly, per `dhariwal2021diffusion` section 4.2 equation 11), ¶4 the
guidance rule. The `step` control walks them one equation per round.
- [ ] **§2 ¶2** diffusion, five pieces sketched, walking sentence 1. Generation stated as
      noise-to-image, training's forward process as image-to-noise with `x_0`, `x_t`, `T` named in
      prose, the closed-form jump (eq A) defining `alphabar_t` because ¶3's identity needs it, the
      noise-prediction objective (eq B), and a closer tying the outputs back to ¶1b's survey. The
      per-step Gaussian is deliberately absent (nothing downstream uses it), and condition
      dropping is ¶4's. Cites `sohldickstein2015deep` and `ho2020denoising` on the forward-process
      sentence.
- [ ] **§2 ¶3** the score view. Why the network's output is a scaled score, and why that is what
      licenses adding two predictions together at all. It may not restate the noise-to-score
      relation, which is already at [iclr2027_conference.tex:96](iclr2027_conference.tex#L96) where
      `eq:poe-composition` first needs it. Cites `song2021scorebased` for the score-matching framing
      and `dhariwal2021diffusion` for the identity itself, section 4.2 equation 11.
- [ ] **§2 ¶4** classifier-free guidance. Condition dropping gives one network two predictions, and
      guidance combines them linearly. This is the first linear combination in the paper, and
      section 3's align block 3.2b leans on it. Cites `ho2022classifier`.
- [ ] **§2 ¶5** close, and the section's claim, now formally stateable because ¶2 to ¶4 built the
      machinery. The linear combination is exact only under conditional independence, Bradley
      2502.04549 states the condition and names the failure modes, CoInD 2503.01145 shows training
      does not produce the property and fixes it in the loss, and no one tests it at inference.
      Ends handing section 3 the assumption. Does not restate the contribution bullets at
      [iclr2027_conference.tex:117-123](iclr2027_conference.tex#L117-L123). CO3 2509.25940 moved to
      ¶1's survey or the baseline discussion; it no longer sits here.

  **The survey in ¶1 may not say the experts are one network.** 2506.08894 formalises the product
  across heterogeneous experts and 2412.17762 superposes different models. One network under two
  prompts is this paper's setting and guidance's, not the literature's. The survey says the methods
  combine the models' own noise predictions, plural.

**If the paper runs long, ¶2 and ¶3 are where this section gets cut back to one paragraph.** Six
paragraphs at five sentences is roughly three quarters of a page for a section carrying no claim of
the paper's own, in front of a section 3 that is already six paragraphs.

**The weakness list is not this section's.** In the map's numbering the introduction's paragraphs
are 2.1 to 2.7, so "2.2, models scale, compositional reasoning stays brittle" is the introduction's
second paragraph, compiled at [iclr2027_conference.tex:88](iclr2027_conference.tex#L88) with four
literal `(Author et al., YYYY)` placeholders. Physical structure, attribute binding, negation and
counting fill those placeholders. Section 2 stating them again would duplicate line 88.

**The section's claim already has its source.** The independence assumption, and the fact that
composition methods build on it without testing it, comes from Bradley et al.,
[arXiv 2502.04549](https://arxiv.org/abs/2502.04549), which states the condition under which
linear score addition is exact and separates two failure modes: the wrong target (a chimera) and
a non-smooth path (collapse). It is registered, it has a bib entry (`bradley2025mechanisms`), and
2.1 is written against it. CoInD ([arXiv 2503.01145](https://arxiv.org/abs/2503.01145)) sits beside
it as the training-time counterpart, the paper that checked whether training actually produces the
assumed independence and then changed the loss so it does.

**2.2's weakness list now has sources too.** Physical structure, attribute binding, and negation
each have a register row, and counting has one pending a single protocol check. See
[open citations](#open-citations) for the mapping and the two constraints those rows put on the
prose.

Parked for this section: Skreta 2412.17762 and 2503.02819, routed here from the introduction.

**What survives from the deleted Background section, and what does not.** Section 3 already
explains what product-of-experts composition computes at one step, so the background material has
shrunk. Each of the six old paragraphs was judged on whether a later section leans on it, since a
background piece nothing downstream needs is cut rather than kept for completeness.

| Old paragraph | Verdict | Why |
|---|---|---|
| training, the noise-prediction objective and condition dropping | keeps its paragraph, §2 ¶1 | Section 3.1 evaluates the network under the unconditional condition and asserts nothing is trained. Neither reads unless one set of weights is known to yield both predictions |
| classifier-free guidance | keeps its paragraph, §2 ¶2 | Section 3's align block 3.2b attaches a weight to each push and calls the result the form used in practice |
| the forward process, fixed rather than learned | one sentence inside §2 ¶1 | The objective cannot be stated without saying what the network predicts, and section 3 never returns to the corruption process. The whole-paragraph loose draft stays in [Loose lines](#loose-lines) |
| the reverse process and the score the network stands in for | cut | Already stated at [iclr2027_conference.tex:96](iclr2027_conference.tex#L96), which is the paragraph that first needs it |
| inference, one sampler step | cut | Nothing downstream needs it |
| the notation everything after uses, gathered in one place | cut | The spine says definitions travel to the paragraph that first needs them |

### 3. ✅ Missing Implications: The Plurality Term

◀ **Needs: section 4** for the notation and for guidance.

The introduction already states the composition rule. This section owns the assumption behind it
and the term it drops, and may not merely restate the introduction.

The probability-space derivations live in the appendix, subsection `app:poe-derivation` under
Additional Method Details: the four-line Bayes route to the prior-corrected product (with
`eq:independence` on its first line), the two-line R_t definition, and the five-line route ending
at `p(x_t|c_1,c_2) \propto p_PoE(x_t|c_1,c_2) R_t`. The main text states each of those results in
one prose sentence, carries no reference to the subsection, and keeps the noise-space blocks, the log-to-noise-prediction block of 3.2b,
the eps_true block of 3.6e, and the two labelled display equations of 3.3.

- [x] **3.1** what PoE computes: the same three forward passes, combined differently, at every
      step. Complete, four sentences placed and one cut.
  - [x] 3.1.1 At each denoising step, product-of-experts composition evaluates the pretrained
        network on the same noisy latent under three conditions: the first prompt, the second
        prompt, and the unconditional condition.
  - [x] 3.1.2 These three predictions are then combined to form the PoE noise estimate given in
        `\eqref{eq:poe-composition}`, by adding the two conditional predictions and subtracting
        the unconditional one.
  - [x] 3.1.3 Each conditional prediction contains the unconditional prediction plus a term
        carrying its own prompt.
  - [~] 3.1.4 cut. It was going to state that the implementation guides each expert with
        w = 6.5 before combining them. Align block 3.2b now derives the weights from the
        regrouping and states the practical form itself, so 3.1.4 would preview an equation the
        reader is about to be given properly.
  - [x] 3.1.5 Three properties of this construction carry into the rest of the section: nothing
        is trained, the network is never shown both prompts at once, and the composed prediction
        is a fixed linear combination of three of its own outputs.
- [ ] **3.2** the derivation, one equation per round: the rule follows from, and only from, the
      concepts being conditionally independent given the image. Owed to 3.1.3, which asserts that
      a conditional noise prediction splits into the unconditional prediction plus a conditioning
      term. 3.2 derives that split rather than restating it in words.
  - [x] 3.2a from the independence assumption to the prior-corrected product. In the main text
        this is one prose sentence carrying the factorised form inline, followed by the
        denominator sentence. The main text does not reference the appendix subsection. The four-line align block
        (same left-hand side throughout, Bayes substituted into each factor in line 2,
        p(c1) p(c2) collected in line 3 and dropped in line 4 because it does not depend on x)
        sits in that appendix subsection, with `eq:independence` on its first line.
  - [x] 3.2b from that product to the sampler's noise prediction, one align block of four lines,
        one move per line. Logarithm, then gradient in x with the constant dropping, then the
        score relation carrying it into noise predictions, then a regrouping into the
        unconditional prediction plus one push per concept. Followed by one sentence attaching a
        weight w_i to each push, giving the form used in practice, which recovers the derivation
        exactly at w_i = 1.

  Paragraph 3.2 is complete. Compiling it lands the correction to
  [iclr2027_conference.tex:96-101](iclr2027_conference.tex#L96-L101) in the same diff. The new
  section goes **after** Problem Setting and Background and **before** Methodology at line 161,
  matching the reading order in the position table. An earlier note in this file put it between
  Related Work and Background, which contradicted the position table.

  **3.2 has no opening sentence.** Every other paragraph in section 3 was walked with one and
  this one was not, so as it stands the paragraph opens on an align block.

  The route follows [arXiv 2206.01714](https://arxiv.org/pdf/2206.01714) rather than inventing
  one, so the paper's algebra and the paper it builds on agree line for line.

  This replaces the align block compiled at
  [iclr2027_conference.tex:96-101](iclr2027_conference.tex#L96-L101). That block asserts
  p(x | c1, c2) = p(x | c1) p(x | c2) on its first line and contradicts it with a 1/p(x) on its
  second. The prior-corrected product with 1/p(x) is right and the first line is what is wrong,
  because the assumption belongs on p(c1, c2 | x), not on p(x | c1, c2). The compile of 3.2 is a
  diff against already-compiled introduction text.
- [x] **3.3** naming r_t as the exact residual the assumption drops, plus the corrected sampler.
      Complete, five sentences placed.
  - [x] 3.3.1 We define the plurality term r_t as the difference between the model's prediction
        under the joint prompt, written eps_Mono, and its product-of-experts prediction, both
        evaluated on the same noisy latent at the same step. The symbol is glossed at first use
        because its explanation (3.5.3) sits three paragraphs later. Carries the display equation
        `r_t = eps_Mono(x_t, t) - hat_eps_PoE(x_t, t)`, labelled `eq:interaction-term`.
  - [x] 3.3.2 The identity eps_Mono = hat_eps_PoE + r_t holds by construction and is not a
        result, whereas the size of r_t, its structure across denoising steps, and its transfer to
        concept pairs held out of training are. Commits ¶5.6 to all three. Register check
        passed: F3 built for size against noise level, F4a to F4e built for structure across
        steps, F8a built for transfer to held-out pairs.
  - [x] 3.3.3 Correcting the sampler means adding a scaled copy of r_t back at each step, with
        lambda = 0 leaving composition uncorrected and lambda = 1 reproducing the joint-prompt
        prediction exactly. Carries the display equation
        `hat_eps_lambda(x_t, t) = hat_eps_PoE(x_t, t) + lambda * r_t`, labelled
        `eq:corrected-sampler`. This is where lambda, the axis every dose figure sweeps, is named.
        Whether lambda is constant across steps or scheduled is left to methodology.
  - [x] 3.3.4 Product-of-experts sampling never sees the joint prompt. The difference r_t
        therefore serves as a signal we can measure, one that establishes what the missing
        correction is, when in the denoising trajectory it matters, and why composition fails
        without it. Two sentences, placed together. The three questions are ¶5.6's own
        structure: F3 for what the correction is, F4a to F4e for when it matters, F2 for why
        composition fails without it. No forward reference to the adapter here, by decision.
  - [x] 3.3.5 r_t is a full noise-prediction vector at every step, so it has a direction as well
        as a magnitude. Replaces the sketch's forward pointer to the rest of the paper, which was
        cut because 3.3.2 already names the three claims and the paragraph should not look ahead
        a third time. Sets up F2, whose controls hold magnitude fixed and change direction.

  Grounded in `run_teacher_residual` at
  [_sampling.py:367-405](../../poe_repair/methods/_sampling.py#L367-L405) and the update
  `eps_t = eps_poe + lam * delta_used` at
  [_sampling.py:593](../../poe_repair/methods/_sampling.py#L593). The code calls the quantity the
  teacher residual and writes it delta_t. The paper calls it r_t. One name has to win.
- [x] **3.4** why concept pairs violate the assumption. Complete, four sentences placed and one
      cut.
  - [~] 3.4.1 cut. It restated in words the assumption align block 3.2a already states in
        symbols with the annotation naming it. The paragraph opens on 3.4.2 instead, whose
        opening words "that assumption" carry the backward reference.
  - [x] 3.4.2 The independence assumption behind the product form fails once the two concepts
        have to share one image, since a patch of pixels that reads strongly as one animal cannot
        read equally strongly as the other. The paragraph's opening sentence. It names the
        assumption by the form that carries it in the main text; `eq:independence` sits in the
        appendix, so the sentence carries no equation reference.
  - [x] 3.4.3 Sampling from that product therefore steers toward whatever image satisfies both
        experts at once, and for two animals competing for the same region that is a single
        blended animal rather than two. Describes a process, not a maximum. A diffusion sampler
        follows scores and does not find the argmax of a density, so "the product is largest at"
        was wrong and is logged as such. Illustrated by F1 and measured by nothing, which 3.4.5
        must say.
  - [x] 3.4.4 In every pair we study both concepts are foreground subjects, rather than a subject
        and a setting that could occupy different parts of the image. Scopes rather than claims. Every earlier candidate asserted a law
        about when composition fails, which the evidence does not carry, and F2b is held out of
        the argument chain by its own register row.
  - [x] 3.4.5 The same network produces two distinct animals when it is given the joint prompt,
        so the failure belongs to the composition rule and not to the model. Replaces the
        sketch's boundary sentence, which would have been section 3's third hedge after 3.3.2
        and 3.3.4. Supported by Figure 1, whose right panel shows the uncorrected composed sample
        beside what joint prompting gives.

  Every causal sentence here needs a register row before it is placed. F2b's claim line is that
  the pair does not fuse merely because the two animals look alike, so 3.4 may not attribute the
  failure to visual similarity.
- [x] **3.5** how r_t is obtained in practice, and the boundary of that proxy. Complete, six
      sentences placed across five slots.
  - [x] 3.5.1 In practice both predictions come from one pass of the network per denoising step,
        evaluated on the first concept, the second concept, the joint prompt, and no prompt.
        "Four-branch pass" is the code's word for how the four conditionings are batched into one
        UNet call and does not appear in the manuscript.
  - [x] 3.5.2 Both are read at the same noisy latent, the one the corrected sampler is currently
        visiting, so r_t is measured along the trajectory it is applied to rather than imported
        from a separate run. Kept because the assumption it forecloses, that r_t is computed once
        along a joint-prompt run and replayed, is the natural one to make and would be a
        different quantity.
  - [x] 3.5.3 The true joint score \nabla_x \log p(x | c_1, c_2) is a property of the world,
        whereas eps_Mono is what this network outputs when its text encoder is handed one prompt
        naming both concepts. So r_t measures a gap between two behaviours of one model, not a
        gap to the correct distribution. Two sentences, placed together.

        **The score is written over the clean image x, not over x_t**, matching paragraph 3.2.
        Whether the move to the noisy latent is legitimate is what 3.6 establishes, so 3.5 may
        not help itself to it.
  - [x] 3.5.4 At lambda = 1 the corrected sampler reproduces the joint-prompt prediction exactly,
        so the most this correction can do is bring composition up to what joint prompting
        already achieves. Follows from 3.3.3 and 3.5.3 together and introduces nothing new.

        **Open question this raises for the limitations section.** If any benchmark pair fails
        under joint prompting, the oracle correction cannot help it, since at full strength it is
        joint prompting. Whether any such pair exists has not been checked.
  - [x] 3.5.5 Joint prompting is the behaviour inference-time composition is trying to reproduce,
        so matching it is what this correction is for. States the target and argues nothing,
        chosen over a version collecting on the introduction's retraining-cost argument and over
        one that framed the ceiling as the experimental question.

  Grounded in the teacher-residual sampler docstring at
  [_sampling.py:395-406](../../poe_repair/methods/_sampling.py#L395-L406): a single four-branch
  UNet call on (A, B, J, empty) per step, with lambda_max = 1 and a constant schedule reducing to
  literal joint prompting.
- [x] **3.6** the exact plurality term, three small align blocks with prose between them.
      Complete: six prose sentences placed across five slots, three blocks, one slot cut.
      Appended after the walk started, not a renumbering. **Reads immediately after 3.2**, before
      3.3.

  Shape B, adopted. Three blocks rather than one, so the payoff line lands in prose instead of
  inside an equation array. Every block moves one step per line and shows the step it takes.

  - [x] 3.6a **block 1, the factor, and the sentence that introduces it.**
        Opening sentences placed: "So far r_t has been defined through one network's behaviour.
        The quantity it stands in for can be written exactly, in terms of the two concepts
        themselves." Two sentences, chosen over versions that opened on the independence
        assumption, on the gap being a single factor, and on the derivation being repeated
        without the assumption.

        **"The same quantity" was trimmed to "the quantity it stands in for."** The original
        asserted r_t and the exact term are one thing, which 3.5 says they are not, and it
        promised a comparison the paragraph then owed. "Stands in for" matches 3.5's proxy
        language, claims no identity, and owes nothing: the blocks deliver the exact form and the
        paragraph is done.
        In the main text R_t is defined inline in prose, as the ratio
        `p(c_1,c_2|x_t) / (p(c_1|x_t) p(c_2|x_t))`. The two-line display (the definition, then
        the identity rearranged as `p(c_1,c_2|x_t) = p(c_1|x_t) p(c_2|x_t) R_t`, shown rather
        than substituted silently) sits in `app:poe-derivation`. The introducing sentence is
        walked here rather than minted separately.
  - [x] 3.6b **the sentence after block 1.** Placed: "R_t equals one exactly when the two
        concepts are independent given the state." The trailing clause "and departs from one by
        however much they are not" is cut. It repeated the vague-quantity move already turned
        down at 3.6a, and what R_t does when independence fails is block 2's and 3.6d's job.
  - [x] 3.6c **block 2, into the joint.** In the main text this is one prose sentence ending at
        `p(x_t|c_1,c_2) ∝ p_PoE(x_t|c_1,c_2) R_t`, unreferenced to the appendix. The five-line
        align block (same left-hand side throughout: Bayes with no assumption, the rearranged
        line substituted, Bayes on each factor, collecting, and recognising the composition
        rule) sits in `app:poe-derivation`.

        **Line 3 leans on the four-line block.** Its annotation points back to
        `eq:independence`, where the per-factor Bayes step is shown in full with p(c_1) p(c_2)
        appearing and being dropped. Both blocks sit in the same appendix subsection, so the
        pointer resolves locally.

        **Variable mismatch, deliberate.** 3.2's derivation is over the clean image x and this
        block is over x_t throughout. That is the drift, which lives in the discussion as 6.2.
        A careful reader will notice the two blocks use different variables three paragraphs
        apart, and nothing in section 3 explains it.
  - [x] 3.6d **the sentence after block 2.** Placed: "At every step the sampler combines the two
        experts as though this factor were one, whether or not it is." Says what happens during
        sampling rather than what the rule assumes, which is the one thing neither 3.6b nor block
        2 has said. Seven earlier candidates were turned down, for repeating 3.6b's R_t = 1, for
        reframing rather than adding, or for reading badly. R_t is deliberately not called the
        plurality term here, since 3.3 gave that name to r_t and 3.5 spends four sentences
        keeping the two apart.

  - [x] 3.6e **block 3, score and noise predictions.** Placed at three lines, one move each:
        logarithm and gradient of the identity `p ∝ p_PoE R_t`, then multiplying through by minus sigma_t,
        then the score relation. Ends at
        `eps_true(x_t, t) = hat_eps_PoE(x_t, t) - sigma_t grad_{x_t} log R_t`.

        **Two things this block leaves unsaid, both owed at compile.** Line 1 drops block 2's
        proportionality constants without a word; 3.2b annotates the same move with "the constant
        drops" and this should point back to it. And eps_true appears for the first time in the
        last line with no introduction, which invites a reader to think it is obtainable. It is
        the noise prediction of the true joint score, which 3.5 says no model gives you.

  - [~] 3.6f **cut.** It was to say how r_t reaches the exact term, in one of three forms: the
        plain restatement, the condition under which the substitution would be exact
        (eps_Mono = eps_true, and nothing else), and the necessity version (R_t cannot be
        evaluated at sampling time). All three closed a loop that 3.6a no longer opens, now that
        its promise is trimmed. The exactness condition is worth keeping somewhere and is the
        strongest of the three; the discussion is where it would go.

  - [x] 3.6g **the guard, with a figure.**
        Sentence 1 placed: "Figure 1b shows one sample from each of two prompt pairs, with the
        size of r_t at every denoising step beneath it, and the correction is present in both
        trajectories while only one sample is wrong." Chosen over a version naming the controlled
        comparison first and a shorter one pointing only at the curves.

        Sentence 2 placed: "R_t is a property of the two concepts and r_t is a property of one
        model's two predictions, so no reading of size carries from one to the other. What the
        curves do carry is timing, and when the correction has to arrive is what we measure next."
        Chosen over "two pairs cannot say whether the size of the correction
        predicts which pairs fail, and this paper does not ask", which stated a limit and gave no
        reason for it.

        **It does three jobs.** It gives the reason size cannot be read across, that r_t's size
        reflects how the joint prompt is encoded as much as how the concepts interact. It
        redirects to timing, which is what F4a to F4e actually measure. And it hands forward without naming a section
        number. "Build" was changed to "measure" once the running order settled: the next section
        applies the oracle correction and measures when it has to arrive, and the adapter is
        built the section after that.

        **Left to the register's caption caps.** That two pairs cannot settle whether size
        predicts failure, that on these two the failing pair's correction is the larger so a size
        reading would point against the paper, and that the two curves differ in shape as well as
        height with one cell each.

        **Stated over r_t throughout, never R_t.** r_t is a difference between two of one
        network's noise predictions. R_t is a ratio of true co-occurrence probabilities and is
        measured nowhere. The guard makes its point on the object the paper can show.

  **Hard limits on section 3, recorded so a later pass cannot widen them.** R_t is never measured
  for any pair. The paper may not claim that R_t not equal to one predicts failure, that pairs
  which compose successfully have R_t near one, or any ordering of pairs by R_t. Two of its own
  figures argue against such a claim: F4b says where the correction is large is not where it
  matters, and F3 says the correction's size follows noise level with one shared curve across
  pairs. Separately, r_t and the exact term minus sigma_t grad log R_t are never shown equal;
  3.5 calls r_t a proxy and 3.6f may not upgrade that to an identity.

  **Pinned, held for later.** log R_t is the pointwise mutual information between the two
  concepts given x_t. The paper does not name it that yet, by decision. The routed pressure-test's
  novelty verdict is the input for that decision.

  **This deliberately redoes 3.2's derivation.** 3.2 imposes independence at its first line and
  gives the rule the field uses. 3.6 imposes nothing and recovers that rule as the case R_t = 1.
  The repetition is the argument, so 3.2 stays as it is.

  The drift between the clean image x and the noisy state x_t is cut from section 3 and lives in
  the discussion as 6.2.

- [x] **3.7 the citation pass, introduction to the end of section 3.** Compiled 2026-08-27, build passed.

  A sequential read of every sentence from [line 86](iclr2027_conference.tex#L86) to
  [line 214](iclr2027_conference.tex#L214), one paragraph per round, each sentence flagged
  **cite** with its source or **no** with the reason. Section 2 is a bare heading and is skipped.
  No new prose unless a citation forces a rewording. The pass compiles once at the end.

  **Nothing resolves until the bibliography comes back.** `\bibliography` and `\bibliographystyle`
  are commented out at [lines 251-252](iclr2027_conference.tex#L251-L252), so every `\citep`
  renders as a bold `?` until that hunk lands. Two sources in this run have no bib entry,
  2506.08894 (shared with the introduction, so registered once) and 2503.01145.

  **The paragraph ledger.** One row per paragraph, in reading order.

  | Paragraph | Line | State |
  |---|---|---|
  | intro ¶1, natural scenes and what generating them requires | 86 | walked, nothing to cite; "usually" left unquantified by decision |
  | intro ¶2, progress and the compositional weaknesses | 88 | placed: s1 `saharia2022photorealistic` + `podell2023sdxl`; s3 reworded to drop "and dynamics" and cites 2306.05720 + 2311.17137; s4 binding and counting 2307.06350, negation 2411.17066; s5 `liu2022compositional` + 2402.01103 on the motivation clause. Counting's register check ran and passed, see Open citations |
  | intro ¶3, the conditioning signal and joint prompting | 90 | placed: the literal `(cite)` becomes `rombach2022high`, the architecture key per its register row; the checkpoint key `podell2023sdxl` already appears in ¶2 and belongs to the methods section. Nothing else in the paragraph cites |
  | intro ¶4, inference-time composition and plurality | 92 | placed: s3 "product-of-experts (PoE) formulation" takes `liu2022compositional` + 2506.08894 (new entry, key `zhang2025product`), the one pre-section-3 home for the formalism key. The often-fails sentence is no by decision, carried by Figure 1, with the failure-mode literature cited where its condition is stated. Hinton's PoE-origin naming has no register row and is left uncredited unless a scout run adds one |
  | intro ¶5, the independence assumption | 94 | placed: s1 takes `bradley2025mechanisms` (2502.04549's registered slot, beat 6), already in the bib. The signpost sentence cites nothing |
  | intro ¶6, the score identity and the composition rule | 96 | placed, with the pass's one elective rewording: "relates to the score via" becomes "is a scaled negative score:", which is the cited equation 11 rearranged one move, then `dhariwal2021diffusion`. Verified against the selection file's verbatim quote. No connective sentence added before the rule, since that step is section 3's opening move and ¶5's signpost already defers it there. The rule sentence cites nothing, Liu credited twice above |
  | intro ¶7, what the linear rule leaves out | 103 | placed, no citation anywhere in it by the author's decision. The interaction-term sentence does not take the CoInD key; CoInD's credit and the relation to it (they remove the term in training, we measure and re-inject it at sampling) belong to the related-work walk. The natural-concepts-interact sentence is motivation carried by Figure 1 and 3.4, uncited by decision |
  | intro ¶8, the cat and dog example | 105 | placed, nothing to cite, all four sentences ours. One mechanical fix staged: the literal "Figure 1" becomes `Figure~\ref{fig:poe-failure}`, the only unreffed figure mention in the file |
  | Figure 1 caption | 110 | placed, nothing to cite, both panels ours and the claim stays inside them |
  | intro ¶9, why the gap matters | 114 | placed: the retraining-infeasibility sentence takes `du2024compositional` (2402.01103), the same key ¶2's motivation clause carries, kept in both places since both state the claim and neither says "exponential". Rest of the paragraph ours |
  | Contributions, three items | 119-121 | placed, nothing to cite by convention, SDXL already keyed in ¶2 |
  | 3.1, what PoE computes | 130 | placed: s1 takes `liu2022compositional`, the section-opening repeat kept for readers who jump straight to section 3. Nothing else cites. The stand-in opener "The rule follows from that assumption and from nothing else" stays bare, its two prose defects recorded below for the 3.2 walk. `zhang2025product` confirmed ¶4-only, its AIS-over-heterogeneous-experts formalism not being the route this derivation takes |
  | 3.2, the derivation | 132-143 | placed. Line 132's stand-in opener is replaced by "Product-of-experts composition is obtained as follows \citep{liu2022compositional}.", chosen from fourteen rounds of candidates: a light functional lead-in, no restatement of the assumption (intro ¶5 states it with Bradley's key, and the block's first line keeps the `eq:independence` label 3.4 points at), no derivation announcement. Nothing inside the align blocks or the denominator sentence cites |
  | the weighted form | 155 | placed: `\citep[equation 11]{liu2022compositional}` on "the form used in practice", the verified verbatim match, answering the whose-practice question with Liu rather than classifier-free guidance |
  | 3.6, the exact term | 157-187 | placed, no keys anywhere in it: the contribution cites nothing, the seam's missing key IS the departure marker after three Liu-keyed paragraphs, and the score identity stays keyed once at its intro first use (the once-or-twice question answered with once). The seam sentence at 157 is replaced: "So far $r_t$ has been defined through one network's behaviour. The quantity it stands in for can be written exactly, in terms of the two concepts themselves." becomes "The gap between the joint conditional and the product form can be written as", running into the R_t block. Chosen for register and because it names no $r_t$, which sidesteps the ordering defect below |
  | Figure 2 caption | 192 | placed, nothing to cite, our samples and curves |
  | 3.3, r_t named and the corrected sampler | 196-210 | placed, nothing to cite, all ours. The CoInD attribution question is settled the same way as intro ¶7: deferred to related work |
  | 3.4, why concept pairs violate the assumption | 212 | placed: the blended-animal outcome sentence takes `bradley2025mechanisms` (the wrong-target failure in their taxonomy) and `dutta2026steer` (CO3's mode-collision account, and this paper's baseline), keys on the outcome, not on our steering explanation. Our scoping sentence stays bare, a key there would credit our study design elsewhere |
  | 3.5, how r_t is obtained | 214 | placed, nothing to cite, implementation prose grounded in the code |

  **Ordering defect, found by the sequential read, owed to the section walk.** The compiled order
  puts 3.6 (lines 157-187) before 3.3 (line 196), so the old seam's "So far $r_t$ has been
  defined" claimed a definition thirty-nine lines below it. The replacement seam sentence no
  longer depends on the order, but the structural question stands: the map's own walk order was
  3.3 before 3.6, and either 3.3 moves up or the map's 3.6 entry records the compiled order as
  final.

  **Findings carried out of the first three rounds, before the pass was made sequential.**

  **2206.01714's equations verified against the ar5iv full text**, correcting the routing in
  [Open citations](#open-citations). Equation 9 is `p(x|c_1..c_n) ∝ p(x) ∏ p(c_i|x)`, our
  `\label{eq:independence}` line at [135](iclr2027_conference.tex#L135). Equation 10 is the Bayes
  substitution, our [136-138](iclr2027_conference.tex#L136-L138). Equation 11 is
  `hat_eps = eps_theta + Σ w_i (eps_theta(·|c_i) − eps_theta)`, which is the weighted form at
  [155](iclr2027_conference.tex#L155). Equation 6 is the energy-based product form and appears
  nowhere in our derivation, so the map's "6, 9, 10 and 11" was wrong on one of four. The
  log-and-gradient step at [145-152](iclr2027_conference.tex#L145-L152) is Liu et al.'s appendix
  equations 17 and 18, not a numbered main-text equation.

  **The weighted form at line 155 is Liu et al. equation 11 verbatim**, `w_i` and all. Its source
  is 2206.01714, not 2207.12598. Classifier-free guidance is the `n = 1` special case, and citing
  it there would credit the general form to the paper that wrote the particular one.

  **2506.08894 does not fit line 130.** Its register row formalises product-of-experts as sampling
  from the product distribution across *heterogeneous* experts via Annealed Importance Sampling.
  Line 130 describes one network evaluated three times and combined linearly, which is Liu et
  al.'s construction. The formalism key belongs with the product distribution itself, or with the
  introduction, which shares it.

  **Two prose findings raised at line 132, referred out of this pass.** "The rule follows from
  that assumption and from nothing else." Its "that assumption" has no antecedent in section 3;
  the referent is [line 94](iclr2027_conference.tex#L94), across a section heading. And "from
  nothing else" asserts the single-premise property that 3.6 spends three align blocks earning,
  twenty-five lines before the paper has it. Both are symptoms of the already-recorded fact that
  **3.2 has no opening sentence** and line 132 is standing in as one. Whether the sentence
  survives is a prose walk, not a citation call, so this pass hangs no citation on it.


Correction owed and carried by 3.2. The compiled derivation at
[iclr2027_conference.tex:97](iclr2027_conference.tex#L97) asserts p(x | c1, c2) = p(x | c1) p(x | c2)
and calls it conditional independence, then line 98 turns the same left-hand side into a
proportionality with 1/p(x) in it. Both cannot hold. The assumption that licenses the rule is
p(c1, c2 | x) = p(c1 | x) p(c2 | x), which is what
[IMMERSE_PoE_Foundations.md:35](../../docs/IMMERSE_PoE_Foundations.md#L35) states correctly. Fixing
the intro block is part of compiling 3.2, and it is a diff against already-compiled text.

▶ **Next: section 5.**

### 4. ✍️ Restoring the Plurality Term Restores Composition

State: skeleton simplified and frozen, walking paragraph 4.1's sketch. Its claim, from
[SPINE.md](SPINE.md): a small amount of the measured plurality term restores composition when
it arrives early in the denoising run, and the full amount arriving late does not. The section
is built around that one claim and the grids that show it. Two things are deliberately not
claims of this section: that more correction composes more, which goes without saying since
the correction is built from the joint prompt's own prediction, and that size-matched
mismatched corrections fail, which shows nothing a reader did not already expect. Both live in
Extras with the strength grid.

**The argument flow, adopted.** Five moves, each a plain claim, in the register of abstract
candidate 3. The question (what does restoring composition require, how much and when). What
we inject (a fraction lambda of the measured term per step, computed from the joint prompt,
so a measurement and not a method). The counting sentence (a detector finds two distinct
animals, manually verified). The three-grids result (a small early amount is enough, the full
amount late does nothing, the heatmap as likely headline, mechanism explicitly left open).
The closer (the correction still comes from the joint prompt, and section 5 learns to produce
it without that prompt). The trajectory-separation move was cut, reasons recorded under 4.4.

- [x] **4.1** the question, and what we inject: compiled, three sentences, result-first shape.
      The compile also trimmed section 3's duplicate ceiling sentence from its proxy
      paragraph, so the ceiling now appears once, at its point of use.
  - [x] s1 placed, result-first: A small fraction of the measured plurality term, applied in
    the first steps of the denoising run, is enough to restore composition.
  - [~] the bridge sentence (section 3 delivered a definition and a measurement) and the
    question sentence (how much and when) both cut, absorbed by the result-first opening.
  - [x] s2 placed, endpoints-first: Adding $\lambda\, r_t$ at each step interpolates between
    plain composition at $\lambda = 0$ and the joint-prompt prediction at $\lambda = 1$.
  - [x] s3 placed and pinned, the ceiling angle in the D2 rendering, swapped in by the user's
    explicit pick: The corrected sampler can therefore do no more than reproduce joint
    prompting, which it equals exactly at $\lambda = 1$. A later round may not reword it
    without asking. Two notes carried. Reconciliation owed with section 3's 3.5.4, which
    states nearly the same ceiling; at a later pass either 3.5.4 is trimmed or the echo is
    kept as deliberate restatement at the point of use. And the term's joint-prompt provenance
    is not stated in this paragraph, it rests on s2's "joint-prompt prediction" endpoint, with
    the full statement living in section 3 and in the closer.

  ¶4.1 is fully placed at three sentences, ready for compile.
- [~] **4.2** absorbed by decision: compose rate is not a proposed definition, it is an
      off-the-shelf detector counting animals, manually verified, and with samples this small the
      count could have been done manually. It gets one or two sentences inside 4.3, not a
      paragraph. The full treatment (definition, validation, the boxes image) lives in Extras.
      The ratio-and-dial piece moves into 4.3 with it.
- [~] **4.3** absorbed by decision: the amount-grows claim goes without saying (the correction
      is built from the joint prompt's own prediction, so getting the answer back when adding
      the answer is unremarkable) and is not argued in the main text. The counting sentence
      and the ratio-and-dial guard move into 4.4. The strength evidence lives in Extras.
- [x] **4.4** the results paragraph, the section's centre: compiled with its figure and
      caption. The compiled s4 differs from the placed wording per the cold read: "In every
      pair we measured, the correction is at its smallest, relative to that pair's typical
      per-step size, in the steps where it matters, so size does not explain the timing."
      Scoped to "we measured" because elephant and penguin, one of the window's eight pairs,
      is in neither size set; verifying it from cached trajectories would license the wider
      claim. Original walking note follows, kept for the sentence record: sketch minted, walking
      sentence 1 of 5. Figure decided: the stacked pair, the sliding-window sample strip (cat
      and dog seed 9) above the eight-pair heatmap, sharing the same nine window-start
      columns, so the picture above any number is the picture that number counted. The dial
      strip is demoted to serving the guard sentence or Extras, since its lambda axis fights
      the window axis. The two frog-and-toad late-start cells are inspected manually before
      the caption exists.
  - [x] s1 placed: A generated image counts as composed when an object detector finds two
    distinct animals in it, and all counts were manually verified.
  - [x] s2 placed, with its motivation clause: To locate when in the run the correction
    matters, we restrict it to a ten-step window, leave the remaining forty steps
    uncorrected, and slide the window across the run in strides of five.
  - reserved for the figure caption: The window is kept narrow so that the sweep can
    distinguish timing from coverage, since a window spanning most of the run would contain
    the early steps at every position. (The fuller width argument, the width-from-size rule
    giving 25 steps and a sweep that cannot fail, goes to Extras with the window code as its
    source.)
  - [x] s3 placed, the two-sided boundary form, unblocked by the frog-and-toad inspection:
    Windows starting within the first five steps restore composition for every pair we tried,
    and windows starting at step 20 or later restore it for none (Figure N). The caption
    carries one reserved sentence naming the two seed-10 cells as detector error, verdict and
    paths in the figure group's card.
  - [x] s4 placed, evidence-first, numbers verified: In every pair the correction is
    relatively smallest in the steps where it matters, so its size does not explain the
    timing. Verified against `cache_analyses/step_collapse.json`, 17 of 17 pairs with early
    medians 0.46 to 0.83 of each pair's typical against 1.01 to 1.19 for the rest.
    Reserved for the caption: the absolute anchor, early-window medians 5.3% (butterfly and
    meadow) and 6.7% (cat and dog) of the composed prediction, from
    `figures/correction-size-over-the-denoising-run.json`; the design note's blanket
    5-to-14% range is retired, it fails cat and dog's second half.
  - [x] s5 placed, the learnability bridge: Why the earliest steps are decisive is open, and
    what we can show is that the correction they need is learnable. Consequence recorded: this
    does half the closer's job early, so when 4.6 is walked its pinned wording shifts weight
    onto the joint-prompt denial (the part s5 does not carry) or accepts the echo knowingly.

  ¶4.4's six sentences are all placed. It opens with "The sweep across all eight pairs and
  four seeds draws the same boundary.", the systematic answer to 4.7's one-pair grids, and its
  window sentence reads "We restrict the correction to a ten-step window". The stacked figure
  is built as one render:
  `scripts/window_samples_over_map.py` writes
  `figures/when-the-correction-arrives/samples-over-the-window-map.{png,pdf,json}`, the
  cat-and-dog strip (seeds 9 to 12, green frame where the scored verdict is composed) over
  the eight-pair map (rows ordered strongest-first at the earliest window, daggers on the two
  error cells). The frog-and-toad cells are confirmed as detector error at detector level, a
  second kept box on the animal's own haunch, evidence in
  `artifacts/results/did-the-detector-miscount-the-late-frog-toad-cells/`. The figure is
  compiled in the tex as `fig:window-map` and stays the section's last figure, after the two
  coverage grids.
      Then the claim and the three
      grids in `figures/when-the-correction-arrives/`, a small amount early is enough and the
      full amount late does nothing. The trajectory-separation support was cut from the flow:
      the separation is continuous from the first steps rather than localized to the early
      band (the pre-registered sharp-window checks failed, steepest rise inside steps 13 to 20
      for 3 of 9 cells in CLIP and 1 of 9 in DINOv2), so it localizes nothing and cannot
      corroborate the window result, and its being cached is a cost fact with no narrative
      standing. If a correction had only worked at steps 20 to 35, the window sweep would have
      shown it and the separation read would not have predicted it, which is the reason the
      windows are the argument and the drift is not. The window sweep locates where
      composition returns and says nothing about why, and the paragraph states that boundary
      plainly: the term's own size profile does not explain the timing, since the relative
      size is at its smallest in exactly the steps where the correction is effective, so the
      section reports where it works, admits the mechanism is open, and hands "but it is
      learnable" to section 5. The population heatmap
      (`seeds-composed-per-pair-as-the-window-start-moves.png`, eight pairs by nine window
      starts, seeds-of-4 in each cell) extends the early-only pattern past cat and dog to
      every pair tried, with two single-seed frog-and-toad cells at late starts flagged for
      manual inspection before any caption cites the map.
- [ ] **4.5** the shared size curve, tentative, waiting on your small-multiples
      check: blocked, may migrate to open section 5, may be cut
- [x] **4.6** the closer, compiled in the eased cost form (C2): Every result
      above used the joint prompt, which composition is not allowed to have. The next section
      removes that need. It discharges the provenance debt parked when ¶4.1's third sentence
      took the ceiling form, and it avoids "learn", which ¶4.4's last sentence owns.
      Revisable if the strength grid graduates from Extras.
- [x] **4.7** the coverage grids, one pair seen two ways before the sweep: compiled with two
      figures. Appended in numbering, reads between 4.1 and 4.4. Front-loaded windows (0-10
      down to 0-50) show ten corrected steps from the start already reach the outcome and
      longer coverage adds nothing; back-loaded windows (10-50 down to 40-50, plus an
      uncorrected row) show starting at ten still composes and starting at twenty or later
      never does. The paragraph closes by scoping, one pair at one seed, illustration rather
      than measurement, which is the sentence 4.4's new opener answers. Green frames in both
      grids mark decodes taken while the correction was active, not detector verdicts, per the
      `contains()` rule in `scripts/longer_correction_grid.py` and
      `scripts/later_start_grid.py`; both grids are cat x dog at seed 12, confirmed against
      the window map's seed 12 row. Figures `fig:window-extends` and `fig:window-starts-later`,
      register rows F4g and F4h, `fig:window-map` last in the section.

**Adopted into the skeleton.** The yardstick-to-dose region carries a ratio piece: the per-step
size ratio ||r_t|| / ||eps_PoE|| with its equation, read as "the missing piece is roughly a
tenth the size of the prediction it corrects", plus the one-sentence guard that this ratio
measures the missing piece while lambda is the dial (a fraction of r_t, never the ratio). Over
a delivery window the per-step sizes reconcile as the delivered total, sum over the window of
lambda_t times size_t, which is the window experiment's own dose-matching measure. A
qualitative dial figure (grid of rendered images with overlays) is routed to /design-figure;
the over-time qualitative need may already be served by the two frame rows (correction on
above off) in how-far-the-corrected-run-separates-from-the-uncorrected-one.pdf, to be judged
by eye when that paragraph is walked.

**The dose figure's controls, reframed by the built figure itself.** The built grid already
dropped the random row; its three controls are the wrong pair's correction, the same pair's
correction from a wrong seed, and the same correction in the wrong order. Read as one ladder,
each rung varies exactly one axis: which concepts (wrong pair), which trajectory (wrong seed),
which time-alignment (wrong order). All three fail, so the correction is specific to the
concepts, the starting state, and the schedule at once. That is the figure's real job: not
"any kick fails" but "the correction is a per-state instruction, not a portable ingredient",
which is what makes section 5's state-conditioned adapter necessary rather than a lookup
table. The paragraph and caption argue that, the AUC stops being the headline, the lambda=1
column gets its by-construction mark, and this figure's numbers come from its own sidecar and
review rows when the paragraph is walked, not from the two-control table in hypothesis-02.
Decided: the grid is not main text for now. It moves to the Extras staging area below, with
its paragraph written beside it, ready to drop in on supervisor approval.

**Extras: figures staged with their paragraphs, not in the main text.** Each entry here is a
figure plus the paragraph that would accompany it, written and held. On supervisor approval an
entry drops into the manuscript as a unit; until then the main text does not reference it.
Residents so far. The strength grid with its three controls and its curve, and the same grid
with detector boxes drawn, both filed with their card in
`figures/how-much-is-added/`; their paragraph argues the correction is a per-state instruction
rather than a portable ingredient, and feeds section 5. And the compose-rate treatment itself,
definition, validation and the boxes image, since the count is an off-the-shelf detector
counting animals, manually verified, not a proposed metric. Open question this creates: the
spine's how-much leg still needs a main-text carrier, candidates being a slim curve-only
figure or the numbers in prose; decided when 4.4 is walked. If the grid graduates, it is
rebuilt to [the dial-figure design note](section-4-dial-figure-design.md): the injection rule
with lambda substituted in each column header, the lambda-1 column footnoted as the
by-construction endpoint, every annotation inside the axes box so nothing clips.

**The controls question, superseded above, kept for the reasoning.** The two fake-correction rows are weaker evidence than their
AUC gap suggests, for a measured reason: r_t vectors are near-orthogonal across pairs and even
across seeds of one pair (cosine about 0.00,
`artifacts/results/which-way-the-correction-points/does-the-subspace-test-predict-transfer/QUERY.md`),
so the wrong-pair control is geometrically equivalent to the random one, and both invite "what
did you expect". The one thing the random row rules out, that any size-matched kick frees the
sampler from the fused mode (a live possibility given stochastic-sampling churn), is worth one
sentence, not a headline row. The section's probative evidence is the dose-response shape (25%
at half strength, 72% at three quarters, so the rise is nonlinear and not an interpolation
triviality) and the timing cliff, where the same real vector is used everywhere and only its
arrival time varies, a control no reviewer can call contrived. Candidate for a control a
practitioner would respect, unrun and uncosted beyond "new sampling, no new machinery": raising
the guidance weight w instead of injecting r_t, the knob anyone would try first. Whether F2
keeps its control rows, and whether it stays main-text at all, is decided when its paragraph is
walked with the dial-figure specs in hand.

**The figure test this section now runs under.** A measurement earns a main-text figure only if
it advances understanding of the dynamics of failure and correction, why linear score addition
breaks one pair over the denoising run and not another, and why the correction repairs it.
Robustness and instrument checks are mentioned in prose, not shown. Under that test beat 8
leans to a one-or-two-sentence mention with F5's main-text slot in question, because the
headline oracle endpoint (94% at lambda 1) is true by construction, the residual contains the
exact correction, so defending the detector on it defends nothing. The evidence that carries
information is the interior doses, the timing cliff, and the direction controls. Candidate
mechanism figures, held to the test rather than assumed to pass it. The two-pair contrast is
`correction-size-over-the-denoising-run.png` (butterfly x meadow against cat x dog, size per
step, three seeds, samples on top): correction present in both while only one sample is wrong.
The direction-oscillation story is inconclusive and one known cell already argues against it,
eagle x hawk's direction is smooth and the pair still fails, so smoothness-tracks-success dies
unless a grid across composing and failing pairs shows otherwise. That grid (ocean x dolphin,
park bench x snowstorm, camel x forest as composing rows; cat x dog, eagle x hawk as failing
rows; same layout as the two-pair contrast) is a proposed experiment, not a built figure: the
three composing pairs have no cached trajectories, so it costs new sampling, and they are
subject-and-setting pairs where the paper's pairs are all foreground-and-foreground, which the
framing must say. The whole mechanism question is routed to a drip-idea reconstruction (routes
table) rather than answered from the existing measurement pile. The window sweep stands.

**The timing family, adopted by eye.** Three built figures stay, each with its own sentence.
F4a, the sliding fixed-width window over four seeds: when the correction arrives decides the
outcome. F4g, windows all starting at 0 and extending later: ten early steps already reach the
ceiling, and more coverage adds nothing. F4h, windows all ending at 50 and starting later:
starting at 10 still composes, starting at 20 or later never does, whatever the coverage. The
register's "not built" rows for these are stale and get corrected when their slots are walked.
Minor improvements are expected on all three before compile (green-border rule quoted from the
figure code into the caption, the seed-11 sketch-style row acknowledged, layout polish); none
blocks the skeleton.
The 44%-margin-or-noise row in hypothesis-05's Still open binds whatever survives of the
instrument checks. Any
third-prompt or language framing of the correction is bound by that review's two language-side
nulls: the pair's difficulty is not visible in text space (75 pairs, four prompt views), and no
language twin of the low-rank structure survives (1.19 to 1.57x against a 2.0x bar). The
text-intervention experiment (adding scene words like "together in a scene" to the composed
prompts) is designed and parked, never run, so it may be named as future work and nothing more.

**Open on the skeleton.** The shared-size-curve beat is blocked on a qualitative check: a
small-multiples figure, one panel per pair, from the cached curves in
`cache_analyses/step_collapse.json` (route emitted, result owed to
`artifacts/results/does-every-pair-share-one-size-curve/`). Its verdict also decides whether
the beat stays here or opens section 5 as motivation. The detector-check beat's placement
(after the first number, or at the end) is undecided. And the walk brief says the paper has no
appendix while the spine still routes the run configuration and reserve figures to one; that
contradiction needs a decision before the yardstick paragraph places its pointer sentence.

**Taught material for the size-curve beat.** An interactive explainer, "How one correction-size
curve is made", lives at
[claude.ai artifact 31277c54](https://claude.ai/code/artifact/31277c54-e582-49dd-80e8-52788af234ba):
six frames from the real tensor to the median-scaled curve, real cached data for
eagle × hawk seed 9 (median relative size 0.094) and lion × tiger seed 1 (median 0.215).
Journey plan 36 of poe-composition-diffusion covers the same measurement on the toy.

**This section uses the correction computed from the joint prompt**, which the composed
sampler cannot have. Section 5 builds the learned one. A paragraph here that reads as a method
rather than as a measurement is arguing the wrong claim.

**Compose rate is defined here, in full, in the paragraph before F2.** It is the dependent
variable of every number in the paper and it is a detector counting instances, not an obvious
quantity. The run configuration goes to the appendix's Experimental Details with one pointer from
that same paragraph.

Figures: F2, F3, F4a and F5 in the main text. F2b, F4b to F4e and F5b to the appendix, which is
what their own register rows already say they are for. F4g and F4h are reserved.

◀ **Needs: nothing.** Every figure it leans on is built.

### 5. 🔧 Learning the Plurality Term

Owned by the `section-5-method` sibling session. Its claim, from [SPINE.md](SPINE.md): a low-rank
adapter on SDXL's cross-attention layers predicts the plurality term from the two concept prompts
alone, so the correction no longer needs the joint prompt it was defined from.

**The framing, in one line.** The correction was defined from a prompt composition is not allowed
to have, so this section removes that dependency, and what makes removing it possible is that the
adapter turns out to have learned the property rather than the pairs. The arc opens on
availability and closes on generality, which is the arc
[abstract candidate 3](abstract_candidate_03.tex) promises.

**The argument flow, adopted.** Five moves, each a plain claim, written in candidate 3's register
so the section's voice is fixed before the sentence walk starts. These are flow sentences and not
placed ones. Each is argued in the walk and most will change.

1. **The dependency, and its removal.** The correction of the previous section is computed from
   the joint prompt, which inference-time composition is denied. We remove that dependency.
2. **The correction is recordable.** Because the plurality term is defined from the joint prompt,
   it can be computed once and stored before any training. This turns a measurement into a
   supervision target. Nothing after this move works without it.
3. **What is trained, and how it is used.** We train a low-rank adapter on the cross-attention
   layers of Stable Diffusion XL so that its composed prediction matches the stored joint-prompt
   prediction at the same state. At inference the correction is the difference between the
   composed prediction with the adapter active and without it. The joint prompt enters as a stored
   target and is encoded nowhere at inference.
4. **The test is transfer.** An adapter that improved only the pairs it was trained on would have
   learned those pairs rather than the property. We therefore hold a set of pairs out of training
   and evaluate on them. This move says in advance what would count as success and what would
   count as failure.
5. **The result, and where it stops.** On pairs held out of training, the corrected composition
   restores plurality at the rate it reaches on pairs the adapter was trained on, and matches the
   correction computed from the joint prompt. The pairs are animals that contend for the same
   image region, and we do not measure whether the adapter reaches other kinds of composition.

Moves 1 and 2 are ¶5.1 and ¶5.2, the argument and then the recording. Move 3 is split across
¶5.3 and ¶5.4, since the architecture is discussed in its own right and carries the section's
figure. Move 4 is ¶5.5. Move 5 is ¶5.6.

**Move 3 says matches where the spine says predicts, deliberately.** The adapter does not emit the
plurality term. Its composed prediction is trained against the stored one, and the correction is
the difference between the adapted and unadapted compositions. "Predicts" is fine in an abstract
and wrong in the paragraph a reimplementer reads, so the two wordings differ on purpose.

**The section does not open result-first, decided.** Section 4 opens on its result and section 5
does not, because section 5 has an architecture to describe and a design to justify before a
number means anything. Move 1's dependency sentence is the section's opening. This is a
deliberate asymmetry with section 4 and not an oversight.

**Where the pairs are named.** ¶5.1 records predictions for a set of pairs, so they are mentioned
there in passing and nowhere else until ¶5.5, which owns the selection criteria, the counts, the
held-out split and the reason for it.

**One claim in move 5 that outruns its evidence.** That the adapter learned the property rather
than the pairs is a reading, not a measurement. What is measured is that the held-out rate matches
the trained-on rate, which is consistent with learning plurality and does not establish it. F7a is
the figure that would speak to the mechanism, and its register row caps its caption because the
effect shows on the control pair too. So either move 5 states transfer as the result and the
property as a reading, or [abstract candidate 3](abstract_candidate_03.tex)'s closing claim that
the adapter "learns a general correction for plurality rather than the appearance of any specific
concept" is a claim this section cannot back. **Deferred by decision**, settled when ¶5.6 is in
view. It blocks no earlier paragraph.

**The paragraph break, adopted.** Five paragraphs. The architecture gets two of them and its own
figure, because a reader has to be able to rebuild the adapter before a compose rate means
anything, and the design gets a full paragraph rather than a clause. The test the method plan sets
is that a reader outside this project could reimplement the correction, so ¶5.2 to ¶5.5 are
judged on whether they leave a hole.

- [x] **5.1** the section transition. Fully placed at four sentences, recast by the user's
      call from the earlier argument arc into a recap-and-turn shape. Ready to compile once a
      compile grain is chosen.

      **Assembled draft.** *In the previous section we measured the plurality term along
      composed runs and showed that adding a small amount of it back, early in the run,
      restores composition. In this section we learn it. We train a model that produces the
      correction at each step, from the current state and the two concept prompts, so a
      corrected run no longer needs the joint prompt. The joint prompt is barred only at
      inference, so the term can still be measured in advance, and those measurements are what
      the model trains on.*

      Recap with the demonstration named, the three-word turn, the requirement and the model
      merged with the payoff stated, and the closer extended to hand into ¶5.2 (the
      measurements are the training data). "Break hybrids" was offered and not used, since the
      compiled text treats hybrid as a description, not a countable object. The earlier
      argument-arc sentences survive in [Loose lines](#loose-lines) and in this entry's
      history.

      **The evaluator sentence is cut.** Ten candidate wordings over five rounds all tried to
      state that computing r_t needs the joint prompt, and the paragraph reads whole without any
      of them, because section 4's compiled closer states the obstacle one line above the
      heading and sentence 2's "from the two concept prompts alone" implies it. A sentence that
      survives deletion is deleted.

      **Assembled draft.** *The previous section showed that a small amount of the plurality
      term, arriving early, restores composition. To make that result usable, the correction
      must be produced during the composed run itself, at whatever state the sampler has
      reached, from the two concept prompts alone.*

      Sentence 1 is the result-restated opening (requirement-first and oracle-status
      alternatives kept in Loose lines). Sentence 2 converts the result into a specification,
      and its three clauses are the paragraph's agenda in order: produced during the run
      (sentence 3 rules out the offline copy), at whatever state the sampler has reached (the
      fact the ruling-out turns on), from the two concept prompts alone (what ¶5.3's adapter
      delivers). It deliberately does not name the joint prompt, which section 4's closer
      already named one line above the section break.

      The paragraph argues its way to the constraint sentence rather than opening on it. The
      arc, reframed away from storage by decision: the result worth keeping, the requirement it
      sets, the term is a function of the state and its only evaluator so far needs the joint
      prompt, therefore learn an evaluator that does not, and the closer, training is possible
      because the constraint is a runtime one. The earlier storing-fails arc was cut because
      caching is how training is implemented in ¶5.2, not why learning is needed, and the
      argument must hold whether or not anything is cached.

      **The state-dependence claim is definitional, not measured.** The plurality term is defined
      in ¶3.3.1 as a difference between two predictions at one noisy state, so saying it belongs
      to that state needs no figure. The dose grid's wrong-seed control measures the same thing
      and is staged in Extras, so this paragraph may not lean on it.
- [x] **5.2** the training data and the pooled adapter. Fully placed at three sentences,
      rebuilt on the user's purpose-first arc after the procedure-heavy version misaligned.

      **Assembled draft.** *The model learns its correction from the residuals themselves. For
      each training pair we therefore store the residual at every denoising step, over several
      runs, so the training data covers whole trajectories rather than isolated steps. We train
      a single adapter on every pair together to learn the correction rule that restores the
      plurality lost during composition.*

      The zoom is deliberate: model in ¶5.1, adapter here, rank-8 cross-attention LoRA in ¶5.3.
      "Residuals" is the right level of description; the four-branch storage detail lives in
      the appendix. The earlier procedure sentences (four stored predictions, branch list) are
      superseded and stay in [Loose lines](#loose-lines). **The trajectory boundary sentence is
      re-homed**: it no longer closes this paragraph, and its fact (training states lie on
      uncorrected runs, corrected runs visit new states) routes to the Discussion's drift
      piece 6.2, recorded there. It may not be dropped from the paper. The correction is defined from the
      joint prompt, which inference-time composition is denied, so it is recorded once, offline,
      before any training. Along an uncorrected product-of-experts trajectory, at each of the 50
      steps, the frozen network is evaluated at the same noisy state under the first prompt, the
      second prompt, the joint prompt and the unconditional condition, and the four predictions
      are stored. The pairs are mentioned here only as the recorded set.

      **The train and inference states are not the same states, and the paragraph must not imply
      they are.** The cache advances by plain product-of-experts, at
      [build_training_cache.py:174-177](../../scripts/build_training_cache.py#L174-L177), so every
      recorded state lies on the uncorrected trajectory. A corrected sampler leaves that
      trajectory at its first corrected step. Section 3's ¶3.5.2 says the oracle correction is
      measured along the trajectory it is applied to, and that sentence does not carry to the
      learned one. Whether this is stated here, in ¶5.4 beside the injection, or in the
      Discussion, is what sketch sentence 5 exists to force.
- [x] **5.3** the adapter. Locked and placed at five sentences. The rank-motivation sentences
      and the parameter-count sentence were cut by the user's call (speculative and unnecessary
      respectively; the counts live in T1, the spectral argument stays barred). The F10
      citation sentence is owed at compile, appended once the rendered image exists.
      **Carries F10.**

      **Assembled draft.** *The adapter is a low-rank adaptation (LoRA) of rank 8 on the query,
      key and value projections of every cross-attention layer, with the base model frozen. At
      each step of the run the UNet takes two inputs, the current latent $x_t$ and the prompt.
      They meet in the cross-attention layers, where the layer projects the image to queries
      and the prompt's token embeddings to keys and values, so each region of the image can
      draw on the tokens it matches. The adapter leaves these three projection matrices
      untouched and adds a small trainable update beside each one, the product of two rank-8
      matrices whose output is summed with the original projection's. Training moves only these
      added matrices, so what the adapter can change is how image regions query the prompt and
      what the prompt's tokens give back.*
- [x] **5.4** the reproduction map. Locked at eight sentences on the user's template arc
      (split, rule with its purpose, collection, storage, objective, settings, compute).
      Carries T1.

      **Assembled draft.** *We split a pool of nineteen animal pairs into eleven for training
      and eight for evaluation. To prevent a held-out gain from coming from an animal the
      adapter has already seen, and to attribute it to the correction rule instead, no animal
      appears in more than one pair anywhere in the pool. For each training pair we then run
      the composed sampler from eight starting noises and, at every step, evaluate the model
      under the joint prompt and under the two concept prompts at the state the run has
      reached. The difference between the joint-prompt and composed predictions is stored as
      that step's residual, giving 4,400 residuals across the 88 runs. The adapter is trained
      so that its composed prediction matches the stored joint-prompt prediction at each state,
      which is the same as supplying the residual without being given the joint prompt.
      Collection runs use guidance scale 7.5 and 50 DDIM steps at 1024x1024 in fp16. Training
      runs for 2,000 epochs of 50 optimizer steps, 100,000 in all, with AdamW at learning rate
      1e-4 and gradient clipping at 1.0. On a single NVIDIA RTX PRO 6000 this takes about seven
      hours and peaks at 23 GB of memory.*

      Sources: pool and split from `pair_pool.json`, counts from `dataset_meta.json`, sampler
      and optimiser from `config.json`, GPU from the W&B metadata, wall-clock and peak memory
      from `history.json`'s logged throughput (4.2 optimizer steps per second) and
      `train/peak_vram_gb`.
- [x] **5.5** inference. Locked at five sentences, reordered by the user's call after the
      two-pass confusion: the method leads (one adapted pass, plain composition, no joint
      prompt, no extra cost), the two-pass form follows as the experimental harness.

      **Assembled draft.** *At inference the correction is not injected at all. We simply run
      plain product-of-experts sampling through the adapted model, with the two concept
      prompts and the null token as the only conditions, so the corrected sampler costs no
      more than the composition it repairs. Training placed the correction inside the adapter's
      weights, so the composition of the adapted predictions lands where the frozen model's
      joint-prompt prediction would have landed. For the experiments we also run each step
      through the frozen model, and the difference between the two composed predictions makes
      the learned correction explicit. Scaling that difference by $\lambda$ before stepping
      gives a dial on the correction, with $\lambda = 0$ recovering plain composition as a
      baseline and $\lambda = 1$ identical to the adapted pass alone.*

      Grounded in [_sampling.py:664-670](../../poe_repair/methods/_sampling.py#L664-L670) and
      the λ = 1 default in `sample_crossbar.py`; the scored transfer runs are mathematically
      the adapted pass alone.
- [ ] **5.6** the results. Sketch held with `X` placeholders by the user's call, numbers filled
      only when the latest checkpoints are scored; the user returns to this paragraph.

      **Sketch, placeholders in place of every number.** *At training step X the adapted
      sampler composes X of held-out cells and X of trained-on cells, where plain composition
      sits at X (Figure F8a). The aggregate is not carried by one easy pair, since every
      held-out pair scores at least X, the weakest being a cat and a dog, and the one pair
      that composes without any adapter, an elephant and a penguin, is marked as a control.
      Against the oracle correction of Section 4, the adapter matches or beats the
      $\lambda=0.75$ rate on X of the eight held-out pairs, without ever encoding the joint
      prompt (Figure F8b). All numbers are read at step X, the last scored checkpoint.*

      The candidate values and their sources (step 60000: 0.96 held-out n=128, 0.96 trained
      n=176, floor 0.03, weakest 0.875, 7 of 8 against the oracle) are in the transfer
      review's answered bar and the F8a and F8b register rows; they are not entered until the
      rescore lands and `numbers` re-derives each one from `compose_rate.json`.

      **The F9 talk-over paragraph, compiled.** *Figure \ref{fig:adapter-samples} shows what
      the compose rate counts. Under plain composition each pair in the middle column
      collapses into a single animal carrying features of both concepts. The adapter-corrected
      sampler on the right returns each pair to two distinct animals, on pairs it never
      trained on.* The target-not-method sentence lives in the caption alone, so it is said
      once.

      **F9 is built and placed**:
      `figures/joint-prompt-poe-and-adapter-samples-for-four-held-out-pairs.pdf` by
      `scripts/joint_poe_adapter_grid.py`, cropped to the four held-out rows by the user's
      call, sidecar carries step 100000, lambda 1, the seeds and the sampler block. One open
      eyeball item: frog × toad's adapter cell is marginal, seeds 10 to 16 on disk to swap
      in.

      **Draft F9 caption.** *Samples for six animal pairs under the joint prompt (left),
      plain product-of-experts composition (middle), and the adapter-corrected sampler
      (right). Rows above the rule are held out of training; rows below it are trained on.
      The three images of a row start from the same noise, and the adapter column is the
      pooled checkpoint at training step 100000 with $\lambda=1$. The joint-prompt column is
      the target the correction was defined from, not a competing method. Six pairs
      illustrate the effect; the rates are in Figure F8a.*

¶5.1 to ¶5.5 are drafted from the design plans, never from the review files. Method and design are
what they are, not what they scored. ¶5.6 is the one paragraph in this section that reads review
files, and it may not be drafted until they are open in front of the walk.

**T1, the settings table, owed by ¶5.5.** One table, not a figure, so it has no register row. It
carries the model (SDXL base 1.0, frozen), the adapter (rank 8, alpha 8, dropout 0, attached to
`attn2.to_q`, `attn2.to_k`, `attn2.to_v`), the optimiser (AdamW, learning rate 1e-4, weight decay
0, gradient clip 1.0, batch size 1), the sampler (DDIM, 50 inference steps, guidance scale 7.5,
1024x1024, fp16), and the data axes (11 training pairs, 8 seeds each, 50 steps per cell, 88 cells,
4,400 recorded steps, 8 pairs held out). All values from
`artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/{config,lora_attach,dataset_meta,pair_pool}.json`.

**The controls ¶5.5 may name, and the one it may not claim.** The uncorrected composition is the
floor and appears in F8a as a dotted line. The correction computed from the joint prompt is the
ceiling and appears in F8b at lambda 0.75 and at lambda 1, the latter true by construction. The
control that answers the real objection, that any pool of the same size would have worked, is the
size-matched mixed pool in
[baseline-01](../../plans/04-does-the-fix-reach-unseen-pairs/review/baseline-01-the-size-matched-control-pool.md).
**It is designed and has not run.** ¶5.5 may state it as the design's intended control and ¶5.6
may not report a result for it. The reserved leave-one-pair-out sweep behind F8 has not run
either.

**One mismatch to resolve before ¶5.5 is drafted.** baseline-01 and the transfer review are
written over fifteen training pairs. The run that exists trained on eleven and held out eight. The
paragraph takes its counts from `pair_pool.json` and the plan text is what is stale.

**What the adapter emits, pinned before any sentence is written.** The adapter does not output the
plurality term. The frozen network and the adapted network are each run on the two concept prompts
and the unconditional condition, each set of three is combined by the same rule, and the predicted
correction is the difference between the two combined predictions. Training minimises squared
error between the adapted combination and the stored joint-prompt prediction, at
[trainer.py:335-336](../../poe_repair/experiments/one_pair_one_seed/trainer.py#L335-L336). The
difference is formed explicitly and scaled only at inference, at
[_sampling.py:667-670](../../poe_repair/methods/_sampling.py#L667-L670). A sentence saying the
adapter predicts or outputs `r_t` would send a reimplementer to a different architecture.

**Section 5 is the paper's last substantive section.** The old Experiments and Results section is
gone, and its figures (F8a, F8b, and the reserved F8) belong to ¶5.6. Sections 4 and 5 are twins,
each stating what it does and then showing what happened, and the thing separating them is which
correction is in play. Section 5's closing boundary therefore hands to the Discussion rather than
to another results section.

The boundary sentence in [Loose lines](#loose-lines) is assigned to section 4 as its closer. One
sentence cannot both close section 4 and open section 5, so ¶5.1 needs an opening of its own that
picks the boundary up without restating it.

The training-set counts (11 pairs by 8 seeds by 50 steps, 88 cells, 4,400 recorded steps, 8 pairs
held out) now have a home in ¶5.3 rather than competing with the method paragraphs.

F9's placement, below.

The section carries one figure, F9, the six-pair grid of joint-prompt, PoE and adapter samples.
The mechanism figures move out of the main text, so nothing here argues from geometry or from
attention.

**Extras: figures staged with their paragraphs, not in the main text.** Same rule as section 4's
staging area. An entry drops into the manuscript as a unit on supervisor approval, and until then
the main text does not reference it. Three residents.

F7a, built, the dot plot of how much the adapter changes what a word paints against where it
looks. Its caption is capped by the effect showing on the control pair too, so it says what the
adapter does to any pair it touches and not why the fix works. That cap is why it reads as
supporting material rather than as the section's carrier.

F7, design committed and not built, the three-band page whose first band is F7a unchanged. Band
two needs the pooled checkpoint rescored at step 100000 for eight pairs, and band three needs
eight seed-9 thumbnails. Neither has been produced, so staging it costs nothing that main-text
placement would not also have cost.

F6, the singular-value spectrum with the random floor shaded. Its argument does not stand as
drawn: the shaded floor gives every row the same expected norm while real ‖r_t‖ spans 4.5x, so it
cannot separate shared directions from uneven sizes, and against random directions carrying the
real norms the pooled stack is 1.4x at k=8 rather than 11x. Staging it does not repair it. It
either gets rebuilt against a norm-matched floor or it is cut, and that decision is still open.

F9 sits in ¶5.6 beside F8a and F8b, since section 5 carries its own results. The schematic it was
once weighed against is F10, which ¶5.2 carries as well, so the two are not alternatives.

### 6. 📝 Discussion

- [ ] **6.1** not yet broken into paragraphs
- [ ] **6.2** the drift between the clean image and the noisy state, moved here from section 3.
      Also carries the training-trajectory boundary re-homed from ¶5.2: the adapter's training
      states all lie on uncorrected runs, a corrected run leaves them at its first corrected
      step, so at inference the model produces corrections at states it never saw in training.
      The independence assumption is stated over the clean image x and used at every noisy latent
      x_t. Those are different assumptions and the derivation carries neither to the other.
      Writing p(c_i | x_t) as an integral over the denoising posterior makes the gap exact: it is
      a covariance between the two concepts' posteriors under p(x_0 | x_t), which vanishes as
      t goes to 0. Blocked on the routed pressure-test for that identity.

◀ **Needs:** the mechanism and limitations plan, writing-06, which has not started.

### 7. 📝 Conclusion

- [ ] **7.1** not yet broken into paragraphs

◀ **Needs:** everything above.

## Loose lines

Navigation: ⬅️ [The draft](#the-draft) | 📋 [TOC](#table-of-contents) | [Next](#blocked-pieces) ➡️

Written down, not yet placed. A loose line survives a session break and is deleted only when it is
placed or explicitly dropped.

| Anchor | The line | What is unresolved |
|---|---|---|
| §2 ¶1b eqref tie | Their combination rule is the one written in `\eqref{eq:poe-composition}`, whose failure this paper traces. | Held loose; its home is now ¶5's close, absorbed into "the combination rule is exact only under independence". Backed by the map's record that 3.2 and 3.6 follow Liu's derivation line for line. |
| §2 ¶2 eq A | The sequence admits a closed form at any step, $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$, labelled `eq:forward`, where $\bar\alpha_t$ is the cumulative noise schedule, decreasing from one toward zero, and nothing in it is learned. | **Walking, candidate offered, not yet placed.** Licensed by Gaussian steps composing into one Gaussian jump. Defines `alphabar_t` for ¶3's identity; `predicted_x0.py:34` reads the same quantity as `alphas_cumprod`. The per-step Gaussian is deliberately omitted. |
| §2 ¶2 eq B | The learned half is the reverse. A network $\epsilon_\theta(x_t, t, c)$ receives the noisy latent, the step, and a conditioning prompt $c$, and is trained to recover the added noise, $\mathcal{L} = \mathbb{E}\,\lVert \epsilon - \epsilon_\theta(x_t, t, c) \rVert^2$. | Sketched, not walked. |
| §2 ¶2 closer | The predictions the methods of the previous paragraph add are exactly these outputs, evaluated at the same noisy latent under different prompts. | Sketched, not walked. Ties ¶2 back to ¶1b and sets ¶3's question. |
| candidate 3 | A single prompt naming several concepts can impose a particular kind of compositionality on a text-to-image model, in which all of them appear together in one scene. | Placed as the old c3.1, then the arc it opened was abandoned for show-then-name. Kept in case a compositionality-first framing is wanted elsewhere; drop it if candidate 3 compiles without it. |
| 4.1 | The full forward-process paragraph, drafted whole before the walk moved to sentences: reversing a fixed corruption process, latent space named once, the per-step Gaussian, the closed-form jump with its reparametrisation, and the point that none of it is learned. | It was drafted as a paragraph rather than sentence by sentence. Re-walk it as 4.1.1 to 4.1.3 and place each one. |
| end of section 4 | The correction we injected comes from the joint prompt, which is exactly what composition is denied, and the next section learns to produce it without that prompt. | The boundary sentence that keeps sections 4 and 5 distinct under the shared plurality banner. Place it as section 4's closing sentence, wording revisable if the strength grid graduates from Extras. |
| 5.1 s1 | ~~The correction of the previous section is computed from the joint prompt, which inference-time composition is denied.~~ | **Cut.** Section 4's closer is already compiled one line above the section break and says "Every result above used the joint prompt, which composition is not allowed to have." This restates it. |
| 5.1 s2 | ~~We remove that dependency by training a model to produce the correction from the two concept prompts alone.~~ | **Cut.** The same closer's second sentence, "The next section removes that need", already says it. |
| 5.1 s3 | ~~Because the plurality term is defined from the joint prompt, it can be computed once and stored before any training begins.~~ | **Absorbed by the placed sentence 1**, which makes the same claim more sharply by naming the constraint as a runtime one. |
| 5.2 s1 | ~~For each concept pair and each seed we run the uncorrected composed sampler to completion, recording at every one of its 50 denoising steps.~~ | **Superseded** by the merged opening placed in ¶5.2, which folds the four-condition list into the same sentence. |
| 5.2 s2 | ~~At each step we evaluate the frozen model at the state the sampler has reached, under the first concept prompt, the second concept prompt, the joint prompt and the unconditional condition, and store all four predictions.~~ | **Superseded** by the same merge. |
| 5.1 alt-B | A correction is only useful to a composed sampler if it can be produced during the run, at whatever state the sampler has reached, from nothing more than the two concept prompts. | Requirement-first opening, not taken for the open. Its content is sentence 2's job now, so it is the base candidate there. |
| 5.1 alt-C | So far the correction has been an oracle, computed from a joint prompt that composition, by definition, does not have. | Oracle-status opening, not taken. The word oracle is wanted at sentence 3, where the recorded copy is ruled out, and in ¶5.6 beside F8b. |
| 5.1 closer | The constraint applies only at inference, so the correction can be measured in advance, while the joint prompt is still available. | Was placed as the section's opening, unplaced by decision: as an opening it was a promise with nothing behind it. It is now the sentence ¶5.1's argument arrives at, and it hands directly to ¶5.2's recording. |
| 5.2 s3 | The recorded states lie on the uncorrected trajectory, which is not the trajectory a corrected sampler follows. | ¶5.2's boundary. Open: whether it belongs here, in ¶5.4 beside the injection, or in the Discussion. It may not be dropped. |
| discussion | Collecting terms in the guided composed prediction gives (1 - 2w) on the unconditional prediction, which is -14 at w = 6.5. Composing two guided experts pushes much harder away from the unconditional prediction than guiding one expert does. The derivation in 3.2 sharpens this: the probability chain licenses w_i = 1 and nothing above it produces the weights at all, so every experiment in this paper runs a sampler seven and a half times away from the composition the maths gives, before any independence question is raised. | No experiment in this paper measures it, so it cannot sit in section 3 as a cause. It belongs in the discussion as an open observation, or nowhere. |

## Blocked pieces

Navigation: ⬅️ [Loose lines](#loose-lines) | 📋 [TOC](#table-of-contents) | [Next](#sessions) ➡️

| Piece | Blocked on | What unblocks it |
|---|---|---|
| 3.6, whether the mutual-information form of the plurality term is novel | the identity itself is two lines of Bayes and is checkable inline; whether anyone has written it before is not | the routed `/pressure-test` returning its novelty verdict. Correctness does not block the walk. |
| 5.1 learning the residual | F9's provenance. The checkpoint step, the seed and the sampler settings behind its six rows are recorded nowhere, and the image does not mark which four rows are held out and which two are trained on | those five facts supplied, then the grid rebuilt as a figure with a sidecar. F6 no longer blocks this piece, having moved to Extras |
| 5.6 the results | F8 is reserved, its leave-one-pair-out sweep has not run | that run finishing. F8a and F8b are built, so the paragraph is not fully blocked. |
| 6.1 discussion | the mechanism and limitations plan, writing-06 | that plan starting |
| 7.1 conclusion | every section above | those sections landing |

## Sessions

Navigation: ⬅️ [Blocked pieces](#blocked-pieces) | 📋 [TOC](#table-of-contents) | [Next](#open-citations) ➡️

One session owns a piece at a time.

| Session | Kind | Owns | State | Returns |
|---|---|---|---|---|
| section-3-citations | walk | the citation pass, intro through section 3 | closed 2026-08-27, compiled, build passed | 15 tex hunks, 7 bib entries, bibliography commands restored; CoInD deferred to related work; ordering defect (3.6 before 3.3) left on the record |
| main | walk | section 4 | closed 2026-08-27, main text compiled (4.1, 4.4 with `fig:window-map`, 4.6) | the compiled section; the 4.5 slot and Extras left open |
| section-5-method | sibling | section 5, Learning the Plurality Term | open, four-paragraph break adopted, no sentence walked | placed paragraphs, compiled |
| section-2-background | sibling | section 2, Background and Related Work | open; ¶1a and ¶1b compiled into the tex, machinery paragraphs ¶2 to ¶5 not walked | ¶1a+¶1b returned; the rest to come |

### Routes out of the section 2 walk

| Route | Serves | Where the result lands | State |
|---|---|---|---|
| /paper-scout, the four foundational slots (forward and reverse process, score-based modelling, classifier-free guidance, and whether the sampler needs its own citation) | §2 ¶2, ¶3 and ¶4 | six rows appended to [the reading register](../../plans/standing/literature/reading-register.md), five entries appended to [iclr2027_conference.bib](iclr2027_conference.bib), and [the selection file](/home-mscluster/mmolefe/goal-setting/learning/poe-derivation-foundations/paper-scout/selection-2026-08-27.md) | returned and integrated. Ran inside this session at the author's instruction. Slot 2 split in two, since the score-matching framing and the noise-to-score identity are different claims in different papers. Slot 4 answered yes, and the citation was routed to the methods section rather than to section 2 |

### Routes out of the section 5 walk

| Route | Serves | Where the result lands | State |
|---|---|---|---|
| diagram, subject lane, "the joined prompt is only a target" | ¶5.2, the architectural figure | Prompt 4a in [the scope's illustrated map](../../plans/diagram-prompts.md), rendered by hand in ChatGPT and saved back beside the manuscript | emitted, awaiting the rendered image |
| candidate-abstract-1 | sibling | abstract candidate 1 | open, sketched, walking c1.1 | `abstract_candidate_01.tex`, compiled |
| candidate-abstract-2 | sibling | abstract candidate 2 | closed, compiled, build passed | `abstract_candidate_02.tex` |
| candidate-abstract-3 | sibling | abstract candidate 3 | done, compiled, build passed | `abstract_candidate_03.tex` |

### Routes out of the section 4 walk

Every route emitted by this walk is a row here the moment it is emitted, so a forgotten tab is
recoverable from this table alone. Integrated means the result was read back and folded into
the walk; a row is deleted only then.

| Route | Serves | Where the result lands | State |
|---|---|---|---|
| picture-speak, "How one correction-size curve is made" | the shared-size-curve beat | [claude.ai artifact 31277c54](https://claude.ai/code/artifact/31277c54-e582-49dd-80e8-52788af234ba) | returned and integrated |
| small-multiples check, one panel per pair from the cached curves | the shared-size-curve beat's verdict, and where the beat lives | `artifacts/results/does-every-pair-share-one-size-curve/` | out; local session `d9fd58b8` (resume with `claude --resume d9fd58b8-d4bf-4b0f-978c-510da1315914`), mid-build as of 2026-08-26 morning, nothing on disk yet |
| /design-figure, the dial figure (grid of rendered images, lambda as a fraction of r_t) | the ratio-and-dial piece | [the dial-figure design note](section-4-dial-figure-design.md) | returned and integrated; the chosen rebuild now targets the Extras grid, the dial-strip alternative is ¶4.4's qualitative candidate |
| /pressure-test, arXiv 2502.04549 against this paper's CLIP-based difficulty and instrument reads | beat 8's scope, and the open why-text-anchoring-works row in hypothesis-05 | chat text in its session; return package pasted back here | to be launched |
| /drip-idea, the mechanism question (which variable, isolated and varied over the denoising run, explains what PoE composition is missing), reconstructed from the smoothness-vs-failure grid and the training-caption story | the section's figure test, and what mechanism figure if any enters the main text | [the idea map](../../artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md), seven claims each carrying a mark | parked mid-walk as of 2026-08-26, nothing settled by a run yet. Beat 8 stays a prose mention until claim 3 (linear score addition has a step-wise signature separating a blending pair from a composing pair) has a verdict, and claims 4 and 5 are instruments with nothing to instrument if it fails. The walk's larger finding is aimed at the discussion rather than at this section's figures: `r_t` is sampler error plus model error added together, per Du et al. [arXiv 2302.11552](https://arxiv.org/abs/2302.11552), nothing here separates them, and this section's timing result is also what a sampler artifact looks like. Three checks outstanding, all reading cached output with no sampling and no queue |
| /frame-hypothesis, the guidance-weight control (does raising w compose, with qualitative grids over the run) | the controls decision in the dose paragraph | a scoped experiment plan, or a feasibility no | to be launched |

## Open citations

Navigation: ⬅️ [Sessions](#sessions) | 📋 [TOC](#table-of-contents) | [Next](#figures-this-draft-leans-on) ➡️

The register is [the reading register](../../plans/standing/literature/reading-register.md), one
row per paper with what we take from it and how deeply it was read. The plan that keeps it current
is [the reading-register plan](../../plans/standing/literature/plans/01-reading-register.md); that
plan file is not the register and holds no rows by design. Sixteen papers are registered, so the
claims below split into those that already have a source and those that do not.

**Claims a register row already covers.** These can be cited as soon as the sentence is placed.

| Claim needing a source | Register row | Where it sits |
|---|---|---|
| product-of-experts composition for diffusion models | 2206.01714 for the technique, 2506.08894 for the formalism | 3.1, and the introduction's compiled text |
| the derivation route itself, Liu et al. 2022, arXiv 2206.01714, equations 6, 9, 10 and 11 | 2206.01714 | 3.2 and 3.6, which follow it line for line and then depart from it |
| the independence condition under which linear score addition is exact, and the two failure modes it separates (wrong target, non-smooth path) | 2502.04549 | 2.1, which is the section's single claim, and the introduction's beat 6 |
| the independence assumption is a property of the trained model that nobody had checked, and training for it changes the outcome | 2503.01145 (CoInD) | 2.1, as the training-time counterpart to this paper's inference-time correction |
| the forward corruption process and the learned reverse | 1503.03585 for the framework, 2006.11239 for the noise-prediction parameterisation | §2 ¶2 |
| the network's output is a score, and the identity that converts a noise prediction into one | 2011.13456 for the framing, 2105.05233 section 4.2 equation 11 for the identity | §2 ¶3, and the score-relation sentence at [line 96](iclr2027_conference.tex#L96) |
| classifier-free guidance | 2207.12598 | §2 ¶4 |
| the sampler every run uses | 2010.02502 | the methods section and Experimental Details, not section 2 |
| models capture physical structure of scenes | 2306.05720 primary, 2311.17137 behind it | the introduction's ¶2.2, [line 88](iclr2027_conference.tex#L88) |
| compositional failure on attribute binding | 2307.06350 | the introduction's ¶2.2, [line 88](iclr2027_conference.tex#L88) |
| compositional failure on counting | 2307.06350, its numeracy check run and passed 2026-08-27 | the introduction's ¶2.2, [line 88](iclr2027_conference.tex#L88) |
| compositional failure on negation | 2411.17066 | the introduction's ¶2.2, [line 88](iclr2027_conference.tex#L88) |
| the combinatorial-coverage argument that motivates composing over scaling | 2402.01103 | the introduction's opening paragraph |
| Stable Diffusion, the architecture that conditions the denoiser on text | 2112.10752 (Rombach et al., CVPR 2022) | the introduction's ¶2.3, the `(cite)` at [line 90](iclr2027_conference.tex#L90) |
| the model the experiments actually run on | 2307.01952 (SDXL, ICLR 2024 Spotlight) | the introduction's ¶2.3 and the methods section |

**Paragraph labels in this table.** `§2 ¶n` is a paragraph of section 2, Background and Related
Work. A bare `2.n` is a paragraph of the introduction, which the map has numbered 2.1 to 2.7 since
that section was walked.

**Two constraints these rows carry into the prose**, both recorded in full on the register rows.
The physical-structure sentence claims structure and not dynamics, because PhyBench (2406.11802)
found text-to-image models lack physical reasoning outside optics. And 2206.01714 may not be cited
for negation or for combinatorial coverage: it defines concept negation but never measures a
pretrained model failing a negated prompt, and the word "exponential" does not appear in it.

**Claims still open.**

| Claim needing a source | Where it sits | What it is waiting on |
|---|---|---|
| which paper covers the other model families | the multi-model generalisation experiment, if it reports them | four other checkpoints appear in the repo (`stable-diffusion-3.5-medium`, `stable-diffusion-2-1-base`, `stable-diffusion-v1-5`, `stable-diffusion-3-medium-diffusers`). SD3 is a different architecture from the LDM that 2112.10752 covers, so it needs its own citation. Unresolved because it is not yet established whether those ids are live experiment paths or leftovers |

**The bib now carries seventeen registered papers**, the original ten plus `chen2023beyond`,
`du2024generative`, `huang2025t2icompbench` (journal unverifiable on arXiv, entered as the arXiv
record), `conwell2024relations`, `du2024compositional`, `zhang2025product` and `dutta2026steer`,
appended by the citation pass. No literal `(cite)` or author-year placeholder remains between the
abstract and the end of section 3. `\bibliography` and `\bibliographystyle` are live. Registered
papers still without entries are the ones no compiled sentence cites yet, among them CoInD
(2503.01145), whose entry is owed to the related-work walk with its deferred attribution.

**One bib entry has no register row.** `saharia2022photorealistic` is in the bib and the
introduction names Saharia et al. in prose at [line 88](iclr2027_conference.tex#L88), with no row in
the register. That gap belongs to the reading-register plan rather than to a section walk.

## Figures this draft leans on

Navigation: ⬅️ [Open citations](#open-citations) | 📋 [TOC](#table-of-contents) | [Next](#compile-log) ➡️

Rows from [figures.md](figures.md). A caption may never claim past its slot's sentence.

| Slot | Status | Section that owns it |
|---|---|---|
| F1 | built and placed | 1, the introduction |
| F1b | built, two rows, samples over the size of r_t per step, shared y axis | 3.6g |
| F2 | built | 4, main text |
| F2b | built, held in reserve | 4, appendix |
| F3 | built | 4, main text |
| F4a | built | 4, main text |
| F4b to F4e | built | 4, appendix |
| F4g, F4h | reserved | 4, appendix if they land |
| F5 | built | 4, main text |
| F5b | built, held beside F5 | 4, appendix |
| F6 | the argument does not stand, slot needs a decision | 5, Extras, do not cite until decided |
| F7 | design committed, not built | 5, Extras |
| F7a | built | 5, Extras |
| F8a, F8b | built | 6 |
| F8 | reserved | 6 |
| F9 | fillable, the run is scored and the image exists as evidence, not as a figure | 5, main text |
| F10 | reserved, the architecture schematic, prompt written and image not yet rendered | 5, main text, ¶5.3 |

## Compile log

Navigation: ⬅️ [Figures](#figures-this-draft-leans-on) | 📋 [TOC](#table-of-contents) | [Next](#rejected-wordings) ➡️

| Paragraph | Landed at | Build |
|---|---|---|
| 2.1 to 2.7, the whole introduction | [iclr2027_conference.tex:84-130](iclr2027_conference.tex#L84-L130) | passed |
| candidate abstract 1, one paragraph | [abstract_candidate_01.tex](abstract_candidate_01.tex) | passed with tectonic, PDF 17.5 KiB |
| the six-change pass after the first compile | across the file | passed, PDF 2.10 MiB, **no overfull lines left**. Removed every `&&\text{...}` annotation column from every align block, which was also what fixed the three overfull lines. Deleted the Problem Setting and Background section. Emptied Related Work, Methodology, Benchmark Design and Experiments, Discussion and Conclusion to bare headings. **Two undefined references remain**, `eq:score` and `eq:adaptation` at line 282 in the appendix's placeholder Proposition, because the equations they point at lived in the sections that were emptied. |
| candidate abstract 3 into the tex as the manuscript abstract, section 3 retitled Missing Implications: The Plurality Term, the term renamed to the plurality term across abstract, introduction and section 3 with the interaction-term anchor clause kept at the first intro use | [iclr2027_conference.tex:81-201](iclr2027_conference.tex#L81-L201) | passed with tectonic, PDF 2.10 MiB, no overfull lines, two underfull vbox warnings at lines 125 and 382 |
| 3.1 to 3.6, the whole interaction-term section, plus two introduction hunks | [iclr2027_conference.tex:155-245](iclr2027_conference.tex#L155-L245), inserted before Methodology | passed, PDF 2.13 MiB. **Three overfull hboxes, all in the new section**, at the three align blocks: 27pt at the assumption block, 31pt at the score-to-noise block, 107pt at the R_t gradient block. The cause is the same in all three, `&&\text{...}` annotation columns on lines whose left side is already near the 3.5in text width. Fixing means shortening the annotations or moving them under their lines. |
| 5.1 to 5.5, the whole method half of section 5, plus the F9 figure (`fig:adapter-samples`) and its talk-over paragraph | [iclr2027_conference.tex:212-241](iclr2027_conference.tex#L212-L241), between `\label{sec:method}` and the Discussion | passed with tectonic, PDF 6.61 MiB, no overfull lines, no undefined references, only the four pre-existing TU/ptm font-shape warnings. ¶5.6's numbers half stays out pending the rescore. |
| 4.1, section 4's opening paragraph, plus the trim of section 3's duplicate ceiling sentence | [iclr2027_conference.tex:216-219](iclr2027_conference.tex#L216-L219) | passed with tectonic, PDF 2.10 MiB, no overfull lines, the two pre-existing underfull vbox warnings at lines 125 and 384 unchanged |
| §2 ¶1a and ¶1b, the whole positioning pair, plus seven bib entries (du2023reduce, skreta2024superdiff, skreta2025feynman, huang2026factordiff, wang2026testtime; zhang2025product and dutta2026steer arrived from the parallel bib pass and my duplicates were removed) | [iclr2027_conference.tex:127-131](iclr2027_conference.tex#L127-L131), after `\label{sec:background}` | passed with tectonic, PDF 3.66 MiB, no overfull lines, every citation resolved. The two pre-existing underfull vboxes persist (now lines 125 and 411 after the shift); `eq:score`/`eq:adaptation` in the appendix placeholder remain the only undefined references, known since the template pass |
| 4.4, the results paragraph, with the timing figure `fig:window-map` and its caption | after ¶4.1, before the section 5 heading | passed with tectonic, PDF 4.14 MiB, no overfull lines; two new mild underfull vbox warnings at lines 263 and 271 from page-breaking around the figure, the two pre-existing ones unchanged |
| 4.6, the closer, two sentences after the figure block | before the section 5 heading | passed with tectonic, PDF 4.14 MiB, no new warnings |
| the citation pass, intro through section 3: 15 hunks (11 citation placements, the ¶6 and 3.2-opener and 3.6-seam rewordings, the "and dynamics" trim, the `Figure~\ref` fix), 7 bib entries appended, `\bibliography`/`\bibliographystyle` restored | across lines 88-212 and 249-250 | passed with tectonic, PDF 3.66 MiB, every `\citep` resolves. Two pre-existing undefined references remain, `eq:score` and `eq:adaptation` in the appendix placeholder. New underfull hbox warnings from the .bbl, cosmetic |

## Rejected wordings

Navigation: ⬅️ [Compile log](#compile-log) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Anchor | The wording | Why it was turned down |
|---|---|---|
| c1.1–c1.11 (revision 1) | Opened on an abstract statement of the physical premise ("A natural image usually shows several distinct concepts..."), reached plurality only at sentence 7, after the failure was already stated in the abstract. Several reworded variants of the opening line were tried (A through P): physical-world framing, capability-first framing grounded in the training distribution, concrete-example framing, expectation-first framing, problem-first framing. None changed the underlying arc, only the wording of one sentence. | Read as paraphrases of the agreed abstract's own opening rather than a genuinely different candidate. The eleven-sentence arc buried the concrete cat-and-dog and camel-and-forest examples the paper actually uses, defined plurality abstractly before showing why it is needed, and gave two audiences (a compositionality expert and a reviewer with no background) nothing concrete to anchor on early. Replaced by revision 2, which opens on the two mechanisms (single-prompt, then inference-time composition) each with a worked example, motivates composition before showing its failure, and only names plurality once the failure mechanism (pixel-region competition) has already been shown. |
| 3.6b | "and departs from one by however much they are not", as a trailing clause. | The same vague-quantity move already turned down at 3.6a. What R_t does when independence fails is block 2's and 3.6d's job, and the guard at 3.6g is where the paper says it will not read values of R_t. |
| 3.6a | Five openings for paragraph 3.6: "instead of assuming the two concepts are independent given the state, we can define the ratio that measures how far from independent they are", the same with "the factor by which they are not", "the exact joint conditional and the product-of-experts one differ by a single multiplicative factor", "product-of-experts composition is not an approximation, it is the exact expression with one factor set to one", and "repeating the derivation without the independence assumption changes one thing". | "How far from independent" implies a distance the ratio does not give, and "the factor by which they are not" does not read. The others open on the assumption or on the gap, which sets the paragraph up but reveals nothing, and the one that reveals most spends 3.6d's punchline in the first line. The placed opening names the contrast between a model-defined quantity and an exact one, which is why the paragraph exists at all. |
| 3.4 | Saying the concepts compete for the same thing three times over, as "the same evidence in the image", "the same region", and "the same place in a scene". | One idea stated in three sentences. Sentence 3.4.3 keeps it and the other two give it up. "Evidence" was also the wrong word, borrowed from statistics for what is a competition over pixels. |
| 3.4 | "fails for real concept pairs" and "the pairs we study are all of this kind". | "Real" implies there are fake pairs; the condition, not the kind of pair, is what varies. "This kind" points back at nothing a reader can name, so the sentence says what the pairs are instead. |
| 3.4.4 | Three versions asserting when composition fails: "the violation is worst when the two concepts compete for the same place in the scene", "if this is right, the failure should track what role the concepts play rather than what they look like", and the same softened to "on this account". | All three assert a law the paper does not measure. The only bearing evidence is F2b, whose register row holds it out of the argument chain and forbids its caption being read as a measurement of the effect. The placed sentence scopes the paper's pairs instead of claiming a law. |
| 3.4.3 | "The product of the two experts is therefore largest not at an image containing both animals but at one that partly satisfies each." | It describes a maximisation that never happens. A diffusion sampler follows scores step by step and does not search for the argmax of a density. The placed version says the sampler steers toward such an image. |
| 3.4.1 | Three plain restatements of the independence assumption as the paragraph's opening sentence: "the image explains each concept on its own with nothing left over that the two concepts share", "each concept can be judged against the image on its own and judging them together adds nothing", and "for a fixed image, whether it shows a cat and whether it shows a dog are two unrelated questions". | All three restate in words what align block 3.2a already states in symbols with the assumption named in its annotation. Every version read strangely because the reader is being handed a definition they already hold. The sentence is cut rather than reworded. |
| 3.3.5 | "and both turn out to matter" as a trailing clause on the direction-and-magnitude sentence. | Fluff. It promises a finding without stating one, and F2 makes the point with evidence later. |
| 3.3.1 | Writing the joint-prompt prediction as `eps_theta(x_t, t \| c_1, c_2)`. | It would claim the model's joint-prompt prediction is the true joint score. The joint prompt is one text string naming both concepts, and the gap between it and the true joint conditional is what the paper is about. |
| 3.1.3 | The trailing clause "so the subtraction removes the duplicate copy that adding two experts introduces". | It duplicates the sentence following align block 3.2a, which says the same thing with a derivation attached. 3.2a keeps the reason and 3.1.3 keeps only the fact. |
| 3.2a | The three-line version with Bayes stated as a side result on `p(c_i \| x)`, then substituted. | The reader leaves the chain at line 2 to look at a different object and comes back at line 3. The placed version keeps one left-hand side on every line. |
| 3.1.1 | "...under three conditions, the first prompt, the second prompt, and no prompt at all." The manuscript's no-colon-inside-a-sentence rule applied, and "the unconditional condition" de-duplicated. | The author kept the colon deliberately. The list reads as an apposition and the colon marks it more cleanly than a comma here. The redundancy in "the unconditional condition" is a separate open question and was not decided. |

## Next step

Navigation: ⬅️ [Rejected wordings](#rejected-wordings) | 📋 [TOC](#table-of-contents)

Section 4's main text is compiled. The next session opens section 5 or the background section,
from this map alone. Standing items for whichever session comes first: integrate the
small-multiples check when it lands (it decides the 4.5 slot), integrate the three routes still
out (routes table under Sessions), run /restyle on section 4 against candidate 3 when polish is
wanted, and take the Extras entries to the supervisors.

## The whole-paper reading pass

Checks that only a front-to-back read can make, written while section 4 landed. Run this list
when the manuscript is read entire, and strike each line as it is settled.

- [ ] The abstract's "a small fraction of this residual, added back at the right stage" and
      ¶4.1's opening sentence are near-twins. At full-paper distance, decide whether the echo
      reads as promise-then-delivery or as repetition.
- [ ] Section 3's proxy paragraph lost its ceiling sentence to the ¶4.1 trim. Check the
      paragraph still closes cleanly without it.
- [ ] The object's name drifts from "the plurality term" (sections 3, ¶4.1) to "the
      correction" (¶4.4 onward). Check the drift binds on one pass or needs one harmonising
      touch.
- [ ] ¶4.4 ends on "learnable" and ¶4.6 hands off with "removes that need"; section 5 will
      open by learning the term. Three adjacent handoffs, check they escalate rather than
      repeat.
- [ ] "Every pair we tried" (the window's eight) against "every pair we measured" (the size
      pool) sit two sentences apart. Check a cold reader keeps the two sets separate, and
      whether the elephant-and-penguin size check has widened the second by then.
- [ ] The lambda-1-is-the-joint-prompt point is made once, in ¶4.1's endpoints sentence. Check
      ¶5.6 never celebrates a lambda-1 number, and that the point is not restated there.
- [ ] The dagger in Figure `fig:window-map` and its caption sentence: check the mark is
      legible at print size and the caption's "daggered" reads without hunting.
- [ ] Sweep the compiled manuscript for any surviving "oracle", "dose", or "instrument" once
      sections 5 to 8 exist; the spine renamed them and no section may reintroduce them.

One decision is queued behind it: F6's slot, whose register row says the argument does not stand,
now staged in section 5's Extras so it no longer blocks that section. It gets rebuilt against a
norm-matched floor or it is cut.

The `section-2-background` sibling session walks section 2 in parallel. ¶1a and ¶1b are
compiled and in the tex, build passed. The machinery pieces ¶2 to ¶5 are deliberately given their
own unhurried sessions, one or two pieces per sitting: ¶2 is mid-walk (sentences 1 and 2 placed,
equation A walking with its candidate held in Loose lines, then equation B and the closer), and
¶3 (score), ¶4 (guidance) and ¶5 (the close) are already mapped in the section entry but
unwalked. One structural decision is queued for the next sitting, before any more prose: whether
¶2 to ¶5 stay paragraphs under the section heading or become named subsections (Diffusion Models,
Score-Based Generative Modelling, Classifier-Free Guidance, and the close), which changes the tex
skeleton but not the walked content. Resume with:

```
/drip-write section 2 of paper/iclr/iclr2027_conference.tex, resume at ¶2 equation A;
the section 2 entry and Loose lines in DRAFT_MAP.md hold the state
```

```
/drip-write section 4, break it into paragraphs
```
