# ✍️ PoE repair (ICLR): the draft, piece by piece

## Recommended prompt (when a section lands)

```
/restyle <section> against <exemplar paper>
```
(For a section that is finished and needs filing rather than restyling: `/polish`.)

## Position in the manuscript

| Step | Section | State | Figures it owns |
|------|---------|-------|-----------------|
| — | [Abstract](#-abstract) | 6 sentences placed; candidate 1 of 3 alternative wordings now walking | none |
| 1 | [Introduction](#1--introduction) | settled and compiled | F1 |
| 2 | [Background and Related Work](#2--background-and-related-work) | not walked, bare heading in the tex | none |
| **3 (done)** | **[The Interaction Term](#3--the-interaction-term)** | **six paragraphs placed and compiled** | **F1b** |
| **4 (current)** | **[Analysing and Correcting Product-of-Experts with the Scaled Residual](#4--analysing-and-correcting-product-of-experts-with-the-scaled-residual)** | **not walked, bare heading in the tex** | **F2, F3, F4a, F5 in the main text; F2b, F4b to F4e, F5b in the appendix** |
| 5 | [Learning the Residual](#5--learning-the-residual) | not walked, bare heading in the tex | F7, F7a; F6 blocked |
| 6 | [Experiments and Results](#6--experiments-and-results) | not walked, bare heading in the tex | F8a, F8b; F8 reserved |
| 7 | [Discussion](#7--discussion) | not walked, bare heading in the tex | none |
| 8 | [Conclusion](#8--conclusion) | not walked, bare heading in the tex | none |

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

Section 3, the interaction term, is compiled and in the tex. Six paragraphs placed, build passing,
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
are not, and the failure is exactly a missing interaction term that a low-rank adapter can learn
and carry to concept pairs it never trained on.

**2. Background and Related Work.** Combining several of one network's own noise predictions
linearly is already standard practice, and the product-of-experts and classifier-free composition
methods that extend it build on the independence assumption without testing it.

**3. The Interaction Term.** The gap between the joint distribution over a correct composed scene
and the product-of-experts approximation to it is a specific, definable term, the interaction
term, and it is what the paper measures and later corrects.

**4. Analysing and Correcting Product-of-Experts with the Scaled Residual.** Adding the measured
interaction term back in causes composition, in an amount that grows with how much is added and
only when it arrives early in the denoising trajectory, while matched controls of the same size do
not.

**5. Learning the Residual.** A low-rank adapter on SDXL's cross-attention layers predicts the
interaction term from the two concept prompts alone, so the correction no longer needs the joint
prompt it was defined from.

**6. Experiments and Results.** The learned correction composes concept pairs it never trained on,
and matches the oracle correction it imitates without ever seeing the joint prompt.

**The split that makes sections 4 and 6 different.** Sections 3 and 4 use the oracle correction,
computed from the joint prompt and unavailable to the composed sampler, as a measurement
instrument. Sections 5 and 6 use the learned correction, which never sees the joint prompt.

## The draft

Navigation: ⬅️ [The spine](#the-spine-one-line-per-section) | 📋 [TOC](#table-of-contents) | [Next](#loose-lines) ➡️

### ✍️ Abstract

State: all six sentences placed, the paragraph is not assembled. Written last by design, so it
waits until the results section can be quoted.

- [x] **1.1** A natural image typically depicts several distinct concepts arranged so that they
      form a single coherent scene, and text-to-image models generate compositional scenes from a
      description of their parts.
- [x] **1.2** Recent methods compose pretrained diffusion models at inference time by sampling
      from a product of experts, one expert per concept.
- [x] **1.3** Product-of-experts composition can fail catastrophically, producing a single blended
      chimera instead of a scene containing both concepts, and we trace this failure to a
      specific, correctable gap.
- [x] **1.4** This gap is the interaction term, the difference between the sample a model
      conditioned on the true joint prompt would produce and the sample the product of experts
      actually produces, and it is exactly what the independence assumption drops.
- [x] **1.5** We train a low-rank adapter on the cross-attention layers of Stable Diffusion XL to
      predict this interaction term at every denoising step without ever seeing the joint prompt,
      and add its prediction back in as a correction.
- [x] **1.6** We show the correction transfers to concept pairs it was never trained on, and that
      adding more of it raises the compose rate toward a scene that composes both concepts
      correctly.
- [ ] **1.7** assemble the six into one paragraph, then `compile`

The numbers are deliberately left out of 1.6. That phrasing is qualitative by choice, not waiting
on a review file.

▶ **Next: section 6**, whose numbers 1.6 may later quote.

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
Candidate 3 is still free, and the open angle is property-first (open on the
specialised-to-a-property claim).

**Every candidate is written for two readers at once.** A reviewer with no diffusion background
follows the argument from the pictures it describes. A reviewer who works on composition finds
the mechanism named exactly, the residual between the joint-prompt prediction and the
product-of-experts prediction, at every denoising step. Neither reader is served by a second
register, so no candidate carries a plain half and a technical half.

| Candidate | Angle | File | State | Session |
|---|---|---|---|---|
| 1 | agreed order, plainer wording | `abstract_candidate_01.tex` | sketched, unwalked | candidate-abstract-1 |
| 2 | failure-first, opens on the hybrid object | `abstract_candidate_02.tex` | sketched, unwalked | candidate-abstract-2 |
| 3 | property-first, opens on what the model is specialised to | `abstract_candidate_03.tex` | ← current, sketched, walking c3.1 | candidate-abstract-3 |

**Candidate 1, sentence by sentence.** Sketched in round 1, walked one sentence per round.

- [ ] **c1.1** ← current. A natural image usually shows several distinct concepts arranged into a
      single coherent scene.
- [ ] **c1.2** models render such scenes from a full description, but the space of scenes grows combinatorially
- [ ] **c1.3** so recent work composes per-concept specialists at generation time instead of training one model over that space
- [ ] **c1.4** composition fails in a specific way
- [ ] **c1.5** and the failure is one hybrid object rather than two objects sharing a scene
- [ ] **c1.6** this work explains and repairs the gap between prompting one model and composing several
- [ ] **c1.7** the missing ingredient is plurality, implicit in the joint prompt and absent from the per-concept prompts
- [ ] **c1.8** plurality is measurable as the per-step residual between the joint-prompt prediction and the composed prediction
- [ ] **c1.9** a low-rank adapter on SDXL cross-attention predicts that residual without seeing the joint prompt
- [ ] **c1.10** adding the prediction back recovers distinct objects and carries to unseen concept pairs
- [ ] **c1.11** the result is a model specialised to an abstract property of a scene, keeping composition's generalisation without losing coherence
- [ ] **c1.12** assemble the eleven into one paragraph, then `compile` to `abstract_candidate_01.tex`

**Candidate 2, sentence by sentence.** Failure-first. Sketched in round 1, walked one sentence
per round.

- [ ] **c2.1** composing pretrained models at generation time returns one hybrid object where the prompt asked for two
- [ ] **c2.2** that failure is worth fixing because composition is the only route that scales, the space of scenes growing combinatorially in the number of concepts
- [ ] **c2.3** each expert is driven by its own short prompt, and nothing in that prompt says the scene holds more than one thing
- [ ] **c2.4** the missing ingredient is named plurality, implicit in the joint prompt and absent from the per-concept prompts
- [ ] **c2.5** plurality is measurable as the per-step residual between the joint-prompt prediction and the product-of-experts prediction
- [ ] **c2.6** a low-rank adapter on SDXL cross-attention predicts that residual at every step from the concept prompts alone
- [ ] **c2.7** adding the prediction back restores distinct objects, and the same adapter carries to concept pairs it never trained on
- [ ] **c2.8** the result is a model specialised to an abstract property of a scene, keeping composition's generalisation without losing global coherence
- [ ] **c2.9** assemble the eight into one paragraph, then `compile` to `abstract_candidate_02.tex`

**Candidate 3, sentence by sentence.** Property-first. It opens on what the trained model ends up
being, a model specialised to a property of a scene rather than to a concept, and reaches the
motivation and the failure afterwards. Sketched in round 1, walked one sentence per round.

- [ ] **c3.1** ← current. a diffusion model is normally specialised to a concept, and this one is specialised to a property of a scene instead
- [ ] **c3.2** the property is plurality, that the scene holds more than one distinct object
- [ ] **c3.3** why anyone needs that property: natural images show several concepts in one scene, and the space of such scenes grows combinatorially, so recent work composes per-concept specialists at generation time
- [ ] **c3.4** what goes wrong: that composition returns one hybrid object where the prompt asked for several
- [ ] **c3.5** where plurality went missing, implicit in the single prompt describing the whole scene and absent from the short prompts driving each expert
- [ ] **c3.6** plurality is measurable as the per-step residual between the joint-prompt prediction and the composed prediction
- [ ] **c3.7** a low-rank adapter on SDXL cross-attention predicts that residual at every denoising step without being conditioned on the joint prompt
- [ ] **c3.8** adding the prediction back recovers distinct objects, and the same adapter carries to concept pairs it never trained on
- [ ] **c3.9** the closing claim, composition's generalisation kept without the loss of global coherence
- [ ] **c3.10** assemble the nine into one paragraph, then `compile` to `abstract_candidate_03.tex`

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

- [ ] **2.1** not yet broken into paragraphs

◀ **Needs: [open citations](#open-citations)** to be filled. A paper not in the register does not
appear.

Parked for this section: Skreta 2412.17762 and 2503.02819, routed here from the introduction.

**What survives from the deleted Background section, and what does not.** Section 3 already
explains what product-of-experts composition computes at one step, so the background material has
shrunk to the diffusion basics, the score relation, and classifier-free guidance. The score
relation is already stated in the compiled introduction. **Which of the six old background
paragraphs survive into 2.1 is undecided**, and they are listed here so the decision is made
rather than defaulted:

- the forward process, fixed rather than learned. A whole-paragraph loose draft exists for it.
- the reverse process and the score the network stands in for.
- training, the noise-prediction objective and the condition dropping that yields an
  unconditional model from the same weights. Section 3 leans on this, since it is why the
  unconditional prediction exists at all.
- inference, one sampler step, what it consumes and what it emits. Nothing downstream needs it.
- classifier-free guidance, the first linear combination of one network's own predictions.
  Section 3's weights paragraph leans on this.
- the notation everything after uses, gathered in one place. **Cut by decision**: the spine now
  says definitions travel to the paragraph that first needs them.

### 3. ✅ The Interaction Term

◀ **Needs: section 4** for the notation and for guidance.

The introduction already states the composition rule. This section owns the assumption behind it
and the term it drops, and may not merely restate the introduction.

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
  - [x] 3.2a from the independence assumption to the prior-corrected product, one align block of
        four lines. Same left-hand side throughout, Bayes substituted into each factor in line 2,
        p(c1) p(c2) collected in line 3 and dropped in line 4 because it does not depend on x.
        Followed by one sentence: the unconditional density is in the denominator because it is
        already contained in both conditional experts, so their product carries it twice and one
        copy has to be removed.
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
  - [x] 3.3.1 We define the interaction term r_t as the difference between the model's prediction
        under the joint prompt and its product-of-experts prediction, both evaluated on the same
        noisy latent at the same step. Carries the display equation
        `r_t = eps_Mono(x_t, t) - hat_eps_PoE(x_t, t)`, labelled `eq:interaction-term`.
  - [x] 3.3.2 The identity eps_Mono = hat_eps_PoE + r_t holds by construction and is not a
        result, whereas the size of r_t, its structure across denoising steps, and its transfer to
        concept pairs held out of training are. Commits section 6 to all three. Register check
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
        without it. Two sentences, placed together. The three questions are section 6's own
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
  - [x] 3.4.2 The independence assumption in \eqref{eq:independence} fails once the two concepts
        have to share one image, since a patch of pixels that reads strongly as one animal cannot
        read equally strongly as the other. The paragraph's opening sentence. It names the
        assumption and points at the equation stating it, rather than restating it or pointing
        back with "that".

        **Requires a label added at compile time.** Line 1 of align block 3.2a carries no label
        yet. It gets `\label{eq:independence}` so this sentence can reference it.
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
- [x] **3.6** the exact interaction term, three small align blocks with prose between them.
      Complete: six prose sentences placed across five slots, three blocks, one slot cut.
      Appended after the walk started, not a renumbering. **Reads immediately after 3.2**, before
      3.3.   ← current

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
        Two lines: R_t defined as `p(c_1,c_2|x_t) / (p(c_1|x_t) p(c_2|x_t))`, then the same
        identity rearranged as `p(c_1,c_2|x_t) = p(c_1|x_t) p(c_2|x_t) R_t`. The rearranged line
        is shown rather than substituted silently. The introducing sentence is walked here rather
        than minted separately, since a block with nothing before it opens on an equation.
  - [x] 3.6b **the sentence after block 1.** Placed: "R_t equals one exactly when the two
        concepts are independent given the state." The trailing clause "and departs from one by
        however much they are not" is cut. It repeated the vague-quantity move already turned
        down at 3.6a, and what R_t does when independence fails is block 2's and 3.6d's job.
  - [x] 3.6c **block 2, into the joint.** Placed at five lines, same left-hand side throughout:
        Bayes with no assumption, the rearranged line substituted, Bayes on each factor,
        collecting, and recognising the composition rule. Ends at
        `p(x_t|c_1,c_2) ∝ p_PoE(x_t|c_1,c_2) R_t`.

        **Line 3 leans on 3.2a.** Its annotation points back to where the per-factor Bayes step
        was shown in full with p(c_1) p(c_2) appearing and being dropped, which is what lets it
        be one line instead of three. If 3.2a ever moves or is cut, this line splits in two.

        **Variable mismatch, deliberate.** 3.2's derivation is over the clean image x and this
        block is over x_t throughout. That is the drift, which lives in the discussion as 7.2.
        A careful reader will notice the two blocks use different variables three paragraphs
        apart, and nothing in section 3 explains it.
  - [x] 3.6d **the sentence after block 2.** Placed: "At every step the sampler combines the two
        experts as though this factor were one, whether or not it is." Says what happens during
        sampling rather than what the rule assumes, which is the one thing neither 3.6b nor block
        2 has said. Seven earlier candidates were turned down, for repeating 3.6b's R_t = 1, for
        reframing rather than adding, or for reading badly. R_t is deliberately not called the
        interaction term here, since 3.3 gave that name to r_t and 3.5 spends four sentences
        keeping the two apart.

  - [x] 3.6e **block 3, score and noise predictions.** Placed at three lines, one move each:
        logarithm and gradient of block 2's last line, then multiplying through by minus sigma_t,
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
  the discussion as 7.2.

Correction owed and carried by 3.2. The compiled derivation at
[iclr2027_conference.tex:97](iclr2027_conference.tex#L97) asserts p(x | c1, c2) = p(x | c1) p(x | c2)
and calls it conditional independence, then line 98 turns the same left-hand side into a
proportionality with 1/p(x) in it. Both cannot hold. The assumption that licenses the rule is
p(c1, c2 | x) = p(c1 | x) p(c2 | x), which is what
[IMMERSE_PoE_Foundations.md:35](../../docs/IMMERSE_PoE_Foundations.md#L35) states correctly. Fixing
the intro block is part of compiling 3.2, and it is a diff against already-compiled text.

▶ **Next: section 5.**

### 4. ✍️ Analysing and Correcting Product-of-Experts with the Scaled Residual   ← current

State: not walked, a bare heading in the tex. Its claim, from [SPINE.md](SPINE.md): adding the
measured interaction term back in causes composition, in an amount that grows with how much is
added and only when it arrives early in the denoising trajectory, while matched controls of the
same size do not.

- [ ] **4.1** not yet broken into paragraphs

**This section uses the oracle correction**, the one computed from the joint prompt and
unavailable to the composed sampler. Section 5 builds the learned one. A paragraph here that
reads as a method rather than as a measurement is arguing the wrong claim.

**Compose rate is defined here, in full, in the paragraph before F2.** It is the dependent
variable of every number in the paper and it is a detector counting instances, not an obvious
quantity. The run configuration goes to the appendix's Experimental Details with one pointer from
that same paragraph.

Figures: F2, F3, F4a and F5 in the main text. F2b, F4b to F4e and F5b to the appendix, which is
what their own register rows already say they are for. F4g and F4h are reserved.

◀ **Needs: nothing.** Every figure it leans on is built.

### 5. 🔧 Learning the Residual

- [ ] **5.1** not yet broken into paragraphs

Drafted from the design plans, never from the review files. Method is what it is, not what it
scored.

### 6. 📊 Experiments and Results

- [ ] **6.1** not yet broken into paragraphs

◀ **Needs:** the figure register slots. Blocked, see [Blocked pieces](#blocked-pieces).

### 7. 📝 Discussion

- [ ] **7.1** not yet broken into paragraphs
- [ ] **7.2** the drift between the clean image and the noisy state, moved here from section 3.
      The independence assumption is stated over the clean image x and used at every noisy latent
      x_t. Those are different assumptions and the derivation carries neither to the other.
      Writing p(c_i | x_t) as an integral over the denoising posterior makes the gap exact: it is
      a covariance between the two concepts' posteriors under p(x_0 | x_t), which vanishes as
      t goes to 0. Blocked on the routed pressure-test for that identity.

◀ **Needs:** the mechanism and limitations plan, writing-06, which has not started.

### 8. 📝 Conclusion

- [ ] **8.1** not yet broken into paragraphs

◀ **Needs:** everything above.

## Loose lines

Navigation: ⬅️ [The draft](#the-draft) | 📋 [TOC](#table-of-contents) | [Next](#blocked-pieces) ➡️

Written down, not yet placed. A loose line survives a session break and is deleted only when it is
placed or explicitly dropped.

| Anchor | The line | What is unresolved |
|---|---|---|
| 4.1 | The full forward-process paragraph, drafted whole before the walk moved to sentences: reversing a fixed corruption process, latent space named once, the per-step Gaussian, the closed-form jump with its reparametrisation, and the point that none of it is learned. | It was drafted as a paragraph rather than sentence by sentence. Re-walk it as 4.1.1 to 4.1.3 and place each one. |
| discussion | Collecting terms in the guided composed prediction gives (1 - 2w) on the unconditional prediction, which is -14 at w = 6.5. Composing two guided experts pushes much harder away from the unconditional prediction than guiding one expert does. The derivation in 3.2 sharpens this: the probability chain licenses w_i = 1 and nothing above it produces the weights at all, so every experiment in this paper runs a sampler seven and a half times away from the composition the maths gives, before any independence question is raised. | No experiment in this paper measures it, so it cannot sit in section 3 as a cause. It belongs in the discussion as an open observation, or nowhere. |

## Blocked pieces

Navigation: ⬅️ [Loose lines](#loose-lines) | 📋 [TOC](#table-of-contents) | [Next](#sessions) ➡️

| Piece | Blocked on | What unblocks it |
|---|---|---|
| 2.1 background and related work | the reading register has no rows | rows read in, see [Open citations](#open-citations) |
| 3.6, whether the mutual-information form of the interaction term is novel | the identity itself is two lines of Bayes and is checkable inline; whether anyone has written it before is not | the routed `/pressure-test` returning its novelty verdict. Correctness does not block the walk. |
| 5.1 learning the residual | F6's slot, whose register row says the argument does not stand | that slot's decision |
| 6.1 experiments and results | F8 is reserved, its leave-one-pair-out sweep has not run | that run finishing. F8a and F8b are built, so the section is not fully blocked. |
| 7.1 discussion | the mechanism and limitations plan, writing-06 | that plan starting |
| 8.1 conclusion | every section above | those sections landing |

## Sessions

Navigation: ⬅️ [Blocked pieces](#blocked-pieces) | 📋 [TOC](#table-of-contents) | [Next](#open-citations) ➡️

One session owns a piece at a time.

| Session | Kind | Owns | State | Returns |
|---|---|---|---|---|
| main | walk | section 4 | open, not yet broken into paragraphs | placed sentences |
| candidate-abstract-1 | sibling | abstract candidate 1 | open, sketched, walking c1.1 | `abstract_candidate_01.tex`, compiled |
| candidate-abstract-2 | sibling | abstract candidate 2 | open, sketched, walking c2.1 | `abstract_candidate_02.tex`, compiled |
| candidate-abstract-3 | sibling | abstract candidate 3 | ← current, sketched, walking c3.1 | `abstract_candidate_03.tex`, compiled |

## Open citations

Navigation: ⬅️ [Sessions](#sessions) | 📋 [TOC](#table-of-contents) | [Next](#figures-this-draft-leans-on) ➡️

The reading register at
[plans/standing/literature/plans/01-reading-register.md](../../plans/standing/literature/plans/01-reading-register.md)
currently holds no rows, so nothing below can be cited yet.

| Claim needing a source | Where it sits |
|---|---|
| product-of-experts composition for diffusion models | 3.1, and the introduction's compiled text |
| the derivation route itself, Liu et al. 2022, arXiv 2206.01714, equations 6, 9, 10 and 11 | 3.2 and 3.6, which follow it line for line and then depart from it |
| models capture physical structure of scenes | 2.2 |
| compositional failure on counting | 2.2 |
| compositional failure on negation | 2.2 |
| compositional failure on attribute binding | 2.2 |
| Stable Diffusion | 2.3 |

Ready to paste:

```
/paper-scout product of experts composition for diffusion models, including the original
product-of-experts formulation and the inference-time composition methods that build on it
```

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
| F6 | the argument does not stand, slot needs a decision | 5, blocked, do not cite until decided |
| F7, F7a | built | 5 |
| F8a, F8b | built | 6 |
| F8 | reserved | 6 |

## Compile log

Navigation: ⬅️ [Figures](#figures-this-draft-leans-on) | 📋 [TOC](#table-of-contents) | [Next](#rejected-wordings) ➡️

| Paragraph | Landed at | Build |
|---|---|---|
| 2.1 to 2.7, the whole introduction | [iclr2027_conference.tex:84-130](iclr2027_conference.tex#L84-L130) | passed |
| the six-change pass after the first compile | across the file | passed, PDF 2.10 MiB, **no overfull lines left**. Removed every `&&\text{...}` annotation column from every align block, which was also what fixed the three overfull lines. Deleted the Problem Setting and Background section. Emptied Related Work, Methodology, Benchmark Design and Experiments, Discussion and Conclusion to bare headings. **Two undefined references remain**, `eq:score` and `eq:adaptation` at line 282 in the appendix's placeholder Proposition, because the equations they point at lived in the sections that were emptied. |
| 3.1 to 3.6, the whole interaction-term section, plus two introduction hunks | [iclr2027_conference.tex:155-245](iclr2027_conference.tex#L155-L245), inserted before Methodology | passed, PDF 2.13 MiB. **Three overfull hboxes, all in the new section**, at the three align blocks: 27pt at the assumption block, 31pt at the score-to-noise block, 107pt at the R_t gradient block. The cause is the same in all three, `&&\text{...}` annotation columns on lines whose left side is already near the 3.5in text width. Fixing means shortening the annotations or moving them under their lines. |

## Rejected wordings

Navigation: ⬅️ [Compile log](#compile-log) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Anchor | The wording | Why it was turned down |
|---|---|---|
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

Break section 4 into paragraphs and walk the first one. It carries the most figures of any section
and has no prose at all.

Two decisions are queued behind it. Which of the six old background paragraphs survive into
section 2, listed in that section's entry. And F6's slot, whose register row says the argument does
not stand, which blocks section 5.

```
/drip-write section 4, break it into paragraphs
```
