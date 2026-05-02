# Caveats and Limits — What this work will not claim

This file is the place where we say up front what this work *cannot* do.
The mathematical foundation referenced throughout is in
[`01_theory.md`](01_theory.md).

## 1. The M1 identifiability bound

**Statement** (formalised in [`01_theory.md`](01_theory.md) §3). Given only
the singleton image-conditional distributions $p(x \mid c_1)$ and
$p(x \mid c_2)$, the joint conditional $p(x \mid c_1, c_2)$ is not in
general recoverable. Therefore the data-side of the interaction term
$g^{\mathrm{IT}}_t = -\sigma_t \nabla \log R_t$ is forbidden to any
intervention that uses only those singleton distributions.

This bound is the reason a "fix everything" claim is impossible. We do not
violate it; we work around it through two channels (text encoder + UNet
internal directions) that the bound's framing does not close.

**Why M2 and C-PoE are not violations.** The bound speaks about
*image-conditional* distributions. M2 acts on text-encoder outputs — fitted
on natural-language joint captions, hence carrying joint-distribution
information that is invisible to the image-marginal-only formalisation.
C-PoE acts on the UNet's singleton classifier directions $u_A, u_B$ —
properties of a model that was itself trained on $p(x, c)$. Both are *gaps
in the bound's framing*, not violations.

## 2. C-PoE can under-condition

C-PoE damps the PoE pull when the two singleton classifier directions
disagree. The mode it is *designed to help* is **destructive cancellation**
— both nudges are real but they cancel, so suppressing them lets denoising
drift to a state where they cooperate.

The mode it *can fail in* is **both nudges genuinely need to push** —
$\alpha(\theta)$ then suppresses both pushes and the output ends up
under-conditioned, containing neither subject confidently. The visible
signature is a vague image with neither cat nor dog clearly present, rather
than a clean two-subject scene or a confident hybrid.

The cooperative pair (butterfly + flower meadow) is the canary for this
failure mode in v1 — if methods *regress* it (image becomes vague), C-PoE's
under-conditioning is the most likely cause.

## 3. M2 may regress to the mean

The M2 synthesizer is trained on text-encoder outputs alone, with the joint
embedding $e_J$ as the supervisory target. SDXL's text encoders were
themselves trained on multimodal contrastive objectives — so $e_J$ encodes
information learned from looking at joint $(A, B)$ images during text-
encoder training. This is C1's whole point.

Risk: if the joint-embedding distribution is multimodal (different valid
spatial configurations of "A and B" yield different valid $e_J$ values), an
MSE-trained synthesizer will *interpolate* across modes rather than choose
one — producing an $\hat e_J$ that lives between configurations and gives
the UNet a smeared, ambiguous conditioning.

Visible signature: M2-replace produces washed-out two-subject images —
both subjects present but indistinct.

If we see this pattern, the fix is *not* more capacity; it is a
behaviour-matching auxiliary loss (cosine similarity between cross-attention
patterns under $\hat e_J$ vs $e_J$ on cached $x_t$ states, computed offline)
or a mode-aware generative target. v1 does not implement these. If MSE-
trained M2 visibly regresses, we add the behaviour-matching loss as a single
targeted change.

## 4. M2 may not work on rare or relational pairs

M2 is text-only-trained. Three classes of pair are most likely to break it:

- **Rare compositions** — pairs whose joint caption is uncommon in the text-
  encoder's training corpus. The encoder has had little opportunity to
  absorb compositional structure.
- **Relational pairs** — captions whose meaning is a specific relation
  ("a cat *holding* a dog"). Text-only training over the
  $\{e_A, e_B, e_\emptyset\} \to e_J$ mapping may not generalise.
- **Highly entangled pairs** — pairs that share most of their description
  in $e_A, e_B$ (e.g. specific dog breeds). The orthogonal residual is small
  in absolute terms even when its ratio is visible; $\hat e_J$ may collapse
  to one of the singletons.

Cat × dog is common, non-relational, modestly entangled — M2 should work.
"a butterfly", "a flower meadow" is similarly common and cooperative — PoE
should work, M2 should not destroy it.

## 5. Single-seed scope

v1 is **N = 2 evidence**: two pairs at one seed. Explicitly a *control
overfit* to test mechanism. It is enough to:

- demonstrate that the methods can produce coherent two-subject images on a
  pair where PoE fails,
- demonstrate that the methods do not regress a pair where PoE works,
- support the qualitative argument about *which channel does what* via the
  mathematical derivation in [`01_theory.md`](01_theory.md).

It is **not** enough to:

- claim cat-and-dog is solved across seeds,
- claim collision pairs in general are solved,
- publish benchmark numbers,
- claim per-seed rankings are stable.

A reviewer will ask either "are other seeds the same?" or "are other pairs
the same?". The first is fixable by running on seeds 1–4 (the upstream
pilot has the same pair at all 5 seeds). The second is fixable by adding
one more collision pair (e.g. lion+tiger, cat+horse). Both extensions are
*data-only*, no code change.

## 6. Sanity floor vs falsifier

Butterfly + flower meadow is a **sanity floor**: PoE works, methods must
not regress. It is *not* a **falsifier**. A falsifier is a pair where:

- PoE fails, AND
- our theory predicts our methods *should also fail*.

The right falsifier is a prior-entanglement pair ("a transparent glass" +
"a dog", "a ballerina" + "a spacesuit") where the failure isn't conflict
between two distinct concepts but the joint mode genuinely not being in the
prior. If the methods help such a pair too, our channel framing is wrong.

v1 does not include a falsifier. Adding one is a follow-up before paper
submission.

## 7. Plan B — what to do if v1 images don't cooperate

Possible v1 outcomes:

1. **All three methods produce coherent cat+dog scenes; cooperative pair
   unchanged.** v1 succeeds.
2. **C-PoE works but M2 looks washed-out (regression-to-mean).** Add the
   behaviour-matching auxiliary loss to the synthesizer trainer; re-train.
3. **Methods produce vague images on cat+dog (under-conditioning).** Lower
   $\gamma$ from 2 → 1; if still vague, accept it as a finding and write
   the diagnostic-only paper around the M1 bound + the channel framing.
4. **Methods help the cooperative pair too, in a way that's clearly
   *changing* it.** That's an under-conditioning regression — same response
   as case 3.

Plan B does not require code beyond v1. If the synthesizer is the bottleneck,
the change is one auxiliary loss term in [`poe_repair/embeddings/train.py`](../poe_repair/embeddings/train.py).

## 8. What this work is *not* the same as

- **Not joint distillation.** We do not train on joint $(A, B)$ images.
- **Not Attend-and-Excite.** A&E maximises per-token attention magnitude;
  C-PoE modulates score composition.
- **Not per-region spatial supervision.** No box / mask inputs.
- **Not cross-model.** SDXL-specific; porting to other backbones is feasible
  but out of scope.
