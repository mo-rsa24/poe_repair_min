# Review: does the same story show up from three other angles?

All five checks have run. This file judges
[../plans/hypothesis-05-the-same-story-from-three-sides.md](../plans/hypothesis-05-the-same-story-from-three-sides.md).
One of its answers fills register slot **F5**; the rest support the central claim rather than
carrying a figure of their own.

The two image-side checks agree with the strength sweep and share none of its scoring machinery.
The two language-side checks come back null: nothing in the prompt's embedding predicts which
pairs are hard, and the direction left over after subtracting the two solo prompts is the prompt
template rather than anything about binding.

## Words this file uses
- **The blend region**: the part of the image-similarity map where one fused animal lands, as
  opposed to where two separate animals land.
- **Additivity gap**: how much the joined prompt "a cat and a dog" differs, in text space, from
  simply adding "a cat" and "a dog". If that gap predicts which pairs need a big correction, the
  difficulty is visible in the text before any picture is made.
- **Binding direction**: whatever is left of the joined prompt after the two separate prompts are
  subtracted. The question is whether every pair leaves the same thing behind.
- **Caption readback**: showing a generated picture to a captioner and seeing which description
  it picks. A blended animal should read back as a blend, a real composition as two animals.

## Run kind
**Tests the claim.** These corroborate the central result from angles that share none of its
machinery, so agreement means something. None of them may overturn it on its own; a disagreement
is a diagnosis, not a verdict.

## Runs

All outputs sit in `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/`.

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| L1 additivity gap + L3 binding direction, 75 pairs, four views of the prompt | Tests the claim | 2026-08-11, in-session | `language_probes.json`, `language_probe_l1_additivity.png`, `language_probe_l3_binding.png` | done, both null |
| quality check, 749 paired `poe.png`/`mono.png` | Tests the claim | 2026-08-11, in-session | `quality_control_cache.json`, `quality_control_cache.png` | done, objection removed |
| manifold slide, 32 cells × 5 doses × 3 rows, CLIP image space | Tests the claim | 2026-08-11, in-session | `manifold_slide_clip.json`, `manifold_slide_clip.png` | done, slides |
| caption readback, same 32 cells, four-way caption bank | Tests the claim | 2026-08-11, in-session | `caption_readback.json`, `caption_readback.png` | done, crosses over |

## Written before the run, answered after

- [x] ⚠️ Does the additivity gap predict how big a correction each pair needs?
      **No.** Over 75 pairs, the rank correlation against the preregistered correction size
      (`relative_norm`) is +0.29 pooled, +0.00 for the CLIP-L sequence, +0.29 for the bigG
      sequence, and +0.32 for the concatenation. The bar was |ρ| ≥ 0.30 with p ≤ 0.05, written
      before the run; exactly one of the four views squeaks past it and the CLIP-L view sees
      nothing at all. A predictor that depends on which view of the prompt you pick is not a
      predictor. The scatter is a broad cloud in every panel.
      This is the null the question anticipated, and it is a finding: the difficulty of a pair is
      not written in its prompt embedding, so the binding information lives in how the model
      processes the two prompts jointly.
      One measurement note worth keeping: the two encoder halves sit at very different scales
      (median ‖e_J‖ is 1191 for CLIP-L against 105 for bigG), so the concatenated view is
      dominated by CLIP-L and is not the average of its two halves.
- [x] ⚠️ Do all pairs leave the same thing behind when the two separate prompts are subtracted?
      **No, not beyond what the prompt template already explains.** Against an isotropic random
      floor the answer looks emphatic: the top direction holds 18-20% of the residual energy
      against a 1.5-2.0% floor, a 10-12× margin. That floor is the wrong comparison. Every prompt
      here has the same shape ("a X and a Y" against "a X" and "a Y"), so a shared direction can
      be the "and", the extra length, or the anisotropy every CLIP text embedding carries.
      Rebuilding the residual with the second solo prompt taken from a **different pair** keeps
      all of that and destroys only the binding. That control reaches 12.8-15.2%, so the real
      pairs beat it by only 1.19-1.57× against a 2.0× bar. The CLIP-L sequence embeddings are
      nearly parallel across all prompts to begin with (mean pairwise cosine +0.999), which is
      where most of the apparent sharing comes from.
      No language-space twin of the low-rank ‖r_t‖ claim survives.
- [x] ⚠️ Is a blended animal wrong content rather than a poor-quality picture?
      **Yes, and this removes the objection.** Over 749 paired cells (same seed, same prompt,
      same sampler, the interaction term the only difference) the content changes hard and the
      quality does not. Compose rate 13.9% → 48.9%, 323 cells gained against 61 lost,
      McNemar p = 3.5e-44. The four content-blind quality proxies all sit inside the ±0.20 sd bar:
      Crete blur −0.13, noise −0.05, contrast +0.04, colourfulness +0.20.
      Two cautions. Colourfulness lands on the bar rather than comfortably inside it. And two
      further proxies were deliberately excluded from the verdict because they cannot answer this
      question: Laplacian sharpness rises with the number of edges, and CLIP's own quality
      preference rises for anything less anatomically distorted, so both move with content by
      construction. CLIP's preference does show a gap (+0.33 sd), which is exactly why it is not
      allowed to decide.
- [x] ⚠️ Do the pictures slide out of the blend region as the strength rises, while a same-sized
      random push does not? **Yes.** Feeds slot **F5**.
      Averaged over the interior doses, the real correction reaches +0.413 along the PoE→Mono
      axis against a 0.30 bar, and rises monotonically at every dose. The same-sized random push
      travels 5% of that. Another pair's correction travels 44%.
      Three things the figure has to say out loud. The λ=0 and λ=1 endpoints are arithmetic, not
      evidence: at full dose the injection adds all of r_t back onto ε_PoE, which is ε_J exactly,
      and the λ=1 picture is the cached mono render to within 1.9 grey levels of 255. Whole-image
      CLIP was already nulled in this repo as a way to tell a blend from a composition, so a
      position on this axis is not a compose score and only the row-to-row comparison carries
      weight. And at interior doses 66-91% of the motion is off the axis, so the axis is a
      summary of where the picture goes, not a description of it.
      The 44% for a mis-aimed correction is the weakest number in this plan. It clears the 50%
      bar but not by much.
- [x] ⚠️ Does the caption readback cross over from a blend description to a two-animal
      description as the strength rises? **Yes, and the controls stay flat.**
      Read at λ=0.75, the largest interior dose, the two-animal description goes from winning 3%
      of cells to 56% (bar: ≥50% and a ≥20-point gain), while the blend description falls 59% →
      25%. The two-animal description overtakes the blend description at λ=0.75. The random push
      gains −3% and another pair's correction gains 6%, against a 50%-of-oracle bar.
      Scored at λ=1 the split is 81% against 16%, but λ=1 is the joint render itself, so that
      number says only that the joint render reads as two animals, which was never in question.
      This one is worth more than it looks: it is CLIP image-text similarity, the same space whose
      whole-image read was nulled for this exact discrimination. Anchoring to text rather than to
      other images recovers the separation that image-to-image distance could not find.
