# The same story from three independent sides

Five checks on whether the interaction term really is what makes composition happen, each
asked in a way that shares no machinery with the strength sweep it is meant to corroborate.

Copies of the run outputs, so the numbers are readable without reaching for `/datasets`.
The originals live in
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/`.

Design: [../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/plans/hypothesis-05-the-same-story-from-three-sides.md](../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/plans/hypothesis-05-the-same-story-from-three-sides.md)
Verdicts: [../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-05-the-same-story-from-three-sides.md](../../plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-05-the-same-story-from-three-sides.md)

## The five answers

| Question | Answer | Files |
|---|---|---|
| Can the prompt's text predict which pairs are hard? | **No** | `language_probes.json`, `language_probe_l1_additivity.png` |
| Do all pairs leave the same thing behind in text space? | **No** | `language_probes.json`, `language_probe_l3_binding.png` |
| Is a blend the wrong picture rather than a bad picture? | **Yes** | `quality_control_cache.*` |
| Do pictures slide out of the blend region as the dose rises? | **Yes** | `manifold_slide_clip.*` |
| Do captions flip from "one blended creature" to "two animals"? | **Yes** | `caption_readback.*` |

Every question was written down before its run, so no answer could be renegotiated afterwards.
Every pass/fail line sits in the source of the script that applied it, not in this file.

## Is a blend the wrong picture, or a bad picture? Yes, the wrong one.

This is the one that earns its keep: it removes "your correction just makes prettier pictures"
as an objection. `quality_control_cache.png`, `quality_control_cache.json`.

749 paired cells, same seed and prompt and sampler, the interaction term the only difference.
The content moves hard and the quality does not.

    content   compose rate      13.9%  ->  48.9%    323 cells gained, 61 lost
                                                    McNemar p = 3.5e-44

    quality   Crete blur        -0.13 sd            bar: within +/-0.20 sd
              noise             -0.05 sd
              contrast          +0.04 sd
              colourfulness     +0.20 sd            <- lands ON the line

Two proxies were deliberately barred from the verdict, because neither can answer the question:
Laplacian sharpness rises with the NUMBER of edges, and two animals have more silhouette than
one; CLIP's own quality preference rises for anything less anatomically distorted, and a chimera
is distorted by definition. CLIP's preference does show a gap (+0.33 sd), which is exactly why
it does not get a vote. Both are in the JSON.

Read `colourfulness` as a caveat, not a pass: +0.1994 against a 0.20 bar.

## Do the pictures slide out of the blend region? Yes. This is figure F5.

`manifold_slide_clip.png`, `manifold_slide_clip.json`. 32 cells, 5 doses, 3 rows.

Each picture is placed on an axis running from the uncorrected blend to the joint-prompt render,
and the question is how far along it sits.

    projection along the axis      dose 0.25   0.50   0.75
    real correction                    0.21   0.38   0.65     mean 0.413, bar 0.30
    another pair's correction          0.16   0.21   0.23     44% of the real travel
    same-sized random push             0.05   0.06   0.05      5% of the real travel

Three things this figure has to admit out loud, all of them in the plot:

- **The endpoints are arithmetic, not evidence.** At full dose the injection adds all of r_t back
  onto eps_PoE, which is eps_J exactly, so the dose-1 picture IS the joint render. Measured at
  1.9 grey levels of 255. The script checks this rather than assuming it. Only the interior doses
  carry the result, which is why the bars are read there.
- **A position on this axis is not a compose score.** Whole-image CLIP was already nulled in this
  repo as a way to tell a blend from a composition. Only the row-to-row comparison survives that
  null.
- **At interior doses 66-91% of the motion is OFF the axis.** The axis says where the picture
  goes, not what it does.

The 44% for a mis-aimed correction is the weakest number in the plan. It clears the 50% bar,
but not comfortably.

## Do captions flip? Yes, and the controls stay flat.

`caption_readback.png`, `caption_readback.json`. Same 32 cells, four kinds of description,
several wordings each, built from the prompts the cell was actually generated with.

    share of cells picking...        dose 0.00   0.25   0.50   0.75
    "a cat and a dog"                      3%    16%    19%    56%
    "one creature, part cat part dog"     59%    41%    50%    25%
    one animal alone                      38%    44%    31%    19%

The two-animal description overtakes the blend description at dose 0.75. The random push gains
-3% and another pair's correction gains 6%, against a bar of 50% of the real correction's gain.

Worth more than it looks: this is CLIP image-text similarity, the same space whose whole-image
read was nulled for this exact discrimination. Anchoring to text rather than to other images
recovers a separation that image-to-image distance could not find.

## The two language checks found nothing

Both nulls, and the plan said in advance that a null here would be a real finding rather than a
disappointment. `language_probes.json`, 75 pairs, one seed each.

**Nothing in the prompt's embedding predicts which pairs are hard.** The additivity gap (how far
"a cat and a dog" sits from "a cat" plus "a dog") correlates with correction size at rho +0.29
pooled, +0.00 for the CLIP-L sequence, +0.29 for the bigG sequence, +0.32 for the concatenation.
The bar was |rho| >= 0.30 at p <= 0.05. Exactly one of four views squeaks past, and one view sees
nothing at all. A predictor that depends on which view of the prompt you pick is not a predictor.

So the difficulty of a pair is not written in its prompt embedding. It lives in how the model
processes the two prompts jointly, which points any follow-up at cross-attention rather than at
the text encoder.

**No shared binding direction survives its control.** Subtract the two solo prompts from the
joint one and ask whether every pair leaves the same thing behind. Against an isotropic random
floor it looks emphatic: the top direction holds 18-20% of the energy against a 1.5-2.0% floor,
a 10-12x margin.

That floor is the wrong comparison, and swapping it changes the answer. Every prompt here has
the same shape ("a X and a Y" against "a X" and "a Y"), so a shared direction can be the "and",
the extra length, or the anisotropy every CLIP text embedding carries. Rebuilding the residual
with the second solo prompt taken from a DIFFERENT pair keeps all of that and destroys only the
binding. That control reaches 12.8-15.2%, so real pairs beat it by 1.19-1.57x against a 2.0x bar.

The CLIP-L sequence embeddings are nearly parallel across all prompts before any subtraction
happens (mean pairwise cosine +0.999), which is where most of the apparent sharing came from.

## Reproducing

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python

$PY scripts/language_probes.py --probe l1 --probe l3   # 75 pairs from cache, ~1 min
$PY scripts/quality_control.py                          # 749 image pairs, ~20 min
$PY scripts/manifold_slide.py                           # 32 cells x 5 doses x 3 rows, ~3 min
$PY scripts/caption_readback.py                         # same cells, caption bank, ~3 min
```

Outputs land in `/datasets/.../cache_analyses/`; copy them here to refresh this folder.

One thing to know before trusting a short run: a 12-cell smoke of the quality check showed
gaps on Crete blur and noise that vanished entirely at the full 749. Do not read the quality
verdict off a subsample.
