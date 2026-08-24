# 🧭 Re-check the numbers behind the paper's headline figure

Task 4 of [the design plan](../plans/hypothesis-02-more-correction-more-composition.md) sends you
here. When you are done, F2's percentages are safe to print and two questions in
[the review file](../review/hypothesis-02-more-correction-more-composition.md) are answered.

## Recommended prompt (when you finish)

```
/analyze-figure paper/iclr/figures/F2-dose-response.pdf
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/hypothesis-02-more-correction-more-composition.md) | the task that sends you here |
| [review](../review/hypothesis-02-more-correction-more-composition.md) | the two questions your result answers |
| **this file** | **the steps, and the reason behind each one** |

## Table of contents

- [Why you are doing this, and where it lands](#why-you-are-doing-this-and-where-it-lands)
- [Words this file uses](#words-this-file-uses)
- [Before you start](#before-you-start)
- [What you need to understand first: the two faults in the counting tool](#what-you-need-to-understand-first-the-two-faults-in-the-counting-tool)
- [1. Confirm what the scorer is reading](#1-confirm-what-the-scorer-is-reading)
- [2. Look at the bad box before choosing a bar](#2-look-at-the-bad-box-before-choosing-a-bar)
- [3. Set the bars in the source code](#3-set-the-bars-in-the-source-code)
- [4. Re-score and re-read](#4-re-score-and-re-read)
- [5. If the direction does not survive](#5-if-the-direction-does-not-survive)
- [6. The five-picture strip](#6-the-five-picture-strip)
- [What you produce](#what-you-produce)
- [Recommended Prompts](#recommended-prompts)
- [Next step](#next-step)

## Why you are doing this, and where it lands

Navigation: 📋 [TOC](#table-of-contents) | [Next](#words-this-file-uses) ➡️

The paper's central claim is that composition fails because one correction term is missing. The
figure that carries that claim is **F2** in `paper/iclr/figures.md`, and its caption will say:
*more of the correction composes more, and two size-matched fakes stay flat.*

The experiment behind F2 has run and the direction survived: turning the correction up from none
to all of it takes the success rate from 6% to 94%, while both deliberately-wrong versions stay
near zero. **What is not yet safe is the exact percentages**, because the tool that counted the
animals had two faults (below). Both are now fixed in source; what remains is the re-score that
applies them and the reading of the result.

So this procedure is the last thing between F2 and the paper. F2 is one of eight slots, and the
writing starts in earnest once five to ten are resolved, which makes this the highest-value hour
available.

**Vocabulary, once.** The experiment turns the correction up in five steps and measures how often
both animals appear. Each step is a *strength setting* (written λ, from 0 = none to 1 = all). A
*cell* is one batch of images for one animal pair, at one strength, from one random seed; every
count below is counting cells. The *oracle* row injects the real correction; the two *control*
rows inject deliberately wrong vectors of the same size, and they are what makes the oracle's
rise mean something.

When you finish, answer the two open questions in
[../review/hypothesis-02-more-correction-more-composition.md](../review/hypothesis-02-more-correction-more-composition.md).
## Words this file uses

Navigation: ⬅️ [Why you are doing this](#why-you-are-doing-this-and-where-it-lands) | 📋 [TOC](#table-of-contents) | [Next](#before-you-start) ➡️

- **GroundingDINO**: the pretrained object detector used to count animals in generated images.
- **λ (lambda)**: the strength setting, how much of the correction is added back, from 0 to 1.
- **AUC**: area under the curve, one number summarising a whole curve so three can be compared
  at a glance.
- **oracle / controls**: the oracle uses the true interaction term; the two controls are
  deliberately wrong versions that should fail, and do.
- **NMS**: the step that merges overlapping detections, so one animal found twice counts once.
- **MIN_BOX_FRACTION**: the size bar, 0.25. A detection must span at least a quarter of the
  image's longer side to count as an animal.

## Before you start

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-you-need-to-understand-first-the-two-faults-in-the-counting-tool) ➡️

**Is the GPU free?**

Step 1 needs no GPU. Steps 2 and 4 load GroundingDINO (the object detector that counts animals in
each image), and this node's card is shared, so check first:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
ps -o user,etime,cmd -p <pid> | tail -1     # whose it is, and how long it has been there
```

If someone else holds most of the 23.5GB, do NOT try anyway: the detector dies with
`torch.OutOfMemoryError` partway through. Two options, both correct:

- **Add `--device cpu`** to the commands in steps 2 and 4. Same boxes, same verdicts, slower
  (the full 480-image re-score takes about an hour rather than a couple of minutes). No queue,
  no waiting.
- **Or submit to biggpu** with `/run-experiment`, and do the writing tasks meanwhile.

## What you need to understand first: the two faults in the counting tool

Navigation: ⬅️ [Before you start](#before-you-start) | 📋 [TOC](#table-of-contents) | [Next](#1-confirm-what-the-scorer-is-reading) ➡️

Both are fixed in source. You are applying the fixes and reading what comes out, so you need to know what each fault did before the steps make sense.

**Fault one: the scorer read pairs that are not part of this sweep.** It collected every image
folder under the output directory rather than only the ones this sweep wrote. Eleven folders hold
a single cell at seed 1, and ten of them are *training* pairs (`a_wolf__x__a_husky`,
`a_lion__x__a_tiger`, `a_cheetah__x__a_cougar` and the rest of the pooled-LoRA training set).
Each has only λ=0 and λ=1 and no control rows. That is why the two end strengths were scored over
44 cells while the three middle ones used 32: the ends and the middle were measured on different
amounts of data, and on a different set of pairs.

Nothing leaked in the training sense. The dose sweep trains nothing: it injects the cached
correction into frozen base SDXL, so no model has ever seen any of these pairs. The damage was to
the comparison, not to the split.

**Fault two: the animal count was not reliably a count of animals.** On the
`an_elephant__x__a_penguin` seed 10 strip, the `random` control row at λ=1 shows one fused
creature and the detector returned two boxes: the animal at 888px, and a 220px box on a shadow
beside its feet. That second box turned a control-row blend into a `compose`, which is the worst
place for a false positive, because the controls are what the oracle's rise is measured against.

Confidence cannot separate that case. The false box scored **0.60**, above the real penguin's
0.54 on the same strip. Size can: every genuine animal there spans 458px or more, every spurious
box 220px or less, on a 1024px image.

## 1. Confirm what the scorer is reading

Navigation: ⬅️ [What you need to understand first](#what-you-need-to-understand-first-the-two-faults-in-the-counting-tool) | 📋 [TOC](#table-of-contents) | [Next](#2-look-at-the-bad-box-before-choosing-a-bar) ➡️

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY -c "
import json
d = json.load(open('/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/dose_curves.json'))
print('cells per row:', d['n_cells'])
seeds = sorted({s['seed'] for s in d['scores']})
print('seeds present:', seeds)
"
```

Expect `32` cells for every row and seeds `[9, 10, 11, 12]`. Those are this sweep's own seeds,
the first four of the held-out block defined in `outputs/animals_compose_transfer/seed_pool.yaml`.
If a seed 1 appears, the seed restriction in step 3 is not taking effect and nothing below is
worth reading.

## 2. Look at the bad box before choosing a bar

Navigation: ⬅️ [1. Confirm what the scorer is reading](#1-confirm-what-the-scorer-is-reading) | 📋 [TOC](#table-of-contents) | [Next](#3-set-the-bars-in-the-source-code) ➡️

Do not pick a threshold from a number alone. Draw the boxes and look at them.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min    # the paths below are repo-relative
$PY scripts/dose_strip.py --pair an_elephant__x__a_penguin --seed 10 --annotate-boxes --device cpu
```

`--annotate-boxes` draws every box the scorer KEPT, labelled with its confidence and its pixel
size, and prints the raw count beside each panel's verdict. Boxes at least a quarter of the image
across are drawn yellow; smaller ones magenta, which is the shape a shadow or a limb takes. Drop
`--device cpu` if the GPU is free.

It writes `dose_strip_<pair>_seed<N>_boxes.png` (a separate file from the plain strip, so the
diagnostic never overwrites the figure candidate).

Read the middle row's λ=1 panel, where two boxes sit on one animal. Every magenta box in that
figure is either spurious or a duplicate of a box already kept; every yellow one is a real animal.
That is the picture the bar in step 3 is set from.

## 3. Set the bars in the source code

Navigation: ⬅️ [2. Look at the bad box](#2-look-at-the-bad-box-before-choosing-a-bar) | 📋 [TOC](#table-of-contents) | [Next](#4-re-score-and-re-read) ➡️

Both bars live in source, not on a command line, so a later change shows up in a git diff instead
of hiding in someone's shell history.

**Fault one:** `scripts/plot_dose_curves.py` carries `SWEEP_SEEDS = (9, 10, 11, 12)` beside its
collection code, so it takes only this sweep's cells. `--all-seeds` gives the old behaviour back,
and the run prints which seeds it used either way.

**Fault two:** `poe_repair/experiments/compose_scorer_validation/detection_scorer.py` carries
`MIN_BOX_FRACTION = 0.25`. A detection must span at least a quarter of the image's longer side to
count as an animal, which is the same line `dose_strip.py` already draws when it colours a box
yellow or magenta, so the diagnostic picture and the scorer agree by construction. `conf` stays at
0.30, because the picture in step 2 shows confidence is not the lever.

```bash
grep -n "MIN_BOX_FRACTION\|conf\|nms_iou" poe_repair/experiments/compose_scorer_validation/detection_scorer.py
grep -n "SWEEP_SEEDS" scripts/plot_dose_curves.py
```

The filter can only remove detections, so it can only push compose rates down. It cannot
manufacture the result the paper claims. Check it on known cells before spending an hour:

```bash
$PY - <<'EOF'
import torch
from pathlib import Path
from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances
R = Path('outputs/interaction_term/dose/pairs/an_elephant__x__a_penguin/seed_10')
for name, p in [("random lam1.00", R/'teacher_residual_const_lam100_random/teacher_residual_const_lam100_random.png'),
                ("oracle lam1.00", R/'teacher_residual_const_lam100/teacher_residual_const_lam100.png')]:
    print(name, count_instances(p, device=torch.device('cpu'), min_box_fraction=0.0)[0],
          '->', count_instances(p, device=torch.device('cpu'))[0])
EOF
```

The false positive on the control drops from 2 to 1, and the real composition stays at 2 or above.

## 4. Re-score and re-read

Navigation: ⬅️ [3. Set the bars](#3-set-the-bars-in-the-source-code) | 📋 [TOC](#table-of-contents) | [Next](#5-if-the-direction-does-not-survive) ➡️

```bash
$PY scripts/plot_dose_curves.py --root outputs/interaction_term/dose/pairs --device cpu
```

Drop `--device cpu` if the card is free. The bars come from the source edits in step 3, so this
command takes no threshold arguments by design: you cannot accidentally re-score with a different
bar than the one in the diff.

It prints the per-row table and the AUCs (one number summarising each curve's total area, so the
three curves can be compared at a glance) and rewrites `dose_curves.json` and `dose_curves.png`.
Both, in one pass: an earlier run used `--no-figure`, which is why the plot on disk can be older
than the scores beside it.

What to check, in order:

- **Cells per row is equal across all five strengths.** If the end strengths hold more cells than
  the middle ones, the seed restriction did not take effect and nothing below is worth reading.
- **The oracle row still rises and both control rows stay near the floor.** That direction is what
  the paper claims. If it survives, answer the first open review question `✅` with the new
  percentages.
- **How far the percentages moved.** Record the new numbers beside the pre-cutoff ones (6% to 94%,
  AUC 0.422 for the oracle against 0.059 and 0.070 for the controls). The size floor should pull
  the controls down more than the oracle, since the controls are where the false positives were.
  If it does, the gap widens and the paper's claim is stronger than the earlier read showed.

## 5. If the direction does not survive

Navigation: ⬅️ [4. Re-score and re-read](#4-re-score-and-re-read) | 📋 [TOC](#table-of-contents) | [Next](#6-the-five-picture-strip) ➡️

Then this was a run testing a scientific claim, and the claim failed its pre-set pass mark. The
standing rule applies: answer the review question `❌` with the numbers, close the plan, and write
one follow-on plan file asking why. Do not loosen either bar until the curve comes back. Both were
chosen in step 3 by looking at actual boxes, and moving one afterwards to rescue a result is
exactly the kind of after-the-fact adjustment this whole setup exists to prevent.

## 6. The five-picture strip

Navigation: ⬅️ [5. If the direction does not survive](#5-if-the-direction-does-not-survive) | 📋 [TOC](#table-of-contents) | [Next](#what-you-produce) ➡️

The strip is the row of five images, one per strength setting, that sits above the curves in
figure F2. It is the qualitative half of the claim: the curves say the rate went up, the strip
shows what "went up" looks like.

```bash
$PY scripts/dose_strip.py --pair an_elephant__x__a_penguin --seed 10 --device cpu
```

That pair carries the strip because it is the one a reviewer cannot dismiss. An elephant and a
penguin share nothing, and PoE still fuses them into a single creature at λ=0, so the failure
cannot be explained away as "the two animals look alike". Seed 10 is the seed whose oracle row
rises monotonically; seeds 9 and 11 do not, and that belongs in the text rather than hidden by the
choice of strip. `a_leopard__x__a_jaguar` seed 9 is the supplementary strip.

Two things in the figure are worth a sentence rather than smoothing over. At λ=0.5 the failure
changes character: it stops fusing and drops the penguin entirely. At λ=1 the panel holds three
animals, not two, which is where the extra count comes from.

**A control we no longer have.** `outputs/animals_compose_transfer/pair_pool.yaml` lists
`an_elephant__x__a_penguin` as the compose-by-default control, the do-no-harm check. It is not
one: it scores 0 of 4 at λ=0, and the four images are single fused creatures, so the scorer is
right and the pool's assumption is wrong. The pool has no working do-no-harm control. That is a
limitations sentence, not a blocker.

## What you produce

Navigation: ⬅️ [6. The five-picture strip](#6-the-five-picture-strip) | 📋 [TOC](#table-of-contents) | [Next](#recommended-prompts) ➡️

| What | Where it lands | What it answers |
|---|---|---|
| the re-scored curves | `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/dose_curves.json` | *Do the curves hold when only this sweep's own cells are scored?* and *Do they hold under a bar chosen against the picture of the boxes?*, both in [the review file](../review/hypothesis-02-more-correction-more-composition.md) |
| the annotated box diagnostic | `dose_strip_an_elephant__x__a_penguin_seed10_boxes.png` | why `MIN_BOX_FRACTION` is 0.25 rather than a number picked from a table |
| the five-picture strip | `paper/iclr/figures/F2-dose-response.pdf` | register slot F2, the paper's headline figure |

## Recommended Prompts

Navigation: ⬅️ [What you produce](#what-you-produce) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Run these when a term here stops meaning anything. Each leaves something you can return to.

- **On λ and the correction itself:** `/drip --math the correction r_t and what the strength
  setting λ scales, one step per message` → `/polish` to file it → `/math-scene` on that file, to
  drag λ and watch the prediction move from PoE toward the Mono target.
- **On the counting rule:** `/demonstrate show me five cells the scorer calls compose and five it
  calls blend, with their boxes drawn, so I can see the rule agree and disagree with my eye`.
- **On the shape of the whole sweep:** `/experiment-atlas the dose sweep: 8 pairs x 4 seeds x 5
  strengths x 3 rows, which cells are missing, and which feed slot F2`.

## Next step

Navigation: ⬅️ [Recommended Prompts](#recommended-prompts) | 📋 [TOC](#table-of-contents)

Go back to [the design plan](../plans/hypothesis-02-more-correction-more-composition.md), tick
task 4, then answer the two re-score questions in
[the review file](../review/hypothesis-02-more-correction-more-composition.md).
