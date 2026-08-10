# Re-check the numbers behind the paper's headline figure

## Why you are doing this, and where it lands

The paper's central claim is that composition fails because one correction term is missing. The
figure that carries that claim is **F2** in `paper/iclr/figures.md`, and its caption will say:
*more of the correction composes more, and two size-matched fakes stay flat.*

The experiment behind F2 has already run and the claim survived: turning the correction up from
none to all of it takes the success rate from 7% to 93%, while both deliberately-wrong versions
stay near zero. **What is not yet safe is the exact percentages**, because the tool that counted
the animals has two faults (below). Until they are fixed, the caption cannot quote a number.

So this procedure is the last thing between F2 and the paper. F2 is one of eight slots, and the
writing starts in earnest once five to ten are resolved, which makes this the highest-value hour
available today.

**Vocabulary, once.** The experiment turns the correction up in five steps and measures how often
both animals appear. Each step is a *strength setting* (written λ, from 0 = none to 1 = all). A
*cell* is one batch of images for one animal pair, at one strength, from one random seed; every
count below is counting cells. The *oracle* row injects the real correction; the two *control*
rows inject deliberately wrong vectors of the same size, and they are what makes the oracle's
rise mean something.

When you finish, answer the two open questions in
[../review/hypothesis-02-more-correction-more-composition.md](../review/hypothesis-02-more-correction-more-composition.md).

## Before you start: is the GPU free?

Step 1 needs no GPU. Steps 2 and 4 load GroundingDINO (the object detector that counts animals in
each image), and this node's card is shared, so check first:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
ps -o user,etime,cmd -p <pid> | tail -1     # whose it is, and how long it has been there
```

If someone else holds most of the 23.5GB, do NOT try anyway: the detector dies with
`torch.OutOfMemoryError` partway through. Two options, both correct:

- **Add `--device cpu`** to the commands in steps 2 and 4. Same boxes, same verdicts, slower
  (minutes rather than seconds per image). No queue, no waiting.
- **Or submit to biggpu** with `/run-experiment`, and do the writing tasks meanwhile.

## The two faults in the counting tool

**Fault one: the scorer reads folders that are not part of this sweep.** It collects every image
folder under the output directory rather than only the ones this sweep wrote, so it also picked up
cells from an earlier quick test. Those extra cells exist only at λ=0 and λ=1. That is why the two
end strengths are scored over 44 cells while the three middle ones are scored over 32: the ends
and the middle are being measured on different amounts of data, which is not a fair curve.

**Fault two: the animal count is not reliably a count of animals.** On one image containing two
cats, the detector reported three. The third was a 162-pixel box at confidence 0.309, just over
the 0.30 cutoff, and almost certainly a limb. The success rule is "at least 2 separate animals
detected", so that extra box does not flip this particular verdict. But the 0.30 cutoff was never
checked against this kind of mistake, which makes it a cutoff we got away with rather than one we
chose.

## 1. Confirm what the scorer is currently reading

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

Expect `44` cells per row, and at least one seed beside the intended 9 to 12. Observed on
2026-08-10: seeds `[1, 9, 10, 11, 12]`, so the stray is **seed 1**, and that is the only one.
If the seeds come back as just 9 to 12, someone has already run this procedure and you can stop
here.

## 2. Look at the bad box before choosing a cutoff

Do not pick the confidence cutoff from a number alone. Find the image with the suspect third box
and look at it.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min    # the paths below are repo-relative
$PY scripts/dose_strip.py --pair a_leopard__x__a_jaguar --seed 9 --annotate-boxes --device cpu
```

`--annotate-boxes` draws every box the scorer KEPT, labelled with its confidence and its pixel
size, and prints the raw count beside each panel's verdict. Boxes at least a quarter of the image
across are drawn yellow; smaller ones magenta, which is the shape a limb takes. Drop `--device
cpu` if the GPU is free.

It writes `dose_strip_<pair>_seed<N>_boxes.png` (a separate file from the plain strip, so the
diagnostic never overwrites the figure candidate).

Read the panel where three boxes sit on two animals. Decide what the smallest box that is
genuinely a whole animal looks like, as a fraction of the image, and note the confidence on the
sliver. Both numbers feed step 3.

## 3. Fix both faults in the source code

Two edits, both in the source rather than on a command line, so that any later change to either
one shows up in a git diff instead of hiding in someone's shell history.

**Fix fault one:** hard-code which folder the scorer reads, so it can never wander into old test
output again. Use the sweep's own seed list `(9 10 11 12)` from
`scripts/mechanism_study/run_dose_sweep.sh`, rather than collecting whatever folders happen to
exist.

**Fix fault two:** set the confidence cutoff beside the existing success rule, using the number
you read off the picture in step 2. Add a minimum box size next to it, because a limb is usually
caught by being too small even when its confidence alone would let it through.

```bash
grep -rn "0.30\|conf\|iou" scripts/plot_dose_curves.py poe_repair/**/compose_scorer*.py | head
```

## 4. Re-score and re-read

```bash
$PY scripts/plot_dose_curves.py --root outputs/interaction_term/dose/pairs
```

Add `--device cpu` here too if the card is still held. The floor and the pinned seeds come from
the source edit in step 3, so this command takes no threshold arguments by design.

It prints the per-row table and the AUCs (one number summarising each curve's total area, so the
three curves can be compared at a glance) and rewrites `dose_curves.json` and `dose_curves.png`.

What to check, in order:

- **Cells per row is now equal across all five strengths.** This is the whole point of fixing
  fault one. If the end strengths still have more cells than the middle ones, the seed
  restriction did not take effect and nothing below is worth reading.
- **The oracle row still rises and both control rows stay near the floor.** That direction is
  what the paper claims. If it survives, answer the first open review question `✅` with the new
  percentages.
- **How far the percentages moved.** Record the new numbers beside the old ones (7% to 93%, AUC
  0.422 for the oracle against 0.059 and 0.071 for the controls). If the oracle curve moves a
  lot, the stray cells were inflating it, and that earns a sentence in the paper's methods.

## 5. If the direction does not survive

Then this was a run testing a scientific claim, and the claim failed its pre-set pass mark. The
standing rule applies: answer the review question `❌` with the numbers, close the plan, and write
one follow-on plan file asking why. Do not loosen the confidence cutoff until the curve comes
back. The cutoff was chosen in step 3 by looking at an actual image, and moving it afterwards to
rescue a result is exactly the kind of after-the-fact adjustment this whole setup exists to
prevent.

## 6. Rebuild the five-picture strip

The strip is the row of five images, one per strength setting, that sits above the curves in
figure F2. It is the qualitative half of the claim: the curves say the rate went up, the strip
shows what "went up" looks like.

```bash
$PY scripts/dose_strip.py --pair a_leopard__x__a_jaguar --seed 9
```

The existing strip was built while the sweep was still incomplete. Rebuild it now that every batch
is present, and answer the second open review question.

## Recommended Prompts

Run these when a term here stops meaning anything. Each leaves something you can return to.

- **On λ and the correction itself:** `/drip --math the correction r_t and what the strength
  setting λ scales, one step per message` → `/polish` to file it → `/math-scene` on that file, to
  drag λ and watch the prediction move from PoE toward the Mono target.
- **On the counting rule:** `/demonstrate show me five cells the scorer calls compose and five it
  calls blend, with their boxes drawn, so I can see the rule agree and disagree with my eye`.
- **On the shape of the whole sweep:** `/experiment-atlas the dose sweep: 8 pairs x 4 seeds x 5
  strengths x 3 rows, which cells are missing, and which feed slot F2`.

## Kept terms

- **GroundingDINO**: the pretrained object detector used to count animals in generated images.
- **λ (lambda)**: the strength setting, how much of the correction is added back, from 0 to 1.
- **AUC**: area under the curve, one number summarising a whole curve so three can be compared
  at a glance.
- **oracle / controls**: the oracle uses the true interaction term; the two controls are
  deliberately wrong versions that should fail, and do.
- **NMS**: the step that merges overlapping detections, so one animal found twice counts once.
