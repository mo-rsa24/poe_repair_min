# Re-score the dose sweep

Read this to completion, do what it says, then answer the two open questions in
[../review/03-dose-response.md](../review/03-dose-response.md).

No GPU is needed for steps 1 to 3. Step 4 loads GroundingDINO, which fits the session node's card
alongside nothing else; check `nvidia-smi` for co-tenants first.

## Why this is being re-run

The sweep's result is not in doubt: the oracle rises from 7% to 93% across λ and both controls stay
near the floor. Two things about the *instrument* are, and both would change the exact percentages
that reach the paper.

The scorer globs the whole dose output tree instead of this sweep's cells, so it swept in 12 older
smoke cells at seeds 1 and 42. Those landed only at λ=0 and λ=1, which is why the endpoints are
scored over 44 cells and the middle doses over 32.

And the count is not reliably a count. On one image with two cats it reported 3 instances, the third
a 162px box at confidence 0.309, just over the 0.30 floor and almost certainly a limb. The rule
("compose iff at least 2 distinct instances") tolerates that, but the floor was never chosen against
this failure, so it is untested rather than justified.

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

Expect `44` per row and seeds including 1 and 42 beside the intended 9 to 12. If the seeds are
already only 9 to 12, this procedure has been run and you can stop.

## 2. Look at the sliver case before choosing a floor

Do not pick a threshold from a number alone. Find the image and look at it.

```bash
$PY scripts/dose_strip.py --pair a_leopard__x__a_jaguar --seed 9 --annotate-boxes
```

Read the panel where three boxes are drawn on two animals. Decide what the smallest box that is
genuinely a whole animal looks like, as a fraction of the image, and note the confidence the model
gave the sliver. Both numbers feed step 3.

## 3. Pin the root and set the floor, in the source

The floor goes in the scorer's source beside the existing rule, not on a command line, so a later
change to it shows up in a diff. Add a minimum box area alongside the confidence floor: a limb is
usually rejected by size where confidence alone lets it through.

```bash
grep -rn "0.30\|conf\|iou" scripts/plot_dose_curves.py poe_repair/**/compose_scorer*.py | head
```

Set the seed restriction the same way, from the sweep's own seed list `(9 10 11 12)` in
`scripts/mechanism_study/run_dose_sweep.sh`, rather than by globbing.

## 4. Re-score and re-read

```bash
$PY scripts/plot_dose_curves.py --root outputs/interaction_term/dose/pairs
```

It prints the per-row table and the AUCs and rewrites `dose_curves.json` and `dose_curves.png`.

What to check, in order:

- **cells per row is now equal across all five doses.** This is the whole point of step 3. If the
  endpoints still exceed the middle, the seed restriction did not take.
- **the oracle still rises and both controls stay near the floor.** The direction is what the paper
  claims. If it survives, answer the first open review question `✅` with the new percentages.
- **how far the percentages moved.** Record the new numbers against the old ones (7% to 93%, AUC
  0.422 / 0.059 / 0.071) in the review file. A large move on the oracle row means the older smoke
  cells were carrying it, which is worth a sentence in the paper's methods.

## 5. If the direction does not survive

Then this is a claim-testing run that failed its bar, and the rule applies: answer the review
question `❌` with the numbers, finish the plan, and write one follow-on plan file asking why. Do
not widen the confidence floor until the curve comes back. The floor was chosen in step 3 against a
picture, and moving it afterwards to rescue a result is the thing the whole arrangement exists to
prevent.

## 6. Regenerate the strip

```bash
$PY scripts/dose_strip.py --pair a_leopard__x__a_jaguar --seed 9
```

The existing strip was built while the sweep was partial. Rebuild it on complete cells and answer
the second open review question.
