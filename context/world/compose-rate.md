# 🌍 Compose rate (and the scorer behind it)

A compose rate is the fraction of seeds, for one pair, that the validated scorer calls a success
("compose") rather than a failure ("blend", see [chimera.md](chimera.md)). The scorer behind it is
a specific, validated detector rule, not a general judgement of whether the picture "looks right".

![One cell the scorer calls a correct compose: a tabby cat and a white labrador, both clearly present](../images/world/03-compose-correctly-scored.png)
*Both animals are visible and separated. The detector found two distinct "animal" boxes here, which
is the whole rule.*

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [What a compose rate is](#what-a-compose-rate-is)
- [What it looks like](#what-it-looks-like)
- [Why the project cares](#why-the-project-cares)
- [How it shows up in the data](#how-it-shows-up-in-the-data)
- [What people get wrong](#what-people-get-wrong)
- [Where this came from](#where-this-came-from)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-a-compose-rate-is) ➡️

- **Instance count**: the validated method. The scorer runs an object detector for the class
  "animal" and counts distinct, non-overlapping instances.
- **Fail rate**: the complement of compose rate for a pair under plain PoE, used when the pool is
  selected precisely because it fails (see [animal-pair.md](animal-pair.md)).
- **NMS (non-max suppression)**: the step that merges overlapping detector boxes so the same
  animal is not counted twice.

## What a compose rate is

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-it-looks-like) ➡️

**The fraction of a pair's seeds where a detector finds at least two distinct animal instances in
the rendered image.** ✅

`artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json`, read directly: `"compose_rule": "COMPOSE iff
distinct-instance-count('animal', NMS iou<0.5, conf>=0.30) >= 2"`, using detector
`IDEA-Research/grounding-dino-tiny`. `"pass": true` and `"separates_hard_pair_both_ways": true`
mark this rule as the validated one, chosen over two rejected alternatives (a whole-image DINOv2 /
CLIP embedding read, and a per-query box-IoU read), both of which failed to separate a known hard
pair (wolf × husky) correctly. ✅

**It is a per-pair rate over several seeds, not a per-image score.** ✅

`artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md` reports it as, e.g., `1.00 (8/8)`: the pair failed
under plain PoE on all 8 tested seeds. A compose rate is the same fraction read the other way
round, under whichever render method (plain PoE, corrected PoE, or the LoRA) is being evaluated.

## What it looks like

Navigation: ⬅️ [What a compose rate is](#what-a-compose-rate-is) | 📋 [TOC](#table-of-contents) | [Next](#why-the-project-cares) ➡️

![One cell the scorer scores compose, but the audit judges an error: two dog muzzles, no cat](../images/world/04-compose-scorer-error-two-of-one.png)
*The detector correctly counted two animal instances and was scored a success. Both instances are
the same dog. This is the one confirmed error the F2 audit found in the strongest-correction
population; see [What people get wrong](#what-people-get-wrong).*

## Why the project cares

Navigation: ⬅️ [What it looks like](#what-it-looks-like) | 📋 [TOC](#table-of-contents) | [Next](#how-it-shows-up-in-the-data) ➡️

**Every headline number the paper prints about whether the fix works is this rate.** ✅

The dose-response claim ("more correction, more composition"), the held-out-seed and held-out-pair
bars, and the group-pooling verdicts in `MASTER_PLAN.md`'s Goals are all compose rates computed by
this scorer. A weakness in the scorer is a weakness in every one of those claims, which is why the
project runs a standing effort (`can-we-trust-the-compose-rate` in `MASTER_PLAN.md`'s background
experiment table) specifically to bound the scorer's own error before trusting it further.

## How it shows up in the data

Navigation: ⬅️ [Why the project cares](#why-the-project-cares) | 📋 [TOC](#table-of-contents) | [Next](#what-people-get-wrong) ➡️

| Column | Stands for | Example | Entry |
|---|---|---|---|
| `n_instances` | the detector's raw count for one image | `2` (example) | [Dictionary § n_instances](../data/02-dictionary.md#n_instances) |
| `label` | the scorer's compose/blend call for one image | `compose` (example) | [Dictionary § label](../data/02-dictionary.md#label-truth) |
| `truth` | the human-assigned ground truth used to validate the scorer, only present in the validation set | `compose` (example) | [Dictionary § label](../data/02-dictionary.md#label-truth) |
| `fail_rate` / `compose_rate` | the fraction of a pair's seeds scored one way | `0.94` (example, the λ=1 headline before the audit's bound) | [Dictionary § fail_rate / compose_rate](../data/02-dictionary.md#fail_rate-compose_rate) |

## What people get wrong

Navigation: ⬅️ [How it shows up in the data](#how-it-shows-up-in-the-data) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**The scorer counts animals, it does not identify them.** ⚠️

`artifacts/results/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md`, its own summary: of 32 cells behind a 94% headline compose
rate at full correction strength, 1 is a confirmed error (two of the same animal, called a
success), 5 more "cannot be called" confidently by eye (mostly waterfowl and raptor pairs that look
alike even to a person), and 12 more are pairs chosen *because* they look alike (leopard/jaguar,
cow/buffalo, frog/toad), which the audit states no better detector can resolve from the image
alone. Its conclusion, quoted directly: "the true rate at λ=1 sits somewhere around 87% to 94%
rather than anywhere near 60%, and the paper should say 94% is an upper bound of about that size
rather than hedging vaguely." ✅

**Querying the detector once per concept would not fix this.** ✍️

The same card tested this idea against itself: asking the detector for "cat" and separately for
"dog" would still fail to resolve the 5 "cannot call" cells and would do no better than guessing on
the 12 look-alike-by-design cells, so it is recorded as considered and not worth building.

## Where this came from

Navigation: ⬅️ [What people get wrong](#what-people-get-wrong) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| The validated compose rule and its detector | Read in `artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json` | 2026-08-24 |
| Fail-rate as the same fraction under plain PoE | Read in `artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md` | 2026-08-24 |
| The scorer's true-error bound at λ=1 | Read in `artifacts/results/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/README.md` | 2026-08-24 |
| The two example images | Read directly, `artifacts/results/can-we-trust-the-compose-score/do-the-successful-cells-contain-both-animals/{01-both-there,02-two-of-one}/` | 2026-08-24 |
| The scorer-trust standing effort | Read in `MASTER_PLAN.md`, background experiment table | 2026-08-24 |
