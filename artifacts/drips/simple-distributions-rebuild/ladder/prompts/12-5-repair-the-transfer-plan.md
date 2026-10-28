# Layer 12, step 5: repair the transfer plan

⬅️ [Layer 12, step 5](../12-the-correction-at-scale.md#5-you-then-the-transfer-plan-repaired-before-it-runs)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`. The walk edits a plan in the other checkout, `~/Playground/PhD/poe_repair_min`, so every path inside the prompt is written from that checkout's root.

```text
/drip-refine-plan
~/Playground/PhD/poe_repair_min/plans/04-does-the-fix-reach-unseen-pairs/plans/hypothesis/04-transfer-as-a-rate-over-fifteen-pairs.md:
settle whether compose rate is binary per pair or a fraction over evaluation seeds (the plan and its
master plan disagree); which 15 pairs it means, since pair_pool.yaml lists 11 trained and 6 held
out, 17 in all; what its x-axis becomes, since leave-one-pair-out holds out exactly 1 of 15 in every
run, so the planned curve from 2/15 to 15/15 cannot be drawn; and what count out of 15 "most of the
15" means

Background
Layer 12 of the composition ladder (artifacts/drips/simple-distributions-rebuild/) measures the
correction's effect on compose rate on held-out animal pairs in poe_repair_min. The transfer plan in
poe_repair_min's scope 04 says how that is read, but four of its terms are undefined or contradict
the files they rest on. This walk repairs them before any run, so step 9's build measures something.
The work happens in the repo ~/Playground/PhD/poe_repair_min.

Objectives
1. Walk the four questions in the skill line, one per piece.
2. Write each answer into the plan's Goals, Description and "What has to pass here".
3. Leave a curve the plan names that can be drawn from 15 one-pair-out runs.
4. Add a CHANGELOG.md entry in poe_repair_min if the direction text changes.

What to read
- ~/Playground/PhD/poe_repair_min/plans/04-does-the-fix-reach-unseen-pairs/plans/hypothesis/04-transfer-as-a-rate-over-fifteen-pairs.md:
  the plan being repaired.
- ~/Playground/PhD/poe_repair_min/plans/04-does-the-fix-reach-unseen-pairs/MASTER_PLAN.md: its
  scope's master plan.
- ~/Playground/PhD/poe_repair_min/artifacts/results/does-the-fix-reach-unseen-pairs/pair_pool.yaml:
  the pair pool the plan means.
- ~/Playground/PhD/poe_repair_min/context/world/compose-rate.md: what a compose rate is.
- ~/Playground/PhD/poe_repair_min/environment/known-failures.md, entry poe-data-001: the seed split.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in the
  poe_repair_min-dog-and-rain repo, and python -m rebuild.<script> runs from toy/ (the guide, "Two
  lanes").
- The scope's master plan defines compose rate over the eval seeds and calls support "held-out pairs
  compose above the do-no-harm baseline across most of the 15, direction intact" ("Reading a result"
  and Goals in plans/04-does-the-fix-reach-unseen-pairs/MASTER_PLAN.md).
- The plan's task 3 collects "compose-rate (binary: yes/no)" per pair (the plan, task 3).
- A compose rate is the fraction of a pair's seeds where grounding-dino-tiny finds at least two
  distinct animal instances, and fail rates are reported as, for example, 1.00 (8/8)
  (context/world/compose-rate.md).
- pair_pool.yaml lists 11 train pairs and 8 heldout entries
  (artifacts/results/does-the-fix-reach-unseen-pairs/pair_pool.yaml).
- 6 of the heldout entries are marked transfer test: leopard×jaguar, frog×toad, eagle×hawk,
  seal×walrus, goose×swan and cow×buffalo (the same file).
- The other 2 heldout entries are cat×dog as the known-failure reference and elephant×penguin as a
  control (the same file).
- The file's header comment says 4 held-out blend pairs, 3 controls and "the full 15", which the list
  does not match: the list holds 17 blend pairs and 1 control (the same file).
- A second file of the same name, artifacts/_shared/cross_pair_pool_configs/pair_pool.yaml, is a
  different pool of 5 train and 5 held-out pairs across groups, and the pool this plan means is the
  one under artifacts/results/does-the-fix-reach-unseen-pairs/ (both files read; the clean-pair-pool
  plan, plans/04-does-the-fix-reach-unseen-pairs/plans/tools/01-the-clean-pair-pool.md).
- The plan's x axis runs 1/15 to 15/15 "fraction held out", while every one of its 15 runs holds out
  exactly one pair (the plan, Description step 4 and Figure Catalog).
- Pass needs at least 13 of 15 runs (the plan, "What has to pass here").
- A run whose distance-reached is zero is delivery-null and left out of the curve (the plan, "What
  has to pass here").
- Runtime is about 8 to 12 hours per run, 5 to 7 days for all 15 (the plan, Considerations).
- biggpu allows one Slurm job per user, so the 15 run one after another or across idle second GPUs
  by the shared-device path (the plan, Considerations and Environment Facts).
- Checkpoints go to /datasets (the plan, Environment Facts).
- Training seeds are 1 to 8 and 17 to 19, and held-out seeds 9 to 16 (environment/known-failures.md,
  entry poe-data-001).
- 12 cache cells in 4 pairs sit in both split folders and resolve held-out first unless split="train"
  is passed (environment/known-failures.md, entry poe-data-001).

Rules
- Edit only this plan.
- A change the master plan needs is proposed for my yes, not written.
- Train nothing.
- No commit.
- Each piece ends by showing the plan's new sentence for one question.

What to hand back
- The plan file with the four answers written into its Goals, Description and "What has to pass
  here".
- The walk's REFINE_WALK.md under ~/Playground/PhD/poe_repair_min/artifacts/ideas/<slug>/.
- A CHANGELOG.md entry in poe_repair_min, if the direction text changes.
- The edited plan, whose compose-rate unit and pair set step 9's build reads.
- The proof: grep -n -i "compose.rate\|pair_pool\|x axis\|out of 15" on the plan prints the four new
  lines.

What you will see
Along the way: one question per piece, each ending on the plan's new sentence and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with the plan's path
and REFINE_WALK.md's; PASS or MISS against 4 of 4 questions answered in the plan and a curve drawable
from 15 one-pair-out runs; what is still open; at most one question for me.

I judge, from the plan: done when each of the four questions has a written answer in it and the curve it names can be drawn from 15 one-pair-out runs.
```

added: the seven sections, from the field-labelled version · every relative path written from ~/Playground/PhD/poe_repair_min, since the walk edits that checkout while the session opens in this repo (from the old Task line, "in repo ~/Playground/PhD/poe_repair_min")
dropped: nothing
