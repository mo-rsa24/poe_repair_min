# Run 2: the adapter's fit on its own corrected path (task 1.3)

Rank 8 at 30k, cat x dog, held-out seeds 9 to 16, λ = 1 at every step, 50 DDIM steps. Slurm job
49421 on mscluster72, 2026-09-03 17:07 to 17:15. Script `scripts/showcase/on_policy_fit_cosine.py`,
output `/datasets/mmolefe/poe_repair_min/outputs/showcase/on_policy_fit_r8_030000/`
(`on_policy_fit.json`, `on_policy_fit_table.md`, one `seed_N_corrected.png` per seed,
`contact_sheet.png`).

At every step of the corrected trajectory the frozen UNet is asked for the true target at the
state actually reached (one joint forward plus the three branches), and cos(Δ̂, Δ_true) is
recorded together with how far the state has drifted from the cached one.

| seed | cos all | early (0-4) | commit (5-24) | late (25-49) | ‖Δ̂‖/‖Δ‖ | state drift @10 | @25 | @49 | cos(live target, cached target), late |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 0.732 | 0.877 | 0.683 | 0.741 | 0.854 | 0.076 | 0.252 | 0.849 | 0.006 |
| 10 | 0.851 | 0.914 | 0.812 | 0.870 | 0.715 | 0.081 | 0.242 | 0.771 | 0.014 |
| 11 | 0.882 | 0.949 | 0.905 | 0.850 | 0.851 | 0.037 | 0.144 | 0.693 | 0.035 |
| 12 | 0.792 | 0.774 | 0.795 | 0.793 | 0.694 | 0.062 | 0.219 | 0.783 | 0.006 |
| 13 | 0.824 | 0.946 | 0.796 | 0.822 | 0.725 | 0.058 | 0.218 | 0.776 | 0.012 |
| 14 | 0.773 | 0.843 | 0.753 | 0.774 | 0.857 | 0.079 | 0.273 | 0.864 | 0.006 |
| 15 | 0.884 | 0.907 | 0.921 | 0.849 | 0.841 | 0.058 | 0.209 | 0.798 | 0.019 |
| 16 | 0.800 | 0.631 | 0.815 | 0.823 | 0.778 | 0.044 | 0.183 | 0.809 | 0.021 |
| **mean** | **0.817** | 0.855 | 0.810 | 0.815 | 0.789 | | | | 0.015 |

Drift is the relative L2 distance between the reached latent and the cached latent at the same
step. Run 1's cached-state cosine for cat x dog was 0.860.

**What it says**

The corrected path leaves the cache almost completely (drift 0.7 to 0.86 by the last step), and
the true target at the reached state is orthogonal to the cached target at the same step
(cosine 0.015 in the late window). The adapter still fits the live target at 0.82 against 0.86
on cached states. So the adapter has learned a rule that holds on states it never saw, and the
cost of leaving the cache is about 0.04 of cosine. Exposure bias is real and small.

The render-time 0.54 and 0.40 measure something else. In
`poe_repair/experiments/cross_pair_lora_pooling/_inline_sampling.py:111` the "direction cosine"
is cos(Δ̂_t, Δ̄_t) against the pool-mean correction across all train pairs and seeds. It is the
share of the correction that lies along the pool average, which is a shared-component measure
and cannot rise to 1 for a state-specific target. It was never a fit, never a generalization
gap, and never the number to chase.

Renders: seeds 9, 10, 12, 13, 16 show a cat and a dog; 11 shows a dog with two cats; 15 shows
two cats; 14 shows one animal beside a child. Seed 14 fits the live target at 0.77, above seed 9,
and still fails, so the one-animal outcome is not a fit failure.
