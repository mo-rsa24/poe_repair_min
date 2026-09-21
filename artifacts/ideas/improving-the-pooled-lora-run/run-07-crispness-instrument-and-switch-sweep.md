# Run 7: an instrument for crispness (task 8.1), and where the adapter should hand off (task 8.2)

Claim 8 says softness is trained in and the adapter can be taught to spend less of itself in the
low-noise steps. Its gate is an instrument that scores "crisp", because the instance count is
blind to it and Laplacian variance was shown unusable. This run tried the one unused idea, a
paired read against each seed's own joint-prompt render, and swept the hand-off step to locate
the window.

Conditions: rank 32 @ 30050, λ 1.2 (the clean-tail grid's value, not the 1.0 of the report's main
results), 50 DDIM steps, cat x dog held-out seeds 9 to 16. Existing renders from
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/` (adapter alone; hand-off at
20 and 30, k=0); new renders at 25, 35, 40, 45 from `scripts/showcase/handoff_switch_sweep.py`.
Output `/datasets/mmolefe/poe_repair_min/outputs/showcase/crispness/`.

## 8.1: the paired high-frequency read is not an instrument

`scripts/showcase/crispness_paired_hf.py`. Share of image energy above a quarter of the Nyquist
radius (wavelengths under about 8 px on a 1024 image), on the render and on the seed's `mono.png`,
reported as a ratio. 1.0 means as much fine detail as the joint prompt drew.

| condition | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | median |
|---|---|---|---|---|---|---|---|---|---|
| plain PoE | 0.383 | 1.658 | 2.831 | 0.820 | 0.935 | 0.453 | 1.159 | 1.301 | 1.047 |
| adapter alone | 0.712 | 3.244 | 0.874 | 2.311 | 1.136 | 0.154 | 0.658 | 1.108 | 0.991 |
| hand-off @20 | 0.916 | 4.204 | 1.135 | 3.204 | 1.683 | 0.238 | 0.812 | 1.289 | 1.212 |
| hand-off @30 | 0.856 | 4.015 | 0.959 | 2.692 | 1.410 | 0.160 | 0.680 | 1.263 | 1.111 |
| joint prompt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

**Rejected.** The medians order adapter-alone below hand-off@30 below hand-off@20, the same
direction DINOv2 gave, but the per-seed values swing from 0.15 to 4.2, so the pairing tamed
nothing. Looking at seed 10 says why: hand-off@20 carries four times the reference's fine detail
because the reference is a smooth photo of two fluffy dogs and the render has fur texture and a
busier background. The read counts texture, exactly as Laplacian variance did. And the DINOv2
grid already recorded hand-off@20 as the worst condition (nearness gain 0.010, two composed seeds
lost), which this read ranks best. That is the surprise case named before the run: "more fine
detail" and "crisper" are different things.

## What looking at the pictures found instead

The DINOv2 primary read, cosine distance to the seed's joint-prompt render, is measured against a
wrong picture on three of the eight seeds. Seed 10's reference is two dogs; the adapter alone drew
a cat and a dog, which is the better composition, and every hand-off keeps it. Seed 14's reference
is two cats. Seed 13's is two dogs. Those are the three cells claim 7's eye pass flagged, and the
0.038 hand-off gain in the clean-tail finding was averaged over them.

So claim 7's one free consequence, that a held-out cell whose joint target shows one concept twice
cannot count as a compose failure, applies to the crispness read as well: it cannot serve as a
nearness reference either. The recorded clean-tail numbers at cutoffs 20 and 30 (nearness gain
0.010 and 0.038, composed 5 and 6 of 8 against 7 for the adapter alone) should be re-read on the
five seeds whose reference is right.

What remains true from the pictures: on seed 10 the hand-off at 30 turns the adapter's slightly
wrong cat face into a proper one with the dog kept, which is the fidelity gain the finding
described, and it is visible without any instrument.

## 8.2: the window is located, and the switch step is 25

`scripts/showcase/handoff_switch_sweep.py` rendered cutoffs 25, 35, 40 and 45 beside the
existing 20 and 30 (k = 0, no corrector), then scored every switch point with the compose count
and the DINOv2 distance to the seed's joint render. Output
`/datasets/mmolefe/poe_repair_min/outputs/showcase/crispness/handoff_switch_sweep.json`.
"Gain" is the fall in that distance against the adapter on all 50 steps; the clean-tail finding's
bar is 0.05.

| adapter on steps | composed, 8 seeds | gain, 8 seeds | composed, 5 good-reference seeds | gain, 5 seeds |
|---|---|---|---|---|
| [0, 20) | 5/8 | +0.010 | 4/5 | +0.022 |
| [0, 25) | 6/8 | **+0.041** | 4/5 | **+0.050** |
| [0, 30) | 6/8 | +0.038 | 4/5 | +0.037 |
| [0, 35) | 7/8 | +0.009 | 4/5 | +0.022 |
| [0, 40) | 7/8 | +0.007 | 4/5 | +0.013 |
| [0, 45) | 7/8 | +0.001 | 4/5 | +0.009 |
| [0, 50) | 7/8 | 0 | 4/5 | 0 |

**The curve is single-peaked at 25.** Gain rises from 20 to 25, then falls smoothly to zero at
50. Handing off earlier than 25 starts to cost composition (seed 10 lost at 20); handing off
later gives the adapter's own low-noise steps back and the gain drains away. So the softness lives
in steps 25 to 50, and the best switch is the step the commit window ends.

**On the five seeds whose reference is a real cat and dog, the gain at 25 is 0.050, on the bar,
with no composition lost at any cutoff** (seed 11 never composes at any switch point, with the
adapter on or off). The two seeds that need the adapter past 30 to reach a count of 2 are 14 and
10, and both have a wrong reference: seed 14's render at 35 and beyond is a child beside a cat,
counted as two animals. So the 7-of-8 that the all-50 render reports is partly the count scorer
passing a picture that should not pass.

Read as claim 8.2 wanted: the window is locatable, it is [25, 50), and a training change that
keeps the adapter out of it (loss masked to steps 0 to 24, hand-off at inference) has a target
to aim at. n = 5 on the clean read, so this is a located lever at the bar, not a cleared one.
