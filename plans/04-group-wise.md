# 🧩 Group-Wise — "group" is a deployable pooling unit

## Description
For each group: train one LoRA on 7 pairs from that group across all 12 seeds, then test it on 3
pairs from the same group it never trained on (only the pair is held out, not the seed). Does it show
both concepts on the unseen pairs, and beat the weaker single-pair test from rung 3? Folds phase file
16. So far only G6 has run, and only as a smoke test.

## Purpose
Serves Objective 4 (Group-Wise) and Definition-of-Done item 4. It answers the question the Scale rung
cannot on its own: is a whole difficulty group the right unit to ship one LoRA for, or is it
per-pair, or one LoRA for everything?

## Goal
For each group (or the honest subset), a pass/fail on transfer to held-out pairs: the held-pair
result (`out_in`) shows both concepts on at least 2 of 3 pairs (≥50%), and beats the single-pair test
from rung 3.

## Tasks
- [x] ✅ G6 within-group smoke: train ~30k steps + mid-training crossbar eval. → G10: `in_in`+`out_in` sampled, 43/43 cells (`artifacts/rung3-group-wise/cross_pair/within_group/g6/main__wandb-ow1jo0xq/`, `eval_crossbar/step_020000/`). Smoke only, not a verdict.  ✓ verified (manifest n_cells_sampled=43, ckpt loads)
- [ ] ⚠️ Build the ~520 missing cache cells for the 40 new within-group pair slugs (seeds 1–12).
  Prompt: `bash scripts/cross_pair_lora_pooling/build_plan16_caches.sh` (idempotent; ~11 h GPU).
- [ ] ⚠️ Run G1 end-to-end (most-opposite to G6), then G2–G4.
  Prompt (`/run-experiment`): `GROUPS=g1 bash scripts/cross_pair_lora_pooling/run_within_group.sh` (leak-guard → train → sample `--quadrants in_in,out_in --out-in-train-seeds 12` → contact sheet → Task D).
- [ ] ⚠️ Finish G6's full tail: sample_crossbar → contact_sheet → Task D at the final checkpoint (not just the mid-training eval).
- [ ] ⚠️ Hold out CONCEPT-DISJOINT siblings (e.g. wolf×husky), not shared-concept ones (lion×dog shares "dog" → near-freebie). Shared-concept transfer does not count as transfer.
- [ ] ⚠️ Read every transfer cell with TWO labelled verdicts: (a) deployable — the image composes (two animals); (b) scientific — Task D cosine of Δ̂_t to the group-mean Δ̄_t^(G). Never sell a high cosine as "it works"; they are separate tiers.
- [ ] ⚠️ On a failed transfer cell, diagnose the mode: magnitude (Task D norm-ratio), timing (cos(t) vs the commitment window), group-coarseness (cell Δ_t vs group-mean), off-manifold drift. Measure the PoE→Mono distance reached and test the prediction it lands BELOW the trained-cell ~40% plateau (delivery is the first-order limiter, not transfer).
- [ ] ⚠️ Cheap check first (graft: Task-D-as-gate) — before the ~30 h train, ask whether the group shares one correction: run each training pair's single-pair LoRA (from rung 1), read off its per-step nudge Δ̂_t, and measure how closely they line up with the group average Δ̄_t^(G). If they line up, the group is a real unit and should transfer; if not, skip the expensive run.
  How: add a `--pre-screen` mode to `task_d_bridge` that reads `artifacts/rung1-overfit/lora/<pair>/seed_42/run__local` per group pair (forward passes only), then run it per group.
- [ ] ⚠️ Which delivery fix wins (grafts B1/B2)? On held-out pairs run four columns: LoRA-only, LoRA + Attend-and-Excite at inference (B1), LoRA with a "keep every subject visible" term added to training (B2), and A&E-only. Expect B1 to transfer better and B2 to look better on trained pairs; the four columns also prove the LoRA is doing the work.
  How (`/run-experiment`): run the 4-way grid on g6's held-out pairs; put the columns in one contact sheet.
- [ ] ⚠️ Per-group classification + rung bucket landing (Good/Mixed/Bad); decide whether the group is a pooling unit.

## Recommended skill
▶ `/run-experiment` ✅ — drives `run_within_group.sh` per group.
   alt: `/analyze-run <wandb_id>` for the per-pair loss buckets (flat/oscillating loss = `multi_pair_trainer` regression — kill early).

## Engagement Instructions
```
$ ls artifacts/rung3-group-wise/cross_pair/within_group/g6/main__wandb-ow1jo0xq/eval_crossbar/step_020000/*.png | wc -l   # 45 (43 cells + baseline)
# Per new group g<N>: within_group/g<N>/main/samples/contact_sheet_out_in.png shows held-out pairs composing on ≥2/3 pairs.
```
