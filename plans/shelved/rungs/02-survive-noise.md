# 🌱 Survive-Noise — one LoRA pooled over seeds generalises to held-out seeds, per group

## Description
Train one LoRA on several starting seeds of the same pair, then test it on four seeds it never
trained on ({9–12}). Start with cat×dog (G6), then do the same for one pair per group (G1–G4). A
seed only re-rolls the starting noise, so this asks one thing: does the fix survive different noise
while the two concepts stay fixed? Folds phase files 07 (the "is the correction the same across seeds?" check,
which landed near "no"), 08 (the cat×dog pool), 10 (per-group), and the image-geometry side-thread
(13, 15-latent) as one optional task.

## Purpose
Serves Objective 2 (Survive-Noise) and Definition-of-Done item 2. It decides one thing: is the fix
a lucky-seed accident, or does it hold across seeds without changing the model.

## Goal
For each group, a pooled LoRA with a clear pass or fail on unseen seeds: both concepts show up on at
least 3 of the 4 held-out seeds.

## Latest status + how to see it
**As of 2026-07-22.** G6 (cat×dog) pool trained to convergence (ep2000 / step 100000), `verdict.json="ok"`. The held-out-seed proof (composes on seeds 9–12) is still pending (enactment generating). G1–G4 pools part-trained (~ep1000–1200 of 2000), no verdicts.

Owning artifacts:
- G6 pool (lives ONLY under /datasets): `/datasets/mmolefe/poe_repair_min/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/` — verdict ok, ep2000, step 100000.
- Repo also has (a different earlier sweep): `taskB__k01_pick1_ep1600__wandb-hbpotmnk`, `taskC__s9_ep1600__wandb-d5b2706v`.

W&B (project `poe-repair-cross-seed`): `pueuo7bl` cat×dog **worked** (verdict ok); `koy9gjis` cat×dog **failed** (failure example); G1 `aoj3oz7s` (dolphin×wave), G2 `yrfw5dio` (dog×oil), G3 `ig20iqul` (mailbox×snowfield), G4 `xcp40234` (typewriter×cactus) — in progress, no verdict.

See it:
```bash
DATA=/datasets/mmolefe/poe_repair_min/artifacts
cat "$DATA/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/verdict.json"   # {"verdict":"ok","epoch":2000,"optimizer_step":100000}
```

## Tasks
- [x] ✅ Cross-seed Δ_t structure diagnostic (N=8). → G06: ~seed noise at the cross-seed mean (`landing_6`); honest prior is a weak seed-mean.  ✓ stated (regenerable, no on-disk artifact)
- [x] ✅ Pooled LoRA on cat×dog (G6), seeds {1–4}. → G07: reached ep 2000, `verdict.json="ok"`. Artifact `/datasets/mmolefe/poe_repair_min/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/`.  ✓ verified (verdict.json="ok", ckpt loads)
- [ ] ⚠️ **[publishable-bar]** Render G6 held-out-seed samples (seeds 9–12) from the pueuo7bl pool + write the held-out verdict — the pending DoD-2 enactment (`pueuo7bl` is trained/verdict-ok but the held-out-seed proof was never rendered).
  Prompt: `$PY scripts/cross_seed_lora_pooling/render_seed_summary.py --pooled-run /datasets/mmolefe/poe_repair_min/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl`
- [ ] ⚠️ **[publishable-bar]** Finish the G1–G4 per-pair pooled runs (currently part-trained ≈ ep1000–1200 of 2000, no verdicts).
  Prompt (`/run-experiment`): `for G in g1 g2 g3 g4; do $PY -m scripts.cross_seed_lora_pooling.run_group --group $G --total-epochs 2000; done` (resolves each group→representative pair; resume via `checkpoints/latest.json` under `artifacts/rung2-survive-noise/cross_seed/<pair>/taskB__k04_ep2000/`).
- [ ] ⚠️ **[publishable-bar]** Task C (per-seed ceiling) + Task D (Δ̄_t bridge) + contact sheets, per group.
  Prompt: `$PY -m poe_repair.experiments.cross_seed_lora_pooling.contact_sheet --pair $PAIR --task B` (and `--task C`); `$PY -m poe_repair.experiments.cross_seed_lora_pooling.task_d_bridge --pooled-run <run-dir>`
- [ ] ⚠️ **[publishable-bar]** Classify each group poor/bad/unknown/good on held-out seeds; write the per-group table.
- [ ] ⚠️ **[optional]** (Side-thread) Latent-manifold / semantic-MDS convergence read for cat×dog: backfill `manifold_cache` (currently 0/7) via `scripts/manifold/sample_with_trajectory.py`, then `scripts/manifold/seed42_phase1.py`. → parks G12 (raw-latent MDS orders by appearance, not co-occurrence).

## Recommended skill
▶ `/run-experiment` ✅ — drives `run_group.py` (Task A / Step 0 / Task B / Task D) per group.
   alt: `/analyze-run <wandb_id>` for the pooled learning curve.

## Engagement Instructions
```
$ cat /datasets/mmolefe/poe_repair_min/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/verdict.json   # {"verdict":"ok",...}
# Per group: samples/heldout/sample_seed_{9,10,11,12}.png show composition (not the PoE chimera); contact_sheet_B.png rendered.
```

**▶ View in the web app:** held-out-seed grids render via `render_seed_summary.py`; the inspector's pair dropdown surfaces per-pair residual/MDS panels.
```bash
$PY scripts/build_lora_manifest.py && bash scripts/run_lora_inspector.sh   # → http://localhost:5050
```
