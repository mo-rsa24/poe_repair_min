# 🌱 Survive-Noise — one LoRA pooled over seeds generalises to held-out seeds, per group

## Description
Train one LoRA on several starting seeds of the same pair, then test it on four seeds it never
trained on ({9–12}). Start with cat×dog (G6), then do the same per group (G1–G4). A seed only
re-rolls the starting noise, so this asks one thing: does the fix survive different noise, holding
the two concepts fixed? Folds phase files 07 (the "is the correction the same across seeds?" check,
which landed near "no"), 08 (the cat×dog pool), 10 (per-group), and the image-geometry side-thread
(13, 15-latent) as one optional task.

## Purpose
Serves Objective 2 (Survive-Noise) and Definition-of-Done item 2. It decides one thing: is the fix
a lucky-seed accident, or does it hold across seeds without changing the model.

## Goal
For each group, a pooled LoRA with a clear pass/fail on unseen seeds: both concepts show up on at
least 3 of the 4 held-out seeds.

## Tasks
- [x] ✅ Cross-seed Δ_t structure diagnostic (N=8). → G06: ~seed noise at the cross-seed mean (`landing_6`); honest prior is a weak seed-mean.  ✓ stated (regenerable, no on-disk artifact)
- [x] ✅ Pooled LoRA on cat×dog (G6), seeds {1–4}. → G07: reached ep 2000, `verdict.json="ok"`. Artifact `artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/`.  ✓ verified (verdict.json="ok", ckpt loads)
- [ ] ⚠️ Finish the G1–G4 per-pair pooled runs (currently part-trained ≈ ep1000–1200 of 2000, no verdicts).
  Prompt (`/run-experiment`): `for G in g1 g2 g3 g4; do $PY -m scripts.cross_seed_lora_pooling.run_group --group $G --total-epochs 2000; done` (resolves each group→representative pair; resume via `checkpoints/latest.json` under `artifacts/rung2-survive-noise/cross_seed/<pair>/taskB__k04_ep2000/`).
- [ ] ⚠️ Task C (per-seed ceiling) + Task D (Δ̄_t bridge) + contact sheets, per group.
  Prompt: `$PY -m poe_repair.experiments.cross_seed_lora_pooling.contact_sheet --pair $PAIR --task B` (and `--task C`); `$PY -m poe_repair.experiments.cross_seed_lora_pooling.task_d_bridge --pooled-run <run-dir>`
- [ ] ⚠️ Classify each group poor/bad/unknown/good on held-out seeds; write the per-group table.
- [ ] ⚠️ (Side-thread, optional) Latent-manifold / semantic-MDS convergence read for cat×dog: backfill `manifold_cache` (currently 0/7) via `scripts/manifold/sample_with_trajectory.py`, then `scripts/manifold/seed42_phase1.py`. → parks G12 (raw-latent MDS orders by appearance, not co-occurrence).

## Recommended skill
▶ `/run-experiment` ✅ — drives `run_group.py` (Task A / Step 0 / Task B / Task D) per group.
   alt: `/analyze-run <wandb_id>` for the pooled learning curve.

## Engagement Instructions
```
$ cat artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/verdict.json   # {"verdict":"ok",...}
# Per group: samples/heldout/sample_seed_{9,10,11,12}.png show composition (not the PoE chimera); contact_sheet_B.png rendered.
```

**▶ View in the web app:** held-out-seed grids render via `render_seed_summary.py`; the inspector's pair dropdown surfaces per-pair residual/MDS panels.
```bash
$PY scripts/build_lora_manifest.py && bash scripts/run_lora_inspector.sh   # → http://localhost:5050
```
