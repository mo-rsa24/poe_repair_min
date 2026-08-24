# Results Summary — LoRA-Fixes-PoE (per rung)

**Generated**: 2026-07-22 (progress-brief over plans/01-05 + inventory/ + W&B).
**One-screen dossier per rung**: current status, what it produced, its W&B runs (worked/crashed), and copy-paste commands to walk the results.

## Read-me-first: two facts that bite

1. **Artifacts are split across two roots, neither complete.** Both `artifacts/` (repo) and `/datasets/mmolefe/poe_repair_min/artifacts/` (canonical) are real directories holding *different* runs. The G6 pool (`pueuo7bl`) lives only under `/datasets`; the cat×dog overfit LoRA only in the repo. So a bare `artifacts/...` path resolves for some runs and silently fails for others. Commands below use the root that actually resolves.
2. **Objective number ≠ artifacts/rungN folder.** Objective 3 (Cross-Pair) has no folder of its own: it reuses rung-2's LoRA. So `artifacts/rung3-group-wise/` = Objective 4, `artifacts/rung4-scale/` = Objective 5.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts
DATA=/datasets/mmolefe/poe_repair_min/artifacts
# W&B: project prime_lab/poe-repair-cross-seed (rungs 1-2), prime_lab/poe-repair-cross-pair (rungs 4-5)
```

---

## Rung 1 — Overfit (Objective 1) ⚠️ partially landed

**Status**: Beachhead landed on cat×dog seed 42 (λ=0 byte-identical to PoE, λ=1 two distinct animals by ~ep600). One more group trained (G4 typewriter×cactus, MDS panels owed). G1–G3 single-seed LoRAs still owed (empty 0-byte stubs). Negative controls all failed as intended, which is what makes the LoRA result mean something. The trained cell plateaus at ~40% of the PoE→Mono distance: delivery, not transfer, is the first-order limiter.

**Produced**
- `REPO/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local/checkpoints/lora_step_062500.pt` (420 LoRA keys). 3.4G run. **run: local, no W&B.**
- `REPO/rung1-overfit/lora/a_typewriter__x__a_cactus/seed_42/run__wandb-wag4z592/` (step 80000; MDS not run). 1.5G.
- `outputs/lora/a_camel__x__a_desert_landscape` (945M) — trained but NOT listed in the plan; needs classifying.
- G1–G3 (`a_dolphin__x__an_ocean_wave`, `a_dog__x__oil_painting_style`, `a_mailbox__x__a_snowfield`): **0-byte stubs, no run.**
- Negative controls: `outputs/group_a_failure/{latent_cnn,latent_unet,frozen_feature_mlp}` — all fail (reported).

**W&B**: `wag4z592` (typewriter single-seed overfit) worked. cat×dog overfit was a local run. Do not confuse `wag4z592` with `xcp40234` (that is the typewriter *cross-seed* run, rung 2).

**Walk it**
```bash
CUDA_VISIBLE_DEVICES="" $PY -c "import torch; sd=torch.load('$REPO/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local/checkpoints/lora_step_062500.pt',map_location='cpu',weights_only=True); print(len([k for k in sd.get('lora_state',sd) if 'lora' in k.lower()]),'lora keys')"   # 420
$PY scripts/build_lora_manifest.py && bash scripts/run_lora_inspector.sh   # → http://localhost:5050 : LoRA-residual + MDS-large tabs
```

---

## Rung 2 — Survive-Noise (Objective 2) ◑ G6 trained, verdict pending

**Status**: G6 (cat×dog) pool trained to convergence (ep2000 / step 100000) with `verdict.json="ok"`. The actual claim (composes on held-out seeds 9–12) is still **pending** (enactment generating). G1–G4 pools part-trained (~ep1000–1200 of 2000), no verdicts yet.

**Produced**
- G6 pool: `DATA/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/` — verdict ok, ep2000, step 100000. **Lives only under `/datasets`.** 3.1G.
- Repo also holds for cat×dog: `taskB__k01_pick1_ep1600__wandb-hbpotmnk` and `taskC__s9_ep1600__wandb-d5b2706v` (a different, earlier sweep).
- G1–G4 pools (part-trained seed banks): `a_dolphin__x__an_ocean_wave` (1.4G), `a_dog__x__oil_painting_style` (1.3G), `a_mailbox__x__a_snowfield` (996M), `a_typewriter__x__a_cactus` (949M).

**W&B** (project `poe-repair-cross-seed`)
- `pueuo7bl` — cat×dog (G6) pool — **worked** (verdict ok).
- `koy9gjis` — cat×dog — **failed** (kept as the failure example).
- `aoj3oz7s` (G1 dolphin×wave), `yrfw5dio` (G2 dog×oil), `ig20iqul` (G3 mailbox×snowfield), `xcp40234` (G4 typewriter×cactus) — **in progress, part-trained, no verdict.**

**Walk it**
```bash
cat "$DATA/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/verdict.json"   # {"verdict":"ok","epoch":2000,"optimizer_step":100000}
$PY scripts/cross_seed_lora_pooling/render_seed_summary.py --pooled-run "$DATA/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl"   # held-out-seed grid
```

---

## Rung 3 — Cross-Pair (Objective 3) ⚠️ code ready, not run

**Status**: Sampler + eval-cache code landed (Plan 12: `sample_heldout --heldout-pair`, `build_eval_cache.py`). **No transfer run executed, no artifacts produced.** Source LoRA is the G6 pool (`pueuo7bl`). This rung is a smoke test only: single-pair→sibling is confounded (a cat×dog-only LoRA saw no variety, so a hit can't be told from a memorised correction that happens to fit). The reviewer-credible transfer test is Rung 4.

**Produced**: none yet. Output will land at `DATA/rung2-survive-noise/cross_seed/a_cat__x__a_dog/heldout_pair/<sibling>/`.

**W&B**: none of its own (inference-only over `pueuo7bl`).

**Walk it (forward-looking, produces the evidence)**
```bash
$PY -m poe_repair.experiments.held_out_seeds.sample_heldout \
  --checkpoint "$DATA/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/checkpoints/lora_step_100000.pt" \
  --pair a_cat__x__a_dog --heldout-pair a_wolf__x__a_husky \
  --out-dir "$DATA/rung2-survive-noise/cross_seed/a_cat__x__a_dog/heldout_pair/a_wolf__x__a_husky"
ls "$DATA/rung2-survive-noise/cross_seed/a_cat__x__a_dog/heldout_pair/a_wolf__x__a_husky/"sample_seed_*.png   # expect seeds 9-12
```

---

## Rung 4 — Group-Wise (Objective 4, `artifacts/rung3-group-wise/`) ◑ G6 smoke only

**Status**: G6 within-group smoke ran (`in_in`+`out_in`, 43/43 cells at a mid-training checkpoint). **Smoke only, not a verdict** (no final-checkpoint crossbar, no Task D). G1–G4 not started; ~520 cache cells missing. Held-out siblings must be concept-disjoint (wolf×husky), not shared-concept (lion×dog shares "dog", a near-freebie).

**Produced**
- `REPO/rung3-group-wise/cross_pair/within_group/g6/main__wandb-ow1jo0xq/eval_crossbar/step_020000/` — 43 cells + baseline (45 PNGs). 402M.
- `within_group/{g1,g2,g3,g4}/` dirs exist but are scaffolds (not run).

**W&B** (project `poe-repair-cross-pair`): `ow1jo0xq` — G6 within-group — **worked (smoke)**.

**Walk it**
```bash
ls "$REPO/rung3-group-wise/cross_pair/within_group/g6/main__wandb-ow1jo0xq/eval_crossbar/step_020000/"*.png | wc -l   # 45
# open contact_sheet_out_in.png in that dir: held-out pairs composing (or not)
```

---

## Rung 5 — Scale (Objective 5, `artifacts/rung4-scale/`) ⏸ trained, crossbar never run

**Status**: `all_groups` LoRA trained to step 30000 on 40 cells (5 pairs × 8 seeds). **The 2×2 crossbar was never sampled**: `cells.jsonl` is absent everywhere, the `out_out` (both-unseen, the headline paper cell) quadrant has no data. A second run died early.

**Produced**
- `REPO/rung4-scale/cross_pair/all_groups/main__wandb-2em6frqv/checkpoints/lora_step_030000.pt`. 477M. No `samples/cells.jsonl`.

**W&B** (project `poe-repair-cross-pair`): `2em6frqv` — **worked to 30k** (incomplete). `0y9un0o4` — **crashed early**.

**Walk it**
```bash
ls "$REPO/rung4-scale/cross_pair/all_groups/main__wandb-2em6frqv/samples/cells.jsonl" 2>/dev/null || echo "ABSENT — this is the gap Rung 5 closes"
# resume + run crossbar (all four quadrants) to produce it:
$PY -m poe_repair.experiments.cross_pair_lora_pooling.sample_crossbar \
  --checkpoint "$REPO/rung4-scale/cross_pair/all_groups/main__wandb-2em6frqv/checkpoints/lora_step_030000.pt" \
  --pair-pool outputs/cross_pair_lora_pooling/pair_pool.yaml \
  --seed-pool-path outputs/cross_pair_lora_pooling/seed_pool.yaml \
  --pair-prompts outputs/cross_pair_lora_pooling/pair_prompts.yaml \
  --out-dir "$REPO/rung4-scale/cross_pair/all_groups/main__wandb-2em6frqv/samples"
```

---

## Rollup

| Rung | Objective | State | Headline evidence | Owed |
|---|---|---|---|---|
| 1 Overfit | 1 | ⚠️ | cat×dog LoRA (062500.pt, 420 keys); controls fail | G1–G3 LoRAs (stubs); G4 MDS |
| 2 Survive-Noise | 2 | ◑ | G6 pool `pueuo7bl` verdict ok (ep2000) | held-out-seed verdict; G1–G4 verdicts |
| 3 Cross-Pair | 3 | ⚠️ | code ready | the run itself (nothing produced) |
| 4 Group-Wise | 4 | ◑ | G6 smoke 43 cells (`ow1jo0xq`) | final crossbar + Task D; G1–G4 |
| 5 Scale | 5 | ⏸ | `all_groups` 30k (`2em6frqv`) | crossbar / `out_out` (`cells.jsonl` absent) |

**One line for the supervisor**: Rung 1 is real, Rung 2's headline pool is trained and passes its training verdict but the held-out-seed proof is still rendering, and Rungs 3–5 are code-ready or partially-trained but have not produced their deciding artifact. The binding constraint is delivery (the ~40% ceiling), not transfer.
