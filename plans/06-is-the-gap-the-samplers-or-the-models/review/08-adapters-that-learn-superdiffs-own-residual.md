# 🎓 Review: adapters that learn SuperDiff's own residual

**Nothing has run yet, and this plan may never run.** It opens only if
[step 49](07-does-the-poe-trained-correction-reach-superdiff.md) says the PoE-trained
correction does not carry over. This file judges
[the design](../plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md).

## Recommended prompt (when the run lands)

```
/analyze-run the SuperDiff-trained adapters, three ranks against the no-adapter and PoE-adapter sheets
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/08-adapters-that-learn-superdiffs-own-residual.md) | the cache, the target, three trainings, six sheets |
| **this file** | **the verdict: gated, not yet run** |
| [step 49's verdict](07-does-the-poe-trained-correction-reach-superdiff.md) | the gate |
| [step 28's verdict](05-what-changes-when-superdiff-leaves-its-own-defaults.md) | the no-adapter baseline sheets |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The SuperDiff cache**: per (pair, seed) in the pool, 200 step files each holding `x_t` and
  the four raw predictions (prompt 1, prompt 2, joint, unconditional) along SuperDiff's own
  trajectory at `kappa` 0.5.
- **Δ_t^SD**: the training target, `ε̃_J − eps_M` with `eps_M` formed from the cached raw
  predictions at `kappa` 0.5.
- **The gate**: step 49's verdict; indistinguishable or hurts opens this plan, transfers closes it
  unrun.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** A fail is a finding about whether SuperDiff's residual is learnable by this
adapter family; an unconverged training is inconclusive, not a fail.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| The gate decision (task 0.3) | — | 2026-09-03 | none | Step 49's verdict, quoted: "Hurts. On every sheet rendered (rank 8 cat×dog, rank 8 butterfly×meadow, rank 16 cat×dog), no seed reaches two separate concepts at any λ … the adapter's correction is the right size but mostly orthogonal to SuperDiff's missing residual (median per-step cosine peaking at +0.34)." | ✅ gate open, this plan runs |
| Cache size statement (task 1.1) | Builds a measuring tool | 2026-09-03 | none | Pool from `phase1_r8_450k/pair_pool.json` + `seed_pool.json`: 11 train pairs (`a_wolf__x__a_husky`, `a_lion__x__a_tiger`, `a_cheetah__x__a_cougar`, `a_horse__x__a_zebra`, `a_donkey__x__a_pony`, `a_crocodile__x__an_alligator`, `a_rabbit__x__a_hare`, `a_dolphin__x__a_porpoise`, `a_crow__x__a_raven`, `a_gorilla__x__a_chimpanzee`, `a_turtle__x__a_tortoise`) × seeds 1–8 = 88 cells, matching the PoE run's `dataset_meta.json` (88 cells, 4,400 steps). A PoE step file is 657,685 bytes (five 4×128×128 fp16 tensors); the SuperDiff cache stores five per step too, so 88 × 200 × 0.66 MB ≈ **11.6 GB**. `df /datasets`: 377T total, 104T used, 273T available (28%). Build cost: 88 renders at 200 steps with the joint row, ~135 s each on an RTX A6000, ≈ 3.3 h on one device | ✅ fits, no step thinning needed |
| One-cell cache check against the builder's own `r_t_sd_norms` (`a_wolf__x__a_husky` seed 1, mscluster109 device 0) | Builds a measuring tool | 2026-09-03 | 1 render, 75.5 s | 200 step files, timesteps 1000 → 5 in steps of 5; target `ε̃_J − eps_M(κ=0.5)` formed from the saved fp16 tensors against the live norm at steps 0, 50, 100, 199: 5.358/5.359, 14.546/14.546, 29.943/29.943, 9.614/9.614, max relative error 3.6×10⁻⁵ (tolerance 2×10⁻²) | ✅ the cache is what the sampler computed |
| Trainer dry run, `--compose superdiff --kappa 0.5`, rank 8, one epoch on the one cached cell (smoke pool), mscluster109 device 0 | Builds a measuring tool | 2026-09-03 | 1 epoch = 50 optimizer steps, ~2 min | `corrector/superdiff/dryrun/sdlora_dryrun_r8/checkpoints/lora_step_000050.pt`: 420 LoRA tensors, all finite, L2 24.0; `config.json` records `compose: superdiff, kappa: 0.5`; 210 modules matched; dataset 200 cached steps from 1 cell | ✅ the whole path runs, cache → step → checkpoint |
| The cache over the pool, 88 cells in two shards on mscluster109 | Builds a measuring tool | 2026-09-03, shard 1 pid 1805839 on device 1, shard 0 pid 1807651 on device 0 | ~77 s per cell, both shards done in ~70 min | `corrector/superdiff/cache/train/<pair>/seed_<n>/`: 88 `meta.json`, 17,600 step files, 11 GB | ✅ complete |
| First launch of all three, 2026-09-03 12:32–12:34 (W&B `nrvy0svn`, `wm9uqvb5`, `ph037jlk`) | Tests the claim | killed at epoch ≤ 5, 12:36 | ~5 min | nothing kept; those three W&B runs stay listed as aborted | ⏹ stopped on purpose: the "commit" loss bucket and both kill criteria read step indices 5–24, the window tuned for 50-step cells, which on a 200-step cache is the opening 10% of the run at high noise, and the "commit loss must halve by step 5000" rule (the one that aborted rank 32's first PoE leg) could then fire spuriously ~1 h in. Relaunched with the window rescaled to the same fraction, steps 20–99 (`--commit-window 20 100`), both kill criteria kept as this project's standing bar |
| Training, rank 8 | Tests the claim | 2026-09-03 12:38, `nohup` pid 1821728 on mscluster109 device 0 (`co3`), log `corrector/superdiff/sdlora_r8_100k.log`, run dir `corrector/superdiff/sdlora_r8_100k/` | 2000 epochs × 50 steps; ~42 s per epoch (first launch's measurement), ETA ≈ 23.5 h | W&B `prime_lab/poe-repair-animals-compose/runs/t5b00h56`; final checkpoint `lora_step_100000.pt` | ⏹ stopped by the user 2026-09-04 01:30 at optimizer step 61,000 (last checkpoint `lora_step_060000.pt`); loss 0.0008, plateaued from ~35k |
| Training, rank 16 | Tests the claim | 2026-09-03 12:38, pid 1821729 on mscluster109 device 1 (`co3`), log `sdlora_r16_100k.log` | ~42 s per epoch, ETA ≈ 23.2 h | W&B `runs/5oz1bsa1` | ⏹ stopped by the user 2026-09-04 01:30 at step 62,000 (`lora_step_060000.pt`); loss 0.0025, plateaued from ~40k |
| Training, rank 32 | Tests the claim | 2026-09-03 12:41, pid 348108 on mscluster112 device 0 (`co3_bw`) until the step-5000 kill; resumed 14:15 as pid 2140758 on **mscluster106 device 1** (RTX 8000, `co3`) because mscluster112 had been taken by another user's process (3.7 GB, 24%) in the meantime; log `sdlora_r32_100k.log` | ~24 s per epoch on the Blackwell for the first 5000 steps; 63.9 s per epoch on the RTX 8000 after the resume, ETA ≈ 34 h from 14:15 unless moved back to a Blackwell node by another resume | W&B `runs/ue205g6e`, continuous across the resume | ⏹ stopped by the user 2026-09-04 01:30 at step 36,000 (`lora_step_035000.pt`); loss 0.0017 |
| Checkpoint strips, all three ranks | Builds a measuring tool | 2026-09-03 17:40, `nohup` pid 523820 on mscluster112 device 0 (`co3_bw`), `corrector/superdiff/scripts/sdlora_sample_watcher.py`, log `corrector/superdiff/sample_watcher.log` | 4 renders per pair per checkpoint, ~3 min per pair on the Blackwell; polls every 5 min, exits once every rank's final checkpoint is rendered | The trainer's inline sampler is off (`--sample-every-epochs 0`) because it renders with the PoE sampler, so this stands in: every checkpoint at a multiple of 10k steps and the final one is rendered through SuperDiff at 200 steps, κ 0.5, adapter on every step at λ=1, seeds 9–12, both pairs. Strip columns: Mono (target) = the joint prompt through SuperDiff's integrator (plan 05's λ=1 tile at κ 0.5), SuperDiff κ 0.5 alone (plan 05's tile), LoRA at this checkpoint. Files `sdlora_r{8,16,32}_100k/samples/superdiff/step_XXXXXX_{cat_dog,butterfly_meadow}.png`; logged to W&B against `ckpt_step` in companion runs `sdlora_r{8,16,32}_samples` (same project), since a second process may not write to a training run. Step 10k, rank 8: cat×dog seeds 10, 11, 12 go from one blended animal to two, seed 9 stays one; butterfly×meadow seeds 9 and 12 show a clear butterfly, 10 and 11 a faint one. Step 10k, rank 16: cat×dog all four seeds two animals, washed-out texture; butterfly×meadow the meadow itself smears, only seed 12 keeps a readable butterfly. Step 10k, rank 32: two silhouettes on every cat×dog seed inside a painterly smear; butterfly×meadow smeared on all four. Across rank at this step: separation rises, texture falls, all at λ=1. Step 20k, rank 8: all four cat×dog seeds two animals (seed 9 now separates); butterfly×meadow seed 11 gains a large butterfly, seeds 10 and 12 hazier than at 10k. Step 20k, rank 16: worse than its own 10k, two silhouettes inside a smear on cat×dog, smear on all four butterfly×meadow seeds. The training curves show nothing matching the smear (per-batch loss 0.0008–0.0017 at 15–20k for both ranks, ‖Δ̂‖ below ‖target‖ at every sampled step), so the smear is a λ=1 rendering-strength read, to be settled by the λ sweep in task 3.1. Step 30k, rank 8: the cleanest strip so far, two distinct animals on all four cat×dog seeds with near-photographic texture on seeds 10 and 12, and a clear butterfly on all four butterfly×meadow seeds for the first time. Step 30k, rank 16: still a smear on both pairs at λ=1, unchanged from 20k; a λ sweep (0.25–1, cat×dog, seeds 9–12) of this checkpoint was rendered to check whether the smear is strength alone. Step 20k, rank 32: a full smear on both pairs at λ=1, worse than its own 10k, the same direction as rank 16. Rank 16 step-30k λ sweep (`preview_sdlora/cat_dog_grid_200_steps_kappa_050_sdlora_r16_step30k_preview.png`): seeds 10 and 12 separate at λ 0.5 with texture intact, seeds 9 and 11 only from λ 0.75 and smear there; against the same sweep at step 10k every column is hazier. So the smear is dose in part (λ 0.5 is usable) and learned in part (it worsens with training at fixed λ), which is the case against ranks 16 and 32 the final sweep in task 3.1 has to settle. Step 40k, rank 8: cat×dog two animals on all four seeds, seeds 9 and 12 clean, seed 11 hazier than at 30k; butterfly×meadow regressed from 30k, no recognisable butterfly on seeds 9 and 12, faint on 10 and 11, every tile more washed out than SuperDiff alone. At λ=1 rank 8 is now drifting the way ranks 16 and 32 did earlier, one checkpoint later. Step 40k, rank 16: unchanged from 30k, two painterly blobs on cat×dog seeds 9, 10, 12 and an unreadable smear on 11; butterfly×meadow faint at best on seeds 10 and 12, none on 9 and 11. Step 50k, rank 16: worse again, cat×dog now one indistinct animal or an unreadable smear on every seed (no seed reads as two bodies), butterfly×meadow faint on 10 and 11, none on 9 and 12. Step 50k, rank 8: cat×dog two bodies on seeds 10, 11, 12 and one on seed 9, all painterly rather than photographic now; butterfly×meadow faint on 10, 11, 12, none on 9, hazier than SuperDiff alone. Rank 8 keeps separation at half training but has lost the texture its 30k strip had. Step 30k, rank 32: no cat×dog seed reads as two bodies (one cat-like face on seed 9, smear on the rest); butterfly×meadow faint on 10 and 11, none on 9 and 12. Rank 32 lost separation earliest of the three. Step 60k, rank 16: one animal per cat×dog seed (the adapter no longer adds a second body at λ=1), butterfly×meadow none on 9 and 10, faint on 11 and 12, painterly. Loss curves to step 61k (`corrector/superdiff/sdlora_loss_curves.png`, total and per bucket, log scale, 25-point mean, from W&B): all three fall then plateau from about step 30–40k; rank 8 lowest in every panel (total ≈ 0.0008 at 61k against 0.0025 for rank 16), rank 32 noisiest with a spike near step 13k. Nothing in the curves rises where the strips degrade, so the loss does not see the smear. Step 60k, rank 8: cat×dog two bodies on seeds 10, 11, 12 (seed 9 one animal plus a blur), softer than SuperDiff alone but still readable; butterfly×meadow a clear butterfly on seed 10 only, faint on 11 and 12, none on 9. Timelines of every strip per (rank, pair), rows = checkpoints, columns = seeds: `corrector/superdiff/timelines/{cat_dog,butterfly_meadow}_r{8,16,32}_timeline.png` | ✅ 15 strips (rank 8 and 16 at 10k–60k, rank 32 at 10k–30k), watcher stopped with the trainings; this evidence is what the stop decision rests on |

With the rescaled window the epoch-1 bucket losses read early 0.0009 / commit 0.0082 / late
0.0069 on both A6000 runs, against 0.0000 / 0.0010 / 0.0075 under the 5–24 window on the same
data: the old window was reading a near-empty high-noise sliver as "commit", which is exactly
what the halving rule would have judged.

**All three were then killed by the halving rule at step ~5000, and the deadline was changed
before any verdict was read.** Commit-bucket loss against its initial value: rank 8 0.0053 from
0.0102 by step 5480, rank 16 0.0057 from 0.0102 by step 5010, rank 32 0.0058 from 0.0101 by
step 5155, i.e. a 44–48% fall where the rule demands 50% by step 5000. That rule was calibrated
on the PoE residual at 50-step cells; the note above, written before launch, anticipated slower
learning here (about 4× fewer passes over the data in the same step budget). A loss that fell
46% and was still falling is not the stall the rule exists to catch, so the deadline is scaled
by the same factor, to 20,000 steps (`--kill-halve-after-steps 20000`); the 0.1 threshold rule
and the halving rule itself are kept. This is a bar changed after seeing a number, and is
recorded as such here and in the scope `CHANGELOG.md`. All three resume from their
`lora_step_005000.pt` checkpoints under the same W&B ids, so the curves stay continuous.
| The 96-cell set and six sheets | Tests the claim | not launched | ~7h | `corrector/superdiff/transfer_sdlora/` | ⚠️ not run: the trainings were stopped before their finals on the strength of the checkpoint strips; the six timelines stand in as the figure output |

The finding this review feeds: [does an adapter trained on superdiffs own residual compose](../../../report/is-the-gap-the-samplers-or-the-models/does-an-adapter-trained-on-superdiffs-own-residual-compose.md).

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ⚠️ **Does an adapter trained on SuperDiff's own residual separate the two concepts at a
      lower λ than SuperDiff alone, where the PoE-trained one did not?** Per (pair, rank): the
      first separating λ per seed, beside the baseline sheet's and step 49's. Pass if lower than
      the baseline on more seeds than not for at least one rank; fail if equal at every rank;
      inconclusive for any rank whose training did not converge.
      **Inconclusive by this plan's own criterion, with a clear direction.** No training reached
      its final step and the 96-cell λ sweep was not run, so the first-separating-λ table does
      not exist. What was measured: at λ=1, rendered every 10k steps, every rank does what the
      PoE-trained adapter never did inside SuperDiff, two bodies on three or four cat×dog seeds by
      step 10k–20k (rank 8 at 30k: all four seeds, and a butterfly on all four meadow seeds).
      Then every rank smears with further training at fixed λ, rank 32 first (no separation by
      30k), rank 16 next (none by 50k–60k), rank 8 last (three of four seeds at 60k, softer each
      checkpoint). Rank 16's step-30k λ sweep put λ 0.5 as usable on two of four seeds. The
      adapter does carry SuperDiff's residual, and the early checkpoints separate where SuperDiff
      alone and step 49's adapters do not, but the finished adapter at full budget would not have
      been the best one, and the user stopped the runs on that reading.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ⚠️ What did step 49 say, quoted, and did that open or close this plan? Quoted in the Runs table (task 0.3 row): "Hurts", which opened this plan.
- [x] ⚠️ How big is the cache, and did it fit without thinning steps? 88 cells, 17,600 step files, 11 GB, every one of the 200 steps kept (Runs table).
- [x] ⚠️ Does the target formed from one cached cell match the composer's own `r_t_sd_norms` for
      that render, to fp16 tolerance? If not, the cache is not what the composer computed.
      **Yes, to 3.6×10⁻⁵** at steps 0, 50, 100 and 199 of `a_wolf__x__a_husky` seed 1 (numbers
      in the Runs table). The saved `x_t` is the UNet input SuperDiff actually fed (its latent
      over `sqrt(sigma_t² + 1)`), so the trainer's forward at the cached `timestep` reproduces
      the cached raw predictions rather than approximating them.
- [x] ⚠️ Did each training's loss flatten before its final step? Which W&B run ids?
      **Flattened, though no run reached its final step.** `t5b00h56` (rank 8), `5oz1bsa1`
      (rank 16), `ue205g6e` (rank 32); curves in `corrector/superdiff/sdlora_loss_curves.png`.
      All three fall and plateau from about step 30–40k; at the stop, total loss 0.0008 (rank 8,
      61k), 0.0025 (rank 16, 62k), 0.0017 (rank 32, 36k). The plateau is where the strips get
      worse, so a flat loss here is not evidence the adapter is done.
      Written before launch: the budget is matched to the PoE runs in optimizer steps (2000
      epochs × 50 = 100k, batch 1, lr 1e-4), not in passes over the data. The SuperDiff cache
      has 200 steps per cell against PoE's 50, so 100k optimizer steps is about 5.7 passes over
      its 17,600 examples where the PoE runs made about 22.7 passes over 4,400. If the loss has
      not flattened by 100k, that is the first suspect, and "inconclusive" is the honest reading
      rather than "fail".
- [ ] ⚠️ At λ=1, is ‖Δ̂‖ per step close to ‖r_t^SD‖ per step from the cache for the same cell? An
      adapter that learned the residual should reproduce its size profile. Not measured: the
      per-step comparison was part of the un-run final set. The strips' sidecars under
      `corrector/superdiff/samples_sdlora/` carry `delta_hat_norms` if anyone wants it.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible.

- [ ] ⚠️ Why does the rendered image get worse while the loss keeps falling? Two readings fit
      and nothing here separates them: λ=1 over-applies a correction fit on trajectories the
      adapter never steered (a dose effect, partly supported by λ 0.5 being usable), or the cached
      residual carries high-frequency texture the adapter learns late (a target effect). A
      per-step ‖Δ̂‖ against ‖r_t^SD‖ on the 30k and 60k checkpoints of rank 8 would tell them apart.
- [ ] 🟡 Would an early checkpoint (rank 8 at 30k) at λ 0.5 be worth keeping as the SuperDiff
      adapter, given it separates all four cat×dog seeds and puts a butterfly on all four meadow
      seeds? Only if the paper needs a SuperDiff-side adapter at all.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ⚠️ **Was the comparison fair?** Same pool, same ranks, same alpha, same modules, same
      training length as the PoE adapters; only the cache (and so the rule) differs. Same seeds,
      `kappa`, steps and noise draws as the baseline and step 49 sheets. **Fair on every axis
      but training length**: stopped at 61k/62k/36k of 100k, so this family was judged on
      earlier checkpoints than the PoE family. That cuts against the PoE side, since these
      adapters were at their best early.
- [x] ⚠️ **Was the measuring tool sound?** Detector on cat×dog only; butterfly×meadow eye-read. **Eye-read only, both pairs**: the detector was part of the un-run final set. Every strip read is in the Runs table and can be re-read from the files.
- [x] ⚠️ **Did the run respect the environment?** Cache sized before building; trainings one at a
      time on biggpu; everything under `/datasets`; W&B holds the curves, this file holds ids and
      verdicts. Yes: cache sized first (11 GB), trainings on three separate devices via `nohup`,
      everything under `/datasets`, curves on W&B with the ids above.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the residual is rule-specific, or model-level | both step 49 and this plan's verdicts, on the same sheets |
| a SuperDiff-trained adapter works | the cache's `kappa`, step count and pool, since the adapter is specific to all three |
| this plan was skipped | step 49's verdict, quoted, as the reason |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open. This file has not been run against.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Wait for step 49's verdict. If it opens this plan, size the cache before anything is built.
