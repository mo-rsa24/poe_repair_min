# 🎲 Review: Feynman-Kac steering on top of the rank-32 correction

This file judges [the design](../plans/baselines/11-feynman-kac-steering-on-top-of-the-rank-32-correction.md):
plan 10's particle sampler with the rank-32 correction in the proposal at λ 0.5 and λ 1.2, read
against the corrected sampler left alone. The bars below were written before the first launch.

## Recommended prompt (when the run lands)

```
/analyze-run the fka run in prime_lab/poe-repair-animals-compose, group fk-steering: the three sheets, compose/*_lam* and agreement/*_lam*
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/11-feynman-kac-steering-on-top-of-the-rank-32-correction.md) | the mixed forward, the two doses, the sheet, the bars |
| **this file** | **the verdict** |
| [plan 10's verdict](10-feynman-kac-steering-on-a-detector-reward.md) | the same steering over plain PoE: null |

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

- **The correction at λ**: `ε_frozen + λ · (ε_lora − ε_frozen)`, plain PoE plus λ times what the
  rank-32 adapter (step 30050) adds to it.
- **Adapter alone**: one particle of that sampler at `eta` 0 from the pinned noise, the showcase
  render at that λ.
- **The control**: K 16 particles of the corrected sampler at `eta` 1.0, never weighted; particle 0
  starts from the pinned noise.
- **Compose rate**: share of the 8 seeds whose finished image the detector counts two or more
  animals in; the picked particle for FK, particle 0 for the control.
- **Headroom**: one minus the adapter-alone compose rate at that λ.
- **The fidelity tie-breaker**: ImageReward (Xu et al. 2023, the human-preference reward the FK
  paper steers SDXL on), scored against the joint prompt "a cat and a dog". The reward is
  `min(count, 2) + 0.5 · sigmoid(ImageReward)`, so the count decides the level and fidelity orders
  particles within a level; it can never lift a one-animal particle over a two-animal one. The
  weight is `FIDELITY_WEIGHT` in `steer.py`. The compose rate the bars judge is the count alone.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Establishes a baseline.** A null says selection adds nothing to the correction at either dose;
a reward blind at step 10 is inconclusive, not a null.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Adapter smoke without the fidelity term, `run_fk_steering.sh full --smoke --adapter-checkpoint <r32 step 30050> --adapter-lambdas 0.5` over SSH on mscluster109 device 1 (`co3`), PID 1859746 | Builds a measuring tool | 2026-09-06 02:40 | 19 s: cat × dog seed 9, K 2, 10 steps, 512² | `outputs/interaction_term/fk_steering/smoke_fka_K2_s10_20260906-024051/`: `adapter.json` (420 tensors, checkpoint step 30050), `adapter_eta0_lam0.5.png`, `ctrl_K2_lam0.5/`, `fk_K2_lam0.5/`, the one-row sheet, `verdict.json` | ✅ the adapter path runs |
| ImageReward install and check, session node CPU | Builds a measuring tool | 2026-09-06 02:50 | `pip install image-reward` then `pip install --no-deps openai-clip` into `co3`; torch 2.5.1, transformers 4.44.1, diffusers 0.29.2 unchanged; checkpoint load 82 s on CPU | scores against "a cat and a dog" on plan 10's seed 9: Mono +0.88, plain PoE −1.03, the steered fused particle −2.08, its step-10 x0-hat −1.50 | ✅ orders clean above blended |
| Adapter smoke with the fidelity term, `run_fk_steering.sh adapter --smoke --adapter-lambdas 0.5`, first over SSH on mscluster109 device 1 (refused by the device guard: another session had taken the card, 16.4 GB at 100%), then as Slurm jobs on `bigbatch` RTX 3090 nodes via `scripts/fk_steering/fk_steering.sbatch`: jobs 50307 (mscluster44), 50309 (mscluster45) and 50321 (mscluster65) failed in one second because those idle nodes' GPUs are faulted (`Unable to determine the device handle for GPU0`, now the third form of `poe-launch-002`); job 50312 on mscluster52 passed at 02:53 (so did 50313, 50314, 50317 on mscluster53, 54, 60); jobs 50311, 50315, 50316 collided on one run directory because they started in the same second and the run id was the timestamp alone (the run id now carries host and pid) | Builds a measuring tool | 2026-09-06 02:51 to 02:57 | 1 min per node: cat × dog seed 9, K 2, 10 steps, 512², adapter at λ 0.5, ImageReward on | `outputs/interaction_term/fk_steering/smoke_fka_K2_s10_20260906-025146/` (mscluster52), logs `logs/fk_steer-503{07..30}.out` | ✅ the adapter path with the fidelity term runs; the sheet carries an `IR` score on every tile |
| Full, Slurm job 50332 on mscluster52 (`bigbatch`, RTX 3090, `co3`), `sbatch --nodelist=mscluster52 scripts/fk_steering/fk_steering.sbatch adapter`, launch commit `39f1964` plus uncommitted edits | Establishes a baseline | 2026-09-06 03:24 | 24 cell-λ × (adapter reference + K 16 control + K 16 steered) with ImageReward reads at every x0-hat and final; about 7.5 h on an A6000, likely 10 h on the 3090 | run dir `outputs/interaction_term/fk_steering/fka_K16_s50_20260906-025826/` (the node's clock reads 02:58 where the session node reads 03:24), PID 737254 on mscluster52 device 0, log `logs/fk_steer-50332.out`, W&B `prime_lab/poe-repair-animals-compose/runs/dfrceppl` (group `fk-steering`); `adapter.json` 420 tensors, checkpoint step 30050 | ◑ running |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ **With the rank-32 correction at λ 0.5 in the proposal, does resampling on the compose
      scorer's read of x0-hat raise the cat × dog compose rate over the corrected sampler's
      particle 0?** Read from `verdict.json`. Support if FK at K 16 composes at least 0.25 more
      often than the control at λ 0.5 (`PASS_MARGIN`). Null if FK is within 0.10 of the control at
      both λ 0.5 and λ 1.2 (`NULL_MARGIN`, over `ADAPTER_LAMBDAS`). Inconclusive if the step-10
      reward disagrees with the final verdict on more than 0.50 of the final particles at λ 0.5
      (`MAX_REWARD_DISAGREEMENT`, `REWARD_BLIND_STEP`). Judged at `ADAPTER_JUDGED_LAMBDA` 0.5,
      K `ADAPTER_K` 16, adapter `ADAPTER_CHECKPOINT` rank `ADAPTER_RANK` 32, all in
      `poe_repair/experiments/fk_steering/steer.py`. The potential, λ 10, the five resample steps
      and `eta` 1.0 are plan 10's, unchanged. The reward carries the ImageReward tie-breaker
      (`FIDELITY_WEIGHT` 0.5) so the picked two-animal particle is the cleanest one; the compose
      rate is judged on the count and the fidelity is read beside it, never in the bar.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ⚠️ Did the smoke attach 420 tensors from checkpoint step 30050 and produce the adapter-alone
      reference, a control, a steered run, a sheet and `verdict.json`? Yes, on mscluster109
      without the fidelity term and on mscluster52 with it (Runs table).
- [ ] ⚠️ What is the adapter-alone compose rate at λ 0.5 and at λ 1.2 on cat × dog? The showcase
      grids put λ 1 at 7 of 8; λ 0.5 is expected part way. Under 5 of 8 at λ 1.2 means the adapter
      did not attach as the showcase had it, and the run stops.
- [ ] ⚠️ Is FK at K 16 above best of 16 at λ 0.5? Equal means picking, not resampling, did the work.
- [ ] ⚠️ Does the control's particle 0 (`eta` 1) compose at about the adapter-alone (`eta` 0)
      rate at each λ? A large gap says the stochastic sampler itself changes the adapter's rate,
      which the caption must carry.
- [ ] ⚠️ Does agreement between the step-10 reward and the final verdict stay above 0.5 at λ 0.5?
- [ ] ⚠️ How many of the five resamples were informative per run at each λ? At λ 1.2 most
      particles should count 2 early, so resampling should rarely separate them.
- [ ] ⚠️ Does the butterfly × meadow sheet keep a clear butterfly over a meadow in every FK tile?
      Its counts are numbers of butterflies, not composition (plan 10's read).
- [ ] ⚠️ Is the picked FK particle's ImageReward above the adapter-alone tile's and the control
      particle 0's, on average over the 8 seeds at each λ? That is what the tie-breaker is for; if
      it is not, the reward read on x0-hat did not carry to the finished image.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible.

- [ ] (none yet)

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Same K, `eta`, noise draws, steps, guidance, resolution,
      adapter and λ between the control and the FK run; only the weighting differs. The `eta` 0
      adapter-alone column is a reference for the eye, never compared numerically to `eta` 1.
- [ ] ⚠️ **Was the measuring tool sound?** The detector scores every final particle and x0-hat
      read; the count is drawn on every tile. Open every FK tile that counts 2 where the control
      counts 1 and record whether it is two bodies or a collar tag.
- [ ] ⚠️ **Did the run respect the environment?** Under `/datasets`, python by node,
      `torch.cuda.is_available()` true on the pinned device, no other process of ours on it,
      node, device and PID here; `adapter.json` with 420 tensors and step 30050.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| selection on top of the correction adds, or does not | the adapter-alone rate at each λ beside the FK rate, so the λ 1.2 gap is read against its headroom |
| the guide is the judge | the best-of-16 column, as in plan 10 |
| this is a combination result | that it neither changes plan 10's null nor sizes the sampler's share |
| the control pair | that the detector counts butterflies there, so its rates are not composition |
| the picked image is cleaner | that ImageReward was in the reward as a tie-breaker, its weight, and the mean score per column |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| Whether half dose plus selection reaches full dose | the full run | the sentence in the paper's corrector section |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Run the smoke, then the full run; fill the Runs table; answer the question above from
`verdict.json`.
