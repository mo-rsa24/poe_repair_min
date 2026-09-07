# 🔁 Review: does the PoE-trained correction reach SuperDiff?

**Nothing has run yet.** Every question below was written before any adapter touched SuperDiff.
This file judges [the transfer design](../plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md).
Its answer opens or closes [step 50](08-adapters-that-learn-superdiffs-own-residual.md).

## Recommended prompt (when the run lands)

```
/analyze-run the SuperDiff transfer sheets, three ranks against the no-adapter kappa 0.5 sheets
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/07-does-the-poe-trained-correction-reach-superdiff.md) | the adapter toggle, the whole-run injection, the 96 cells, the six sheets |
| **this file** | **the verdict: not yet run** |
| [step 28's verdict](05-what-changes-when-superdiff-leaves-its-own-defaults.md) | the no-adapter κ=0.5 sheets every sheet here is read against |
| [step 50's verdict](08-adapters-that-learn-superdiffs-own-residual.md) | what this file's answer gates |

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

- **The adapter**: one of the three LoRAs trained to predict `eps_J − eps_PoE` along PoE
  trajectories, rank 8 (450k steps), 16 and 32 (100k), alpha equal to rank.
- **Δ̂**: `eps_M`(adapter on) − `eps_M`(adapter off), SuperDiff's blend with the adapter's
  contribution isolated, at `kappa` 0.5.
- **λ**: the fraction of Δ̂ added on every one of the 200 steps.
- **The baseline sheet**: step 28's `cat_dog_grid_200_steps_kappa_050` and its butterfly twin,
  the same grid with no adapter; its λ=0 column is reused here.
- **Transfers / indistinguishable / hurts**: the three pre-registered outcomes, defined in the
  design's claim.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tests the claim.** The result decides whether step 50 runs. A null (indistinguishable) is a
finding about the correction being rule-specific, not a failed run.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Attach preflight, rank 8 at 100k (not the final checkpoint) on SuperDiff's UNet, CPU, mscluster85 | Builds a measuring tool | 2026-09-03 | seconds | 210 matched modules, 420 LoRA tensors loaded, 420 tensors in the checkpoint | ✅ the module names match; task 1.1 still runs on the three final checkpoints |
| Attach check, three checkpoints on SuperDiff's UNet, mscluster109 device 1 | Tests the claim | 2026-09-03 | seconds | 210 matched modules for rank 8 (`lora_step_410000.pt`), rank 16 (`lora_step_085018.pt`), rank 32 (`lora_step_090050.pt`) | ✅ equal, non-zero. **These are the latest checkpoints at launch, not the final ones**: the trainings were still running (450k, 100k, 100k targets); the plan's partial-pass guidance covers this and the sidecars record the exact checkpoint per cell |
| Inertness check, rank 8 attached, 20 steps, `a_cat__x__a_dog` seed 9, mscluster109 device 1 | Tests the claim | 2026-09-03 | 3 renders | λ=0 render sha `79238174f0b3e244` = no-adapter render sha `79238174f0b3e244`; λ=1 ‖Δ̂‖ per step min 10.0, max 78.3, none zero | ✅ the off pass is the baseline; the adapter does something |
| The 96-cell transfer set | Tests the claim | 2026-09-03, `nohup` pid 1793168 on mscluster109, `CUDA_VISIBLE_DEVICES=1`, log `corrector/superdiff/transfer.log` | 49 cells at ~121s each | `corrector/superdiff/transfer/grid/`; 3 of 6 sheets (rank 8 both pairs, rank 16 cat×dog) | ⏹ stopped by the user at 49 of 96 after the third sheet: all three read hurts on every seed, and the mechanism was already measured (cosine), so the remaining 47 cells could not change the verdict |

The finding this review feeds: [does the poe trained correction carry into superdiff](../../../report/is-the-gap-the-samplers-or-the-models/does-the-poe-trained-correction-carry-into-superdiff.md).

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ⚠️ **Does an adapter trained against PoE separate the two concepts inside SuperDiff at a
      λ where SuperDiff alone does not?** Per (pair, rank): the first λ at which each seed shows
      two separate concepts, beside the same number from the baseline sheet. Transfers if the
      adapter's λ is lower on more seeds than not; indistinguishable if equal; hurts if higher.
      Eye read on both pairs, validated detector on cat×dog only.
      **Hurts.** On every sheet rendered (rank 8 cat×dog, rank 8 butterfly×meadow, rank 16
      cat×dog), no seed reaches two separate concepts at any λ, where the no-adapter sheets
      reach it on all four seeds by λ=0.75 (cat×dog) or 0.25 (butterfly). The first separating
      λ is therefore "never" for the adapter against 0.5–0.75 for the baseline, on 12 of 12
      seeds. Mechanism measured on one cell: the adapter's correction is the right size but
      mostly orthogonal to SuperDiff's missing residual (median per-step cosine peaking at
      +0.34). Ranks 16-butterfly and 32 were not rendered: the run was stopped at 49 of 96 with
      the verdict already fixed. Detector read not taken: the images are too degraded for an
      instance count to mean anything, and the eye read is unambiguous. **Step 50 opens.**

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ⚠️ Did every checkpoint attach with a non-zero, and equal, matched-module count?
      **Yes: 210 for each of rank 8, 16, 32** (see the Runs table). A second rank on the same
      UNet needs `delete_adapters("lora")` first, or diffusers refuses the duplicate name; the
      driver does that per rank.
- [x] ⚠️ Is λ=0 with the adapter attached byte-identical to the baseline render? If not, the
      off pass is not the baseline and every sheet measures the bug.
      **Yes**, same sha256 on a 20-step render of `a_cat__x__a_dog` seed 9 at κ 0.5.
- [x] ⚠️ Is ‖Δ̂‖ non-zero on every step at λ=1, and how does its per-step profile compare with
      ‖r_t^SD‖ from step 28's sidecars? An adapter whose Δ̂ has the wrong size scale cannot be
      expected to transfer, and that would be worth saying before the verdict.
      **Non-zero on all 20 steps, 10.0 to 78.3.** Step 28's `r_t^SD` on the same pair and seed
      ran 8.3 to 17.6 over 50 steps, so Δ̂ is the same order of size and larger early. The
      200-step profiles from the transfer sidecars (`delta_hat_norms` against `r_t_sd_norms`)
      are the real comparison, once the set finishes.
- [x] ⚠️ Did the three trainings finish at their final step, and were the final checkpoints the
      ones used? A mid-run checkpoint changes what "the adapter" means.
      **No: the set was launched on the latest checkpoints while the trainings were still
      running**, rank 8 at 410k of 450k, rank 16 at 85k of 100k, rank 32 at 90k of 100k, by the
      user's call to not wait. Every cell's sidecar records the checkpoint path. If the finals
      change the verdict, the set is re-run on them; the caption says which was used.
- [ ] ⚠️ How long did the first cell take, with two UNet passes on every step?

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible.

- [ ] 🟡 The first four cells (cat×dog seed 9, rank 8 at 410k, λ = 0.25 to 1) degrade from
      λ=0.25 and wash into low-contrast mush from λ=0.5, a faint animal ghost at λ=1. That is
      the pre-registered "hurts" shape on one seed and one rank; the sheet decides, not this
      strip. Read 2026-09-03 while the set was 4 of 96 in.
- [ ] 🟡 Is it amplitude or direction? Amplitude is ruled out: per step, median ‖Δ̂‖ against
      median ‖r_t^SD‖ on the same cell runs 10.9 vs 11.5 (steps 0–19), 19.5 vs 46.0 (20–99),
      27.5 vs 85.1 (100–179), 49.7 vs 47.5 (180–199). The adapter's correction is the right
      size or smaller throughout. What is left is direction: a residual learned along PoE's
      50-step DDIM trajectory pointing the wrong way on SuperDiff's 200-step stochastic one, and
      a right-sized wrong-direction nudge on every step accumulating. The number that would
      settle it is the per-step cosine between Δ̂ and `r_t^SD`. The loop now records it
      (`delta_hat_cos_r_t` in the sidecar, when the render is run with both the adapter and
      `form_r_t=True`; import-checked, not yet run). One cell on a free GPU, `a_cat__x__a_dog`
      seed 9, rank 8, λ=1, is the check: a cosine near zero or negative through the middle of
      the run is "wrong direction"; near one is "right direction, and the failure is elsewhere".
      **Run (mscluster109 device 0, `lora_step_420000.pt`, the latest at that moment; the sheet
      used 410000): median cosine +0.10 over steps 0–19, +0.34 over 20–99, +0.18 over 100–179,
      +0.06 over 180–199; per-step minimum −0.40, maximum +0.66.** Mostly orthogonal, weakly
      aligned at best in the middle of the run, never close to one. Sidecar:
      `corrector/superdiff/transfer_diag/cosine/.../superdiff_200steps_kappa0.50_with_rt_lorar8_lv1.00.json`,
      field `delta_hat_cos_r_t`. So the rank-8 failure is direction, not size: the PoE-trained
      residual and SuperDiff's missing residual are different vectors at the same state.
- [ ] 🟡 **First sheet, cat×dog at rank 8 (`cat_dog_grid_200_steps_kappa_050_lora_r8`, eye
      read): hurts, on all four seeds.** Every seed is degraded at λ=0.25 (a noisy chimera) and
      washed into low-contrast mush from λ=0.5, with at most a faint animal ghost at λ=1. No
      seed reaches two separate concepts at any λ, where the no-adapter sheet reaches it on all
      four by λ=0.75. Per-rank verdict recorded here; the plan's verdict waits for ranks 16 and
      32 and the butterfly pair.
- [ ] 🟡 **Second sheet, butterfly×meadow at rank 8
      (`butterfly_meadow_grid_200_steps_kappa_050_lora_r8`, eye read): hurts, on all four
      seeds.** Same shape as cat×dog: degraded at λ=0.25, washed out from λ=0.5, no seed's
      butterfly ever sharper or larger than the no-adapter sheet already had it at that λ. So
      rank 8 hurts on the easy pair as well as the hard one; ranks 16 and 32 remain.
- [ ] 🟡 **Third sheet, cat×dog at rank 16 (`cat_dog_grid_200_steps_kappa_050_lora_r16`,
      checkpoint `lora_step_085018.pt`, eye read): hurts, on all four seeds, a shade worse than
      rank 8.** Noisy chimera at λ=0.25, heavy wash-out already at λ=0.5, near-uniform mush at
      λ=1. No seed reaches two separate animals at any λ. Doubling the rank did not change the
      direction of the correction, only, if anything, how much of it there is to accumulate.
- [ ] 🟡 Note for the interpretation either way: SuperDiff's own missing residual is large
      mid-run (‖r_t^SD‖ 46 to 85 over steps 20–179 against ‖eps_M‖ ≈ 230–260), so most of what
      it lacks sits in the middle of the run, where the PoE window never injected.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Between a transfer tile and its baseline tile only the adapter
      differs: same seed, same `kappa` 0.5, same 200 steps, same guidance, same prompts, same
      noise draws (the generator is reseeded identically).
- [ ] ⚠️ **Was the measuring tool sound?** The validated detector covers cat×dog only; the
      butterfly×meadow sheets are eye-read, as step 28 recorded.
- [x] ⚠️ **Did the run respect the environment?** One render process on the shared GPU, everything
      under `/datasets`, checkpoints read from their run folders and never copied.
      **Yes, on mscluster109 device 1** (an RTX A6000 at 784 MiB used, 0% before launch),
      chosen by reading `nvidia-smi` per device over SSH rather than trusting `sinfo`: the three
      trainings run through the SSH `nohup` path, so Slurm shows their nodes as idle, and
      mscluster111 reads `[GPU requires reset]`, the hardware fault `poe-launch-002` describes.
      Two things learned for the environment notes: `/tmp` is node-local, so a script in this
      session's scratchpad is invisible from another node (the scripts now live under
      `corrector/superdiff/scripts/` on `/datasets`); and the transfer log is read over SSH on
      the launch node, since NFS on the session node can lag behind it.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the correction transfers, or does not | that it was tested on two pairs at four seeds, three ranks, one `kappa`, one step count, injected over the whole run |
| any rank ordering | that rank order is reported, not claimed |
| the butterfly×meadow row | that it is eye-read only |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

Nothing open. This file has not been run against.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Wait for the three final checkpoints, attach and print the counts, prove λ=0 inert, then launch
the 96 cells.
