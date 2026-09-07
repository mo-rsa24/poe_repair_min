# 🎯 Review: noise search on top of the adapter

This file judges [the design](../plans/baselines/12-noise-search-on-top-of-the-adapter.md): eight
perturbations of each held-out seed's starting noise, rendered with the rank-32 step-30050
adapter at λ 1.2, the best kept by compose count and then sharpness. All three runs are done and the
verdict is written below; the bars were written before the first launch.

## Recommended prompt (when the run lands)

```
/analyze-run the two noise-search-adapter runs in prime_lab/poe-repair-animals-compose, group noise-search-adapter: the sheets, compose/*, sharpness/* and secondary/*
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/12-noise-search-on-top-of-the-adapter.md) | the adapter sampler, the verifier, the sheet, the bars |
| **this file** | **the verdict** |
| [plan 11's verdict](11-zero-order-search-over-the-initial-noise.md) | the same search on plain PoE |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Choices made without an answer in the room](#choices-made-without-an-answer-in-the-room)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The adapter pivot**: the seed's cached noise rendered with the correction at λ 1.2 on every
  step; the unperturbed corrected render and the baseline.
- **The plain pivot** and **Mono**: the same noise with plain PoE and with the joint prompt, both
  rendered before the adapter is attached; the two sharpness ceilings.
- **Sharpness**: variance of the second differences of the greyscale 1024 px render (plan 07's
  Laplacian variance). Compared only per seed, as ratios.
- **The kept image**: of eight candidates rendered with the adapter, the one with the highest
  compose count (clipped at 2), then the highest sharpness, then the lowest index.
- **The sharpness ratio**: kept over adapter pivot, per seed; the verdict reads its median over
  the 8 seeds.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Establishes a baseline.** A null says the corrected render's softness is not a property of the
starting noise and hands the fidelity question to the schedule and tail experiments; a break says
a sharpness verifier costs animals.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Smoke, `GPU=1 bash scripts/noise_search/run_noise_search.sh adapter-smoke` over SSH on mscluster107 device 1 (Quadro RTX 8000, `co3`), bash PID 56620, python PID 56767 | Builds a measuring tool | 2026-09-06 02:38 | cat × dog seed 9, N 2, σ 0.3, 10 steps, 512² | `outputs/interaction_term/noise_search/smoke_adapter_N2_s10_20260906-023829/`: `mono_eta0.png`, `poe_pivot.png`, `detach_check.json` (0.0 grey levels) with its PNG, `adapter_pivot_lambda1.2.png`, `sigma_0.3/c0.png`, `c1.png`, `run.json`, the one-row sheet with its sidecar, `summary.json`, `verdict.json`; the log printed `LoRA rank=32 attached: n_matched=210 n_loaded=420 checkpoint_step=30050` and `torch_sees=Quadro RTX 8000`; 18 s for the seed after the load. Images are noise at 10 steps; the verdict it wrote is plumbing. Log `outputs/interaction_term/noise_search/logs/adapter_smoke.log` | ✅ done |
| Full, cat × dog, `GPU=1 bash scripts/noise_search/run_noise_search.sh adapter-full --pairs a_cat__x__a_dog` over SSH on mscluster107 device 1 (Quadro RTX 8000, `co3`), bash PID 57820, python PID 57966, launch commit `7916d92` with a dirty tree | Establishes a baseline | 2026-09-06 02:42 | 8 seeds × (Mono + plain pivot + adapter pivot + 2 σ × 8 candidates with two UNet passes per step), 50 steps, 1024²; about 4 h | run dir `outputs/interaction_term/noise_search/zo_adapter_N8_s50_20260906-024225/`, log `.../logs/adapter_full_a_cat__x__a_dog.log`, W&B `prime_lab/poe-repair-animals-compose/runs/kqj26oiz` (group `noise-search-adapter`) | ✅ done 04:31, 1 h 49 min, about 13 min per seed; 8 seed folders each with the three references, `adapter_pivot_lambda1.2.png` and two `sigma_*` folders of 8 PNGs plus `run.json`; the sheet, `detach_check.json`, `summary.json`, `verdict.json`. Both-ness failed on the launch node (the xformers CPU kernel, as in plan 11) and was rerun 04:33 on mscluster85 with `XFORMERS_DISABLED=1`. Filed as `artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-adapter-*` |
| Floor check, `GPU=1 bash /datasets/.../noise_search/floor/run_floor.sh` on mscluster109 device 1, PID 1867250 | Builds a measuring tool | 2026-09-06 03:55 | plain PoE seed 9 twice in one process, no adapter, 50 steps, 1024² | `outputs/interaction_term/noise_search/floor/mscluster109/floor.json`: 0.0 grey levels between the two | ✅ done |
| Full, butterfly × meadow, the same launch line with `--pairs a_butterfly__x__a_flower_meadow` on mscluster109 device 1 (RTX A6000, `co3`; 786 MiB of another user's idle allocation on the card, 0% utilisation, allowed by the shared-device rule), bash PID 1860284, python PID 1860430 | control | 2026-09-06 02:43 | the same on the control pair | run dir `outputs/interaction_term/noise_search/zo_adapter_N8_s50_20260906-024309/`, log `.../logs/adapter_full_a_butterfly__x__a_flower_meadow.log`, W&B `6oai0d99` | ✅ done 03:52, 1 h 9 min, about 8 min per seed; the same files; no verdict because the judged pair is absent |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ✅ **With the adapter attached, does the best of eight nearby starting noises give a
      two-animal render sharper than the cached noise's, on cat × dog seeds 9 to 16, without
      losing the compose rate?** **Support on the bar, half the seeds by eye.** `verdict.json`:
      the adapter pivot composes on 6 of 8 (0.75; seeds 11 and 14 count 1 through this sampler),
      the kept image on 8 of 8 at both σ; median sharpness ratio kept over adapter pivot 1.47 at
      σ 0.1 and 2.88 at σ 0.3, both above the 1.10 gain with compose held. Opened tile by tile
      against the adapter pivot (the eye read, 04:35): a cleaner cat and dog on seeds 10, 12, 14
      and 16 at both σ, about equal on 13 and 15, worse on 9 (the clean photo replaced by drawn
      cats with no dog) and 11 (a busier sketch with more heads). Seed 14, where the adapter
      fails, gets two animals at both σ. So 4 of 8 seeds gain, 2 hold, 2 lose, and the losses are
      the sharpness key rewarding drawn texture, which the median ratio cannot see. Read from `verdict.json`. Support if at either σ the kept
      compose rate is within one seed of the adapter pivot's (`MAX_COMPOSE_DROP` 0.125) and the
      median over seeds of kept-over-adapter-pivot sharpness is at least 1.10
      (`MIN_SHARPNESS_GAIN` 0.10). Null if compose holds at both σ and the median ratio is within
      0.05 of 1 at both (`NULL_SHARPNESS_BAND`). Breaks if the kept rate falls more than one seed
      below the adapter pivot's at a σ. Inconclusive otherwise. Constants in
      `poe_repair/experiments/noise_search/adapter.py`. Fixed: the rank-32 step-30050 checkpoint
      at λ 1.2 on every step, N 8, one round, σ 0.1 and 0.3, DDIM `eta` 0, 50 steps, guidance 7.5,
      1024², the same candidate draws as plan 11. The adapter pivot is expected to compose on 7 of
      8 (seed 14 fails), so support needs at least 6 of 8 kept images composing.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ Did the smoke run produce the Mono and plain references, `detach_check.json`, the adapter
      pivot, two candidates with counts and sharpness, a one-row sheet, `summary.json` and
      `verdict.json`, and did the log print the attach line and the device? **Yes, all of it**
      (Runs table, PID 56767).
- [x] 🟡 Is the adapter inert when disabled (`detach_check.json`,
      `mean_abs_diff_disabled_vs_before_attach`, 0 to 255 scale)? Expected zero or a few grey
      levels. Anything larger means attaching changed the UNet and the references cannot be
      trusted. Smoke, 10 steps at 512²: **0.0**. Butterfly × meadow full run, 50 steps at 1024²,
      seed 9: **6.0**, read while the run was in flight. The references are clean regardless,
      because they were rendered before the attach; what 6.0 leaves open is whether the disabled
      adapter leaks or the sampler is nondeterministic run to run and 50 steps amplify it. The
      floor is measured at harvest: plain PoE seed 9 rendered twice in one process with no adapter
      attached, on the same device, and its difference read beside the 6.0. The cat × dog run's
      own check on mscluster107 (Quadro RTX 8000), the same code at the same 50 steps and 1024²:
      **0.0**. The floor on mscluster109 (`outputs/interaction_term/noise_search/floor/mscluster109/floor.json`,
      03:57, PID 1867250): plain PoE seed 9 rendered twice in one process with no adapter,
      **0.0**, so that card is deterministic run to run and the 6.0 is not noise. Opened, the two
      A6000 renders are the same meadow, flowers and butterfly placement with a different wing
      pattern and pose, a late divergence of the kind plan 11's seed 12 showed between two
      samplers. The reading that fits: attaching the adapter changes the card's numerics slightly
      (a different kernel choice once the LoRA weights sit in memory), which is exact on the
      Quadro RTX 8000 and a few grey levels on the A6000 after 50 steps. Consequences: the
      references are clean because they were rendered before the attach; every adapter-on render
      in a run goes through the attached UNet on both branches, so within a run the comparison is
      consistent; and "inert when disabled" is exact on one card and within-scene on the other.
      Marked 🟡 rather than ✅ because the cause is inferred from where the 0.0s and the 6.0 fall,
      not tested by re-attaching and disabling in the floor process.
- [x] ✅ Is the adapter pivot the same scene as the existing λ 1.2 render of the same seed
      (`adapter_pivot_mean_abs_diff_vs_r32` per seed)? Plan 11 saw up to 14.5 grey levels between
      the two DDIM code paths with the same scene; the layout match is the read. **Per seed 9 to
      16: 4.8, 9.5, 3.4, 3.7, 2.0, 11.9, 7.7, 3.2 grey levels.** On the sheet every adapter pivot
      is the scene the λ 1.2 column of plan 11's sheet shows; the count differs on seeds 11 and 14
      (1 here against 2 and 1 there), which is the late-divergence effect on a sketch seed and the
      failing seed.
- [x] 🟡 How far below the plain pivot and Mono does the adapter pivot sit on sharpness, per seed
      (`secondary.median_ratio_plain_over_adapter` and `median_ratio_mono_over_adapter`)? This is
      the room the search has; a ratio near 1 means there is no softness to recover on that
      measure. **Median plain over adapter 0.81, Mono over adapter 1.16.** On this measure the
      corrected render is not softer than plain PoE on the median seed; it is on seeds 11, 14 and
      15 (sketches, where plain has more line work) and not on the photo seeds 9, 10, 12 and 16.
      So Laplacian variance does not read the softness the corrected photos show to the eye, and
      the room it reports is edge count rather than clarity.
- [x] 🟡 On how many seeds does the kept image reach the plain pivot's sharpness, and Mono's
      (`sigma_<σ>_kept_reaches_plain_band`, `_kept_reaches_mono_band`)? The claim in words is
      "snaps into a clean cat and dog": reaching Mono's sharpness with a count of 2 is that. **By
      the number: plain band 5 of 8 at both σ, Mono band 5 of 8 at σ 0.1 and 6 of 8 at σ 0.3. By
      eye: 4 of 8** (seeds 10, 12, 14, 16), and on those four the kept tile is a clean cat and
      dog crisper than the pivot's. The number overcounts because seed 9's drawn cats and seed
      11's busy sketch clear the band on edges alone.
- [x] ✅ Does the search find a composing candidate on seed 14, where the adapter pivot fails?
      **Yes at both σ**: 7 of 8 candidates count 2 or more at σ 0.1 and 5 of 8 at σ 0.3, and the
      kept tiles show a cat and a dog (under a bicycle at σ 0.1, beside a cartoon figure at σ 0.3)
      where the pivot shows a child with one dog.
- [x] ✅ Does σ 0.3 give a larger median sharpness ratio than σ 0.1? The prediction is yes, more
      spread in the candidates; the risk is that the larger stride also loses animals, which the
      compose read catches. **Yes, 2.88 against 1.47, and no animals lost: 8 of 8 at both.** The
      candidate compose fraction is 0.94 at σ 0.1 and 0.89 at σ 0.3, so nearly every nudge of the
      corrected run still composes.
- [x] ✅ How does the kept image under plan 11's rule (count then detector confidence) compare
      (`sigma_<σ>_kept_by_confidence_compose_rate`)? If the sharpness key changes nothing, the
      picks are driven by the count alone. **Also 8 of 8 at both σ**, so the compose gain over the
      pivot comes from having eight candidates, and the sharpness key only decides which composing
      candidate is shown.
- [x] 🟡 Both-ness of the kept tiles against the adapter pivot's (`bothness.json`): a sharper
      kept tile that drops toward the PoE band (about 0.2) has traded the second animal for
      edges; the eye read confirms. **Means: adapter pivot 0.415, kept 0.391 at σ 0.1 and 0.375
      at σ 0.3, Mono 0.523, plain 0.200.** The kept means sit a little under the pivot's, and the
      drops are where the eye says: seed 9 falls from 0.46 to 0.29 and 0.17 (the drawn cats), seed
      14 stays at 0.29 and 0.21 (a cat and dog, small in a busy scene). The four seeds the eye
      calls cleaner sit at 0.38 to 0.47, the corrected band.
- [x] ❌ Butterfly × meadow, by eye: does every kept tile keep a butterfly over flowers? The count
      is not this pair's read. **The correction itself is what changes this pair.** Plain PoE from
      the cached noise keeps a photographic butterfly over flowers on all 8 seeds. The adapter
      pivot at λ 1.2 turns every seed into an illustration and loses the butterfly on seeds 12
      (faint), 13 (an empty painted valley), 15 and 16 (a small butterfly among painted flowers);
      the detector counts 0 on 12, 13, 15 and 16. The kept tiles keep a butterfly on all 8 seeds
      at both σ, always as several crisp illustrated butterflies (counts 2 and 3), and seed 14 at
      σ 0.3 is a repeating botanical print. Mean sharpness: Mono 65, plain 69, adapter pivot 503,
      kept higher still, so on this pair the edge measure is reading illustration line work, not
      clarity. Read from the sheet
      `zo_adapter_N8_s50_20260906-024309/noise_search_adapter_a_butterfly__x__a_flower_meadow_sheet.png`
      on 2026-09-06 03:55, before the judged run finished.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold.**

- [x] ❌ Seed 9 of cat × dog, read while the run was in flight (2026-09-06 03:05): the adapter
      pivot is a clean cat and dog side by side (count 2, sharpness 43), and the σ 0.1 pick that
      the verifier preferred (count 2, sharpness 132, three times the pivot's and above Mono's
      75) is a drawn, fur-textured large cat holding a kitten: no dog, and the extra "sharpness"
      is hatching and fur. Of the eight candidates, six sit at 34 to 58 like the pivot and two at
      120 and 132; the verifier picked the outlier style. So the sharpness key rewards edge
      texture over the clean two-animal image, the failure the words section warned of, and a
      support on the median ratio can be a loss by eye. Every kept tile is opened at harvest and
      the count of "cleaner cat and dog by eye" is reported beside the ratio (raised by the seed 9
      numbers and tiles).

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** The adapter pivot and every kept tile are rendered by the
      same batched adapter sampler at the same λ, steps, guidance, resolution and `eta`; only the
      starting noise differs, at mean cosine 0.995 (σ 0.1) and 0.958 (σ 0.3). The plain pivot and
      Mono were rendered before the attach and never entered the verdict.
- [x] ❌ **Was the measuring tool sound?** Sharpness counts edges, and even compared per seed it
      preferred drawn texture over a clean photo on seed 9 and a busier sketch on seed 11, and it
      does not read the corrected photos' softness at all (plain over adapter 0.81). The count did
      its job as a veto. The eye read on every kept tile is the measurement this plan reports; the
      ratio is the number that triggered the bar.
- [x] ✅ **Did the run respect the environment?** Outputs under
      `/datasets/.../noise_search/`, `co3` on both nodes, `torch_sees` printed for both cards, the
      references rendered before the attach, nodes, devices and PIDs in the Runs table; 13 and
      8 min per seed, so nothing fell back to the CPU; the only fallback was both-ness, rerun.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#choices-made-without-an-answer-in-the-room) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a picked noise gives a crisper two-animal render, or does not | the median per-seed sharpness ratio with its measure named, the compose rates, N, σ, the seed count |
| the search reaches the plain-PoE or Mono band | that the band is per seed, that sharpness counts edges, and the eye read |
| the method is Ma et al.'s zero-order search | plan 11's caveats: the perturbation is this project's, one round is the smallest version |
| the search composes 8 of 8 where the adapter alone composes 6 of 8 | that plan 11's count-then-confidence rule reaches the same 8 of 8, so the gain is eight candidates and not the sharpness key; and that two of the eight kept tiles are worse by eye |
| the kept renders are crisper | that Laplacian variance counts edges, that it prefers drawn styles (seed 9) and busy sketches (seed 11), and that it does not read the corrected photos' softness (median plain over adapter 0.81); the eye count of 4 of 8 cleaner goes beside the ratio |
| the control pair was not harmed by the search | that the adapter itself turns the pair into illustration and loses the butterfly on 4 of 8 seeds by count, and that the search returns a butterfly on all 8 as illustrated multiples |
| the adapter is inert when disabled | exact on the Quadro RTX 8000, six grey levels within the same scene on the A6000 with a 0.0 floor, cause inferred (`noise-search-adapter-detach-checks.json`) |

## Choices made without an answer in the room

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

**The verifier is count then sharpness**, because the count alone is saturated at 7 of 8 with the
adapter and would pick a random candidate; the count stays first so a sharper one-animal render
can never win.

**Sharpness is plan 07's Laplacian variance, compared per seed as a ratio**, because that measure
counts edges and the sketch seeds dominate any band across seeds.

**The bars are a 10% median gain for support and a 5% band for null**, chosen before the run as
the smallest gain a reader could see on a sheet and the drift two renders of the same noise show
between samplers.

**The two pairs run on two devices**, one run dir each, because the adapter doubles the cost.

**The same candidate draws as plan 11**, so the two plans see the same neighbourhood.

## Still open

Navigation: ⬅️ [Choices made without an answer in the room](#choices-made-without-an-answer-in-the-room) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [ ] A verifier that reads clarity rather than edges (a DINOv2 distance to the seed's Mono
      render, or a no-reference sharpness that ignores line work) would make the pick honest on
      seeds 9 and 11; the candidates are on disk, so re-picking costs no render.
- [ ] The both-ness read is in the repo copy and not in W&B run `kqj26oiz`.
- [ ] Why attaching the adapter shifts the A6000's plain output by six grey levels when the
      Quadro RTX 8000's does not; untested beyond the floor.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

The finding is at
[does searching over the noise sharpen the corrected render](../../../report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-noise-sharpen-the-corrected-render.md).
For the showcase: on seeds 10, 12, 14 and 16 the kept tiles are the pictures to show, and on the
rest the adapter pivot stays. The candidates for every seed are on disk, so a better verifier can
re-pick without rendering.
