# 🖼️ The same rule on four models: Figure 2 as a grid

This plan redraws Figure 2 of the manuscript as a grid of three prompt pairs by four Stable
Diffusion models. Each cell shows the joint prompt on the left and uncorrected product-of-experts
composition on the right, all at seed 42. The figure then shows the same composition rule on
several models, where the current one shows two pairs on SDXL only.

## Recommended prompt (after this plan completes)

```
/drip-proofread resume artifacts/drips/proofread-section-3/PROOF_MAP.md at chunk 11, with the new Figure 2 at paper/overleaf-iclr/figures/same-rule-on-four-models.pdf
```

## Recommended skill

▶ `/run-experiment` ✅ for the launch; `/drip-proofread` ✅ to walk the new caption.

## Position in the plan tree

**No step number: nothing in the running order waits on this.** The section 3 proofread walk waits
on it at chunk 11 (Figure 2's caption). Registering it in the `## Running order` table of the
[repo root MASTER_PLAN.md](../../../../MASTER_PLAN.md) is task 0.2.

Design only. What the renders showed, cell by cell, lives in
[the paired review file](../../review/12-the-same-rule-on-four-models.md).

## Table of contents

- [What this asks, in one line](#what-this-asks-in-one-line)
- [Words this plan uses](#words-this-plan-uses)
- [The figure](#the-figure)
- [What we expect before rendering](#what-we-expect-before-rendering)
- [Considerations](#considerations)
- [Environment Facts This Plan Depends On](#environment-facts-this-plan-depends-on)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [The check before moving on](#the-check-before-moving-on)
- [Figure Catalog](#figure-catalog)
- [Code references](#code-references)
- [Next step](#next-step)

## What this asks, in one line

⬅️ [Top](#-the-same-rule-on-four-models-figure-2-as-a-grid) | 📋 [TOC](#table-of-contents) | [Next](#words-this-plan-uses) ➡️

Does the pattern in Figure 2 hold on four Stable Diffusion models? The pattern is that the same
composition rule gives both concepts on one pair and a hybrid on another.

## Words this plan uses

⬅️ [Previous](#what-this-asks-in-one-line) | 📋 [TOC](#table-of-contents) | [Next](#the-figure) ➡️

**Joint prompt**

One string naming both concepts, "{a} and {b}", sampled with ordinary classifier-free guidance.
The manuscript's reference (introduction ¶1).

**PoE**

Uncorrected product-of-experts composition, equation 1 of the manuscript: one conditional
prediction per concept plus the shared unconditional one, with the model's usual guidance weight
on each conditional term. No correction, no adapter.

**Hybrid**

One animal carrying features of both, where two animals were asked for. The manuscript's name for
the failed object (introduction ¶3).

## The figure

⬅️ [Previous](#words-this-plan-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-we-expect-before-rendering) ➡️

```
                 SD 1.4          SD 2.1          SDXL            SD 3.5
               joint | PoE     joint | PoE     joint | PoE     joint | PoE
butterfly ×    [img] [img]     [img] [img]     [img] [img]     [img] [img]
flower meadow
camel ×        [img] [img]     ...
forest
cat × dog      [img] [img]     ...
```

- **Rows:** the three prompt pairs, labelled at the left edge in the pair's own words.
- **Columns:** the four models, labelled across the top, with "joint" and "PoE" under each model
  name so the pair order inside a cell reads without a legend.
- **One cell:** two images of one pair on one model at seed 42, the joint prompt on the left and
  PoE on the right.
- **Data:** the PNGs under `/datasets/mmolefe/poe_repair_min/outputs/same_rule_four_models/`, one
  `meta.json` per image carrying model id, sampler, steps, guidance, resolution, seed and prompts.
- **Output:** `paper/overleaf-iclr/figures/same-rule-on-four-models.pdf`, full text width.

## What we expect before rendering

⬅️ [Previous](#the-figure) | 📋 [TOC](#table-of-contents) | [Next](#considerations) ➡️

These are written before any image exists, and the review file records each cell against them.

| Pair | Joint prompt, expected | PoE, expected |
|---|---|---|
| butterfly × flower meadow | both concepts, all four models | both concepts, all four models |
| camel × forest | both concepts, all four models | both concepts, all four models (Liu et al.'s success case) |
| cat × dog | two animals, all four models | one hybrid, all four models |

**What would surprise us:** PoE giving two separate animals for cat × dog on any model, or a
joint-prompt cell failing to show both concepts. Either is shown as rendered, and the caption is
narrowed to what the grid shows. The figure is never re-seeded to remove a surprise.

## Considerations

⬅️ [Previous](#what-we-expect-before-rendering) | 📋 [TOC](#table-of-contents) | [Next](#environment-facts-this-plan-depends-on) ➡️

**Each model runs at its own defaults, and the rule is the same on all of them**

SD 1.4 and SD 2.1 (base) run at 512 × 512 with DDIM, 50 steps, guidance 7.5. SDXL runs at
1024 × 1024 with DDIM, 50 steps, guidance 7.5, the settings every SDXL result in the paper uses.
SD 3.5 medium runs at 1024 × 1024 with its flow-matching Euler sampler at the settings its model card
recommends (40 steps, guidance 4.5); the earlier joint-prompt-only script
`scripts/showcase/sd35_joint_prompts.py` used 50 steps and 7.0, so its renders are not this column. Each PoE cell uses the same guidance weight as the joint-prompt cell
beside it, so within a cell the only change is the composition rule.

**PoE on SD 3.5 is the same rule on velocities**

SD 3.5 predicts a velocity rather than noise. At a fixed step the velocity is an affine function
of the noise prediction with the same coefficients for every prompt, so
$v_u + w\sum_i (v_i - v_u)$ is equation 1 written in velocities. The render script applies it
there without further derivation, and the review file states this as the reason.

**Seed 42 means a different starting noise on each model**

The four models have different latent shapes (4 × 64 × 64 for SD 1.4 and 2.1 at 512, 4 × 128 × 128
for SDXL, 16 × 128 × 128 for SD 3.5), so no single noise tensor can be shared across columns. SDXL
cells borrow the cached seed-42 starting latent of cat × dog, as every SDXL figure in the paper
does, because that cache cannot be regenerated from `torch.manual_seed`. The other three models
draw their noise from `torch.Generator().manual_seed(42)` at their own shape. Within one column,
all three rows share one starting noise.

**One seed, 42, fixed before rendering**

Only seed 42 renders, and the figure shows it whatever it looks like. One seed per cell shows what
the rule can do on each model; it cannot show how often, so the caption says one sample per cell.

**The camel pair comes from Liu et al.'s released code, not their paper**

`"a camel" "a forest"` is the conjunction example for GLIDE in the README of
`energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch`
(`scripts/image_sample_compose_glide.py --prompts "a camel" "a forest" --weights 7.5 7.5`). The
paper's text never mentions a camel. So a caption can say the pair is the example from Liu et al.'s
code, and may not say it is from their paper. The same repository's Stable Diffusion script
defaults to `CompVis/stable-diffusion-v1-4`, DDIM, 50 steps, scale 7.5, which is the SD 1.4
column's setting here.

## Environment Facts This Plan Depends On

⬅️ [Previous](#considerations) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

- Python is `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`, by absolute path, in the job
  script ([environment overview](../../../../environment/overview.md)).
- Output goes to `/datasets` only, with the `df` guard on that filesystem
  ([storage](../../../../environment/storage.md)).
- One RTX 3090 on `bigbatch` holds any one of the four models in fp16. The job loads them one at a
  time and frees each before the next ([nodes](../../../../environment/hpc/nodes.md)).
- The cluster's Hugging Face cache (`~/.cache/huggingface/hub`, checked 2026-09-22) holds
  `CompVis/stable-diffusion-v1-4`, `stabilityai/stable-diffusion-xl-base-1.0`,
  `stabilityai/stable-diffusion-3.5-medium`, and SD 2.1 base only as the mirror
  `Manojb/stable-diffusion-2-1-base`, which has no `model_index.json`. The script loads SD 1.4 and
  SD 2.1 part by part, tries `stabilityai/stable-diffusion-2-1-base` first and the mirror second,
  and records the id that loaded in every image's `meta.json`. Task 1.4 reports which one it was.
- The cluster checkout's `outputs/` is a real folder on `/home-mscluster`, not a link to
  `/datasets`. The renders go to `/datasets`, and the job copies only the three sheet PDFs into
  `outputs/same_rule_four_models/`, where `scripts/cluster.sh pull` reaches them.
- Slurm does not create the `--output` log folder, so the launch line makes it first. The job
  excludes the faulted `bigbatch` nodes and `mscluster72`/`73`, which cannot see `/datasets` from
  inside a job ([nodes](../../../../environment/hpc/nodes.md)).
- Launch is `scripts/cluster.sh run "sbatch ..."` from the laptop. It refuses to run with
  uncommitted tracked changes.

## Tasks

⬅️ [Previous](#environment-facts-this-plan-depends-on) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

**For Claude to execute.** Ask Claude to do these.

### 0. 🧭 Check this plan before working from it

- [ ] **0.1** Check this plan conforms and its instructions are concrete.
  - Paste: `/verify-plan @plans/03-does-the-correction-cause-composition/plans/figures/12-the-same-rule-on-four-models.md`
  - Done when: the report comes back clean, or its proposals have been applied.
- [ ] **0.2** Add this plan to the background pool of the root running order.
  - Paste: `/integrate-plans @plans/03-does-the-correction-cause-composition/plans/figures/12-the-same-rule-on-four-models.md`
  - Done when: the root `MASTER_PLAN.md` lists it, with no step number.

▶ **Next: [task 1.1](#1--pin-the-prompts-and-check-the-models-load)**.

### 1. 🔧 Pin the prompts and check the models load

- [x] **1.1** Confirm the camel pair's wording in `liu2022compositional` (arXiv 2206.01714), and
      write the three pairs' concept prompts and joint prompts into the review file before any
      render.
  - Done when: the review file's `Prompts` table has three rows, the camel row quoting its source.
- [x] **1.2** Write `scripts/same_rule_four_models.py`. It has one `--model` switch
      (`sd14`, `sd21`, `sdxl`, `sd35`), a `--column` of `joint` or `poe`, `--pairs` and `--seeds`,
      and a `--sheet` mode that draws the grid. The SDXL path calls the repo's own
      `run_method("mono" | "poe", ...)` with the cat × dog starting latent; the other three run a
      plain loop over their diffusers components with the combination above.
  - Done when: `--dry-run` on the laptop prints the 24 cells per seed it would render with the
    settings per model.
- [x] **1.3** Write `scripts/same_rule_four_models.sbatch` from `scripts/figure_candidates.sbatch`:
      disk guard on `/datasets`, GPU guard, `co3` path check, models in the order SD 1.4, SD 2.1,
      SDXL, SD 3.5, with the joint column before PoE for each.
  - Done when: `bash -n` passes on the script.
- [ ] **1.4** Commit the plan, review file and both scripts, then check all four models load on a
      node before rendering anything.
  - Command: `scripts/cluster.sh run "mkdir -p /datasets/mmolefe/poe_repair_min/outputs/same_rule_four_models/logs && sbatch --export=ALL,CHECK_LOAD=1 scripts/same_rule_four_models.sbatch"`
  - Done when: the log ends `rc 0`, with one `[load]` line per model; the SD 2.1 line's id goes into
    the review file's settings table.

▶ **Next: [task 2.1](#2--render-and-draw)**.

### 2. 🚀 Render and draw

- [ ] **2.1** Launch through the one launch path.
  - Command: `scripts/cluster.sh run "sbatch -p batch --exclude=mscluster124,mscluster129 --export=ALL,SEEDS=42 scripts/same_rule_four_models.sbatch"`
  - Expected runtime: under half an hour for 24 images (3 pairs × 4 models × 2 columns, seed 42).
  - Done when: `scripts/cluster.sh jobs` no longer lists the job and the log ends in `=== done`.
- [ ] **2.2** Pull the renders and draw the grid.
  - Command: `scripts/cluster.sh pull outputs/same_rule_four_models`, then copy
    `outputs/same_rule_four_models/seed-42.pdf` to `paper/overleaf-iclr/figures/same-rule-on-four-models.pdf`.
    The job draws the sheet on the cluster.
  - Done when: the job log's `[sheet]` line says `missing 0 of 24`.
- [ ] **2.3** Read every seed-42 cell against the table in
      [What we expect before rendering](#what-we-expect-before-rendering), and write each read into
      the review file: both concepts, one hybrid, or something else, in words.
  - Done when: the review file has 24 rows filled.

▶ **Next: [instruction 3.1](#3--look-at-the-grid)**.

### Close out. 🔄 Record what this plan taught

- [ ] **C.1** Return to the proofread walk at chunk 11 with the recommended prompt at the top of
      this plan. The walk swaps the figure file and caption in the manuscript. This plan writes
      nothing into `iclr2027_conference.tex`.

## Instructions

⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#the-check-before-moving-on) ➡️

**For you to follow manually.** Do these yourself.

### 3. 👁️ Look at the grid

◀ **Needs: [task 2.3](#2--render-and-draw)**.

- [ ] **3.1** Open `paper/overleaf-iclr/figures/same-rule-on-four-models.pdf` in VS Code's PDF
      viewer and zoom to 200%.
  - Read the row labels on the left and the model names across the top without looking anywhere
    else. If either needs the caption to make sense, note which one.
  - Look at the cat × dog row's PoE images: each should be one animal.
- [ ] **3.2** Say whether the grid goes into the paper as drawn, or name the cell or label to change.

## The check before moving on

⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#figure-catalog) ➡️

```bash
OUT=/datasets/mmolefe/poe_repair_min/outputs/same_rule_four_models
scripts/cluster.sh sh "find $OUT -name '*.png' | wc -l"        # 24
scripts/cluster.sh sh "cat $OUT/sd35/a_cat__x__a_dog/seed_42/poe.json"   # steps 40, guidance 4.5, rule velocities
ls -la paper/overleaf-iclr/figures/same-rule-on-four-models.pdf
```

**Pass criteria**

- 24 PNGs, each with its `meta.json`.
- The seed-42 sheet has 24 filled images, and labels sit on the rows and columns.
- The review file has every seed-42 cell read in words.

**Fail criteria (STOP)**

- A model id fails to load: stop and say which one. Do not swap in another checkpoint silently.
- A joint-prompt cell is pure noise or a black image: the sampler is wrong for that model, not
  the prompt.

**When you get results, answer**
[the review file](../../review/12-the-same-rule-on-four-models.md).

## Figure Catalog

⬅️ [Previous](#the-check-before-moving-on) | 📋 [TOC](#table-of-contents) | [Next](#code-references) ➡️

| Figure | File | What it shows | Status |
|---|---|---|---|
| Figure 2, main text | `paper/overleaf-iclr/figures/same-rule-on-four-models.pdf` | three pairs by four models, joint and PoE per cell, seed 42 | owed |

## Code references

⬅️ [Previous](#figure-catalog) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- `poe_repair/run.py`, `run_method`: the SDXL joint (`mono`) and PoE paths the SDXL column reuses.
- `poe_repair/config.py`: SDXL guidance 7.5 and 50 steps.
- `scripts/figure_candidates.py`: `NOISE_PAIR = "a_cat__x__a_dog"`, the borrowed starting latent.
- `scripts/showcase/sd35_joint_prompts.py`: loads SD 3.5 medium on the cluster.
- `scripts/correction_size_over_the_run.py`: draws the current two-image Figure 2, which this
  replaces.

## Next step

⬅️ [Previous](#code-references) | 📋 [TOC](#table-of-contents)

Task 1.4: commit and run the load check on a node. The dry-run passed on the laptop on 2026-09-22.
