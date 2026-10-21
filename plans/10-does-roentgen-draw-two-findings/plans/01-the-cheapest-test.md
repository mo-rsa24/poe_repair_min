# 🧪 The cheapest test: does RoentGen-v2 draw two findings, and does composing two single-finding versions draw them better

Moved: this plan now lives in ~/PhD/medical-imaging/compositional_diffusion/a-two-findings-in-one-film/a1-does-roentgen-draw-two-findings/; this copy is kept for the record.

## Recommended prompt (after this plan completes)

```
/sync-plan-tree @plans/10-does-roentgen-draw-two-findings/plans/01-the-cheapest-test.md — pair table, plain and composed both-finding rates measured
```

## Position in the plan tree

| Step | Plan | What it does |
|------|------|-------------|
| **1 (current)** | **[the cheapest test](01-the-cheapest-test.md)** | **Measures whether a pretrained chest X-ray generator fails at pairs of findings that rarely co-occur in its labels, and whether Liu-style composition recovers them** |

This scope has no master plan yet. It is a feasibility test for a medical use of [product-of-experts composition](../../../context/world/poe-composition.md), separate from the animals paper.

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: where you are](#quick-context-where-you-are)
- [The claim](#the-claim)
- [Tasks](#tasks)
- [Instructions](#instructions)
- [What has to pass before this runs](#what-has-to-pass-before-this-runs)
- [Outputs](#outputs)
- [Environment facts this plan depends on](#environment-facts-this-plan-depends-on)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

## Quick context: where you are

**The experiment**

Prompt RoentGen-v2 (stanfordmimi/RoentGen-v2 on Hugging Face, MIT licence, Stable Diffusion 2.1 base) for two chest findings together and for each alone. A detector scores how often both findings appear. Then train one adapter per finding, compose the two at sampling time by summing their noise predictions per step (Liu et al. 2022), and score again.

**Why it exists**

The animals work shows that composing two single-concept experts draws "a cat and a dog" where one prompt fails. Nothing found in the literature measures whether a medical generator fails the same way, so this test is the cheapest way to learn whether a medical use case exists at all.

## The claim

**Hypothesis:** for finding pairs that rarely co-occur in the NIH labels, the plain pair prompt draws both findings less often than the composed adapters do.

**The pair.** Cardiomegaly and Pneumothorax: 49 images carry both, against 131 expected if independent (ratio 0.37), with 2,776 and 5,302 images carrying each. The control pair is Cardiomegaly and Effusion: 1,063 images carry both (ratio 3.2).

**The measure.** A finding counts as drawn when the detector's score for it passes the value that gives 90% specificity on real NIH images without that finding. The both-rate of a prompt type is the fraction of its 64 images (fixed seeds) where both findings count as drawn. P is the plain pair prompt's both-rate on the rare pair, C the same on the control pair, K the composed both-rate on the rare pair.

**Falsification, written as constants in the scorer source in task 3.1 before any image is generated.**

- Null: P is at least 0.5 × C. The generator already draws the rare pair about as well as a common one, no medical problem is shown, and task 5.1 is not run.
- Support: P is under 0.5 × C, K is at least P + 0.20, K is at least 0.8 × C, and each finding's single-finding rate in composed images stays at least 0.8 of its rate in the alone prompts.
- Inconclusive: the detector's AUROC on real NIH images is under 0.75 for either finding.

## Tasks

For Claude to execute. Ask Claude to do these.

### 1. 📊 Build the pair table

- [x] **1.1** Download the NIH ChestX-ray14 label file to `data/nih_cxr14/Data_Entry_2017_v2020.csv` (112,120 rows, 9 MB, not committed). ✓ verified (112,121 lines including the header, 2026-10-06)
- [x] **1.2** Write `scripts/medical_pair_table.py`: per finding, the count of images; per pair, the count with both, the count expected if the two were independent, and the ratio of observed to expected. Writes `artifacts/results/does-roentgen-draw-two-findings/pair-table.csv`. ✓ verified (14 findings printed, Pneumothorax 5,302, 91 pair rows written, 2026-10-06)
  - Verify: the script prints 14 findings and their counts, and Pneumothorax is 5,302.

▶ **Next: [instruction 2.1](#2--pick-the-pair-and-approve-the-bar)** (you pick the pair from the table).

### 3. 🔧 Prompt RoentGen-v2 and score it

◀ **Needs: [instruction 2.1](#2--pick-the-pair-and-approve-the-bar)** done, so a pair and a bar exist.

- [x] **3.1** Write the bar into the scorer source as named constants, as chosen in instruction 2.1: `scripts/medical_pair_bar.py`. ✓ verified (four made-up cases returned problem-shown, null, inconclusive, support, not-supported as intended, 2026-10-06)
- [ ] **3.2** On mscluster107, generate for the chosen pair and a frequent control pair: the pair prompt, finding A alone, finding B alone, each at fixed seeds. Checkpoints and images go to `/datasets/mmolefe/poe_repair_min/`.
- [ ] **3.3** Score each image with TorchXRayVision `densenet121-res224-all` as the primary detector and `densenet121-res224-pc` (trained on neither MIMIC nor NIH) as a secondary read, using `scripts/medical_pair_score.py`. Chosen as the cached detector with the highest lower AUROC across Cardiomegaly and Pneumothorax on 4,999 real NIH images (one shard, `images_001`); six were measured. AUROC for Cardiomegaly, Pneumothorax: all 0.901, 0.771; pc 0.874, 0.744; nih 0.886, 0.749; mimic_nb 0.746, 0.693; chex 0.764, 0.693; mimic_ch 0.723, 0.672. The primary clears the 0.75 bar; the secondary misses it for Pneumothorax by 0.006 and is reported beside the primary, not instead of it. The primary saw NIH and MIMIC images in training, so its AUROC here is optimistic and it is not independent of the generator. Report the both-finding rate per prompt type.

▶ **Next: [instruction 3.4](#3--read-the-plain-prompt-images)**.

### 5. 🔧 Compose single-finding adapters

◀ **Needs: [instruction 3.4](#3--read-the-plain-prompt-images)** done.

- [ ] **5.1** Train one adapter per finding on mscluster107 with a [disk guard](../../../environment/storage.md#the-disk-guard-rule-and-the-failure-that-motivates-it) on `/datasets`: Cardiomegaly from the 2,727 images carrying it without Pneumothorax, Pneumothorax from the 5,253 carrying it without Cardiomegaly. Only runs if task 3.3 shows the null did not hold.
- [ ] **5.2** Sample with the two adapters composed by summing noise predictions per step; same seeds as task 3.2.
- [ ] **5.3** Score as in task 3.3 and write the comparison.

▶ **Next: [instruction 5.4](#5--read-the-composed-images)**.

## Instructions

For you to follow manually. Do these yourself.

### 2. 🖱️ Pick the pair and approve the bar

◀ **Needs: [tasks 1.1 to 1.2](#1--build-the-pair-table)** done.

- [x] **2.1** Pick the pair and the bar, as written under [the claim](#the-claim): Cardiomegaly and Pneumothorax as the rare pair, Cardiomegaly and Effusion as the control. Delegated to Claude on 2026-10-06, open to a veto before task 3.1.
  - Check it: open `pair-table.csv` in VS Code and find the two rows (ratios 0.373 and 3.224).

▶ **Next: [task 3.1](#3--prompt-roentgen-v2-and-score-it)**.

### 3. 🖱️ Read the plain-prompt images

- [ ] **3.4** Open the contact sheet for the pair prompt and each single prompt. Write down, per image, whether you see one finding, both, or neither.
  - What it means: if you already see both in most images, the null is likely and task 5.1 may not be worth the training time.

### 5. 🖱️ Read the composed images

- [ ] **5.4** Open the composed contact sheet next to the plain pair sheet at the same seeds and write which draws both more often.

## What has to pass before this runs

- The NIH label file downloads and has 112,120 rows.
- The bar constants exist in the scorer source before any generation (task 3.1).
- The pretrained detector's labels include both chosen findings (checked first in task 3.3).

## Outputs

- `artifacts/results/does-roentgen-draw-two-findings/pair-table.csv`
- contact sheets and the both-rate table, filed with a card entry per `ARTIFACT_TREE_FORMAT.md`.

## Environment facts this plan depends on

- Work runs on mscluster107 (2× Quadro RTX 8000, 49 GB each, verified idle by `nvidia-smi` this session); `environment/hpc/nodes.md` still lists it as down.
- Python is the `co3` environment at its absolute path; a `cxr` environment also exists and is unexamined.
- Checkpoints and images go to `/datasets/mmolefe/poe_repair_min/`, never `/home-mscluster`.
- The RoentGen-v2 weights are gated on Hugging Face (MIMIC-CXR credentialing terms accepted on the user's account 2026-10-06); a download of `model_index.json` from mscluster107 succeeded with the token in `~/.cache/huggingface/token`. Set `HF_HOME` under `/datasets/mmolefe/poe_repair_min/` so the weights do not land on `/home-mscluster`.
- The `co3` environment on mscluster107 has torch 2.5.1, diffusers 0.29.2, peft 0.19.1, accelerate and torchxrayvision 1.5.2.

## Next step

If the composed adapters draw both findings more often than the plain pair prompt on the rare-joint pair and equally on the control pair, scope the use case in a master plan. If not, record the null as a finding.

## Error Matrix

| Failure | Cause | Fix |
|---|---|---|
| Pair table has 0 rows for a pair | label spelled differently (`Pleural_Thickening`) | match the label strings from the file's own list |
| Generation runs on CPU without error | node GPU in a bad state | assert `torch.cuda.is_available()` after pinning the device |
| Detector never fires on either finding | labels missing from the detector | choose another detector or train one on NIH |
