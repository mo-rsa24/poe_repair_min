# Walk map: the cheapest test (does RoentGen-v2 draw two findings)

**Plan being walked:** [the cheapest test](../../../plans/10-does-roentgen-draw-two-findings/plans/01-the-cheapest-test.md)

**Session:** `f5a87047-c4c9-464a-b11b-972a4056f6ee` (the paper search that preceded the build ran as `0ac4f7d7-37fb-49ed-b8af-c955bdf4ba58`)

**Returns land in:** `routes/` beside this file, one self-addressed file per breakaway or handoff, named by its row id in the table below. The walker is this map's only writer.

## Where the walk is

Piece 4 of 5 (the detector). The 320-image run with one-word prompts is scored and cannot answer the question: both-rate 0% on the rare pair and on the control, and the single-finding prompts drew Pneumothorax in 0 of 64 and Effusion in 1 of 64. The authors' released images, scored with the same detector, show it can read their pneumothorax and effusion images. Next: regenerate with prompt text taken from the released dataset.

## Walk in this session

Depth: pieces 1 to 5 of a cut written at the open. Choices: read the authors' code first, plan repairs after piece 4, GPU work on mscluster107 (confirm the allocation and that the cluster checkout matches before launching), piece 2 is 48 images (3 findings x 16 seeds). The 25% single-finding gate stays the user's to settle and is not in `scripts/medical_pair_bar.py`.

1. Read the authors' inference code: done. Source read at commit 039e34b of StanfordMIMI/RoentGen-v2 (guidance 3 in their demo config, 75 steps, bf16, PNDM scheduler, empty-string unconditional prompt, prompts like "28 year old WHITE female. No evidence of acute cardiopulmonary disease ."). Reader confirmed by hand: "I see something like '28 year old WHITE female. No evidence of acute cardiopulmonary disease .'". No prediction was given for which difference matters most.
2. Regenerate the three single findings with the authors' settings and wording, 16 seeds each.
3. Score them and compare with the detector's reading of the released images.
4. Rare pair and control, joint prompt against PoE, 64 seeds.
5. Repair plan 10 and file the results.

## Ticked

- Tasks 1.1 and 1.2: pair table, 91 pairs.
- Instruction 2.1: pair (Cardiomegaly with Pneumothorax) and control (Cardiomegaly with Effusion) chosen, delegated.
- Task 3.1: bar constants in `scripts/medical_pair_bar.py`.

## Open

- Task 3.2 and 3.3: rerun generation with the authors' impression wording, then score with the primary detector and the secondary.
- Tasks 5.1 to 5.3 and instructions 3.4, 5.4.
- Plan repairs owed: group numbering (Tasks 1, 3, 5 beside Instructions 2, 3, 5), missing skeleton sections, paired review file, three `✓ verified` lines to move there.

## Out of the session

| Id | Kind | What it serves | State |
|---|---|---|---|
| none | | | nothing running on mscluster107 |

## Closing pass still owed

Diagrams none. Artifacts to file: the smoke sheet and the scores under `artifacts/results/does-roentgen-draw-two-findings/`, with a card entry. Review file and finding after the verdict.
