# 📖 Read map: My Eye, My Brain

The walk's persistent state. Read it first on every invocation, together with a listing of
`routes/`, and write it before every round ends. Only the walking session writes this file.

## Position in the walk

| Field | Value |
|---|---|
| Paper | My Eye, My Brain: Development and Validation of an Accessible AI-Enabled Retinal Screening System for the Early Detection of Ocular and Systemic Disease in Underserved South African Populations. Samuel Sephiri. A PhD research proposal for the University of Pretoria or the University of the Witwatersrand, undated. No identifier. |
| Journey | none. No row of `~/goal-setting/learning/JOURNEYS.md` covers retinal screening or diagnostic-accuracy studies. The walk lives in this repository's `artifacts/drips/`, beside the walk of the Composability Gap proposal. |
| Paper's folder | `artifacts/drips/read-my-eye-my-brain/` holds everything: `paper/`, `read-ahead/`, `checks/`, `routes/` |
| Where the reader types | `$HOME/playground/deep-learning/my-eye-my-brain/`, not created yet |
| Current chunk | 5 of 65: abstract > sentence 5, "The second involves the design" (open, prediction asked)   ← current |
| States | 1 checked, 0 trusted, 2 read, 1 left open, 0 out of reach, 0 disagrees, 61 not reached |
| Object | stage 1, built 2026-10-03 (all three steps matched): one screening day at one clinic, 100 patients, 20 with disease by the expert's grade |
| Read-ahead | by hand. `read-ahead/00-INDEX.md` and `read-ahead/00-abstract.md` exist. Each later section's file is written before that section opens. |
| Owner | the session that opened the walk on 2026-10-02 |

## Table of contents

- [Position in the walk](#position-in-the-walk)
- [Quick context: what this is](#quick-context-what-this-is)
- [The claim list](#the-claim-list)
- [The object and what it reaches](#the-object-and-what-it-reaches)
- [The chunk record](#the-chunk-record)
- [The pieces](#the-pieces)
- [Terms and parts](#terms-and-parts)
- [Notation pairs](#notation-pairs)
- [Disagreements and slips](#disagreements-and-slips)
- [Parked, and what it waits on](#parked-and-what-it-waits-on)
- [Runs](#runs)
- [Routes and threads](#routes-and-threads)
- [Timings](#timings)
- [Next step](#next-step)

## Quick context: what this is

Navigation: ⬅️ [Position](#position-in-the-walk) | 📋 [TOC](#table-of-contents) | [Next](#the-claim-list) ➡️

**The paper, in one line**

A plan to build a phone-based retinal camera and a disease-detection model together, and to test
the model on that camera's photographs in rural South African clinics.

**Why it is being read**

To understand it, in the reader's words on 2026-10-02: "I want to understand it." A round is
therefore for what the proposal says and how its claims would be tested. Slips are named where
they sit because they change the reading. Feedback for the author is not the purpose.

**Where the walk is**

The object is approved. Step 1 of 3 is done: the reader's `clinic_day.py` printed 100 20 24. The reader asked `why` 20 of 100 have disease and was told the count is invented, with 5 as the alternative that makes each missed patient worth 20 points. The reader is on step 2, waiting on `check 16 4 8 72`. The reader asked whether the abstract will be walked and interacted with; told yes, one sentence per round after step 3, and that the pasted paragraph is chunks 1 and 2.

## The claim list

Navigation: ⬅️ [Quick context](#quick-context-what-this-is) | 📋 [TOC](#table-of-contents) | [Next](#the-object-and-what-it-reaches) ➡️

One row per abstract sentence, in order. The state is the state of the chunks that pay it off.

| # | The sentence, first words | Pays off at | Test on the object | State |
|---|---|---|---|---|
| 1 | "This study proposes the design, development, and validation" | sections 4.1 and 8 | none: framing | read |
| 2 | "The research addresses two closely coupled problems" | section 3 gaps 1 and 3; sections 6.2, 6.3 | one model's table on two populations | checked |
| 3 | "The study is structured as two interdependent research streams." | section 8 | none: framing | read |
| 4 | "The first involves the curation, fine-tuning, and clinical validation" | objectives 1 and 2; section 8.1 | two models on the same patients | left open |
| 5 | "The second involves the design and prototyping" | objective 3; sections 8.1 and 6.3 | the share of photographs too poor to grade | not reached |
| 6 | "The two streams are validated jointly through an integrated field study" | objective 4; section 8.2; gap 2 | the same patients photographed by two cameras | not reached |
| 7 | "The study further maps its research outputs" | objective 5; sections 6.5, 8.3, 12 | the subgroup table and the range around each fraction | not reached |
| 8 | "Expected contributions include a validated, population-appropriate" | sections 7 and 13 | none new: follows 4, 5 and 6 | not reached |

## The object and what it reaches

Navigation: ⬅️ [The claim list](#the-claim-list) | 📋 [TOC](#table-of-contents) | [Next](#the-chunk-record) ➡️

Approved by the reader on 2026-10-02 as proposed. A stage is added only with the reader's word. Every
count in the object is invented by the walk, because the proposal gives no numbers.

| Stage | The object | What it reaches | Earlier numbers it must still print |
|---|---|---|---|
| steps | a list of 100 (expert, model) pairs; the list counted into a two-by-two table | nothing in the proposal: the shapes and names | |
| 1 | one clinic day as a two-by-two table: 100 patients, 20 with disease by the expert's grade | "clinically acceptable diagnostic accuracy" (5.1); sensitivity and specificity (8.2) | |
| 2 | the same patients, each with the model's score and a cut-off that moves | AUC (8.2); "clinically acceptable" as a choice of cut-off | stage 1's two fractions at the stage 1 cut-off |
| 3 | two tables for the same patients: two cameras, two models, or two populations, plus the photographs too poor to grade | abstract sentences 2, 4, 5, 6; gaps 1 and 2; secondary questions 1 and 3 | stage 1's two fractions |
| 4 | each table split into subgroups, with the range each fraction could take and the patients needed to narrow it | objective 4; section 6.5; section 9's power calculation | stage 1's two fractions |

Out of reach, with the reason: every statement about published work in sections 2 and 6 (the
cited papers are not in the folder); the device's optics (the proposal gives no design
parameter); the regulatory statements in section 6.5 (the SAHPRA document is not in the folder);
any trained model on real images (no data, no code, no reported results).

## The chunk record

Navigation: ⬅️ [The object](#the-object-and-what-it-reaches) | 📋 [TOC](#table-of-contents) | [Next](#the-pieces) ➡️

Every chunk in the proposal's order. One table per section, added when the section's read-ahead
file is first read.

### 1. Abstract

| # | Kind | Printed number | First words | State | What the reader did | Number, and where to recount it |
|---|---|---|---|---|---|---|
| 1 | sentence | | "This study proposes the design" | read | read the sentence; nothing to check | |
| 2 | sentence | | "The research addresses two closely coupled" | checked | went straight to `check`, no prediction given in chat; typed both clinics and the gap, matched | 0.2, `clinic_day.py`, re-run from an empty file redoes it |
| 4 | sentence | | "The first involves the curation" | left open | gave a critique of the ordering and the language instead of a prediction or a check; typed `next` | |
| 5 | sentence | | "The second involves the design and prototyping" | not reached | asked for `ungradable`, the two policies for photographs too poor to grade | |
| 2 | sentence | | "The research addresses two closely coupled problems" | not reached | | |
| 3 | sentence | | "The study is structured as two" | not reached | | |
| 4 | sentence | | "The first involves the curation" | not reached | | |
| 5 | sentence | | "The second involves the design" | not reached | | |
| 6 | sentence | | "The two streams are validated jointly" | not reached | | |
| 7 | sentence | | "The study further maps its research outputs" | not reached | | |
| 8 | sentence | | "Expected contributions include" | not reached | | |

## The pieces

Navigation: ⬅️ [The chunk record](#the-chunk-record) | 📋 [TOC](#table-of-contents) | [Next](#terms-and-parts) ➡️

Every piece of code the walk has asked for or will ask for. Answer keys with an output are ran;
the rest are written before their round.

| Piece | Chunk | Lane | Lines | Must print | The reader's printed | Answer key |
|---|---|---|---|---|---|---|
| `patients` | step 1 | typed | 6 | 100 20 24 (patients, with disease, referred) | 100 20 24, matched 2026-10-02. The reader counted by adding the lengths of the four blocks, where the key counts over `patients`. The reader named the (0, 0) block `healthy_model_misses`. | `checks/step1_patients.py`, ran, shown after the match |
| `count_table` | step 2 | typed | 6 | 16 4 8 72 (caught, missed, false alarm, cleared) | 16 4 8 72, matched 2026-10-03, counted over `patients` with `and` and `not` on the 0 and 1 values | `checks/count_table.py`, ran, shown after the match |
| `sens_spec` | step 3, stage 1 | typed | 4 | 0.8 0.9 | 0.8 0.9, matched 2026-10-03. The reader guards an empty row by returning 0; told that 0 reads as "caught none" where the truth is "nothing to measure", which matters at stage 4's small subgroups | `checks/sens_spec.py`, ran, shown after the match |
| `two_tables` | 2 | typed | 8 | clinic A 0.8/0.9, clinic B 0.6/0.9, gap 0.20 | clinic A (0.8, 0.9), clinic B (0.6, 0.9), gap 0.2, matched 2026-10-03. Left a dead `def clinic_b(patients): pass` above the real assignment, unused | `checks/two_tables.py`, ran, shown after the match |
| `paired_change` | 4 | typed | 10 | 2 | | `checks/paired_change.py`, ran, shown after the match |
| `ungradable` | 5 | typed | 10 | dropped: (0.8, 0.923), counted as referral: (0.8, 0.75) | | `checks/ungradable.py`, ran, shown after the match |
| `extremes` | step 3, example | shown, asked for by the reader | 6 | refer everyone 1.0 0.0; refer nobody 0.0 1.0 | | `checks/extremes.py`, ran, output in `extremes.out` |
| `run_inference` | 1, via r2 | typed by the agent, not the reader | ~40 | 4 raw scores | 0.5715 0.2231 0.1974 0.8440, ran | `routes/dataset-model-return.md`; venv at `~/playground/deep-learning/my-eye-my-brain/dataset-smoke-test/` |
| `interval` | 7 | verified by hand | about 6 | the range around one fraction | | not written |

## Terms and parts

Navigation: ⬅️ [The pieces](#the-pieces) | 📋 [TOC](#table-of-contents) | [Next](#notation-pairs) ➡️

Standing is one of: worked, planned and not worked, partly, missing. Looked up on 2026-10-02 in
`JOURNEYS.md` and in each journey's plan files and ticked boxes.

| Term or part | The paper teaches it at | Standing in the tree, and where | Holding answer given |
|---|---|---|---|
| sensitivity, specificity, AUC | assumed | missing as a taught topic in any registered journey. The phrase appears in `machine-learning/classical-ml-from-scratch/MASTER_PLAN.md`, whose `plans/` is empty | |
| power calculation | assumed | planned and not worked: `mathematics/statistics/hypothesis-testing`, plan 02, 0 of 17 boxes ticked across the journey's three plans | |
| comparing two proportions | assumed | planned and not worked: the same journey, plan 01 | |
| the range around an estimated fraction | assumed | missing: `mathematics/statistics/sampling` has a master plan and no plan files | |
| two graders agreeing | assumed | planned and not worked: `data-engineering/data-for-ml`, plan 02, 0 of 20 boxes ticked | |
| fine-tuning | assumed | missing: `ai-engineering/llm-applications/model-adaptation` has a master plan and no plan files | |
| fundus imaging | named 14 times, never described (first at line 89) | missing | chunk 1 |
| diabetic retinopathy, glaucoma | named, never described | missing | |
| the closest existing work to "Sam" | assumed the reader knows the field | r1 names Bellemo et al. 2019 (Lancet Digital Health), a Singapore-trained ensemble CNN validated with zero African training images on 1,574 Zambian patients | chunk 1 |
| SAHPRA, Class C or D | glossed at section 6.5 | missing | |

## Notation pairs

Navigation: ⬅️ [Terms and parts](#terms-and-parts) | 📋 [TOC](#table-of-contents) | [Next](#disagreements-and-slips) ➡️

The proposal has no symbols and there is no journey, so there is no `NOTATION.md` to pair against.

| The paper writes | The journey writes | Meaning | First met at chunk |
|---|---|---|---|

## Disagreements and slips

Navigation: ⬅️ [Notation pairs](#notation-pairs) | 📋 [TOC](#table-of-contents) | [Next](#parked-and-what-it-waits-on) ➡️

A disagreement is the reader's number against the paper's claim. A slip is a fault in the printed
paper. The read-ahead found eleven candidate slips, listed in `read-ahead/00-INDEX.md`. Each moves
into this table in the round where it sits.

| Chunk | What the paper says | What was found | Evidence | Label |
|---|---|---|---|---|
| 1 (found at `term fundus`) | "fundus images", "fundus photographs" throughout | the proposal never says whether a result counts eyes or patients, though screening usually photographs both eyes (recalled) | search of `paper/proposal.md` for "both eyes", "per eye", "per patient" finds none | ran |
| 4 | objective 2 (§4.2): fine-tuning and its own benchmark are one step; §3 gap 1 asserts existing models may not generalise | the reader's own critique: no step tests an existing model on African data first, before committing to build Sam; the proposal's own §6.2 citation (Bellemo et al. 2019) already ran that test and found "clinically acceptable" results, which the proposal never engages with. Separately, "curation... of a deep learning model" misapplies a data verb to the model; objective 1 shows curation belongs to the dataset. | the reader, 2026-10-03, against lines 121, 123 and r1's finding | reader's judgment |
| 1 | "an integrated, low-cost ... retinal screening system" | "low-cost" is never given a figure anywhere in the document | search of `paper/proposal.md` for any currency or cost figure finds none | ran |
| 2 | sentence 2: "near-total absence of AI diagnostic models... validated... specifically on African patient populations" | holds for IDx-DR, EyeArt, Medios AI, Google ARDA (no African validation found for any); does not hold for SELENA+ (2019/2021 Zambia validation, active Tanzania partnership) or for Bellemo et al. 2019, the proposal's own §6.2 citation, which is that same Zambian validation | `routes/model-survey-return.md`, parts 1 and 3, each claim labelled quoted/paraphrased/not found with a source | reported by r1 |
| 4 | §5.1's "lightweight" model, §4.2 objective 2's "multi-class detection" of DR and glaucoma together | a real architecture fork: RETFound (ViT-Large, both diseases, not lightweight) versus EfficientNet-B0 baselines (lightweight, DR only, would need a second model or head for glaucoma); the proposal does not say which side it is on | `routes/model-survey-return.md`, part 2 and "Changes for the walk" | reported by r1 |
| 1 | "a deep learning model" (sentence 4), fundus photographs (sentence 1) | a real fundus photograph exists (4 APTOS 2019 images, a Hugging Face mirror) and a real pretrained DR-grading model (ResNet-18, 11.2M params, MIT-licensed) loaded cleanly and ran on all 4; its raw scores were 0.5715, 0.2231, 0.1974, 0.8440, but the model card never states a grade convention and the mirror's label scheme does not match APTOS's own ordinal order, so no grade was read off | `routes/dataset-model-return.md`; ran: `run_inference.py` in the playground, state-dict loaded with no missing or unexpected keys | reported by r2 |

## Parked, and what it waits on

Navigation: ⬅️ [Disagreements and slips](#disagreements-and-slips) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Chunk | The part that left | Where it went | The sections or the ask | Returned |
|---|---|---|---|---|

## Runs

Navigation: ⬅️ [Parked](#parked-and-what-it-waits-on) | 📋 [TOC](#table-of-contents) | [Next](#routes-and-threads) ➡️

| # | Anchor chunk | What it executed | State | Finding |
|---|---|---|---|---|

## Routes and threads

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#timings) ➡️

No read-ahead agent was dispatched: the walking session read ahead by hand.

| # | Anchor | The ask | Return file | State |
|---|---|---|---|---|
| r1 | chunk 1, term fundus | background agent: survey IDx-DR/EyeArt/Medios/ARDA/SELENA+, two open baselines, one recommended paper, SAHPRA guidance content | `routes/model-survey-return.md` | integrated, folded into the disagreements table |
| r2 | chunk 1, term fundus | background agent: download the smallest reachable public fundus dataset, open real images, load a pretrained DR model, run inference on `clinic_day.py`'s object | `routes/dataset-model-return.md` | integrated, folded into the pieces and disagreements tables |

## Timings

Navigation: ⬅️ [Routes and threads](#routes-and-threads) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

One row per round. This is the only measure of how long a paper takes.

| Sitting | Chunk | Start | End | Minutes | Typed by the reader | Pasted |
|---|---|---|---|---|---|---|
| 1, 2026-10-02 | setup and read-ahead of the abstract | 15:17 | 15:22 | 5 | nothing | nothing |
| 1, 2026-10-02 | the claim index, sent twice: once as a wall, once as a table after the reader asked for the skill to split it | 15:24 | 15:28 | | "I want to understand it. next" | nothing |
| 1, 2026-10-02 | the object, for approval | 15:28 | 15:30 | 2 | "next", then "approve" | nothing |
| 1, 2026-10-02 | step 1, the list of 100 patients | 15:30 | 15:39 | 9 | `clinic_day.py`, 6 lines; then "why then check 100, 20, 24" | nothing |
| 1, 2026-10-02 | step 1: `why` answered, check matched, answer key shown | 15:39 | 15:41 | | "next", and asked whether the abstract's paragraphs will be walked and interacted with, pasting abstract paragraph 1 | nothing |
| 1, 2026-10-02 | step 2, the list counted into a two-by-two table | 15:41 | the sitting ended with the step typed; the reader's next message came 2026-10-03 03:45 | | `count_table` in `clinic_day.py`, 6 lines | nothing |
| 2, 2026-10-03 | step 2: check matched, answer key shown | 03:45 | 03:47 | | "check" | nothing |
| 2, 2026-10-03 | `hand` on the two fractions, last line held; `next` queued | 03:47 | 03:51 | | "hand then next"; the reader then typed "next" without giving the two decimals, so `hand` was left unanswered | nothing |
| 2, 2026-10-03 | step 3, the two fractions as a function | 03:51 | 03:55 | 4 | `sens_spec` in `clinic_day.py`, 4 lines; `count_table` rewritten to the answer key's form | nothing |
| 2, 2026-10-03 | step 3: check matched, answer key shown; object complete at stage 1 | 03:55 | 03:57 | 2 | "check" | nothing |
| 2, 2026-10-03 | two background agents dispatched (r1, r2); skill edit to drip-read's `ground` errand, grounding a real-world sketch in a fetched source | 03:57 | 04:35 | | asked for a diagram, existing models, a reachable dataset and pretrained model, background agents, and a skill improvement | nothing |
| 2, 2026-10-03 | `example` on step 3: the two extreme models, `checks/extremes.py` | 03:57 | 04:10 | | asked to see "they pull against each other" with `clinic_day.py` | nothing |
| 2, 2026-10-03 | chunk 1, abstract sentence 1 | 04:10 | 04:22 | 12 | "next" | nothing |
| 2, 2026-10-03 | chunk 1: `term fundus` | 04:22 | 04:35 | 13 | "term fundus" | nothing |
| 2, 2026-10-03 | r1 and r2 return, integrated into the map | 04:35 | 04:48 | 13 | "integrate", then an environment-tooling question answered outside the walk | nothing |
| 2, 2026-10-03 | chunk 2, abstract sentence 2 | 04:48 | 05:11 | 23 | `clinic_b` in `clinic_day.py`, 2 lines added plus one unused `def clinic_b` stub; "check" | nothing |
| 2, 2026-10-03 | chunk 3, abstract sentence 3 | 05:11 | 05:17 | 6 | "next" | nothing |
| 2, 2026-10-03 | chunk 4, abstract sentence 4 | 05:17 | 05:41 | | critique of the ordering and the language, then "next" with no check | nothing |
| 2, 2026-10-03 | chunk 5, abstract sentence 5 | 05:41 | open until the reader's next message | | "next" | nothing |

## Next step

Navigation: ⬅️ [Timings](#timings) | 📋 [TOC](#table-of-contents)

Waiting on the reader's prediction for chunk 5, then `check` on `ungradable.py`'s two policies
(dropped: sens 0.8 spec 0.923; counted as referral: sens 0.8 spec 0.75). After it, chunk 6,
abstract sentence 6, where `two_tables` is reused for the two cameras.
