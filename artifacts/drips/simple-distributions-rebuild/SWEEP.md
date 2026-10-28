# Sweep: bringing the rebuild walk up to the current formats

This file scores every markdown file in this walk folder against the formats in `~/.claude` as they stand on 2026-10-28. The files here were written on 2026-10-18 to 2026-10-21, and since then the prompt format, the teaching triad, the vault-note presentation rules, the drip linking rules and the route-file template all changed. The checklist below holds only those changes, one rule per line with its source commit, so the retrofit edits against a written rule instead of by taste. Nothing in the folder has been edited yet. The score table shows where each file stands, and the batches at the end order the retrofit from the cheapest and safest edits to the most involved.

**How to rerun the score.** From this folder, `python3 -I sweep-score.py .` prints the same table as plain text (a dot is not applicable, a number is a miss count). After each batch, rerun it and the batch's columns should read zero.

**File kinds used below.** *Guide* is `PROMPTS.md`. *Ladder* is each `ladder/*.md`, a layer of pasteable prompts. *Step* is each of `01` to `05` at the folder root, the long files a `drip-execute-plan` walk follows. *Walk map* is `REBUILD_WALK.md`. *Coverage* is a `.quiz-prep/quiz-*.md` preparation file, and *solutions* is a `.quiz-prep/*-solutions.md` file. *Report* is a `video-scout/` pulse report.

## Table of contents

- [The checklist](#the-checklist)
- [The score](#the-score)
- [The batches](#the-batches)
- [Changes left out, and why](#changes-left-out-and-why)

## The checklist

Each rule is one change made to `~/.claude` since 2026-10-18 that applies to files of these kinds. "Commit" is the `~/.claude` commit that introduced it.

| Id | The rule | Source and commit | Applies to | How it is checked |
|---|---|---|---|---|
| P1 | A pasteable prompt reads in seven plain sections, in order: Background, Objectives, What to read, What is already known, Rules, What to hand back, What you will see. | `PROMPT_FORMAT.md`, "How it reads", `21b60a1` (2026-10-20) | guide, ladder, step | Per prompt (a `text` block of 5 or more lines that opens on a skill line or carries field labels), all seven headings present. The guide's "The prompt shape" paragraph, which still describes the nine-field form (line 140), counts as one miss. |
| P2 | No field names written as prose: the old labels (`Task:`, `Starts from:`, `Settled facts:`, `Outputs:`, `Target:`, `Progress:`, `Proof:`, `Limits:`, `Hand-off:`) become the plain sections of P1. | `PROMPT_FORMAT.md`, "How it reads", `21b60a1` | guide, ladder, step | Count of lines inside prompts matching `^(Task\|Starts from\|Settled facts\|Outputs\|Target\|Progress\|Proof\|Limits\|Hand-off):`. |
| P3 | "What you will see" has an along-the-way part (one question or a yes at each stop) and an at-the-end closing report in five parts: what was asked, what was done with paths, PASS or MISS with numbers, what is open, at most one question. | `PROMPT_FORMAT.md`, "How it reads", item 7, `21b60a1` | guide, ladder, step | Per prompt, the words "At the end" or "closing report" present. |
| P4 | A prompt ends on one line saying who judges it, from what, and when it is done. | `PROMPT_FORMAT.md`, "How it reads", `21b60a1` | guide, ladder, step | Per prompt, the last non-blank line names a judge (contains "judge"). |
| T1 | Every concept that has a `🧸 **Toy:**` line carries at least one `🔤 **The name:**` line, which takes each word of the name apart on that toy. | `TEACHING_TRIAD_FORMAT.md` v3, "Words mean things"; `LEARNING_PLAN_SKELETON.md`; `21b60a1` | step | Per `## Step`, a 🧸 line with no 🔤 line is a miss. Same grep as the triad's own check. |
| T2 | The name step sits after the moving picture and before the build, and before the formula's display block: the formula is read back in the name's words, never shown first. | `TEACHING_TRIAD_FORMAT.md` v3, "Where it sits"; `TEXTBOOK_FORMAT.md`, "A name is taken apart before its formula"; `21b60a1` | step | Per `## Step`, the first 🔤 line comes after the "words together: the formula" heading or after the `🔨 **Build` line. |
| T3 | The first toy is an ordinary one from everyday life; a toy built on the project's own numbers comes second, under a heading such as "Run it back on your work". | `TEACHING_TRIAD_FORMAT.md`, "The story is ordinary by default", `e14fd6a` (2026-10-21) | step | Per `## Step`, the first 🧸 line names a project artefact (the notebook, `rebuild/out`, a `.json` file, the ring, the bar, the quadrant, cat or dog, example 9, overlap, the mosaic) and the file has no "Run it back" section. |
| S1 | Every display equation that brings a new term is written plain, then again with an `\underbrace` label on each part. | `VAULT_NOTE_FORMAT.md` v2, "Every equation is taken apart", `08d8627` (2026-10-21); `STUDENT_PAGE_FORMAT.md`, "Every equation is interpreted", `ce671f4` (2026-10-20) | step | A `$$` block with no `\underbrace` and no underbraced block within 15 lines. An upper bound: a block that brings no new term is counted too. |
| S2 | A "**What the equation says.**" chunk follows each taken-apart equation. | `VAULT_NOTE_FORMAT.md` v2, `08d8627`; `STUDENT_PAGE_FORMAT.md`, `ce671f4` | step | Per heading section, `\underbrace` present without "What the equation says". |
| S3 | One colour key per page, declared once near the top, and each quantity keeps its colour everywhere. | `STUDENT_PAGE_FORMAT.md`, "One colour key per page", `ce671f4`; `VAULT_NOTE_FORMAT.md` v2 | step | `\textcolor` used and no "colour key" or "Colours on this page" block. |
| S4 | Every chart, table or figure the reader learns from is followed by "How to read this" and "What to look for". | `STUDENT_PAGE_FORMAT.md`, "Every visual is read for the student", `ce671f4`; `VAULT_NOTE_FORMAT.md` v2, "A toy table for every section" | step | Each `<table`, `![…](…)` or `<img` without both phrases in the 40 to 50 lines after it. |
| S5 | Inside an HTML table, symbols are written in HTML (`<sub>`, `<i>`, `<b>`, `<code>`), never `$…$`, `**` or backticks. | `VAULT_NOTE_FORMAT.md` v2, `08d8627` | step | Lines between `<table` and `</table>` containing `$…$`, `**` or a backtick. |
| S6 | A paragraph holds at most three short sentences and one idea, opened by a bold label naming its job. | `VAULT_NOTE_FORMAT.md` v2, "Short chunks, spaced", `08d8627` | step | Prose paragraphs (outside code, tables, lists and headings) with more than three sentences. |
| S7 | A quantitative figure is drawn by code, with the exact script in a code block directly above it. | `VAULT_NOTE_FORMAT.md` v2, "Figures are drawn by code first", `08d8627` | step | Each `.png` or `.svg` image with no `python` code block among the 6 non-blank lines above it. |
| L1 | Links sit on the word being read; what has no word (a video moment, a scene frame, a second source) goes in one short `🔗 **Where else this lives**` block at the heading's foot, one or two links per kind. | `skills/drip/references/navigation.md`, the `links` control, `4ab0fa4` (2026-10-21); `plans/one-list-of-connections-per-heading.md`, `bef249c` (2026-10-24); `VAULT_NOTE_FORMAT.md` v2, "Links sit on the reading" | step | Each markdown table holding YouTube links that sits outside a `<!-- links: -->` block. |
| L2 | No link inside rendered maths: an equation is linked through the name written beside it, never inside `$…$`. | `skills/drip/references/navigation.md`, "Linking by default", `6ed953f` (2026-10-24) | all kinds | Inline `$…$` spans containing `](` or `\href`. |
| L3 | A word that has a concept card links to the card's `#where-to-learn-this` section, with the card's `<sub>` marks (numbered textbook pages, ▶ videos, ◆ our plans) straight after the link, once per section. | `skills/drip/references/navigation.md`, "Concept cards come first", `6ed953f` | step, ladder, guide | Distinct card words (from `~/.claude/cache/term-index.json`, kind `card`) in the file's prose, less those already card-linked. An upper bound: 15 wrong-sense words (map, rest, port and the like) are excluded, and the rest still need the linker's wrong-sense check. |
| E1 | A walk borrows an interpreter listed in `~/.claude/ENVIRONMENTS.md`; nothing is installed into a playground, and a new shared environment needs a yes. | `skills/plumb/SKILL.md`, "Reuse before install", `9389876` (2026-10-28) | step, ladder, guide, walk map | Lines matching `pip install`, `-m venv`, `conda create`, `uv venv` or `virtualenv`. |
| R1 | The chain's file names its whole-chain objective and outcome as `**Objective (the whole chain):**` and `**Outcome (the whole chain):**`. | `skills/skill-route/references/route-file-template.md`, `407931d` (2026-10-19) | guide | Both lines present. The guide holds the same content as "The goal they serve" and "Where it ends". |
| R2 | Each step names its own `**Objective:**` (with its number), `**Outcome:**` (the file or result held at its close) and `**Serves:**` (its part of the chain's outcome). | `route-file-template.md`, `407931d` | ladder | Per `### N.` step, each of the three lines missing is one miss. Applied because the ladder files play the route file's part (an Achieve line, two lanes, a prompt per step); see batch 9. |
| R3 | Each step carries a learning-path table (`\| # \| \| Open \| Textbook \| Watch \|`) listing the rungs to open first, earliest first. | `route-file-template.md`, `407931d` | ladder | The table header present once per file. The ladder's **Study first** line holds the same sources as prose. |
| Q1 | Line 2 of a coverage file is the stamp `Standard: QUIZ_PREP_FORMAT v1`. | `QUIZ_PREP_FORMAT.md` v1, "The coverage file", `fca5278` (2026-10-18); `STANDARDS.md`, `quiz-prep` row | coverage, solutions | Line 2 equals the stamp exactly. |
| Q2 | A section per question, each with a node table carrying Tier and Min columns. | `QUIZ_PREP_FORMAT.md` v1; `STANDARDS.md` conformance check; `fca5278` | coverage | Per `Q` heading, a `\| Node \| Leans on \| Tier \| Min \|` header present. |
| Q3 | The file ends with a Gaps table, and no question is blocked without being listed there. | `QUIZ_PREP_FORMAT.md` v1; `STANDARDS.md`; `fca5278` | coverage | A heading containing "Gaps" present. |
| Q4 | Every link and anchor resolves. | `STANDARDS.md`, `quiz-prep` conformance check, `fca5278` | coverage, solutions | Each local link's file exists on this machine and each `.md` anchor matches a heading. Links into `/home-mscluster` are skipped, since that path exists only on the cluster. |
| Q5 | A paper question names its PDF page. | `STANDARDS.md`, `quiz-prep` conformance check, `fca5278` | coverage, solutions | Links into `papers/` or `arxiv.org/pdf` carry `#page=`. |

## The score

`✓` meets the rule, a number is the count of misses, and `·` means the rule does not apply to that kind of file. The last column gives, per missed rule, the line of its first miss.

| File | P1 | P2 | P3 | P4 | T1 | T2 | T3 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | L1 | L2 | L3 | E1 | R1 | R2 | R3 | Q1 | Q2 | Q3 | Q4 | Q5 | Misses | Where to look (rule: first line) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `PROMPTS.md` | 1 | ✓ | ✓ | ✓ | · | · | · | · | · | · | · | · | · | · | · | ✓ | 16 | ✓ | 2 | · | · | · | · | · | · | · | **19** | P1: 140, R1: 1 |
| `REBUILD_WALK.md` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | ✓ | · | ✓ | · | · | · | · | · | · | · | · | **0** | none |
| `01-one-expert-from-scratch.md` | ✓ | ✓ | ✓ | ✓ | 1 | ✓ | 3 | 3 | ✓ | ✓ | 8 | ✓ | 22 | 10 | 18 | ✓ | 60 | ✓ | · | · | · | · | · | · | · | · | **125** | T1: 555, T3: 580, S1: 775, S4: 645, S6: 5, S7: 257, L1: 276 |
| `02-composing-and-sampling.md` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 1 | ✓ | ✓ | ✓ | ✓ | ✓ | 12 | 4 | 14 | ✓ | 43 | ✓ | · | · | · | · | · | · | · | · | **74** | T3: 2052, S6: 3, S7: 558, L1: 98 |
| `03-ideal-experts-and-the-seven-kinds.md` | ✓ | ✓ | ✓ | ✓ | 1 | ✓ | 3 | ✓ | ✓ | ✓ | ✓ | ✓ | 11 | 2 | 9 | ✓ | 23 | ✓ | · | · | · | · | · | · | · | · | **49** | T1: 1734, T3: 953, S6: 3, S7: 495, L1: 78 |
| `04-the-parts-world.md` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 4 | ✓ | ✓ | ✓ | ✓ | ✓ | 4 | ✓ | 8 | ✓ | 27 | ✓ | · | · | · | · | · | · | · | · | **43** | T3: 275, S6: 3, L1: 116 |
| `05-the-overlap-dial.md` | ✓ | ✓ | ✓ | ✓ | 1 | 3 | 5 | ✓ | ✓ | ✓ | ✓ | ✓ | 13 | 3 | 7 | ✓ | 10 | ✓ | · | · | · | · | · | · | · | · | **42** | T1: 635, T2: 706, T3: 63, S6: 3, S7: 1309, L1: 53 |
| `ladder/00-setup.md` | 1 | 9 | 1 | 1 | · | · | · | · | · | · | · | · | · | · | · | ✓ | ✓ | ✓ | · | 9 | 1 | · | · | · | · | · | **22** | P1: 37, P2: 39, P3: 37, P4: 37, R2: 22, R3: 1 |
| `ladder/01-one-diffusion-expert.md` | 4 | 21 | 4 | 4 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 11 | ✓ | · | 18 | 1 | · | · | · | · | · | **63** | P1: 42, P2: 45, P3: 42, P4: 42, R2: 34, R3: 1 |
| `ladder/02-composing-two-experts.md` | 7 | 48 | 7 | 7 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 10 | ✓ | · | 25 | 1 | · | · | · | · | · | **105** | P1: 38, P2: 40, P3: 38, P4: 38, R2: 34, R3: 1 |
| `ladder/03-ideal-experts-and-the-seven-kinds.md` | 5 | 35 | 5 | 5 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 7 | ✓ | · | 21 | 1 | · | · | · | · | · | **79** | P1: 36, P2: 38, P3: 36, P4: 36, R2: 32, R3: 1 |
| `ladder/04-the-parts-world.md` | 5 | 35 | 5 | 5 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 8 | ✓ | · | 24 | 1 | · | · | · | · | · | **83** | P1: 38, P2: 44, P3: 38, P4: 38, R2: 34, R3: 1 |
| `ladder/05-the-overlap-dial.md` | 6 | 39 | 6 | 6 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 11 | ✓ | · | 28 | 1 | · | · | · | · | · | **97** | P1: 38, P2: 45, P3: 38, P4: 38, R2: 34, R3: 1 |
| `ladder/05a-reading-stop-projective-composition.md` | 6 | 44 | 6 | 6 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 4 | ✓ | · | 15 | 1 | · | · | · | · | · | **82** | P1: 37, P2: 39, P3: 37, P4: 37, R2: 33, R3: 1 |
| `ladder/06-guidance-as-a-dial.md` | 6 | 40 | 6 | 6 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 4 | ✓ | · | 24 | 1 | · | · | · | · | · | **87** | P1: 40, P2: 45, P3: 40, P4: 40, R2: 38, R3: 1 |
| `ladder/07-a-without-b.md` | 6 | 40 | 6 | 6 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 2 | ✓ | · | 24 | 1 | · | · | · | · | · | **85** | P1: 40, P2: 45, P3: 40, P4: 40, R2: 38, R3: 1 |
| `ladder/08-the-interaction-term.md` | 7 | 48 | 7 | 7 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 7 | ✓ | · | 23 | 1 | · | · | · | · | · | **100** | P1: 40, P2: 45, P3: 40, P4: 40, R2: 36, R3: 1 |
| `ladder/08a-correction-prior-art.md` | 3 | 21 | 3 | 3 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 3 | ✓ | · | 14 | 1 | · | · | · | · | · | **48** | P1: 33, P2: 35, P3: 33, P4: 33, R2: 29, R3: 1 |
| `ladder/09-the-correction-learned-back.md` | 6 | 39 | 6 | 6 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 6 | ✓ | · | 21 | 1 | · | · | · | · | · | **85** | P1: 40, P2: 47, P3: 40, P4: 40, R2: 36, R3: 1 |
| `ladder/10-latent-space-and-vaes.md` | 5 | 35 | 5 | 5 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 3 | ✓ | · | 19 | 1 | · | · | · | · | · | **73** | P1: 36, P2: 38, P3: 36, P4: 36, R2: 32, R3: 1 |
| `ladder/11-the-text-conditioned-testbed.md` | 3 | 21 | 3 | 3 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 4 | ✓ | · | 8 | 1 | · | · | · | · | · | **43** | P1: 33, P2: 37, P3: 33, P4: 33, R2: 31, R3: 1 |
| `ladder/12-the-correction-at-scale.md` | 8 | 62 | 8 | 8 | · | · | · | · | · | · | · | · | · | · | · | ✓ | 8 | ✓ | · | 27 | 1 | · | · | · | · | · | **122** | P1: 48, P2: 54, P3: 48, P4: 48, R2: 44, R3: 1 |
| `.quiz-prep/quiz-0.md` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | ✓ | · | · | · | · | · | ✓ | ✓ | ✓ | ✓ | ✓ | **0** | none |
| `.quiz-prep/quiz-1-1-solutions.md` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | ✓ | · | · | · | · | · | ✓ | · | · | ✓ | ✓ | **0** | none |
| `.quiz-prep/quiz-1-1.md` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | ✓ | · | · | · | · | · | ✓ | ✓ | ✓ | ✓ | ✓ | **0** | none |
| `.quiz-prep/quiz-1-2-solutions.md` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | ✓ | · | · | · | · | · | 1 | · | · | ✓ | ✓ | **1** | Q1: 2 |
| `.quiz-prep/quiz-1-2.md` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | ✓ | · | · | · | · | · | ✓ | ✓ | ✓ | ✓ | ✓ | **0** | none |
| `video-scout/pulse-2026-10-18.md` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | ✓ | · | · | · | · | · | · | · | · | · | · | **0** | none |
| `video-scout/pulse-2026-10-20-phases-2-to-5.md` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | ✓ | · | · | · | · | · | · | · | · | · | · | **0** | none |
| `video-scout/pulse-2026-10-20.md` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | ✓ | · | · | · | · | · | · | · | · | · | · | **0** | none |
| **All files** | 79 | 537 | 78 | 78 | 3 | 3 | 16 | 3 | 0 | 0 | 8 | 0 | 62 | 19 | 56 | 0 | 267 | 0 | 2 | 300 | 15 | 1 | 0 | 0 | 0 | 0 | **1527** | |

The walk map and the three video-scout reports meet every rule that applies to them. Every miss sits in the guide, the ladder, the step files or one solutions file.

## The batches

Ordered by risk first and gain second. Each batch ends by rerunning `python3 -I sweep-score.py .` and checking that its columns read zero for its files. The edit counts are estimates of separate edits; together the batches clear all 1527 misses.

| # | Batch | Files | Rules | Edits (estimate) | Misses cleared |
|---|---|---|---|---|---|
| 1 | Two labels and one stamp | `PROMPTS.md`, `.quiz-prep/quiz-1-2-solutions.md` | R1 (rename "The goal they serve" and "Where it ends" to the whole-chain Objective and Outcome lines), Q1 (move the stamp from line 3 to line 2) | 3 | 3 |
| 2 | Concept-card links, by the linker | the five step files, `PROMPTS.md`, `ladder/01` to `ladder/12` | L3, run through `concept-cadence` (its `link-concepts.py`): a dry run first, link-only writes, each wrong-sense word skipped on its card | up to 267 links, no prose | up to 267 |
| 3 | Small repairs in the step files | `01-one-expert-from-scratch.md` (T1 at line 555, S1 from line 775, S4 from line 645), `03-ideal-experts-and-the-seven-kinds.md` (T1), `05-the-overlap-dial.md` (T1, T2 from line 706) | T1, T2, S1, S4 | 17 | 17 |
| 4 | Prompt rewrite, the trial | `ladder/00-setup.md` (1 prompt), `ladder/01-one-diffusion-expert.md` (4 prompts), and the guide's "The prompt shape" paragraph (line 140) | P1, P2, P3, P4: each field-labelled prompt rewritten into the seven sections, every settled fact kept, its `added:` and `dropped:` lines updated | 6 | 46 |
| 5 | Prompt rewrite, the rest | `ladder/02` to `ladder/12`, with `05a` and `08a` (73 prompts) | P1, P2, P3, P4, in the form batch 4 settled | 73 | 726 |
| 6 | One connections block per heading | the five step files (56 video tables: 18, 14, 9, 8 and 7) | L1: each video table becomes a `🔗 **Where else this lives**` block at its heading's foot, at most two videos a heading, the rows left out named in the walk's `video-scout/` reports so none is lost | 56 | 56 |
| 7 | Short chunks and figure scripts | the five step files | S6 (62 paragraphs over three sentences, split), S7 (19 figures with no script above, each given the script that draws it) | 81 | 81 |
| 8 | Ordinary toys first | `01` (3 steps), `02` (1), `03` (3), `04` (4), `05` (5) | T3: a new everyday toy per step, its numbers computed by script, and the present project toy moved under a "Run it back on your work" heading | 32 | 16 |
| 9 | Route-file shape on the ladder | `ladder/00` to `ladder/12` (15 files) | R2 (Objective and Serves per step, Outcome where missing), R3 (a learning-path table per layer from `journey_lookup.py --path`) | 315 | 315 |

**Batch 9 needs a decision before it starts.** The route-file template rules files named `plans/route-*.md` that `skill-route` writes. The ladder files are scored against it because they play that part, with an Achieve line, two lanes and one prompt per step. If they are not route files, R2 and R3 become not applicable, the batch is dropped, and the total falls from 1527 to 1212.

**Batch 8 carries the most risk.** Each new toy brings new worked numbers, which have to print from a script the way the present ones print from `python -m rebuild.walk_numbers`, and the quizzes in `.quiz-prep/` quote the present toys.

<details>
<summary>Changes left out, and why</summary>

## Changes left out, and why

**Walk behaviour in chat.** `skills/drip/references/navigation.md` gained the `draw`, `plumb`, `scene`, `keep`, `links` and `learn` controls, the "While it runs" block, the `path` line, pillar links in a walk's messages, the mode-index lookup for side routes, and the handling of `hld-cadence` questions. `skills/drip/SKILL.md` and `skills/drip-execute-plan/SKILL.md` gained `draw` and `plumb` and the rule that a textbook shelf page counts as teaching a word in a reply. All of these govern what a walk says or dispatches, not what these files hold. Two of them do reach files and are scored: the `links` control (L1) and the concept-card rule (L3).

**The hld-diagram skill.** Its `SKILL.md` and references (`walk.md`, `scene-for-another-system.md`, `design-from-docs.md`, `compose-page.md`, `registry.json`, the scene template) govern drawing, scenes and the HLD gallery. No rule there applies to a guide, a ladder, a step file or a coverage file.

**plumb's reply linking and its start from a scene.** Both govern a plumb walk's chat replies and the prompts a scene writes.

**PILLAR_CADENCE_FORMAT.md and RUNBOOK_FORMAT.md.** The queue claim and runbook pictures; this folder holds no queue file and no runbook.

**The vault-note stamp and the vault's own textbook copies.** `standard: vault-note v2` in frontmatter binds vault notes only, and so does the rule that textbook links go to the vault's copy. Files outside the vault use the learning tree's path by `FUNDAMENTAL_LINKS_FORMAT.md`, which these files already do. That rule was scored as L4 at first and then dropped, since every one of its misses was a correct link.

**Three student-page rules with no mechanical check.** "Highlight what the step uses", "Make the image speak" and "One picture, several annotated copies" (`STUDENT_PAGE_FORMAT.md`, `ce671f4`) need someone to look at the table or figure. They are left for the reading pass in batches 3 and 7.

**The triad's "every term met on the way is linked".** It overlaps L3 and was not scored separately.

**The figure style rule in OUTPUT_GUIDE.md** (`figkit/FIGURE_STYLE.md`, `4956544`) governs the images themselves, not the markdown that embeds them.

**Files that did not change in the window.** `~/.claude/CLAUDE.md` has no commit since 2026-10-18, and `FUNDAMENTAL_LINKS_FORMAT.md` last changed on 2026-10-15. `QUIZ_PREP_FORMAT.md` was committed earlier on 2026-10-18 (`fca5278`) than the hour `git log --since=2026-10-18` starts from, so the log does not list it; its rules are scored because the `quiz-prep` row of `STANDARDS.md` binds the coverage files to it.

</details>

<details>
<summary>How the counts were taken, and where they are soft</summary>

## How the counts were taken

**One script, read-only.** `sweep-score.py` beside this file computes every number in the score table from the files as they stood on 2026-10-28. It writes nothing unless given an output path.

**What counts as a prompt.** A `text` code block of five or more non-blank lines that opens on a skill line (`/drip`, `/drip-execute-plan`), opens on "In the background", or carries the old field labels. Box-drawn flow diagrams are excluded. One-line invocations are not counted, and there are 78 prompts in the ladder files.

**Upper bounds.** L3 counts distinct card words per file before the linker's wrong-sense check, and the rule asks for a link once per section, so the true count can move either way. S1 counts every unbroken display block, including ones that bring no new term.

**Rules applied by extension.** S1 to S7 come from the vault-note and student-page formats. They are applied to the step files because `VAULT_NOTE_FORMAT.md` v2 takes its section form from Step 4 of `01-one-expert-from-scratch.md`, and the `keep` control files a piece into a step file in that breakdown style. R1 to R3 are applied to the guide and ladder as batch 9 says.

**Q4 skips two kinds of link.** Web links are not fetched, and links into `/home-mscluster` are skipped because that path exists only on the cluster.

</details>
