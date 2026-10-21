# Read-ahead index: My Eye, My Brain

The walking session's own read of the whole proposal, done by hand because the document is about
3,300 words. Every statement carries a label: **quoted** (it is in the proposal, with the line in
`paper/proposal.md`), **ran** (a number from a script that executed, with the file), **recalled**
(from memory, unverified).

## The paper

| Field | Value | Label |
|---|---|---|
| Title | My Eye, My Brain: Development and Validation of an Accessible AI-Enabled Retinal Screening System for the Early Detection of Ocular and Systemic Disease in Underserved South African Populations | quoted, lines 3 to 5 |
| Author | Samuel Sephiri (MBA, BSc Honours, BSc Mathematical Sciences) | quoted, lines 9 to 11 |
| Kind | a PhD research proposal, not a paper: it reports no results | quoted, line 1 |
| Venue | University of Pretoria or University of the Witwatersrand; field, faculty, supervisor and date are all still to be confirmed | quoted, lines 17 to 25 |
| Files | `paper/my-eye-my-brain-proposal.docx` (the original, 33,575 bytes) and `paper/proposal.md` (pandoc's conversion, 267 lines) | ran |
| LaTeX source | none: the document is a Word file, so words come from the conversion and there are no printed equation or figure numbers | ran |
| Code | none released and none referred to | ran: no link in the text |
| Figures and equations | none; one table (the timeline, section 11) | ran: `checks/count_chunks.out` |

**What the proposal does not contain** (ran: searches of `paper/proposal.md`, 2026-10-02)

- No accuracy target. "Clinically acceptable" appears at lines 135 and 155 and is never given a number.
- No sample size, no count of patients, images or sites.
- No cost figure for the device, in any currency.

## The sections

Counts are ran, by `checks/count_chunks.py`, output in `checks/count_chunks.out`. Sentence counts
are approximate. The abstract is walked one sentence per chunk (8), so the walk has 65 chunks:
8 abstract sentences, 48 body chunks, 9 reference entries.

| Printed section | File | Paragraphs | List items | Tables | Sentences | Chunks |
|---|---|---|---|---|---|---|
| 1. Abstract | `00-abstract.md` | 3 | 0 | 0 | 8 | 8 (by sentence) |
| 2. Introduction and Background | `02-introduction.md`, not written | 3 | 0 | 0 | 8 | 3 |
| 3. Problem Statement and Research Gap | not written | 2 | 3 | 0 | 8 | 5 |
| 4.1 Aim | not written | 1 | 0 | 0 | 1 | 1 |
| 4.2 Objectives | not written | 0 | 5 | 0 | 5 | 5 |
| 5.1 Primary research question | not written | 1 | 0 | 0 | 1 | 1 |
| 5.2 Secondary research questions | not written | 0 | 4 | 0 | 4 | 4 |
| 6.1 to 6.5 Literature Review | not written | 5 | 0 | 0 | 16 | 5 |
| 7. Significance and Contribution | not written | 1 | 0 | 0 | 4 | 1 |
| 8, 8.1 to 8.3 Research Design and Methodology | not written | 5 | 0 | 0 | 12 | 5 |
| 9. Population, Sampling, and Data Sources | not written | 1 | 0 | 0 | 3 | 1 |
| 10. Ethical Considerations | not written | 0 | 5 | 0 | 5 | 5 |
| 11. Research Timeline | not written | 1 | 0 | 1 | 2 | 2 |
| 12. Translation and Impact Pathway | not written | 1 | 0 | 0 | 2 | 1 |
| 13. Scope and Limitations | not written | 0 | 3 | 0 | 3 | 3 |
| 14. Preliminary Resource Requirements | not written | 1 | 5 | 0 | 6 | 6 |
| References (preliminary) | not written | 9 | 0 | 0 | | 9 |

Each section's file is written by the walking session before that section opens.

## The claim list

One row per abstract sentence. The full blocks are in `00-abstract.md`.

| # | First words (quoted, `paper/proposal.md`) | Pays off at | Leans on | Test on the object |
|---|---|---|---|---|
| 1 | "This study proposes the design, development, and validation" (line 87) | sections 4.1 and 8 | fundus imaging; "low-cost"; "early detection" | none: a framing sentence |
| 2 | "The research addresses two closely coupled problems" (line 87) | section 3 gaps 1 and 3; sections 6.2 and 6.3 | generalise; imaging hardware | stage 3: one model's table on two populations, and whether the difference is larger than the noise in either |
| 3 | "The study is structured as two interdependent research streams." (line 89) | section 8, which names three phases | | none: a framing sentence |
| 4 | "The first involves the curation, fine-tuning, and clinical validation" (line 89) | objectives 1 and 2 (4.2); section 8.1 | fine-tuning; benchmark datasets; clinical validation | stage 3: a baseline model and a fine-tuned model on the same patients |
| 5 | "The second involves the design and prototyping" (line 89) | objective 3; section 8.1 paragraph 2; section 6.3 | fundus imaging device; non-specialist staff | stage 3, in part: the share of photographs too poor to grade. The optics are out of reach |
| 6 | "The two streams are validated jointly through an integrated field study" (line 89) | objective 4; section 8.2; section 3 gap 2; section 6.2 | curated datasets; imaging workflows | stage 3: the same patients photographed by two cameras |
| 7 | "The study further maps its research outputs" (line 91) | objective 5; sections 6.5, 8.3 and 12 | SAHPRA; AI/ML-enabled medical device | stage 4: the subgroup table the regulator asks for |
| 8 | "Expected contributions include a validated, population-appropriate" (line 91) | sections 7 and 13 | validated; population-appropriate | none new: its state follows sentences 4, 5 and 6 |

## The object

**The proposal's own smallest experiment** (quoted, section 8.2, line 185): "Images captured with
the prototype device will be evaluated against expert human grading (reference standard) ...
Performance will be assessed using standard diagnostic metrics (sensitivity, specificity, AUC),
with subgroup analysis by relevant demographic and imaging-condition variables".

**The object proposed from it**: one screening day at one clinic. 100 adults are photographed.
An expert grader says which of them have disease worth referring, and the model says which it
would refer. Every number in it is invented by the walk, because the proposal gives none.

**Why it passes the three tests**

- Computable on paper in under five minutes: every quantity is a count out of 100, 20 or 80.
- Drawable: a two-by-two grid of counts, and later a row of patients sorted by the model's score.
- Contains what makes the proposal hard: a clinic sees few diseased patients, so the headline
  number is a fraction with a small denominator, and every comparison the proposal promises
  (population against population, camera against camera, subgroup against subgroup) is a
  difference between two such fractions.

**The steps up to it**: a list of 100 (expert, model) pairs; the same list counted into a
two-by-two table; the two fractions read off the table.

**The stages**

| Stage | The object | What it reaches |
|---|---|---|
| 1 | one clinic day as a two-by-two table: 100 patients, 20 with disease by the expert's grade | "clinically acceptable diagnostic accuracy" (5.1); sensitivity and specificity (8.2) |
| 2 | the same 100 patients, each with the model's score from 0 to 1 and a cut-off that moves | AUC (8.2); "clinically acceptable" as a choice of cut-off |
| 3 | two tables for the same patients: two cameras, or two models, or two populations, plus a count of photographs too poor to grade | abstract sentences 2, 4, 5 and 6; section 3 gaps 1 and 2; secondary questions 1 and 3 |
| 4 | each table split into subgroups, with the range of values each fraction could plausibly take, and the number of patients needed to narrow it | objective 4; section 6.5's subgroup requirement; section 9's power calculation |

**Out of reach, with the reason**

- Every statement about the published literature (sections 2 and 6): the cited papers are not in
  the folder. Each is trusted, or sent out with `route` to fetch the reference.
- The device's optics (section 6.3, secondary question 2): the proposal gives no design
  parameter to compute with.
- The regulatory statements (section 6.5): the SAHPRA document is not in the folder.
- A trained model on real images: no data, no code, and the proposal reports no results.

Answer keys for the object's pieces are written to `checks/` and run once the reader approves
the object, since their numbers depend on the approved counts.

## The flagged terms

Used before being taught, or never taught. Standing in the reader's tree is filled in by the map.

| Term | Taught later, or assumed | Label |
|---|---|---|
| fundus imaging, fundus camera | assumed throughout | quoted, first at line 89 |
| diabetic retinopathy, glaucoma | named at line 95, never described | quoted |
| fine-tuning | assumed | quoted, line 89 |
| generalise | assumed | quoted, line 87 |
| public benchmark datasets | named at line 179 (EyePACS, APTOS, IDRiD, Messidor, AIROGS, RFMiD, BRSET), never described | quoted |
| sensitivity, specificity, AUC | assumed | quoted, line 185 |
| reference standard | assumed | quoted, line 185 |
| subgroup performance analysis | assumed | quoted, lines 127, 167, 185 |
| power calculation | assumed | quoted, line 193 |
| purposive/convenience sampling | assumed | quoted, line 193 |
| oculomics | glossed at lines 97 and 163 | quoted |
| SAHPRA, Class C or D | glossed at line 167 | quoted |
| POPIA | spelled out at line 199, not described | quoted |
| "lightweight" model | appears once, in the primary question (line 135), nowhere else | ran: search |
| "multi-class" detection | appears once, in objective 2 (line 123); section 6.1 speaks of "multi-label" frameworks | ran: search |

## The sentences that assume experience

| Sentence (first words, line) | What the author has seen that the reader may not have |
|---|---|
| "a smartphone-based, low-cost fundus imaging device" (89) | what a fundus photograph looks like from a desktop camera and from a phone held to a lens |
| "evaluated against expert human grading (reference standard)" (185) | how a grader scores a photograph for diabetic retinopathy, and how often two graders disagree |
| "a well-documented failure mode in which AI systems validated on curated datasets underperform" (89) | one real deployment where this happened; the references list Beede et al. 2020 (line 257) |
| "Class C or D" (167) | how a medical-device regulator sorts devices by risk |
| "A purposive/convenience sampling approach" (193) | how patients are recruited at a clinic, and what that does to who ends up in the sample |
| "in-store pharmacy wellness clinics" (225) | what a screening visit at a retail pharmacy clinic involves |

## Slips and internal disagreements found on the read-ahead

Each is raised in the round where it sits, never before.

| Where | What the text says | What does not fit | Label |
|---|---|---|---|
| Section 11, line 209 | "the two research streams described in Section 8" | section 8 describes three phases; the two streams are named in the abstract | quoted |
| Section 11 table, lines 218 and 220 | Year 3 is "H1 2029 -- H2 2029"; Year 3.5 to 4 is "H2 2029 -- 2030" | H2 2029 is in both rows | quoted |
| Sections 4.2 and 3 | the objectives are numbered 4 to 8 in the conversion | the list continues the numbering of section 3's three gaps; whether Word displays 1 to 5 is unknown | ran: pandoc conversion |
| Table of contents, lines 29 to 83 | every entry has page number 1 | the field was not updated; Word may redraw it on opening | ran: pandoc conversion |
| Abstract line 87 against section 6.2 line 155 | models trained on non-African data "may not generalise reliably" | the proposal's own review reports a Singapore-trained model reaching "clinically acceptable sensitivity and specificity" on a Zambian population | quoted |
| Title, line 5, against section 13, line 233 | "Early Detection of Ocular and Systemic Disease" | systemic and brain health are "not a core deliverable of this thesis" | quoted |
| Line 95 against lines 89 and 123 | age-related macular degeneration is one of three named diseases | the model targets diabetic retinopathy, glaucoma and "related retinal pathology" | quoted |
| References, lines 255 and 257 | Xia et al. 2024 and Beede et al. 2020 are listed | neither name appears in the body | ran: search |
| Section 6.4, line 163, and section 6.3, line 159 | "a 2026 analysis in The Lancet Digital Health"; "Other published work" | neither has an entry in the references | ran: read of lines 251 to 267 |
| Abstract line 91 | "an openly documented low-cost device design" | "openly" appears nowhere else | ran: search |
| Section 11 table against section 8.1 | ethics approval for the field study is in Year 2 | new South African patient data is collected in phase 1, and the table's Year 1 lists "ethics application" and "data assembly" together | quoted |
