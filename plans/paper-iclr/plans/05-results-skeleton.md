# 🧱 Experiments and results: the frame, not the prose

## What this asks, in one line
Write the results section with every number as XX and every figure as its reserved slot, so 'what should I run next' becomes 'what is still blank'.

## Description
Build the experiments and results sections as structure with named placeholders.
No results prose is written in phase 1, because most of the numbers do not exist.

## Purpose
A skeleton makes the shape of the argument visible and shows exactly which
missing number blocks which paragraph. Writing results prose against numbers
that do not exist produces text that has to be thrown away. Serves DoD 7, and
sets up DoD 8 for a later phase.

## Goal
Experiments and results sections present in the `.tex`, each subsection carrying
a named placeholder that says which run closes it.

## Environment Facts This Plan Depends On
- Sections go in `paper/iclr/iclr2027_conference.tex`; build with
  Ctrl+Shift+P → Build with recipe → `tectonic`. See `docs/ENVIRONMENT.md`,
  "Paper: where the LaTeX lives and how it is built".
- What exists: the pooled held-out read (out_out 0.96 at step 60k) from
  `animals-compose-transfer` plan 03a, and the negative controls already
  reported in the root scope (group-A, internal-force).
- What is owed: `animals-compose-transfer` owes the leave-one-pair-out run, the
  mixed-pool contrast, and the 70k-100k scoring. `interaction-term` owes plans
  01-11 (only plan 00, the instruments, is complete).
- The paper's transfer number is always cited with its checkpoint. That rule
  comes from `interaction-term` DoD item 7, which names plan 03a as its owner.
- Placeholders depend on the placeholder-figure macro from plan 00. Without it,
  an owed figure breaks the build.

## Tasks
- [ ] write the experiments section structure: one subsection per experiment
      the spine names, each stating the question and the setup
- [ ] write the results structure: one subsection per claim, each holding a
      named placeholder rather than prose
- [ ] label each placeholder with the run that closes it, matching the owed
      rows in FIGURES.md
- [ ] write the one results paragraph that CAN be written today: the pooled
      held-out transfer read, cited with its checkpoint

## Success/Failure Outcomes
- **the skeleton**
  - Success: the PDF builds, and every gap is visible and labelled with the run
    that fills it.
  - Failure: a placeholder with no named run behind it. That is an unowned gap;
    either name the run or cut the subsection.
- **the transfer paragraph**
  - Success: the number appears with its checkpoint and its held-out condition
    stated.
  - Failure: the number quoted bare. A transfer number without its checkpoint is
    the specific dishonesty this project has already committed to avoiding.

## Next

1. `/draft-section results`: one piece per register slot, in register order; its `numbers`
   control re-derives every figure from its review file before any compile. Or by hand:
   one subsection stub per register slot (F1 to F8 in `paper/iclr/figures.md`): the slot's claim
   as the topic sentence, the placeholder-figure macro from plan 00, every number as XX.
2. Build the PDF and read it: the blank spots ARE the remaining run order.
3. When a slot turns fillable (plan_pulse reports it at session start), replace its XX numbers
   from the review file it names, never from memory.

## Engagement Instructions
```bash
cd paper/iclr && /home-mscluster/mmolefe/.local/bin/tectonic iclr2027_conference.tex
ls -la iclr2027_conference.pdf     # expect a PDF even with placeholders present

# the two counts should match: every owed figure has a placeholder, and every
# placeholder is an owed figure. A mismatch means an unowned gap either way.
grep -c "PLACEHOLDER" iclr2027_conference.tex
grep -c "owed" FIGURES.md
```
Manual check, by eye: open the PDF and confirm each placeholder renders as a
labelled grey box naming its slot, not as a broken reference or a blank.
