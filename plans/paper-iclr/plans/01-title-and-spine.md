# 🧭 The title and the order the story is told in

## Description
Commit one title, and write down which claim each section carries and why it
comes where it does.

## Purpose
Every section inherits from the spine. Writing the method before deciding
whether the paper leads with the measurement or the fix produces prose that has
to be rewritten. Serves DoD 4.

## Goal
A title in the `.tex`, and `paper/iclr/SPINE.md`: one line per section naming
the single claim it carries, in reading order.

## Environment Facts This Plan Depends On
- The title is the `\title{...}` line in `paper/iclr/iclr2027_conference.tex`
  (currently the stock "Formatting Instructions for ICLR 2027"). SPINE.md is a
  new file in `paper/iclr/`, prose only, not part of the build. See
  `docs/ENVIRONMENT.md`, "Paper: where the LaTeX lives and how it is built".

## Tasks
- [ ] ⚠️ write the one-sentence claim of the paper, the sentence a reader should
      be able to repeat afterwards
- [ ] ⚠️ decide the lead: does the paper open on the measurement (r_t is real,
      small, shared) or on the fix (a LoRA transfers to unseen pairs)
- [ ] ⚠️ write SPINE.md: one line per section, each naming its single claim
- [ ] ⚠️ commit a title in the `.tex` (a working title is fine; it is cheap to
      change and expensive to keep deferring)

## Success/Failure Outcomes
- **the spine**
  - Success: each section line names one claim, and the claims read in an order
    where none depends on a later one.
  - Failure: a section carries two claims, or the results section is doing work
    the method should have done. Split the section rather than widening the line.

## Recommended skill
▶ `/restyle` ✅ once the title candidates exist: match them against an ICLR
   exemplar you paste, without changing what they claim.

## Engagement Instructions
```bash
cat paper/iclr/SPINE.md                  # expect one claim per section, in order
grep -c "Formatting Instructions" paper/iclr/iclr2027_conference.tex   # expect 0
grep "\\\\title" paper/iclr/iclr2027_conference.tex   # then read it: is it OUR title?
```
The count is the real gate: it fails while the stock title is still in place. The
second line is for your eye, since no command can tell a good title from a bad one.
