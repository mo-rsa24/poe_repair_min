# 📄 The abstract

**Step 22 of 22.** Waits on steps 20 and 21. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

| Step | Plan | Status |
|---|---|---|
| 21 | [writing-06-mechanism-and-limitations](writing-06-mechanism-and-limitations.md) | ⚠️ |
| **22** | **this plan** | **⚠️** |

## What this asks, in one line
Write the abstract last, from the spine and the finished sections, because an abstract written first describes the paper you hoped for.

## Description
Write the abstract, after the spine is locked and the method and introduction
exist.

## Purpose
Written first, an abstract locks a story the figures then have to serve. Written
after the spine, it reports a story already decided. It is a separate plan from
03 only because of that ordering. Serves DoD 6.

## Goal
An abstract in the `.tex` that states the claim, the evidence, and the honest
limit, inside the ICLR word budget.

## Environment Facts This Plan Depends On
- The abstract goes in `paper/iclr/iclr2027_conference.tex`, in the existing
  `\begin{abstract}` block (currently stock ICLR filler text). Build with
  Ctrl+Shift+P → Build with recipe → `tectonic`. See `docs/ENVIRONMENT.md`,
  "Paper: where the LaTeX lives and how it is built".
- Depends on plan 01 (the spine) and plan 03 (method and intro) landing first.
- Numbers quoted here must be ones that exist. Today that is the pooled held-out
  read (out_out 0.96 at step 60k, from `does-the-fix-reach-unseen-pairs` plan 03a), and
  it is always cited with its checkpoint. Anything from `does-the-correction-cause-composition` is
  owed until its runs land.

## Tasks
- [ ] draft the abstract: the problem, the measurement, the fix, the transfer
      result, the limit
- [ ] check every number in it exists and is cited with its checkpoint
- [ ] /restyle pass against a named ICLR exemplar
- [ ] check it against the ICLR word or line budget

## Next

1. `/draft-section abstract`. It refuses to run before SPINE.md exists, and its numbers piece
   stays blocked while any cited slot is unfilled, which is the enforcement of "write it last".

## Success/Failure Outcomes
- **the abstract**
  - Success: a reader knows what was claimed, what was shown, and what was not,
    without reading further.
  - Failure: it promises evidence the paper does not yet carry. Weaken the
    sentence, do not defer the check.

## Recommended skill
▶ `/restyle` ✅ against a pasted ICLR abstract whose voice you want. It preserves
   every claim exactly, and flags rather than edits if a style change would move
   one.

## Engagement Instructions
```bash
cd paper/iclr && /home-mscluster/mmolefe/.local/bin/tectonic iclr2027_conference.tex
grep -A5 "begin{abstract}" iclr2027_conference.tex   # expect real prose, not the stock text
```
