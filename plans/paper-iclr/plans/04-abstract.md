# 📄 The abstract

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
- Depends on plan 01 (the spine) and plan 03 (method and intro) landing first.
- Numbers quoted here must be ones that exist. Today that is the pooled held-out
  read (out_out 0.96 at step 60k, from `animals-compose-transfer` plan 03a), and
  it is always cited with its checkpoint. Anything from `interaction-term` is
  owed until its runs land.

## Tasks
- [ ] ⚠️ draft the abstract: the problem, the measurement, the fix, the transfer
      result, the limit
- [ ] ⚠️ check every number in it exists and is cited with its checkpoint
- [ ] ⚠️ /restyle pass against a named ICLR exemplar
- [ ] ⚠️ check it against the ICLR word or line budget

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
