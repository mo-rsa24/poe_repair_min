# ✍️ Method and introduction, written by hand

## What this asks, in one line
Write the two sections that depend on no unfinished run: the method (settled) and the introduction (follows from the spine).

## Description
Write the two sections that do not depend on unfinished runs: the method, and
the introduction.

## Purpose
The method is settled (cache r_t, train a rank-8 cross-attention LoRA on it,
inject at inference without ever encoding the joint prompt), so it can be
written today. The introduction follows from the spine. Together they are most
of the paper's non-results prose. Serves DoD 6.

## Goal
Both sections present in the `.tex` as real prose, building to a PDF.

## Environment Facts This Plan Depends On
- All prose goes in `paper/iclr/iclr2027_conference.tex`. The `.sty`, `.bst`,
  and `natbib`/`fancyhdr` files are the ICLR distribution and are not edited.
  Check `math_commands.tex` before defining a macro; it likely exists already.
  See `docs/ENVIRONMENT.md`, "Paper: where the LaTeX lives and how it is built".
- Build to check the prose renders: Ctrl+Shift+P → Build with recipe →
  `tectonic` (the default recipe fails, no `latexmk` here), or run tectonic in
  the terminal.
- Prose is written by hand. `/restyle` is the only skill pass, and it runs after
  a section exists, never as a drafter.
- The glossary at the bottom of the root `MASTER_PLAN.md` holds the project's
  agreed plain definitions (PoE, chimera, Mono-free, r_t, lambda, the crossbar).
  Reuse that wording so the paper and the plan tree do not drift apart.

## Tasks
- [ ] write the method: the cached residual, the rank-8 cross-attention LoRA,
      the injection at inference, and why it is Mono-free
- [ ] write the introduction against SPINE.md, ending on the contributions
- [ ] write the related-work paragraph placement (it may be a section or fold
      into the intro; the spine decides)
- [ ] /restyle pass over both, against a named ICLR exemplar

## Next

1. `/draft-section method`: drafted from the design plans piece by piece, compiled into the tex
   with the diff shown each time. The map lives in `paper/iclr/DRAFT_MAP.md` and survives
   session breaks.
2. `/draft-section intro`, claim by claim against SPINE.md; its `cite` control pulls from the
   reading register and cues /paper-scout where the register is empty.
3. `/restyle` against a named ICLR exemplar, after both sections exist.

## Success/Failure Outcomes
- **the method section**
  - Success: a reader outside this project could reimplement the correction from
    it. Every symbol is defined where it first appears.
  - Failure: it describes what was run rather than what the method is. Move the
    run details to experiments.
- **/restyle**
  - Success: the voice is consistent and no claim changed.
  - Failure: it flags that a style change would force a claim to shift. Keep the
    claim, change the style.

## Recommended skill
▶ `/restyle` ✅ after each section is drafted: paste an ICLR paper whose voice
   you want, and let it match register without touching the claims.

## Engagement Instructions
```bash
cd paper/iclr && /home-mscluster/mmolefe/.local/bin/tectonic iclr2027_conference.tex
grep -c "section{Method\|section{Introduction" iclr2027_conference.tex   # expect 2
```
