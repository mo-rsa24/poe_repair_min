# Paper-ICLR

## Mission
Write and submit the ICLR 2027 paper arguing that PoE's compositional failure is a
measurable quantity, not a mystery. Multiplying two predictions asks for one image
that is both concepts at once; the sentence means two things side by side. The gap
between those readings is `r_t`, cached and concrete. It is small, shared across
pairs, and concentrated in a narrow noise band, which is why a rank-8
cross-attention LoRA learns it once and fixes pairs it never saw, without ever
encoding the joint prompt. This scope owns the manuscript: the compile, the title,
the story order, the figure layout, and every word. It reads figures from
`interaction-term` and `animals-compose-transfer` and produces none of its own.

## Objectives
(Direction. Each phase is a state the manuscript is in.)
1. **Buildable** — the template compiles to a PDF from this repo, with the bib
   wired and a figure-include convention that survives figures arriving later.
2. **Rough draft** — a PDF with the real title, a locked story order, the figure
   layout decided, method and intro written, and honest placeholders where the
   numbers are still owed.
3. **Evidenced** — placeholders replaced by real figures and real numbers as the
   two result scopes deliver them, in the order the figure layout set.
4. **Defensible** — the honesty caveats written in, the print-gated claims through
   their /pressure-test passes, the prose restyled to a consistent voice.
5. **Submitted** — anonymised, within the page limit, checklist cleared, uploaded.

## Goals
(Checkpoints. Measurable.)
1. Build: `tectonic paper/iclr/iclr2027_conference.tex` produces a PDF with no
   unresolved references, run from a clean checkout.
2. Title and spine: one title committed in the `.tex`, and a written story order
   naming which claim each section carries.
3. Layout: every figure slot in the paper named, each tagged with its owning scope
   and plan file, and marked "have it" or "owed".
4. Draft: method, intro, and abstract are real prose that a reader unfamiliar with
   the project can follow; experiments and results are structured skeletons with
   named placeholders, not prose.
5. Evidence: each placeholder closed by the figure its layout entry named, cited
   with the checkpoint it came from.
6. Defence: both /pressure-test verdicts folded in; the three honesty caveats
   present in the text; the mechanism section written to match `interaction-term`
   plan 02's verdict.
7. Submission: anonymous copy compiles, page limit met, OpenReview submission
   confirmed.

## Expected Outcome
An ICLR 2027 submission. At minimum, a paper whose causal claim (dose, direction,
timing) is figure-backed and whose transfer claim is one honest held-out number
cited with its checkpoint. At most, that plus the universality evidence
(cross-model replication, sampler sweep) as a second contribution. The fallback
if the runs do not land in time is a workshop submission with the same spine and
a narrower claim, decided at the phase-3 boundary rather than at the deadline.

## Definition of Done
1. ✅ Toolchain confirmed: `tectonic` builds this template on the cluster
   (verified 2026-08-05, 73 KB PDF from the stock template)
2. ⚠️ Template de-stubbed: the sample title, the Cranberry-Lemon authors, and the
   23-line stub bib replaced with real content [inferred; the template is
   currently unmodified from the ICLR distribution]
3. ⚠️ Figure-include convention written down: one path rule that works for figures
   that do not exist yet [inferred; figures live in two other scopes]
4. ⚠️ Title committed and the story order written.
5. ⚠️ Figure layout table: every slot, its owning scope and plan, have-it or owed.
6. ⚠️ Method, intro, and abstract drafted by hand.
7. ⚠️ Experiments and results sections structured with named placeholders.
8. ⚠️ Every placeholder closed with its real figure and number.
9. ⚠️ Mechanism section and the three honesty caveats written (moved here from
   `interaction-term` plan 09).
10. ⚠️ Both /pressure-test verdicts folded into the wording.
11. ⚠️ /restyle pass over the full draft against a named ICLR exemplar.
12. ⚠️ Anonymous build compiles, page limit met, submitted.

## Reads From (produces no figures of its own)
- `plans/interaction-term/` — the causal figure cascade (plan 10) and the
  mechanism verdict (plan 02). Its DoD item 7 also owns the 100k transfer number
  this paper cites.
- `plans/animals-compose-transfer/` — the transfer figures F2–F5 (plan 05) and the
  pooled held-out read (plan 03a).

## Sub-Scopes
(None yet.)

## Plans
(Phase 1, the rough draft. Later phases get their plan files when phase 1 lands.)
- ◑ 00-compile-the-template.md — the build works; de-stubbing and the figure-path
  rule remain (DoD 1-3)
- ⚠️ 01-title-and-spine.md — gates 03, 04, 05 (DoD 4)
- ⚠️ 02-figure-layout.md — the phase-1 deliverable; its owed column is the run
  order handed back to the two result scopes (DoD 5)
- ⚠️ 03-draft-method-and-intro.md (DoD 6)
- ⚠️ 04-abstract.md — after 01 and 03 (DoD 6)
- ⚠️ 05-results-skeleton.md — placeholders, not prose (DoD 7)
- ⚠️ 06-mechanism-and-caveats.md — moved here from `interaction-term` plan 09 on
  2026-08-05; blocked on that scope's plans 02 and 09 (DoD 9)

## Running order

This scope keeps no order of its own. The single flat order across every scope
and level is the `## Running order` table in the repo root `MASTER_PLAN.md`.

## Environment Context
See `docs/ENVIRONMENT.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in this scope.

Scope-specific: no system LaTeX exists on the cluster nodes (`pdflatex`,
`xelatex`, `latexmk`, `bibtex` are all absent from PATH). The build path is
`tectonic` at `/home-mscluster/mmolefe/.local/bin/tectonic`, which fetches
packages on demand and therefore needs network access on first build.
