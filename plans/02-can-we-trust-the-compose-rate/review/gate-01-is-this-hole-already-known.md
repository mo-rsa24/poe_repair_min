# 🔍 Review: has someone already said this?

**Unanswered.** This file judges
[../plans/gate-01-is-this-hole-already-known.md](../plans/gate-01-is-this-hole-already-known.md),
the check that decides whether the rest of this scope gets built. `idea-01` and `gate-02` do not
start until the questions below are answered. `instrument-01` runs regardless.

## Recommended prompt (to run the search)

```
/paper-scout metrics that count separate instances of each named concept in a generated image
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/gate-01-is-this-hole-already-known.md) | the claim to write down first, and the benchmarks to search |
| **this file** | **the verdict: already known, said informally, or not addressed** |
| [what it gates](idea-01-what-the-current-benchmarks-score.md) | the bake-off, which does not start until this returns |
| [what reads it](gate-02-promote-or-close.md) | the decision that writes the scope's ending |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The pre-registered bar](#the-pre-registered-bar)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **Presence metric**: one that asks an image "is there a cat? is there a dog?" and scores yes
  or no per concept. A cat-dog fusion answers yes to both, because it really does carry cat
  features and dog features.
- **Count metric**: one that counts how many distinct things are in the picture. Ours is a count
  metric. Two dogs counts as two animals, so a repeat scores the same as a correct pair.
- **The claim**: that neither family alone measures what a person means by "it composed", and
  the reason is different in each case.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Not a run: a literature check before print.** Judged by whether it could have come back
"already known", not by which answer it gave.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

No compute. The work is reading, and the output is the verdict below.

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| the literature search | Literature check | | reading only, no GPU | the verdict and one line per benchmark examined | not started |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [ ] ⚠️ Is the claim already named AND measured in a published benchmark?
      Three answers allowed and they are not the same. *Already known and named*: a paper states
      both halves and reports a number for how often it happens. This scope shrinks to a methods
      paragraph in `writing-06`, and `idea-01` and `gate-02` are cancelled. *Said informally,
      never measured*: a paper mentions the limitation in passing with no number attached. The
      scope continues and the contribution becomes the measurement. *Not addressed*: the scope
      continues at full size. This is the only question whose answer may cancel work.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ For each benchmark examined, what question does it actually ask of an image?
      T2I-CompBench (arXiv 2307.06350), TIFA, VQAScore, Davidsonian Scene Graph, VISOR, and the
      attribute-binding work each get one line naming the question their scorer puts to the
      picture. A benchmark that turned out irrelevant still gets its line, saying why.
- [ ] ⚠️ Does any published metric ask something other than presence or count?
      If one asks for separate instances of each named concept, that is our proposed metric
      already published and the answer to the bar is "already known". If one asks a question
      neither family covers, name it, because it may be a better idea than ours.
- [ ] ⚠️ Does the verdict say what happens next, by filename?
      It must name `idea-01-what-the-current-benchmarks-score` and `gate-02-promote-or-close`
      and say for each whether it proceeds or is cancelled. Without that line the gate has not
      gated anything.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the search itself raised. **Nothing here may ever become a bar**, because it was written
with the answer already visible. Nothing yet: the search has not run.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** A search that stops at the first paper confirming the hole
      is open produces the answer this scope wants. Every benchmark on the list gets its line,
      including the ones that turned out irrelevant, so the coverage is visible rather than
      asserted.
- [ ] ⚠️ **Was the instrument sound?** Was the claim written down before the search? Task 1 of
      the plan requires the paragraph first. Check `git log` puts it before the verdict. A claim
      written after reading the literature can be shaped to survive it.
- [x] ✅ **Did the run respect the environment?** Not applicable. No compute, no output directory,
      no flags. The work is reading.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| neither presence nor count metrics measure what a person means by "it composed" | the one line per benchmark examined, so the claim rests on a visible search rather than on absence of memory |
| whichever verdict this returns | the verdict itself belongs in `writing-06` under every outcome, including "already known", where it becomes a citation rather than a contribution |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| everything in this file | the literature search | [idea-01](idea-01-what-the-current-benchmarks-score.md) and [gate-02](gate-02-promote-or-close.md), both of which wait on this verdict. [instrument-01](instrument-01-the-three-state-labelled-set.md) runs regardless |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Write the claim paragraph, commit it, then run the search. The commit order is itself one of the
questions above.
