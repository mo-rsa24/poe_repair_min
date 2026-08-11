# Review: has someone already said this?

Unanswered. This file judges
[../plans/gate-01-is-this-hole-already-known.md](../plans/gate-01-is-this-hole-already-known.md),
the check that decides whether the rest of this scope gets built. `idea-01` and `gate-02` do not
start until the questions below are answered. `instrument-01` runs regardless.

## Words this file uses
- **Presence metric**: one that asks an image "is there a cat? is there a dog?" and scores yes
  or no per concept. A cat-dog fusion answers yes to both, because it really does carry cat
  features and dog features.
- **Count metric**: one that counts how many distinct things are in the picture. Ours is a count
  metric. Two dogs counts as two animals, so a repeat scores the same as a correct pair.
- **The claim**: that neither family alone measures what a person means by "it composed", and
  the reason is different in each case.

## Run kind
**Not a run: a literature check before print.** Judged by whether it could have come back
"already known", not by which answer it gave.

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| | | | | |

## The questions

- [ ] ⚠️ **The bar.** Is the claim already named AND measured in a published benchmark?
      Three answers allowed and they are not the same. *Already known and named*: a paper states
      both halves and reports a number for how often it happens. This scope shrinks to a methods
      paragraph in `writing-06`, and `idea-01` and `gate-02` are cancelled. *Said informally,
      never measured*: a paper mentions the limitation in passing with no number attached. The
      scope continues and the contribution becomes the measurement. *Not addressed*: the scope
      continues at full size. This is the only question whose answer may cancel work.
- [ ] ⚠️ For each benchmark examined, what question does it actually ask of an image?
      T2I-CompBench (arXiv 2307.06350), TIFA, VQAScore, Davidsonian Scene Graph, VISOR, and the
      attribute-binding work each get one line naming the question their scorer puts to the
      picture. A benchmark that turned out irrelevant still gets its line, saying why.
- [ ] ⚠️ Does any published metric ask something other than presence or count?
      If one asks for separate instances of each named concept, that is our proposed metric
      already published and the answer to the bar is "already known". If one asks a question
      neither family covers, name it, because it may be a better idea than ours.
- [ ] ⚠️ Was the claim written down before the search?
      Task 1 of the plan requires the paragraph first. Check `git log` puts it before the
      verdict. A claim written after reading the literature can be shaped to survive it.
- [ ] ⚠️ Does the verdict say what happens next, by filename?
      It must name `idea-01-what-the-current-benchmarks-score` and `gate-02-promote-or-close`
      and say for each whether it proceeds or is cancelled. Without that line the gate has not
      gated anything.
