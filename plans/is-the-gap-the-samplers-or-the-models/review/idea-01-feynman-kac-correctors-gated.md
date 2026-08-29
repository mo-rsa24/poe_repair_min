# 🔍 Review: are Feynman-Kac correctors built here or cited here?

**Nothing has run yet, and this plan may close without running.** Every question below was written
before the paper was read in full. This file judges
[the Feynman-Kac design](../plans/idea-01-feynman-kac-correctors-gated.md). If
[the gate](hypothesis-02-what-is-left-once-the-chain-settles.md) returns a null, the corrector arm
moved nothing, a second corrector family is a related-work paragraph, and this plan closes unrun
with the reason recorded. That is a completed plan, not an abandoned one.

## Recommended prompt (when the read lands)

```
/unpack-paper https://arxiv.org/abs/2503.02819
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/idea-01-feynman-kac-correctors-gated.md) | the gate, the read, the cost estimate, and the built-or-cited decision |
| **this file** | **the verdict: not yet run** |
| [the gate's verdict](hypothesis-02-what-is-left-once-the-chain-settles.md) | whether this plan runs at all |
| [the dose-axis verdict](baseline-02-three-rules-on-one-dose-axis.md) | whether the corrector rows are worth extending to a second family |

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

- **Feynman-Kac correctors**: a sequential Monte Carlo method for sampling correctly from product
  distributions, which is exactly the problem this scope is about. Skreta et al.,
  [arXiv 2503.02819](https://arxiv.org/abs/2503.02819), ICML 2025 Spotlight.
- **A full read**: this project's reading register distinguishes reading an abstract from reading
  the method. A full read can say what the method does at each noise level, not only what it
  claims.
- **The cost estimate**: what would have to be implemented in this repo, against what already
  exists after the Langevin corrector was built, and what it would buy.
- **Closing unrun**: recording that the plan was deliberately not executed, with the reason. It is
  an outcome, and it is written down like any other.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Explores.** There is no bar, and this file says so plainly rather than inventing one. Per this
project's run conventions, a run that tries an idea may not change any claim: its outcome may
propose an experiment and nothing more. What it must produce is a decision with a reason.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-pre-registered-bar) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Full read of arXiv 2503.02819 and a cost estimate | Explores | not launched | no GPU, no queue, one read | the promoted register row and this file's decision | ⚠️ gated on the bar at step 26 |

## The pre-registered bar

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

**There is no bar here, deliberately.** This plan explores, so nothing it returns may move a claim,
and inventing a threshold would give it an authority it should not have. The one thing it cannot do
is close with a decision whose reason is unwritten: a decision with no reason is unreviewable in
three months and gets remade from scratch.

## Written before the run, answered after

Navigation: ⬅️ [The pre-registered bar](#the-pre-registered-bar) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [ ] ⚠️ Are Feynman-Kac correctors built here or cited here, and why? The answer includes the case
      "not run, because the gate returned a null", with the branch named.
- [ ] ⚠️ Does a usable implementation exist now? None was found when this scope was designed, which
      is the whole reason this is a read rather than a wiring job. One appearing since changes the
      cost estimate by an order of magnitude. What was searched, and what was found, including
      "nothing".
- [ ] ⚠️ What would have to be implemented in this repo, against what already exists after the
      Langevin corrector was built, and roughly how large is each piece? A built-or-cited decision
      made against a feeling rather than a number is not a decision.
- [ ] ⚠️ What does Soiffer et al.'s result do to the answer? It says existing correctors,
      Feynman-Kac included, reduce the gap without closing it. If that holds, building a second
      family buys a smaller gap and not a closed one, which is a reason and belongs in the decision.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a bar**, because anything written here is written with the answer
already visible. Empty until the read happens.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [ ] ⚠️ **Was the comparison fair?** Not applicable: nothing is compared, and this file says so
      rather than leaving the check unanswered.
- [ ] ⚠️ **Was the instrument sound?** The instrument here is a read. It is sound if the register
      row can answer what the method does at each noise level; an abstract standing in for the full
      read cannot cost the implementation, and a row that overstates itself is worse than one that
      admits the gap.
- [ ] ⚠️ **Did the run respect the environment?** No GPU, no queue, nothing written under
      `/datasets`. The only environment dependency is network access to fetch the paper.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| the related-work sentence on Feynman-Kac correctors | whether they were tried here, and if not, the cost that decided against it |
| any claim about what correctors can do | that one family was tested, the simplest, and which others were read but not run |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether Feynman-Kac correctors are built or cited | a full read of arXiv 2503.02819, gated on the gate at step 26 and the dose grids at step 29 | nothing. The default is cited |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Read the gate's branch and decide whether this plan runs at all. Record either "running, because
the gate returned X" or "closed unrun, because the gate returned a null", with the branch named.
