# 💡 Earn the plurality name: the dog-times-plurality test

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| **1 (current)** | **needs a check** | **the PoE combination rule in code: whether same-prompt-twice PoE reduces to the single expert, which collapses the three candidate forms into one** |
| 2. the validated scorer judges a two-dogs scene under the same rule | needs a check | the rule counts instances of the query "animal", class-agnostic, so two dogs are two instances; no one-class positive appears in `scorer_validated.json`, so a small hand-labelled two-dogs set is the check |
| 3. the bar is written before any run | open | this walk writes it, after claim 1 pins the form |
| 4. a fail demotes the name with a small tex edit, not a rewrite | holds | the anchor clause is verbatim at `paper/iclr/iclr2027_conference.tex:103`; plurality appears at ~10 sites (abstract, two contributions, three section titles), all renames, no evidence moves |
| 5. the oracle probe runs today from existing infrastructure | needs a check | `build_eval_cache.py` takes `--prompt-a/--prompt-b/--joint-prompt` freely; the check is `slugify("a dog","a dog")` not colliding and the inject script accepting the degenerate pair |

Load-bearing: claim 1. Without one faithful operational form, no run can earn or demote the name.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [The claims](#the-claims)
- [What the words are](#what-the-words-are)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Sources](#sources)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

Richard Klein's earn-the-name condition (dog times plurality gives two dogs, or the name narrows back to interaction term) is runnable in this framework, and its outcome decides whether the enacted rename in `paper/iclr/` stands.

**Where the walk is**

Parked at claim 1, round 1: pinning the operational form of "dog times plurality". The candidate forms may collapse into one via the PoE combination algebra, and the one named check (the combination rule in the sampling code) was not yet run. No runs are outstanding.

**What compile would produce today**

Neither destination yet. The walk owes the pinned form, the pre-registered bar, and the run design before the verdict can exist.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

The paper names the residual `r_t` (the model's joint-prompt prediction minus its PoE prediction) the plurality term, and the abstract defines plurality as every concept a prompt names appearing as its own distinct object. Richard's condition from the 2026-08-20 meeting: if the adapter applied to dog alone yields two dogs, the name is earned; if not, plurality narrows to the conflicting-pair setting and the general object stays an interaction term. The test has an operational form in this pipeline (the adapter conditions on two concept prompts, so the degenerate pair "a dog", "a dog" is in its input domain but outside its training distribution of conflicting distinct pairs), a validated scorer that counts animal instances class-agnostically, a cheap oracle first probe (`r_t` computed from a joint prompt like "two dogs"), and a demotion path already present in the tex as the interaction-term anchor clause.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. 🔍 "Dog times plurality" has one faithful operational form here  ← current

Mark: needs a check. Three candidates were on the table: same-prompt-twice PoE with the adapter on versus off; single-expert plus adapter; and "dog times dog times plurality" as Richard literally said. If the PoE combination is an equal-weight average of the two expert predictions, then two identical experts average to the single expert, and all three candidates are one computation: sample "a dog" through the pipeline with the degenerate pair, adapter off (baseline, expect one dog) versus adapter on (test). The named check is the combination rule in the sampling code, including guidance weights.

### 2. 🔍 The validated scorer judges a two-dogs scene under the same rule

Mark: needs a check. `scorer_validated.json` shows the rule is COMPOSE iff distinct-instance-count("animal", NMS iou<0.5, conf>=0.30) >= 2. The query is the class-agnostic word "animal", so two dogs count as two instances with no change to the rule. But every positive in the validation set is a two-class scene; no two-dogs (one-class, two-instance) image was ever labelled. The check is a small hand-labelled one-class set, for example joint-prompt "two dogs" anchors, before the scorer's verdict on the test is trusted.

### 3. ⬜ The bar is written before any run

Mark: open. Compose rate over how many seeds earns the name, what demotes it, and what is inconclusive, with numbers, written before the first sample. Waits on claim 1 (the form) and claim 2 (the instrument).

### 4. ✅ A fail demotes the name with a small tex edit, not a rewrite

Mark: holds. `paper/iclr/iclr2027_conference.tex:103` already carries the fallback verbatim: the gap "is an interaction term, the quantity a product of independent experts drops. In our setting it is narrowed to plurality". Plurality appears at roughly ten sites (the abstract at line 81, contributions at 119-120, section titles at 127, 216, 219). Demotion is a bounded rename sweep plus re-weighting that anchor clause; no figure, number, or evidence block moves.

### 5. 🔍 The oracle probe runs today from existing infrastructure

Mark: needs a check. `build_eval_cache.py` takes `--prompt-a`, `--prompt-b`, `--joint-prompt` as free arguments, so ("a dog", "a dog", joint "two dogs") is directly parameterizable. Two named checks: whether `slugify("a dog","a dog")` produces a usable non-colliding slug, and whether `interaction_term_inject.py`'s pair handling and canary accept a degenerate pair.

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| dog times plurality | operationalization of a construct | mapping a named concept (plurality) to one measurable procedure; the name is earned only through the procedure | confident, standard measurement-theory term |
| the adapter on the degenerate pair | out-of-distribution transfer to the diagonal | the adapter trained on conflicting distinct pairs is queried at (c, c), a point its training set never contains | confident on the framing |
| the plurality term | the interaction term / residual r_t | the joint-prompt prediction minus the PoE prediction, the quantity a product of independent experts drops | confident, the project's own term |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 1 | read the PoE combination rule in the sampling code (which line combines the two expert predictions, and the guidance weights) | equal-weight average: the three candidate forms are one computation and the definition is pinned; anything else (unequal weights, per-expert guidance): the forms differ and the choice must be argued |
| 2 | hand-label a small one-class set (joint-prompt "two dogs" anchors) and run the validated rule on it | rule separates one dog from two dogs: the instrument carries over; it does not: the scorer needs a one-class validation pass before any verdict counts |
| 5 | run `slugify("a dog","a dog")` and a smoke `build_eval_cache.py` on the degenerate pair | clean slug and cache: the oracle probe is launchable; collision or assert: a small patch is needed first and the "runnable today" claim narrows |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|

## Sources

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Source | What it gives the idea | Confidence |
|---|---|---|
| `artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json` | the validated rule verbatim, including the class-agnostic "animal" query and the all-two-class validation set | read this session |
| `paper/iclr/iclr2027_conference.tex` lines 81, 103, 119-120, 127, 196, 216, 219 | every site the plurality name touches, and the interaction-term anchor clause verbatim | read this session |
| `scripts/build_eval_cache.py` | prompts enter as free CLI arguments, so the degenerate pair is parameterizable | read this session |
| meeting transcript 2026-08-20, 30:25-32:18 | Richard's condition and fallback wording, quoted verbatim in the walk's brief | supplied by the user |
| `artifacts/ideas/merge-supervisor-structure/IDEA_MAP.md` claims 3 and 4 | the enacted rename this walk's verdict feeds; both claims rest on the name | read this session |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Resume at claim 1, round 1: settle the operational form. The first move on resume is the named check, reading the PoE combination rule in the sampling code; an equal-weight average collapses the three candidate forms into one and settles the claim.

**Where the verdict lands when this walk finishes**

Two integration edges, both via `integrate` in their own walks. Into [the merge-supervisor-structure walk](../merge-supervisor-structure/IDEA_MAP.md), whose claims 3 and 4 rest on the plurality name: name earned means those claims stand as written; name demoted means their wording narrows and the tex rename sweep follows. Into [the three-product-comparison walk](../three-product-comparison/IDEA_MAP.md), whose missing-implications algebra uses the same interaction-term-versus-plurality naming, so the verdict fixes which word that walk's candidate section-3 claim carries.
