# ✅ Meeting checklist: what Richard and Devon want to see

From the meeting of 20 August 2026 (Devon Jarvis, Richard Klein, Molefe Molefe). Each item restates what was asked, simplified but faithful, with the speaker and timestamp. Deep-link any moment as `https://app.fireflies.ai/view/01KZW6VY321PPBVJT7KP0YJPZ1?t=<seconds>`. The full transcript is [meeting-transcript.txt](meeting-transcript.txt), the Fireflies summary is [meeting.md](meeting.md).

---

## Position in the paper work

| # | Topic | Items | Weight |
|---|-------|-------|--------|
| 1 | [The paper's size and shape](#1-the-papers-size-and-shape) | 5 | 🔴 shapes every section |
| 2 | [The abstract](#2-the-abstract) | 9 | 🔴 due first (candidates by email) |
| 3 | [Writing style](#3-writing-style) | 3 | 🔵 habits for every pass |
| 4 | [The name: plurality or interaction term](#4-the-name-plurality-or-interaction-term) | 11 | 🔴 the load-bearing claim |
| 5 | [Experiments they want to see](#5-experiments-they-want-to-see) | 11 | 🟠 section 3's evidence |
| 6 | [Timeline and process](#6-timeline-and-process) | 4 | 🔵 the real deadline |

---

## Table of contents

- [Position in the paper work](#position-in-the-paper-work)
- [Colour legend](#colour-legend)
- [1. The paper's size and shape](#1-the-papers-size-and-shape)
- [2. The abstract](#2-the-abstract)
- [3. Writing style](#3-writing-style)
- [4. The name: plurality or interaction term](#4-the-name-plurality-or-interaction-term)
- [5. Experiments they want to see](#5-experiments-they-want-to-see)
- [6. Timeline and process](#6-timeline-and-process)
- [Where this already connects in the repo](#where-this-already-connects-in-the-repo)

---

## Colour legend

Navigation: ⬅️ [Previous](#table-of-contents) | 📋 [TOC](#table-of-contents) | [Next](#1-the-papers-size-and-shape) ➡️

| Badge | Meaning |
|-------|---------|
| 🔴 | must land before submission; the paper's claim depends on it |
| 🟠 | an experiment to run |
| 🟡 | a decision or check to make, no run needed |
| 🟢 | only if it is essentially free; drop it the moment it hurts |
| 🔵 | a working habit or process step |

---

## 1. The paper's size and shape

Navigation: ⬅️ [Previous](#colour-legend) | 📋 [TOC](#table-of-contents) | [Next](#2-the-abstract) ➡️

**Length and what moves to the appendix**

- [ ] 🔴 **Main text at 9 pages or fewer.** Confirm the venue's limit yourself (Devon, 04:45, 05:21).
- [ ] 🔴 **The draft is heading toward too much.** Much of what exists reads as appendix material (Devon, 05:57, 06:12).
- [ ] 🔴 **Section 3 loses the raw maths.** Formalize it into a statement or theorem in the main text; the derivation goes to the appendix (Devon, 06:21).

**One clear story**

- [ ] 🔴 **One big point, nothing beyond it.** Simplify the story until it is really clean (Devon, 06:41).
- [ ] 🟡 **No wild speculation about meaning.** This is not the paper for that (Devon, 06:49).

---

## 2. The abstract

Navigation: ⬅️ [Previous](#1-the-papers-size-and-shape) | 📋 [TOC](#table-of-contents) | [Next](#3-writing-style) ➡️

**The message it must carry**

- [ ] 🔴 **The core message:** a mathematical composition of experts cannot add information, and each expert individually believed there was one object in the scene, so the composition cannot recover two (Devon and Richard, 14:39 to 14:53).
- [ ] 🔴 **Each specialist gets its own prompt, unmissably.** Right now that is only touched on, not in your face (Richard, 14:27).
- [ ] 🟡 **Richard's phrasing to build on:** the missing notion of plurality is *implicit in the text prompt describing the full scene but missing from the individual text prompts used to guide each expert* (Richard, 13:06, 20:23; Devon, 20:53). He flagged the wording as not final, but it makes clear each expert's prompt implicitly encodes non-plurality (21:16).
- [ ] 🔴 **The cold-reader test.** Someone who does not already know the work must come away knowing what was done. Richard's read on the current draft: if he did not already know, the abstract would not tell him (Richard, 14:58 to 15:12).

**Naming the methods**

- [ ] 🔴 **Say "product of experts" explicitly**, for example "product of experts does X"; grounding in a concrete method is fine (Richard, 16:14, 17:40).
- [ ] 🔴 **Then widen it:** later in the abstract, say the work covers PoE, SuperDiff, and so on, so it does not imply only one composition method was studied (Richard, 17:55).

**Who it is written for**

- [ ] 🔴 **Write for the diffusion-literate outsider**, someone who knows what diffusion is and can follow "product of experts", not the three people who already know this work. The more accessible, the better, while staying clear about what was done (Richard, 18:50 to 19:35).

**The candidate-abstracts task**

- [ ] 🔵 **Two or three candidate abstracts, voted by email** between the three of you (Devon, 21:37; email chosen, Richard, 22:25).
- [ ] 🔵 **The writing method:** read last week's abstract once to load its cadence into working memory, then put it away and do not look again; no copied sentences or phrasing. Keep the mini-paper flow (intro, background, method and contribution, results, conclusion) and fold in "across PoE and SuperDiff" and "the missing concept can be reintroduced at composition time" (Devon, 22:50 to 24:21).

---

## 3. Writing style

Navigation: ⬅️ [Previous](#2-the-abstract) | 📋 [TOC](#table-of-contents) | [Next](#4-the-name-plurality-or-interaction-term) ➡️

**Citations**

- [ ] 🔵 **One giant references file.** Drop the whole proposal's BibTeX in; BibTeX only includes what you cite, so it costs nothing and you stop breaking flow to hunt references (Richard, 25:49 to 26:29).
- [ ] 🔴 **Every citation ends as a proper BibTeX cite command.** The process does not matter; the end state does (Richard, 26:45).

**Adjectives**

- [ ] 🔵 **No load-bearing adjectives.** "Noisy latent" forces the reader to guess which property is meant. If the noise matters, say "latent with the added noise", so noise becomes the noun (Devon, 27:06 to 28:34).

---

## 4. The name: plurality or interaction term

Navigation: ⬅️ [Previous](#3-writing-style) | 📋 [TOC](#table-of-contents) | [Next](#5-experiments-they-want-to-see) ➡️

**The agreed framing**

- [ ] 🔴 **The name decides the reviewer's mindset**, either "fantastic and fundamentally cool" or "these guys are speaking about a nonsense" (Devon, 28:44 to 29:38).
- [ ] 🔴 **The framing both agreed on:** the maths says PoE can only work with what the individual expert prompts imply; the *interaction term* is what the maths says; *plurality* is the specific concept this paper studies through it (Devon, 43:06 to 44:04; Richard agreed, 44:12).
- [ ] 🟡 **That framing is the reviewer escape route.** A reviewer who does not buy that the term does only plurality has nothing to attack: the paper never claims that (Richard, 44:12 to 44:26).
- [ ] 🔴 **Say the limits outright** in the discussion or limitations: other interactions exist (prepositions, relationships), this paper deliberately studied one important one clearly, future work may cover others (Devon, 57:14 to 57:33; Richard, 57:42).

**Richard's test: the name must be earned**

- [ ] 🟡 **He likes "plurality"**, and if it holds he would call it *prompt plurality*. But first confirm it actually is plurality (Richard, 29:58 to 30:16).
- [ ] 🟠 **The test itself:** if dog × cat × plurality gives two different animals, then what does dog × plurality give? **If it gives two dogs, the name is earned** (Richard, 30:25 to 30:59).
- [ ] 🟡 **The fallback if it fails:** if applying it to a single concept does not give more than one of it, be careful about claiming plurality and use the more generic interaction claim (Richard, 31:39 to 32:03).
- [ ] 🟡 **The reviewer read on specificity:** the more specific the claim, the more interesting the paper, but the burden of proof rises with it. Seeing "plurality", Richard would say: I agree it includes plurality, now prove that is what it is (Richard, 32:10 to 32:23, 39:13 to 39:52).

**The null-space check**

- [ ] 🟠 **Where plurality is already implied, the correction should do nothing.** Butterfly on a flower needs no plurality added, so the term should sit in the null space (Devon, 33:16 to 34:11). Richard's version: where one expert alone could have generated the whole scene, the term should do less (34:31 to 34:48).
- [ ] 🟡 **The butterfly complication goes in the discussion.** Single-butterfly prompts already give multiples on some seeds; it does not contradict the point but complicates it, and the correction doing nothing there is the good outcome (Devon, 38:12 to 38:46). Send example seed images so it can be checked by eye (38:03).

**Claiming generality**

- [ ] 🟡 **A generality claim buys a demonstration debt.** If the paper says the term is more general than plurality, a reviewer wants at least one other interaction shown. The bar is low: one main interaction plus one or two side examples, never ten (Richard, 45:51, 46:05 to 46:28).

---

## 5. Experiments they want to see

Navigation: ⬅️ [Previous](#4-the-name-plurality-or-interaction-term) | 📋 [TOC](#table-of-contents) | [Next](#6-timeline-and-process) ➡️

**The section 3 control: where the implied information lives**

- [ ] 🔴 **Section 3 demonstrates and controls the phenomenon** before any learned fix appears; establishing what is being looked at is itself a big part of the contribution (Devon, 50:16 to 50:45).
- [ ] 🟠 **The control run:** PoE on "cat and thing" × "dog". First, does "cat and thing" alone give a cat plus some invented second object? Then, does adding "and thing" to one prompt fundamentally change what PoE produces, collapsing the "thing" into the dog? **If yes, that is sufficient control** to show joint prompting and PoE carry different implied information, and that PoE just needs plurality somewhere in it to use it (Devon, 47:35 to 48:31).

**Dominance and intersection: one prompt or both**

- [ ] 🟠 **Does the plurality cue need to be in one prompt or both?** Compare "cat and thing" × "dog and thing" against "cat and thing" × "dog" (Richard, 48:38 to 48:54).
- [ ] 🟡 **The underlying question:** does plurality dominate singularity or the other way round, and must the cue intersect both experts' distributions? Run it if it is easy to run (Richard, 49:07 to 49:34; Devon posed the intersection form, 49:20).

**The inverse: removing plurality**

- [ ] 🟠 **Train the inverse of the correction**, a remove-plurality mode (Devon proposed it, 40:35 to 40:44).
- [ ] 🟠 **The silver bullet:** prompt something already plural (a group of dogs, or seed 42's multi-butterfly scene), subtract the correction, and watch it collapse to a single subject. If that works, it is the most compelling evidence for the plurality claim (Richard, 40:45 to 40:59; Devon, 41:00 to 41:17).

**A second interaction type, only if cheap**

- [ ] 🟢 **One cheap demonstration of another interaction** (a preposition such as "on", which itself bundles plurality plus a spatial relation) would support the generality sentence (Richard, 52:31 to 52:55, 1:03:00).
- [ ] 🟢 **The rule for running it:** it must be essentially free. If it does not hurt, do it; if it hurts the schedule, drop it and the paper claims only plurality was proven (Richard, 1:04:05 to 1:04:12).

**A warning about one proposed figure**

- [ ] 🟡 **The mono-versus-PoE table is a trap.** Rows like "cat", "cat with", "cat next to a dog" make an effective table, but it immediately invites "how well does the technique work on each of those?", trapping the paper into running all those experiments. Use with care (Richard, 47:12 to 47:34).

**Deferred to later papers**

- [ ] 🟡 **Ordering and directionality are paper 2.** Cat on dog versus dog on cat: the PoE maths is commutative and cannot represent it, and the LoRA cannot bind to one object over another. Write it down, keep it out of this story (Richard, 54:05 to 55:44).
- [ ] 🟡 **The syntax-tree version is paper 3.** A syntax tree whose edges are LoRAs, where the tree language needs is the tree PoE needs (Devon, 56:37; Richard, 56:44 to 57:07).

---

## 6. Timeline and process

Navigation: ⬅️ [Previous](#5-experiments-they-want-to-see) | 📋 [TOC](#table-of-contents) | [Next](#where-this-already-connects-in-the-repo) ➡️

- [ ] 🔴 **The real deadline is one to two weeks early.** Abstract due 18 September, paper 25 September; a near-complete draft early enough lets an outside reader (a Steve, a Benji) name the weak spots a reviewer would pick on, which makes an enormous difference at big venues (Richard, 1:01:26 to 1:02:30).
- [ ] 🔵 **Devon passes on the introduction** as soon as it is readable (Devon, 20:11).
- [ ] 🔵 **Write this meeting's ideas down properly** (Devon, 57:47).
- [ ] 🔵 **Share the Fireflies record and summaries** so the discussion feeds the writing and the experiment design (57:54 to 58:10).

---

## Where this already connects in the repo

Navigation: ⬅️ [Previous](#6-timeline-and-process) | 📋 [TOC](#table-of-contents) | [Top](#position-in-the-paper-work) ⬆️

- Richard's earn-the-name test is the parked [earn-the-plurality-name walk](../../artifacts/ideas/earn-the-plurality-name/IDEA_MAP.md), resuming at the PoE combination-rule check.
- The plurality renames from this meeting are enacted in [the tex](iclr2027_conference.tex), tracked by [the supervisor-structure merge walk](../../artifacts/ideas/merge-supervisor-structure/IDEA_MAP.md).
- Three candidate abstracts exist: [candidate 1](abstract_candidate_01.tex), [candidate 2](abstract_candidate_02.tex), [candidate 3](abstract_candidate_03.tex).
