# 🧭 Label a set of renders without knowing which condition made each one

[Plan 02's instruction 5.1](../plans/tools/02-the-blind-label-pass.md#5--label-one-seed-by-hand-and-check-the-blinding-holds) sends you here to check the tool, and [plan 03](../plans/baselines/03-what-the-stacked-sampler-alone-does.md#3--label-both-samplers-blind) and [plan 04](../plans/hypothesis/04-the-parent-and-the-four-children.md#5--label-every-run-blind-and-write-the-verdicts) send you here for the real reads. When you are done, one label file exists for the set you were given, and nothing you did could have told you which picture came from which condition.

## Recommended prompt (when you finish)

```
/analyze-run <the W&B run id of the set you just labelled>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/tools/02-the-blind-label-pass.md) | the strips, the blinding, the three labels, the two secondary reads |
| [review](../review/02-the-blind-label-pass.md) | the questions your labelling answers about the tool itself |
| **this file** | **the steps, and the reason behind each one** |

## Table of contents

- [Why you are doing this, and where it lands](#why-you-are-doing-this-and-where-it-lands)
- [Words this file uses](#words-this-file-uses)
- [Before you start](#before-you-start)
- [What you need to understand first](#what-you-need-to-understand-first)
- [1. Open the set you were given, and check it is blind](#1-open-the-set-you-were-given-and-check-it-is-blind)
- [2. Label every tile, one strip at a time](#2-label-every-tile-one-strip-at-a-time)
- [3. Run the join and read the counts](#3-run-the-join-and-read-the-counts)
- [If it does not work](#if-it-does-not-work)
- [What you produce](#what-you-produce)
- [Next step](#next-step)

## Why you are doing this, and where it lands

Navigation: 📋 [TOC](#table-of-contents) | [Next](#words-this-file-uses) ➡️

**What is at stake**

Every verdict in this scope is a label taken here. The detector that counts animal-shaped regions calls most of checkpoint 30050's unclean renders composed, so no automatic measurement in this repository can currently see the defect the scope exists to fix. Your eye is the instrument, and these steps are what make it a fair one.

**What is already settled, and what is not**

The three labels and their definitions are fixed and are not open for adjustment while you label. What is not settled is which condition made any given picture, and it stays that way until you have finished and run the join.

**Why now**

The set you were given cannot be read any other way, and the plan that produced it is waiting on the count.

## Words this file uses

Navigation: ⬅️ [Why you are doing this](#why-you-are-doing-this-and-where-it-lands) | 📋 [TOC](#table-of-contents) | [Next](#before-you-start) ➡️

- **A strip**: one row of four picture tiles, all from the same random seed. One of them is the picture a single prompt naming both animals drew, one is the picture with no correction, one is checkpoint 30050's, one is the run being judged. Which is which is hidden.
- **A tile id**: the opaque name a tile carries instead of its condition. You record labels against ids.
- **The mapping**: the file that says which id was which condition. It lives in a different directory and you do not open it. A later step does.
- **The three labels**: **clean**, **unclear**, **not two**. Their full definitions are in [what you need to understand first](#what-you-need-to-understand-first) below, and you read them before you start rather than trusting your memory of them.
- **The join**: the command that resolves ids to conditions after labelling and prints the counts.
- **A set name**: the run and the sampler joined by a hyphen, for example `baseline-shipped`, `baseline-stacked`, `P-stacked`, `C1-shipped`. The plan that sent you here names the set and its root directory; every command below writes `$ROOT` where that root goes.
- **The set root**: the directory the sending plan gives. [Plan 02](../plans/tools/02-the-blind-label-pass.md) uses `/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/read`, [plan 03](../plans/baselines/03-what-the-stacked-sampler-alone-does.md) uses `.../improve_r32/baseline`, and [plan 04](../plans/hypothesis/04-the-parent-and-the-four-children.md) uses `.../improve_r32/<run>` with one root per run. Export it once before you start:

  ```bash
  export ROOT=<the root the sending plan named>
  ```

## Before you start

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#what-you-need-to-understand-first) ➡️

- [ ] 🖥️ **The set exists and carries no condition names.**

  ```bash
  ls $ROOT/strips/
  python3 -c "import json,sys; t=json.load(open(sys.argv[1])); print(len(t), 'tiles'); print(sorted(t[0].keys()))" \
    $ROOT/tiles.json
  ```

  ✅ **One strip image per seed, and the printed key list holds an id, a seed and a sampler but no condition, run name or checkpoint path.** The set is blind as built.
  ❌ **A key naming a condition, a run or a checkpoint appears**: stop. The set is not blind and labelling it would waste your time. Report it against [the tool's review file](../review/02-the-blind-label-pass.md) and ask for a rebuild.

- [ ] 🖥️ **No labels exist for this set yet.**

  ```bash
  ls $ROOT/labels.json 2>/dev/null || echo "none yet, good"
  ```

  ✅ **The file is absent, or holds no entry for this set's ids.** You are starting fresh.
  ❌ **Entries for this set already exist**: somebody has labelled it. Do not add to it. The collector refuses a second label per id by design, and overwriting would mix two readings into one table.

- [ ] 👁️ **You have twenty uninterrupted minutes and a screen you can see detail on.**

  ✅ Judging an unresolved limb or a blended muzzle is the whole task, and a laptop screen at a glance will not show either.
  ❌ **You do not**: come back later. A rushed set is worse than an unlabelled one, because nothing downstream will know it was rushed.

## What you need to understand first

Navigation: ⬅️ [Before you start](#before-you-start) | 📋 [TOC](#table-of-contents) | [Next](#1-open-the-set-you-were-given-and-check-it-is-blind) ➡️

**The three labels, in full. Read them now, not from memory.**

**clean** means both of the pair's named animals are present, each one is clearly itself, and nothing in the picture is unnatural on inspection. Inspection is the operative word: a picture that is plausible at a glance and wrong on looking is not clean.

**unclear** means either two animals are present but you cannot say they are the two the pair names, or both are clearly there and the picture carries something unnatural: features blending across the two bodies, a limb that does not resolve, a tail that belongs to nothing, an object from outside anything the training pool contains.

**not two** means one animal, a single creature blending both, or the same animal twice.

**Why the boundary between unclear and not two matters more than the others.** A run gets no credit for turning a **not two** into an **unclear**, and the verdict rule counts only clean seeds. So the case you should slow down on is the one where you cannot decide whether you are looking at two animals or at one animal drawn confusingly. Count how often that happens; the review file asks for it.

**Why the joint target is on the strip at all.** One of the four tiles is what the joint prompt itself drew on that seed. On some seeds that picture is wrong: it shows two dogs where the pair names a cat and a dog. It is on the strip because the adapter was trained toward it, so a seed whose own reference is wrong is a seed where nothing could have done better. You label it exactly like the others, and the join is what reveals which one it was.

**Why you must not try to guess which tile is which.** Not because guessing is against the rules, but because a guess you believe becomes the label. The shuffle exists so that the picture is the only evidence available to you.

## 1. Open the set you were given, and check it is blind

Navigation: ⬅️ [What you need to understand first](#what-you-need-to-understand-first) | 📋 [TOC](#table-of-contents) | [Next](#2-label-every-tile-one-strip-at-a-time) ➡️

Before labelling anything, spend one minute trying to break the blinding. If a tile is identifiable, everything after this is wasted.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
python3 scripts/showcase/blind_label.py --show --set <the set name you were given>
```

✅ **Four tiles per row, captions carrying ids only, and the tile that looks like a joint-prompt render is not in the same position on every row**: the shuffle is working and you can start.

❌ **One column is consistently the odd one out across every row**: that is not necessarily a leak, since one condition really may look different, but check the ids differ per row before continuing. If a single id repeats down a column, the shuffle did not run and the set must be rebuilt.

- [ ] Write down whether anything about the presentation told you a tile's identity, for [the review file](../review/02-the-blind-label-pass.md).

## 2. Label every tile, one strip at a time

Navigation: ⬅️ [1. Open the set](#1-open-the-set-you-were-given-and-check-it-is-blind) | 📋 [TOC](#table-of-contents) | [Next](#3-run-the-join-and-read-the-counts) ➡️

One strip at a time, four labels each, and no going back to revise an earlier strip once you have seen a later one. Revising backwards is how a labeller drifts toward internal consistency and away from the definitions.

```bash
python3 scripts/showcase/blind_label.py --label --set <the set name you were given>
```

✅ **The collector advances strip by strip and writes each label as you give it.** At the end it prints the number of tiles labelled, which should equal four times the number of seeds in the set.

❌ **It refuses a tile, naming an id already present**: that id was labelled before. Stop and check whether you are re-labelling a set somebody else has already done.

- [ ] ⚠️ Count the tiles where you hesitated between **unclear** and **not two**, and write the number down now, while you remember it. The review file asks for it and it cannot be reconstructed afterwards.

## 3. Run the join and read the counts

Navigation: ⬅️ [2. Label every tile](#2-label-every-tile-one-strip-at-a-time) | 📋 [TOC](#table-of-contents) | [Next](#if-it-does-not-work) ➡️

This is the first moment anyone learns which tile was which. It runs only after every label is in.

```bash
python3 scripts/showcase/blind_label.py --join --set <the set name you were given>
```

✅ **A table with one row per tile carrying its label, its resolved condition, the region count and the both-names outcome, and a printed clean count per condition.** That count is what the plan that sent you here is waiting for.

❌ **The join reports an id with no label, or a label with no id**: the set and the label file are out of step and the counts cannot be trusted. Do not fix it by hand; report it and rebuild.

- [ ] Write the clean count per condition into the review file the plan that sent you here names.

## If it does not work

Navigation: ⬅️ [3. Run the join](#3-run-the-join-and-read-the-counts) | 📋 [TOC](#table-of-contents) | [Next](#what-you-produce) ➡️

**A tile is identifiable without the mapping.** Stop labelling. The blinding is not a property of the tool yet, and labels taken under it cannot be used. This is a failure of [plan 02](../plans/tools/02-the-blind-label-pass.md), not of the set you were given.

**You cannot decide between two labels on many tiles.** Finish the set anyway, using the definitions rather than your sense of what the run ought to have produced, and record the count of hesitations. A high count bounds every verdict built on this set, and that is worth knowing.

**The strips are too small to judge.** Open the individual renders under the run's `renders/` directory instead of the strip image; the strip is a convenience, not the evidence. Say in the review file that you did this, because it changes what "at sheet size" means for [plan 05's sheet](../plans/figures/05-the-comparison-sheet.md).

**Somebody has already labelled this set.** Do not add to their file. Two readings of one set is a useful thing to have, but it is a second file and a comparison, not an append.

## What you produce

Navigation: ⬅️ [If it does not work](#if-it-does-not-work) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What | Where it lands | What it answers |
|---|---|---|
| the label file for this set | `$ROOT/labels.json` | the clean-seed count every verdict in this scope rests on |
| the joined table | `$ROOT/labels_joined.csv` | which condition each label belonged to |
| the hesitation count | the review file the sending plan names | [how often the unclear-against-not-two boundary was hard to call](../review/02-the-blind-label-pass.md#written-before-the-run-answered-after) |

## Next step

Navigation: ⬅️ [What you produce](#what-you-produce) | 📋 [TOC](#table-of-contents)

Go back to the plan that sent you here, tick its instruction, and write the counts into its review file: [plan 02](../plans/tools/02-the-blind-label-pass.md), [plan 03](../plans/baselines/03-what-the-stacked-sampler-alone-does.md) or [plan 04](../plans/hypothesis/04-the-parent-and-the-four-children.md).
