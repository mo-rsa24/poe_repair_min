# Walk map: today, does a third concept bring the wet camel back?

**Plan:** [today-the-third-concept](../../../plans/10-what-poe-loses-when-it-composes/today-the-third-concept.md)
**Branch:** exec/dog-and-rain (commits per tick, never pushed)
**On step:** 5, call every tile
**Depth:** full

## Stop point

**Bar:** step 6's counted-seed table for camel and dog, read against the plan's pass marks, with one verdict line carrying its numbers. Checked by `$PY scripts/local_sdxl_poe.py tally` printing the table.
**Time box:** about 2.5 hours, one sitting (the plan's own estimate).

| Sitting | Start | End |
|---|---|---|
| 1 | 2026-10-22 10:41 | |

## Ticks

| Step | Evidence |
|---|---|
| 3, render | 128 tiles, camel and dog × 8 seeds × 8 columns, exit 0 at 13:09 |
| 4, sheets | camel-sheet.jpg and dog-sheet.jpg, 8 × 8 |
| 1, card | 1851 MiB in use; step 2's render finished all seven tiles with no out-of-memory error |
| 2, seven camel tiles | seven camel photographs, none black or noise; tiles 49 to 72 s, 405 s for the seed, 8 min 16 s with the model load |
| 1, setup log | ENV OK 2.5.1+cu121 0.31.0 True RTX 4070 Laptop; WEIGHTS OK; SETUP EXIT 0 |

## Parked

## Repairs

- Step 4 builds the sheets twice, after the camel and after the dog, so the camel's calls overlap the dog's render. The timing run showed 6¾ min per seed, which put 8 seeds about 20 min over the box without the overlap.

## Decisions

| Question | Options | Who chose | Why |
|---|---|---|---|
| GPU memory before rendering | close Chrome and Teams first; try as is | you, default | "defaults and I closed chrome and teams"; card went 2225 to 1851 MiB, still over the plan's 1 GB |
| pass marks had no row for the camel on exactly 2 recovered seeds | a inconclusive for the camel; b supported allows 2; c falsified starts at 2 | you, before any render | "a" |
| seed count | 8; 7; 6 | you | "8" |
| tile 7 wet where 5 and 6 are dry: a cause is needed, or push is enough | add PoE camel × wet × wet now; park it in the causal route | you, after seeing camel seeds 1 to 6 | "add wet twice"; pass marks for column 8 written into the plan before it rendered; costs ~15 min past the box |
| cat × dog sheet | after the dog on the laptop; now; on the cluster | you | "the cluster is back up so you can try use biggpu or bigbatch"; both joint wordings kept (the recommended default) |
| depth | full; to 2 | you, default | every step feeds step 6's table |

## Out of the session

| Id | Kind | Serves | Command | State |
|---|---|---|---|---|
| bx8capx2u | run | step 2 | `$PY scripts/local_sdxl_poe.py render --animals camel --seeds 1`, log scratchpad/step2-camel-seed1.log | landed, exit 0, 10:54:38 to 11:02:54 |
| render-8 | run, detached (nohup setsid), pid 225613 | step 3 | `$PY scripts/local_sdxl_poe.py render --animals camel dog --seeds 1 2 3 4 5 6 7 8`, log laptop-third-concept/render.log | stopped 12:04 by the walk after 4 dog tiles, so column 8 renders first (your ask to see it) |
| 73075 | cluster job, bigbatch mscluster56 | your side request, outside the plan: cat × dog joint and PoE sheet | `PAIRS="a cat\|a dog" COLUMNS="single mono poe" SEEDS="1..8" sbatch scripts/concept_pair_sheet.sbatch`, cluster checkout on exec/correction-loss-01 | running |
| 73077 | cluster job, after 73075 | the same sheet with plurals: "cats" alone, "dogs" alone, joint "cats and dogs", PoE "cats" × "dogs" | `PAIRS="cats\|dogs" COLUMNS="single mono poe"`; replaces 73076, cancelled while pending | pending on 73075 |
| col8-queue | run, detached waiter | step 3, column 8 | waits for pid 225613 to exit, then the same render command; log laptop-third-concept/render-col8.log | landed 13:09:12, exit 0 |

## Closing pass still owed

All of it.
