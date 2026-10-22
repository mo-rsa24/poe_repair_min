# Walk map: today, does a third concept bring the wet camel back?

**Plan:** [today-the-third-concept](../../../plans/10-what-poe-loses-when-it-composes/today-the-third-concept.md)
**Branch:** exec/dog-and-rain (commits per tick, never pushed)
**On step:** 3, render the rest
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
| depth | full; to 2 | you, default | every step feeds step 6's table |

## Out of the session

| Id | Kind | Serves | Command | State |
|---|---|---|---|---|
| bx8capx2u | run | step 2 | `$PY scripts/local_sdxl_poe.py render --animals camel --seeds 1`, log scratchpad/step2-camel-seed1.log | landed, exit 0, 10:54:38 to 11:02:54 |
| render-8 | run, detached (nohup setsid), pid 225613 | step 3 | `$PY scripts/local_sdxl_poe.py render --animals camel dog --seeds 1 2 3 4 5 6 7 8`, log laptop-third-concept/render.log | in flight, started 11:15:47 |

## Closing pass still owed

All of it.
