# Walk map: wet camels and wet dogs in the training captions

**Plan:** [today: camel and dog in the training data](../../../plans/10-what-poe-loses-when-it-composes/today-camel-and-dog-in-the-training-data.md)
**Mode:** default walk, depth full (the plan is a single afternoon; every step feeds step 8)
**On step:** 3, read the camel captions with wet words (you)

## Stop point

| | |
|---|---|
| Bar | step 8's seven talking points written, each with its number and its picture; `tally.png` and `fig-causal-renders.png` named for the slide |
| Time box | 1 sitting, about 3 h (the plan's own estimate) |

| Sitting | Start | End |
|---|---|---|
| 1 | 2026-10-22 10:41 SAST | |

## Preflight

| Check | Result |
|---|---|
| `scripts/census/laptop_slice.py` | present, `--help` runs, five stages |
| token | `~/.cache/huggingface/token` missing, `HF_TOKEN` unset: step 0 is needed |
| network | huggingface.co answers 401 unauthenticated in 0.33 s |
| disk | `/` 592 GB free of 1007 GB; the slice needs about 14.5 GB |
| step 7 renders | all four files present |

## Ticks

| Step | Evidence |
|---|---|
| 2 count | exit 0; 2 files, 32,776,409 captions in 24 s; camel 12,291 (239 with a wet, weather or water word: wet 7, weather 22, water 213), dog 225,544 (4,822: wet 781, weather 651, water 3,456) |
| 1 download | 2 files, 3.62 GB each, 16,388,209 and 16,388,200 rows; 1.6 and 1.5 MB/s; download stopped by the watcher at 12:11 |
| 0 token | confirmed by hand: curl printed "200"; `~/.cache/huggingface/token` is `-rw-------`, 38 bytes |

## Parked

## Repairs

- `scripts/census/laptop_slice.py` `count()` reads each file one row group at a time. The whole-file read was killed for memory (exit 137) with an SDXL render from the third-concept walk holding 3.1 GB and swap at 7 of 8 GB; peak after the change is 1.0 GB.

## Decisions

| Decision | Options | Who chose | Why |
|---|---|---|---|
| download 2 files, not 4 | 4 files (~2 h) · 2 files (~1 h, the plan's rule) · 1 file (~30 min) | the plan's own rule, applied by the walk; open to the user's veto | first file measured at 2.1 MB/s, under the plan's 5 MB/s cutoff; every later number covers 2 of 128 files |

## Out of the session

| Id | Kind | Serves | State |
|---|---|---|---|
| bkbyc0nc2 | run: `/home/molef/miniconda3/bin/python3 -u -I scripts/census/laptop_slice.py download --files 4`, log in the session scratchpad `download.log` | step 1 | done 12:11, folded in |
| b4v21nc87 | watcher: kills the download (pid 203103) once 2 `part-*.parquet` files exist | step 1 | fired 12:11, folded in |

## Closing pass owed

- file `artifacts/results/what-the-training-captions-show/` with its card entry
- Phase 4 retrospective
