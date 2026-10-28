# Separation steering walk

Request: build and run separation steering on plain SDXL PoE for "dog" and "cat". No plan file; the
request carried its own cut, bar and run settings.

## Stop point

Bar: steered tiles show one dog and one cat on at least 4 of seeds 1 to 7 at some eta. Result: 0 of 7
clean, 1 borderline. Time box: not set; one sitting, about 45 minutes.

## Decisions

| Decision | Chosen | Options | Who |
|---|---|---|---|
| Loss, steps, device, guard, paths | as written in the request | none offered | user |
| Token position | 1 (one-word prompt: start, word, end) | 2 as in the "a cat" scripts | Claude, from the tokenizer |
| Loss scale for the fp16 backward | 2^20 | 1024 gave a zero gradient for the overlap term | Claude, from calibration |
| Step sizes | 2.5e4, 1.2e5, 5e5 (1%, 5%, 20% of latent norm per step at mean gradient norm 1.1e-4) | step sizes from the overlap term alone (presence terms would then move the latent by half its norm) | Claude, written to the card before rendering |
| How to reach the cluster | copy scripts to /datasets, import the cluster checkout, push nothing | cluster.sh sync (switches another session's branch, refuses with this checkout's uncommitted files) | Claude |
| Plain PoE column | re-rendered by the same script and device | earlier and_vector tile | Claude |

## Runs

| What | Where | State |
|---|---|---|
| calibration, seed 1, scales 1024 and 2^20 | mscluster109 device 1, logs/calibrate*.log | done |
| identity check, seed 1 | device 0, logs/identity.log | done |
| render, 4 columns x 8 seeds | device 1, logs/render.log, 01:22 to 01:32 | done |

## Owed

Card and sheet are filed under artifacts/results/what-the-and-vector-is-made-of/. The idea map is
not edited; the user runs `integrate` in the drip-idea walk.
