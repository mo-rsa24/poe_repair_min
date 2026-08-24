# The four cached-trajectory analyses, as a page you can drive

Seven claims from
[../review/hypothesis-04-what-the-cached-runs-already-show.md](../review/hypothesis-04-what-the-cached-runs-already-show.md),
each one a thing you operate rather than read. Drag the noise level and the curve, the readout and
the five expert pictures move together. Every ratio is recomputed in the browser from the loaded
arrays, and every panel shows the file it read and when that file was written.

## Run it

```bash
npm install
npm run dev          # http://localhost:5173
```

Over SSH, forward the port: `ssh -L 5173:localhost:5173 mscluster109`.

Each claim is in the URL (`#C3`, `#C5`, `#owed`), so a panel can be linked to and reloaded.

## Rebuild the data

The app reads one generated file, `src/data/result.json`. Nothing in `src/` holds a number.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY loader/build_data.py
```

Rerun it after any of the four analyses is re-run, or after the review file is edited. The loader
is read-only on the repo and fails loudly rather than emitting a zero: if a sentence it quotes has
been reworded, or a file it reads has moved, the build stops and names what it could not find.

`vite.config.ts` lists the two roots the dev server may read. Change those if the outputs move.

## The four analyses and where they live

The plan's Engagement Instructions name `outputs/...` in the repo. The scripts actually write to
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses/`, which is what the
loader reads.

| claim | analysis | file |
|---|---|---|
| C1 | correction size against noise level | `refresh_20260810/prereg/snr_collapse.json` |
| C2 | the same cells under the raw measure | `refresh_20260810/raw/snr_collapse.json` |
| C3 | where the two paths separate | `refresh_20260810/fork_curve.json` |
| C5 | does the correction push along the motion | `plausibility_climb.json` |
| C6 | low-rank against a matched random floor | `spectrum.json` |
| C7 | does the fitted subspace carry to unseen pairs | `artifacts/results/which-way-the-correction-points/does-the-subspace-test-predict-transfer/result.json` |

C4 has no file. It is drawn as an empty slot carrying the run that fills it.

## What was re-run to build this

Two analyses were re-run because the page needed evidence the review file asserts but no file held:

- `fork_curve.py --root outputs/interaction_term/dose/pairs` over every eligible cell. The default
  roots point at `/datasets`, where these cells do not live, which is why the original read covered
  19 of them. The elbow is step 16 either way.
- `snr_collapse.py --pool --normalize own-median`, which is the raw size measure. It puts a file
  behind the review's claim that the two measures disagree about the peak.

Both wrote to `cache_analyses/refresh_20260810/`. The 2026-08-05 originals are untouched, and C3
lets you switch between the two coverages.

## Which arms each claim has

The three-way mono against PoE against LoRA is only complete on C7. Every claim page states its
arms rather than drawing a line for one that was never sampled.

- **C3** has PoE (`lam000`) and Mono (`lam100`), a full 51-step path each. No LoRA path was sampled.
- **C5** has the PoE path only. Its two comparisons are a random vector and the right correction
  taken at the wrong step.
- **C7** has all three: plain PoE at 0%, the teacher as the target, and the adapter at 96.9%.
- **C1** and **C2** measure the correction to PoE, so there is no second path in them.

## The pictures, and the one that does not move

- **C1, C2, C5** move a real picture: five decoded views (no prompt, each prompt alone, PoE, joint)
  at the point in the run you are pointing at. 495 frames, 3 pairs at 3 seeds, every fifth step.
- **C3 does not.** Its two frames are the end of each path. The saved trajectories hold only the
  noisy latent, and the fork sits at step 16 of 51, where a noisy latent still looks like noise.
  Gap G1 on the Owed page names the re-sample that fixes it.

3.4GB of cells are not copied into this app. `public/experts` and `public/forkcells` are symlinks.

## Three places the page disagrees with the review file

All three are on the Owed page, and none was written back into the review file.

- The fork elbow is read over 19 cells; 43 are eligible. The elbow is step 16 on both.
- The review says 15 of 19 cells land between steps 13 and 20. Recomputed, it is 17 of 19.
- Two files answer the climb question over different populations (38 cells and 34). The review
  quotes the 38. Both are shown, never averaged.
