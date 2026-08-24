# The dose-response result, as a page you can drive

**ran under** Node/Vite, `npm run dev` on a cluster node, port 5173 forwarded over SSH; last run
under this README's own instructions below (stated, not independently re-verified this move).

**built from** `dose_curves.json`, 440 cell PNGs and 320px thumbnails under
`poe_repair/paths.py`'s `HOW_MUCH_CORRECTION_IS_NEEDED` root (still on `/home-mscluster` and
`/datasets`, verified by `loader/build_data.py`'s own `paths.resolve(...)` calls), the compose
scorer's size floor read live from `detection_scorer.py`, and the nine claims in
`plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/review/hypothesis-02-more-correction-more-composition.md`.

**why** to make the correction-strength result operable rather than only readable: scrub the
strength slider and the curve, the three rates, and the pictures move together, each one
recomputed in the browser and checked against the file it came from.

**depends on** stated as of this move (2026-08-24), moved here from
`plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/scene/` as part
of the retrofit sweep's pile 1c:
- `node_modules/` and `dist/` were **not** moved; they still sit at the old path above, now
  orphaned from the source that built them (a loose end for a later pass, not fixed here).
- `public/full`, `public/figures`, `public/paperfigs` are symlinks with **absolute** targets
  (`/home-mscluster/...`, `/datasets/...`), so they still resolve correctly after this move.
- `loader/build_data.py` computes `SCOPE = SCENE.parent` from its own file location to find the
  review/plan/procedure files it reads (`REVIEW`, `PLAN`, `PROCEDURE` below). Before this move
  `SCOPE` pointed at `.../does-the-correction-cause-composition/`, where those files live; after
  this move `SCOPE` resolves to `artifacts/scenes/`, which does not hold them. The loader is not
  edited here (moving files is in scope, editing source is not) — running it from the new location
  will fail loudly on those three paths until a later pass points them back at the plan scope.
- The "Rebuild the data" and "Two places..." sections below still say `../review/...` and similar
  relative phrasing inherited from the old location; read those against the path above, not the
  new one.

Nine claims from
[../review/hypothesis-02-more-correction-more-composition.md](../review/hypothesis-02-more-correction-more-composition.md),
each one a thing you can operate rather than read. Scrub the strength and the curve, the three
rates and the pictures move together. Every rate is recomputed in the browser from the 480 cell
records and checked against the values `dose_curves.json` carries; the size floor is read from
`detection_scorer.py` at build time, not typed in.

## Run it

```bash
npm install
npm run dev          # http://localhost:5173
```

Over SSH, forward the port: `ssh -L 5173:localhost:5173 <this node>`.

## Rebuild the data

The app reads one generated file, `src/data/result.json`. Nothing else in `src/` holds a number.

```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY loader/build_data.py                 # data file and 320px thumbnails, about 30s
$PY loader/build_data.py --no-thumbs     # data file only, about 2s
```

Rerun it after a re-score, after the review file is edited, or after the cells move off
`/home-mscluster`. The loader takes `--curves`, `--images` and `--figures` if any of those roots
change, and `vite.config.ts` lists the roots the dev server is allowed to read.

The loader is read-only on the repo. It fails loudly rather than emitting a zero: if a sentence it
quotes has been reworded, or a file it reads has moved, the build stops and names what it could
not find.

## What is drawn, and what is not

- 6 of the 9 claims are measured: their numbers come from `dose_curves.json`, the 440 PNGs, or a
  constant read from source, and every panel shows the file it read.
- 3 claims (the size floor's before-and-after counts, the harness canaries, the control-separation
  distances) are answered in the review file but no file in the repo holds their numbers. They are
  drawn apart from the rest, quoted with their line numbers, and each carries the run that would
  turn it into a measured claim.
- Nothing unmeasured is plotted. A claim without an artifact gets an empty slot, never a
  placeholder curve.

## Two places the page disagrees with the review file

Both are flagged in red where they appear, on claim C7.

- The review calls cat × dog the only pair whose two controls score exactly 0.000. Five of the
  eight pairs do. Cat × dog is still the strongest pair by area, which is the other reason given,
  and that one holds.
- The review says seeds 9 and 11 of elephant × penguin are not monotone. On these records only
  seed 9 falls; seed 11 goes 0, 0, 0, 100%, 100%, which never decreases.

## The images

3.4GB of cells are not copied into this app. `public/full` and `public/figures` are symlinks to
where they live, and `public/thumbs` holds 416 JPEGs at 320px for the grids. Full size opens on
click.
