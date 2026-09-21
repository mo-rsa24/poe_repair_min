# Drift log

Where the tree was found to disagree with reality, and when.

## 2026-08-30: three journeys built for this repo were unreachable from it

The curriculum was not lost, it was unfindable from the place the work happens, which is the same
thing. `speciation-before-divergence` holds 12 plans and exists to answer the gap between the
correction's window and measured divergence, and scope 05, which asks exactly that question, did
not point at it. `trajectory-manifold-by-hand` holds 15 plans and exists to defend the manifold
figure's caption, and scope 01, which owns that figure, did not point at it.
`poe-composition-diffusion`, the parent journey with 57 leaves, was reached from nothing. All five
links that did exist pointed at just two journeys.

- Six links added, on words already in the sentences, so no prose was rewritten. Every in-scope
  journey is now reachable from at least one working document. 19 links in total, all resolving.
- The `learning-map.md` prompts are stale in a way rendering would hide: they draw four journeys
  and a single `plans/` tile. They are marked as needing regeneration before `/render-diagrams`
  touches them.
- The coarse route is still missing: this repo's `CLAUDE.md` journeys section names one journey of
  the six. That section is owned by `JOURNEY_FORMAT.md`, not by the weave, so it is reported here
  rather than drafted.

## 2026-08-30: the first rename attempt broke links, and was reverted rather than patched

Renaming 35 files into kind subfolders needs every inbound link recomputed, not string-replaced: a
plain substitution assumes only the target moved, but files that reference each other were both
moving, so the same substitution ran twice on some paths and produced `plans/figures/figures/...`
and `checks/02-....md` pointing nowhere. `git mv` marks a moved file as a rename in `git status`,
so a revert built from `grep '^ M'` silently skipped every renamed file's bad content, and the
first attempt looked clean until a full link-resolution pass, not a keyword grep, caught it.

The fix: for every moved file, its content was restored from the commit before any of this ran
(read at its old path, written to its new one), and the link rewrite was redone as a resolver that
recomputes each link from where the linking file used to sit to where its target now sits, rather
than pattern-matching text. Zero dangling links to an old filename remain anywhere in the repo,
verified with a full walk, not a sample.

## 2026-08-30: a plan scope was carrying 7 GB of build output

`find plans -name "*.png"` returned 1388 images, which is what surfaced it. Scope 03 held two Vite
scene apps: `scene-h04/`, a live app whose source is 316 KB beside a 3.5 GB `dist/` and 92 MB of
`node_modules/`, and `scene/`, whose source a previous tidy pass had already moved to
`artifacts/scenes/how-much-correction-is-needed/` while leaving 3.5 GB of build residue behind with
3377 files still tracked in git.

- Both moved to `artifacts/scenes/`, which is the kind that owns them. Nothing was deleted, because
  the ask was to archive rather than remove.
- `.gitignore` had no entry for `dist` or `node_modules` anywhere, which is how a build bundle came
  to be committed at all. It has one now, and the 1309 tracked build files were dropped from the
  index while staying on disk.
- Six images remain under `plans/`, all of them real diagrams: the project map's render, one each
  for scopes 01, 03 and 06, and two beside a scope 04 plan.

## 2026-08-30: work that had finished was still sitting in the live tree

`plans/` is defined as holding only work still to do, and `plans/.walk/` as holding unfinished
walks. Both had drifted: a completed scope and seven shelved ones sat in the live tree, and two
walks that had compiled and handed off their ledgers were still in `.walk/`.

- The relocations and the walk retirements are listed in `RENAMES.md`. Nothing was deleted.
- The five parked scopes had no resume block. Each now opens with one, so a reader arriving in six
  months learns what was last done and what to read before restarting, rather than guessing from
  ticked boxes.
- `plan_pulse.py` reports the same nine checks with the same counts before and after, so the moves
  changed location and nothing else.

## 2026-08-29: the plan tree had a parent layer that carried no order

`plans/` showed two live folders, and one of them hid six scopes a level down. The root
`MASTER_PLAN.md` already owned the mission, the one running order and the scope-state table, so
the parent `closing-the-compositional-gap/` carried no order of its own and cost every reader a
hop. A seventh scope of equal standing, `is-the-gap-the-samplers-or-the-models/`, sat at the top
level, so the listing showed neither the whole set nor any sequence.

- All seven scopes now sit directly under `plans/`, numbered `01` to `07` in the order a reader
  meets them: the shipped adapter, the instrument, the causal claim, the transfer claim, the
  commitment probe, the framing threat, the manuscript. The rename table is in `RENAMES.md`.
- The number is a reading order and nothing else. Step order stays interleaved in the root
  `## Running order`, where a plan's step number is permanent.
- The parent's unique prose (the run-kind filename vocabulary, the figures-first rule) moved into
  the root `MASTER_PLAN.md`; the emptied shell is at
  `artifacts/plans/archived/closing-the-compositional-gap/`.
- Links were resolved against each file's old location and re-emitted from its new one, because a
  string swap would have left every escaping relative link inside a promoted scope two levels too
  deep. Dangling links were counted at the checkpoint commit and again after: the move added none.

## 2026-08-29: two sessions authored the showcase scope concurrently

Two live sessions populated the same scope in parallel. One (this log's author) merged the
walk ledgers into `plans/closing-the-compositional-gap/plans/showcase-the-trained-lora/` and
wrote plans 01 to 03 with reviews; the other absorbed those files, authored its own 04 and 05,
grew the ledger further, and renamed the scope to `showcase-the-trained-adapter/`. Arbitration
by the user, recorded here so neither session re-litigates it:

- The other session finishes the scope; this session stands down from writing into it.
- Once that session goes quiet, the scope is renamed back to `showcase-the-trained-lora`
  (the repo's struck-words list says prose uses "lora"), with a RENAMES.md row and a
  reference sweep. The rename never happens while the other session is writing.
- References that currently look stale on purpose, because they anticipate the rename-back:
  the root `MASTER_PLAN.md` running-order rows 31 to 35, the parent scope's Sub-Scopes entry,
  and `runbook/looking-at-what-a-run-produced/reading-a-training-run.md` line 8, all naming `showcase-the-trained-lora`.
- Done 2026-08-29, after 20 quiet minutes: the scope renamed back, references swept
  (root running order, parent Sub-Scopes, scope-internal paths and title). The drips folder
  `artifacts/drips/showcase-the-trained-adapter/` keeps its name; it is the walk's working
  folder, not the scope.
- The shipped-artifact identity is `phase1_r8_100k` (animals-compose); the cross-seed run
  `pueuo7bl` is a replication datum only. Decided twice, independently, and merged.
