# Retrofit ledger

One dated section per sitting. This is the one file in the sweep that carries history.

## Sitting 2026-08-23

**Where we are.** The sweep stopped at its census because `artifacts/` meant two contradictory
things, and the walk that would have decided it had never compiled. That walk is now resumed and
running at `plans/.walk/retrofit.md`. One of its eight piles is fully settled: the run bytes, which
was the blocker. Seven piles and the look pass remain, and nothing has moved yet.

Shape: `artifacts/<kind>/<grouping>/` per `~/.claude/ARTIFACT_TREE_FORMAT.md` v2.
Git: 49 uncommitted paths. `/datasets` verified reachable, 348T free. `/datasets` is not a git
repository, so every move there is permanent.

| Stage | State |
|---|---|
| 0 census | verified |
| the walk | pile 1 of 8 settled, piles 2 to 8 open |
| 1 to 7 | not started, blocked on the walk compiling |

---

## The three constraints everything else follows from

**Names say what varied.** The `rung1` to `rung4` prefix is gone, and so is any vague root or
vague suffix. A name states what was measured against what was varied and on which data. It never
states the claim, because a claim can die and leave a lie on disk.

**Every artifact gets opened.** No disposition is inferred from a filename. A set of
coordinate-named cells counts as one thing, judged once and shown by one cell, which is what makes
4,259 images on the mount finishable in eleven judgements.

**Animal pairs are the boundary, and group-cut work stays out.** The 20 animal pairs are in. The 7
pairs from the earlier cross-taxonomy era stay. Anything cut by group stays, which costs 833M and
strands no per-pair directory, because every group-cut folder is a pooled run.

---

## 1. Renaming

Nine run families and two rung levels. The names come from documents already in the repo, so none
is invented.

| Now | Becomes | Why the old name fails | Grounded in |
|---|---|---|---|
| `artifacts/rung1-overfit/lora/` | `artifacts/results/can-lora-learn-a-residual-that-corrects-poe/one-pair-one-seed/` | `rung1` is a private label; `lora` names the technique, not what varied | the folder holds one pair, one seed |
| `artifacts/rung2-survive-noise/cross_seed/` | `artifacts/results/can-lora-learn-a-residual-that-corrects-poe/held-out-seeds/` | says the same thing twice, once in words that mean nothing | the level below already carried the axis |
| `outputs/group_a_failure/` | `artifacts/results/residual-dynamics/correction-outside-the-unet/` | `group_a` is a private label and `failure` states the claim | `docs/results-archive/group-a-failure.md`: "a standalone network... SDXL frozen end-to-end" |
| `outputs/animals_compose_transfer/` | `artifacts/results/does-the-fix-reach-unseen-pairs/` | names the mechanism, not the question | `fail_rate.md`: 16 animal pairs with train, control and reference roles |
| `outputs/residual_diagnostics/` | `artifacts/results/residual-dynamics/residual-between-mono-and-poe/` | "diagnostics" of what, against what | `docs/results-archive/residual-diagnostics.md`: "r_t = ε̃_Mono − ε̃_PoE" |
| `outputs/conditioning_window/` | `artifacts/results/when-the-correction-must-arrive/cfg-window-without-lora/` | collides with `interaction_term/window`, and nothing says which is the baseline | `docs/results-archive/conditioning-window.md`: "the no-LoRA baseline" |
| `outputs/conditioning_window_lora/` | `artifacts/results/when-the-correction-must-arrive/cfg-window-with-lora/` | same collision, other half | its paired half |
| `outputs/poe/` | `artifacts/results/poe-blends-instead-of-composing/poe-baseline-samples/` | three letters that could mean the whole project | `outputs/INDEX.md`: "λ=0 exemplars for F1" |
| `outputs/compose_scorer/` | `artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/` | reads as the scorer's code, not its evidence | `outputs/INDEX.md`: "the contract `scorer_validated.json`" |

**On the mount, seven more names are owed and four are already fine.**

```
already fine:   window
renamed:        dose            → how-much-correction-is-needed
                dose_matched    → same-total-correction-different-window
                xhat0_readback  → predicted-clean-image-per-step
                cache_analyses  → dissolved; its 16 figures go to the questions they answer
owe a name:     cross   experts   reprobe   window_seeds
                overcorrection_grid   live_curves_smoke_run
```

The six cannot be named until they are opened, which is what the look pass is for.

**The shape, as settled after the census.** `artifacts/` gains `results/` and loses `figures/`,
which `results/` absorbs. Grouping is by question, never by medium, so one question's folder holds
its figures, its numbers and its checkpoints together. The split inside it is what can be
committed: a curve is small and lives in the repo, a checkpoint is not and lives on the mount,
named by the card. `scenes/`, `drips/`, `notes/`, `decks/` and `videos/` stay.

```
artifacts/results/<the-question-the-experiments-asked>/
  <figure>.png + <figure>.json          committed
  <what-was-held-out>/<pair>/<run-tag>/  on the mount, named by the card
```

**The rule that blocked all of this is replaced.** `outputs/INDEX.md` says today:

> Folder names are not renamed: scripts write into these exact paths.

The replacement: a run family's folder name is renamable, because no script spells it. Every path
comes from one module that names each family once, so a rename is one edit there plus a move on
disk. `poe_repair/config.py` already resolves the output root from `POE_REPAIR_OUTPUT_ROOT` and
stops there; it gains one constant per family, 159 files import them, and the hardcoded
`/datasets/mmolefe/poe_repair_min/outputs/...` strings become that same root so one script runs
against either filesystem.

**This is the first task and nothing moves before it is green.** 159 code files touch `outputs/`:
`interaction_term` 73, `animals_compose_transfer` 25, `compose_scorer` 20, `residual_diagnostics` 7,
`conditioning_window` 7, `conditioning_window_lora` 5, `group_a_failure` 3, `presentation` 2,
`poe` 1.

---

## 2. Moving

**Run bytes into one root, grouped by the question they answer, on whichever filesystem already holds them.** The same relative path on
both, so one name means one thing.

```
artifacts/rung1-overfit/lora/a_cat__x__a_dog/
  →  artifacts/results/can-lora-learn-a-residual-that-corrects-poe/one-pair-one-seed/a_cat__x__a_dog/
outputs/poe/
  →  artifacts/results/poe-blends-instead-of-composing/poe-baseline-samples/
/datasets/.../outputs/interaction_term/
  →  /datasets/.../artifacts/results/   (same relative path on both filesystems)
```

**The mount's mirrored `artifacts/` merges in too**, so the collision is gone from both sides.

**`interaction_term` is one family split across two filesystems, and it merges onto the mount.**

```
repo  outputs/interaction_term/          mount  outputs/interaction_term/
  canary          1.4M                     cross              10G
  direction_wall  2.6M                     dose              6.3G  ← the real one
  dose               0  ← empty husk       window            6.0G  ← the real one
  dose_smoke      7.7M                     window_seeds      693M
  noise_slice      32M                     live_curves       398M
  noise_slice_smoke 5.1M                   overcorrection    303M
  seed_signature  256K                     experts           148M
  window           16M  ← partial          reprobe           104M
                                           dose_matched       70M
                                           xhat0_readback     15M
                                           cache_analyses    6.5M
```

Six folders exist only in the repo and five only on the mount. `dose` in the repo is a zero-byte
husk left by the 6.3GB migration. `window` is 16M against 6.0G and gets diffed before either is
touched. `outputs/INDEX.md` still calls the repo copy 3.4G, which it has not been since the move.

**Scene source out of the plan tree.** A built scene is an artifact and `PLAN_TREE_FORMAT.md` has no
slot for one.

```
plans/.../does-the-correction-cause-composition/scene/{src,public,loader,package.json}
  →  artifacts/scenes/<grouping>/
```

Each gains the four-key source header: `ran under`, `built from`, `why`, `depends on`.

**Run logs to the folder that holds run logs.**

```
results/mechanism_study/dose_sweep.log   →  logs/dose_sweep.log
```

Ten files, filenames unchanged because they already say which run wrote them. `results/` then holds
nothing and goes, dropping the top-level count by one.

**The 33 image clusters into groupings**, after the look pass names them. The eight grouping names
already agreed stay, because the words are yours and only the shape changed:

```
what-poe-is-and-what-it-drops        poe-blends-instead-of-composing
how-much-correction-is-needed        when-the-correction-must-arrive
how-big-the-correction-is            which-way-the-correction-points
does-it-transfer-to-unseen-pairs     can-we-trust-the-compose-score
```

**The mount is referenced, never symlinked into.** Each grouping card names the mount path in prose
and commits one representative cell that the card embeds. A card that renders only while the mount
is up is not a card.

---

## 3. Archiving

Out of the working tree, into `artifacts/_quarantine/`, not destroyed.

| What | Size | Why |
|---|---|---|
| `recap/` and `results/recap_landed/` | 12.6M | A superseded summary of the July LoRA era. `RECAP.md` still says "G6 enactment: pending Slurm job recap_g6", which never landed. The convention archives a superseded recap rather than filing it, because filing it makes it look current. |

---

## 4. Deleting

Every one of these is untracked, so there is no undo. Each gets its own confirmation with its
version-control status stated first.

| What | Size | Regenerable by | State |
|---|---|---|---|
| `scene/node_modules`, `scene-h04/node_modules` | 196M | `npm install` from `package-lock.json` | proposed |
| `scene/dist` | 3.4G | `npm run build`, because `public/` holds its 9.1M | proposed |
| `scene-h04/dist` | 3.5G | **unknown** | **held, not proposed** |

**Why one is held.** `scene-h04/public` is empty and its `dist/experts` holds 495 images, exactly
the count in the mount's `interaction_term/experts`. So the pictures came from there and nothing in
the repo says how. Delete it before reading the loader and the scene cannot be rebuilt, only
re-derived by someone who works it out again.

---

## 5. Staying, as decisions rather than skips

| What | Size | Why |
|---|---|---|
| the group-cut folders: `within_group/g6`, `all_groups`, `recap_landed/gen/g6_survive_noise` | 833M | cut by group. None holds a per-pair directory, so nothing per-pair is stranded. |
| `within_group/{g1,g2,g3,g4}` | 4.0K each | empty husks |
| the 7 non-animal pairs | mixed | the earlier cross-taxonomy era: `a_ballerina__x__a_spacesuit`, `a_camel__x__a_desert_landscape`, `a_dog__x__oil_painting_style`, `a_dolphin__x__an_ocean_wave`, `a_mailbox__x__a_snowfield`, `a_park_bench__x__a_sand_dune`, `a_typewriter__x__a_cactus` |
| `data/pilot/` | 42M | genuine input, not output. `poe_repair/runtime.py:116` walks it for pair discovery and `_eval_common.py:32` reads it. `data/` is a named folder in the convention. |
| `composition/` | 118M | five cloned third-party repositories. Vendored files are never renamed. |
| `scripts/`, `poe_repair/`, `tests/` | 7M | code stays where the tooling expects it |

**One correction worth keeping visible.** `outputs/group_a_failure/` is **not** group-cut. "Group A"
names a family of correction methods (latent-CNN, latent-UNet, frozen-feature MLP) on cat/dog seed
42, not a pair taxonomy. It moves and renames.

**And the group taxonomy is the container, not a sibling.**
`data/pilot/seed_42/a_cat__x__a_dog/summary.json` says `"group_label": "Group 6 - Coherent
Collision"`. Group 6 is where cat×dog lives. The exclusion was kept anyway, on the measured cost
above.

---

## 6. What is too large, and where it should not be

**The repo is 36G on `/home-mscluster`, and 35G of that is run bytes and build output.** The
project's own rule in `CLAUDE.md` is that large artifacts go to `/datasets` only, because
`/home-mscluster` hit 100% once and silently killed checkpointing.

| On `/home-mscluster` today | Size | Should be |
|---|---|---|
| `outputs/` | 20G | `/datasets`, except what a card commits |
| `artifacts/` (the run bytes) | 8.0G | `/datasets` |
| the two built scenes | 7.1G | 6.9G deleted, 588K of source kept |
| `paper/` | 184M | stays, but see the zips |
| everything else | ~700M | stays |

**Four specific piles of waste, largest first.**

**One run kept 13 full checkpoints.** `outputs/group_a_failure/checkpoints/direct_eps/direct_eps_overfit_catdog_hg/snapshots/`
holds `step_0000600.pt` through `step_0002600.pt` at about 100M each, 1.3G for one training run of
one pair. Whether every step is needed or only the last is a question for the look pass.

**11 W&B run directories are duplicated on disk, 2.7G.** W&B is the tracker and already holds these
runs, so the local `wandb/run-*/` copies are a second copy of something the project's own convention
says W&B owns.

**The mount holds 73G nothing references.** The earlier survey found `synthesizer/` at 63G and the
four `veracity_cfg_*` folders at 9.7G named by no document anywhere, and `training_cache/` empty
with 31 references pointing at it.

**`paper/` carries 100M of Overleaf zips**, two of them superseded:

```
iclr-overleaf-20260820T104938Z.zip   61M   Aug 20
iclr-overleaf.zip                    36M   Aug 13
iclr-overleaf-slim.zip              2.3M   Aug 20
```

---

## 7. What is not decided yet

**Six piles.** `docs/`; the loose investigation folders (`evidence/`, `show-me/`, `captures/`,
`diagrams/`, `flow-map-images/`); `dl-scene/` and `pressure-tests/`; the code grouping; the record
folders (`inventory/`, `learning-captures/`, `todoist-staging/`); and the loose root files.

**The look pass**, which is minted and has a shape but no budget. Six sittings against three
questions:

1. What does it show? What is on y, what is on x, what one point is.
2. Does any document name it?
3. Is it the same picture as something later?

```
sitting  cluster                                  images  unit
1        paper/iclr/figures + drafts                  39  per image, 20 have .json sidecars
2        evidence/f2-lambda1-audit                    32  one set, one cell shown
3        docs/evidence, 7 subfolders                  22  mixed
4        evidence/h05, f5b, show-me, drips,           13  per image
         flow-map-images, diagrams
5        data/pilot, dl-scene, scene public,          17  per image
         recap figs, the 2 loose root images
6        the mount, 11 families                    4,259  11 sets, 11 cells shown
```

**Three things held for a look before they can be settled.** `outputs/presentation/` (6M of old
slides, no pairs). The two `window` copies (16M against 6.0G, same name). `scene-h04/dist`, whose
deletion is blocked on reading the loader.

**Three absent documents** the sweep will build once the walk compiles: `context/`, `runbook/`, and
`environment/` as a folder rather than the single environment file that preceded it.

---

## 8. What happens first, in order

1. **`/rename`**, plus the new paths module. Nothing moves until this is green, because 159 files
   spell the paths by hand.
2. **The look pass**, six sittings, so the seven owed names exist.
3. **`compile`** the walk into `plans/retrofit-poe-repair-min.md`.
4. **`/retrofit-repo`** against that plan, stages 1 to 7.

Renames this sitting: none. Nothing has moved.
