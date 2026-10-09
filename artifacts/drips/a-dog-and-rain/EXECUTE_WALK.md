# A dog and rain: execute walk

Walked with `/drip-execute-plan --request`, no plan file (the reader's choice). The question: when
SDXL is asked for a dog and rain, does the joint prompt draw a wet dog while plain PoE draws a dry
dog with rain behind it, and what do the per-concept guidance directions, the text embeddings and
the image embeddings show about why.

**Branch** `exec/dog-and-rain`, local worktree `/home/molef/PhD/poe_repair_min-dog-and-rain`,
cluster worktree `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`. Launch with
`CLUSTER_DIR=<cluster worktree> scripts/cluster.sh run "..."` so the shared cluster checkout is
never switched off the other sessions' branch.

**Decisions** seeds 1, 2, 4, 9, 10. Prompts stay "a dog and {B}" for every B word, so one word
changes between sheets. SuperDiff on SDXL and SD 1.4 only (the SuperDiff paper's own model); SD 2.1
and SD 3.5 SuperDiff cells grey. A tile's wet or dry call is by eye, with the CLIP "a wet dog" minus
"a dry dog" score beside it.

**Bar, written before any render** the reader's guess holds if the joint prompt draws a wet dog on
at least 3 of 5 seeds and PoE on at most 1 of 5 (the 6-of-10 and 3-of-10 bar, scaled to five seeds).

**Toy** seed 10, `a_dog__x__rain`.

## The cut

| # | Piece | State |
|---|---|---|
| 1 | SDXL sheet: dog alone, rain alone, joint prompt, PoE, SuperDiff, 5 seeds | four columns rendered (job 65969); SuperDiff at 200 steps rendering (job 65970) |
| 2 | One seed through the steps: the dog push and the rain push drawn as two arrows in their own exact 2D plane, PoE as their sum, the joint push projected in with its out-of-plane length, SuperDiff's weight per concept | open |
| 3 | Embeddings: CLIP text vectors of the B words drawn in the plane of "a dog" and "rain"; each render scored wet against dry | open |
| 4 | The sheet again for raining, wet and water, plus the wet-minus-dry score per word and method | open |
| 5 | Seed 10, models as rows (SD 1.4, SD 2.1, SDXL, SD 3.5), same columns | open |

## Out in the world

| id | serves | what | state |
|---|---|---|---|
| 65969 | piece 1 | `PAIRS="a dog\|rain" SEEDS="1 2 4 9 10" sbatch scripts/concept_pair_sheet.sbatch`, log `/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/logs/pairsheet-65969.out` | done, 25 tiles; its 50-step SuperDiff tiles are a shattered texture, kept as `superdiff.png` |
| 65970 | piece 1 | `COLUMNS=superdiff ... sbatch --partition=biggpu --nodelist=mscluster110 scripts/concept_pair_sheet.sbatch`, SuperDiff at 200 steps, tiles `superdiff_200.png` | submitted |

## For the close

- Repair: SuperDiff runs at its own 200 steps. At 50 its stochastic sampler leaves a shattered
  texture, the same failure the 50-step cells of the SuperDiff-defaults finding show.

- Piece 1: any grid of methods is one renderer per column plus a shared starting noise per row, so a
  difference along a row is the method and nothing else.
