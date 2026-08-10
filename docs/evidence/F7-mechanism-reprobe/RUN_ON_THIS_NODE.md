# Run this on the compute node

You opened VS Code on `mscluster109` or `mscluster111` to use the idle GPU
directly, instead of queueing. This file is the handoff: it assumes the reader
knows nothing about the session that wrote it.

Written 2026-08-05.

## The one command

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
bash scripts/mechanism_study/run_sweep_on_this_node.sh
```

That is the whole job. It checks for a GPU, runs 64 cells, and scores them.
Roughly 4 to 8 hours (each cell is a 50-step SDXL trajectory with two extra
forward passes at three of the steps).

To survive the terminal closing:

```bash
nohup bash scripts/mechanism_study/run_sweep_on_this_node.sh \
  > results/mechanism_study/sweep_$(hostname).log 2>&1 &
tail -f results/mechanism_study/sweep_*.log
```

Interrupting is safe. A cell whose manifest exists is skipped, so Ctrl-C and
re-run picks up where it stopped. Running it twice by accident does no harm.

## What it is measuring

Plan 02 of the `does-the-correction-cause-composition` scope, the mechanism re-probe. It decides
Goal 6 of that scope's master plan.

The question: when the trained LoRA is switched on, does it change **what** a
word paints, or **where** that word looks? The scope's account says the fix
lives in the value channel (what), not the attention weights (where). That
finding currently rests on a single seed of a single pair, and Goal 6 asks
whether it survives held-out pairs and seeds.

Each cell runs one denoising trajectory, and at steps 10, 25 and 40 captures
the attention weight map and the painted-content map for each of the pair's two
subject words, once with the adapter off and once with it on at the identical
latent state.

8 pairs (the 6 unseen transfer pairs plus the known-failure reference and the
dissimilar control) x seeds 9 to 16 = 64 cells.

## Reading the verdict

The script scores automatically when it finishes. To re-score at any point,
including partway through:

```bash
python scripts/mechanism_study/reprobe_table.py
```

It prints a per-pair table and a verdict, and writes
`/datasets/mmolefe/poe_repair_min/outputs/interaction_term/reprobe/verdict.json`.

**The bar was pre-registered before the sweep ran** (it is in the scorer's
source, `MIN_MEDIAN_RATIO` and `MIN_FRACTION_ABOVE_ONE`):

> median ratio >= 1.2x **and** >= 75% of rows above 1

Both conditions, so one strong pair cannot carry a weak average.

- **Replicates** → the paper's mechanism section proceeds.
- **Does not replicate** → the section shrinks to an honest negative paragraph,
  which the plan already provides for. **Do not loosen the bar to rescue it.**
  A negative here is a result, not a failure.

## The measure, and why it is not the obvious one

This matters more than anything else in this file, because the obvious measure
gives the opposite answer.

The natural thing is `||on - off|| / ||off||` for each map. Under that measure
the weight maps change **1.70x more** than the content maps, which contradicts
the hypothesis.

That reading is wrong. Weight maps are row-stochastic; content maps carry value
magnitudes; the two norms are not on the same footing. And the adapter dims the
attention weights by about 25% overall (raw sum 287 -> 215 on the cell that was
checked). That uniform gain change swamps the spatial-pattern change the
hypothesis is actually about.

The fix: fit the single best rescale `alpha` of the off-map onto the on-map,
then separate

- **gain**, `|alpha - 1|`, how much is just uniform brightness
- **pattern**, `||on - alpha*off|| / ||on||`, what a rescale cannot explain

and compare the **pattern** terms. On that measure content changes 1.5 to 2x
more than weight, supporting the hypothesis.

`gain_and_pattern()` in `value_probe.py` computes this per cell during capture,
so the scorer reads it rather than re-deriving it. Full argument with the
guards (shuffled-map control, denominator check, raw sums) is in
`docs/evidence/F7-mechanism-reprobe/measure-fairness.md`.

## What is already known

Four cells were run in-session before the sweep, under a different output root
(`outputs/attn_mechanism/value_probe/`), so the sweep will redo them under its
own root. Their numbers:

| pair | median content/weight pattern ratio |
|---|---|
| a_frog__x__a_toad | 2.03 |
| a_cat__x__a_dog | 1.89 |
| an_eagle__x__a_hawk | 1.55 |
| a_seal__x__a_walrus | 1.27 |

28/28 rows above 1, median 1.84x, with 5.7x headroom over a shuffled-map noise
floor. Encouraging, but four cells on one seed each is not replication, which
is the entire point of running 64.

`a_seal__x__a_walrus` reads lowest. Worth watching: it is the pair whose second
subject the tokenizer splits into `wal` + `rus`.

**One cell of the real sweep is already done**, to prove the path works:
`a_leopard__x__a_jaguar` seed 9. It reads **1.15x, below the 1.2 bar**, and the
scorer correctly returns "does not replicate" on that single cell.

Do not read that as the answer. One cell of 64 decides nothing, and the four
smoked pairs above range 1.27 to 2.03. But it is a useful sign that the bar can
fail rather than rubber-stamping whatever arrives, and it is a warning that the
result may be closer than the smoke suggested. Let the 64 cells speak.

## Two bugs that were fixed before this sweep

Both would have silently produced a wrong verdict.

**The token map was hardcoded to index 2.** That is the subject in "a cat" and
"a dog", but three pool pairs split into pieces: `a_seal__x__a_walrus`
(`wal`+`rus`), `a_gorilla__x__a_chimpanzee` (`chim`+`pan`+`zee`), and
`a_dolphin__x__a_porpoise`. The probe would have measured a word fragment with
no error raised, and `a_seal__x__a_walrus` is in this sweep. The map is now
derived per pair from the tokenizer and verified before sampling; the probe
averages over the pieces. Checked across all 19 pool pairs, zero mismatches.

**The measure was unfair**, as described above.

## If something goes wrong

**"no CUDA device visible"** — the node has no GPU or it is not exposed to this
session. Fall back to the queue: `sbatch
scripts/mechanism_study/value_probe_sweep.sbatch`.

**Warning that the GPU is already in use** — someone else's process is on the
card. Two SDXL processes on one GPU will run out of memory. Check `nvidia-smi`;
either wait or use a different node. This is the real hazard of running outside
Slurm, since Slurm does not know this work exists and may schedule someone onto
the node.

**A few cells fail** — they are named in the log and counted at the end.
Re-running the script retries only those.

**Out of memory partway through** — re-run. Finished cells are skipped, so it
resumes.

## Queued jobs from the earlier session

Two jobs may still be pending. If the node run finishes the work, cancel them
so they do not redo it:

| job | partition | why it is stuck |
|---|---|---|
| `26881` | biggpu | the same sweep. biggpu allows one job per user and `24806` holds the slot |
| `26906` | bigbatch | a 3-minute probe testing whether bigbatch has a usable GPU |

```bash
squeue -u mmolefe          # check
scancel 26881              # only if this node's run has finished the cells
```

`26881` is harmless if it does start: it skips cells that already exist.

## After the verdict

Mark plan 02 in `plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/plans/hypothesis-01-what-the-fix-changes-inside-the-model.md`, record
the verdict either way, and update the Plans list in
`plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/MASTER_PLAN.md`.

Plans still open in that scope: 03 (dose-response, the causal headline), 04
(window timing), 05 (cache analyses, task 1 done, five tasks left and all
cache-only), 06, 07, 08, 09, 10, 11.

Plan 05's remaining tasks need no GPU at all, so they are the natural thing to
do on a login node while a sweep runs.
