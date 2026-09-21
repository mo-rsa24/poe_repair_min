# Handoff prompt: launch the parent and children on the v57 pool

Paste the block below into a new session. Everything it needs is already on disk; nothing has to
be rendered or judged first.

---

Train the rank-32 pooled adapter's parent and three children on the newly curated pool, in
`/home-mscluster/mmolefe/Playground/PhD/poe_repair_min`. Read
`plans/08-improve-the-rank-32-pooled-adapter/run-names.md` first for what each run varies.

**The pool.** `artifacts/_shared/cross_pair_pool_configs/cells_v57.json`, 29 pairs and 43 cells,
every cell judged good by eye. 22 animal pairs (23 cells) and 7 object pairs (20 cells). Its
pool, prompt and seed yamls are the `v57` generation in the same folder; pass `POOLGEN=v57` so the
trainer can resolve the object pairs' prompts, which live in the yaml and not in each cell's
`meta.json`.

**The runs.** One axis each, launched with `scripts/showcase/train_pool_run.sh`:

| RUN | CELLS | STEPRANGE | extra |
|---|---|---|---|
| `pool43-all50` | `cells_v57.json` | unset (all 50) | the parent |
| `pool43-early25` | `cells_v57.json` | `"0 25"` | |
| `pool43-all50-orth3` | `cells_v57.json` | unset | `EXTRA="--orth-weight 3.0"` |
| `pool43-animals` | `cells_v57_animals.json` | unset | drops the 7 object pairs |

Common to every arm, and they must match or the comparison is void:

    POOLGEN=v57 WD=1e-2 BATCH=1 EPOCHS=600 SAMPLEEVERY=25
    SAMPLETRAIN=a_lion__x__a_meerkat,a_typewriter__x__a_cactus SAMPLETRAINCELLS=1
    SAMPLEHELD=a_cat__x__a_dog,an_elephant__x__a_penguin SAMPLEHELDCELLS=2

`SAMPLETRAIN` must be overridden: the launcher defaults to `a_giraffe__x__a_lion`, which is not in
v57, so the default would label an untrained pair as in-pair.

**Read the result on** `eval/tracking/learned_actual_cosine/bucket_early` for
`out_out/a_cat__x__a_dog/seed_09` in W&B project `prime_lab/poe-repair-animals-compose`. Reference
points measured already: 0.46 to 0.51 for the adapter that composes, 0.34 to 0.46 for the four
runs that did not. Do not read `eval/compose_rate`: two runs differing only in weight decay
reported 0 and 1 on identical data, so it is noise here. Do not read the `/mean` of any eval
metric across runs either, because the tracking sets differ between run generations; compare the
same named cell.

**Facts already measured, so do not re-derive them.**

Training on all 50 cached steps beats steps [0, 25): `v54d_C2_allsteps` 0.458 against `v54d_P`
0.389 on cat x dog seed 9, with zero differing config fields between the two. That is why the
parent is all-50 and `pool43-early25` is the child.

Weight decay does not matter: 0.0 gives 0.289 and 1e-2 gives 0.317 on the same pool.

Batching is not a speedup: batch 2 costs 1.9x the wall clock for 2x the samples, and batch 4 runs
out of memory on a 47 GB card. Gradient checkpointing is slower, not faster, and is only for the
24 GB 3090s. Evaluation and inline sampling are 7.7% of wall clock, so leave sampling on.

Each arm is about 10 hours for 30,000 steps at 1.28 s/step.

**Environment.** biggpu (106, 107, 109, 110, 112) allows one Slurm job per user, so launch there
over ssh with `cd <repo> && setsid nohup env ... &` and pin the GPU index; the launcher's own
guard refuses a card another user holds. mscluster108 is down and mscluster111 reports
`[GPU requires reset]` and silently falls back to CPU, so never use it. bigbatch takes sbatch with
`-w <node>` and has no GRES, so pin the node or two jobs share a card; exclude
mscluster44,45,50,51,65,74,83. Check a log within a minute of launching.

**Two guards that must stay as they are.** `KILLAFTER` defaults to disabled, and it must: the
commit-bucket kill rule is checked every step and killed four healthy runs at step 6409 with exit
code 0. And never `pkill -f` a pattern that appears in your own command line; kill by explicit PID.

Do not run the old 88-cell reference pool. That comparison is deliberately out of scope here.

---
