# Layer 12, step 9: the build

⬅️ [Layer 12, step 9](../12-the-correction-at-scale.md#9-you-the-build)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan --request in ~/Playground/PhD/poe_repair_min: read the cached ε_joint, ε_A, ε_B
and ε_∅ for the 25 pairs as the layer 8 quantity, with the layer 9 correction as the reference
behaviour, sampled with tuned MALA; bar: the correction's effect on compose rate on held-out pairs,
reported beside the interaction term per pair

Background
This is layer 12's build in the composition ladder (artifacts/drips/simple-distributions-rebuild/),
its last layer. Layer 8 defined the interaction term and layer 9 trained a correction on the toys;
here I read the same term from poe_repair_min's SDXL cache and measure the correction's effect on
compose rate on held-out animal pairs. poe_repair_min is read and never written: I type every block
into my own script. I have passed quiz 12.2.

Objectives
1. Cut this into 4 to 6 pieces, and build, teach and test one piece per round.
2. I type every block myself into ~/Playground/PhD/simple-distributions-walk/walk12.py and run it
   with the python the node table in ~/Playground/PhD/poe_repair_min/environment/hpc/nodes.md names.
3. Read the cached ε branches for the 25 pairs as layer 8's ε-space term.
4. State how "tuned MALA" runs on the SDXL branches, or name the sampler used instead.
5. Print one row per held-out pair: compose rate without and with the correction, and the
   interaction term as the mean bound over steps 10 to 40, and save the rows as a CSV beside
   walk12.py.
6. Tick each piece in the step file and walk map the --request open writes.

What to read
- Step 5's refined transfer plan,
  ~/Playground/PhD/poe_repair_min/plans/04-does-the-fix-reach-unseen-pairs/plans/hypothesis/04-transfer-as-a-rate-over-fifteen-pairs.md:
  the compose-rate unit, the held-out pair set and the threshold.
- The layer 12 --math drip folder under artifacts/drips/ (step 7): my notation.
- walk08.py's ε-space term, and walk09.py with toy/rebuild/phase9_correction.py: layers 8 and 9.
- ~/Playground/PhD/poe_repair_min/runbook/running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md:
  how a checkpoint is rendered.
- poe_repair_min's report does-interaction-strength-predict-which-pairs-blend.md and its figure
  explainer: the cache and the bound.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The cache is /datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/, where one cell is
  one pair and seed (poe_repair_min report does-interaction-strength-predict-which-pairs-blend.md,
  "Where this came from"; its figure explainer).
- Each cell holds residuals/step_NNN.pt per step with eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond and
  timestep, for all 50 DDIM steps of the uncorrected run at guidance 7.5 (the same report; its figure
  explainer).
- The 25 pairs are the keys of pairs in
  /datasets/mmolefe/poe_repair_min/outputs/showcase/interaction_bound/interaction_bound.json, each
  with n_seeds, fail_rate and median_bound_by_step over steps 0 to 49 (the file).
- The cache folder now holds 1,165 pair folders under train/ and 58 under heldout/, far more than
  storage.md's "18 train + 58 heldout", so the 25 are named from that file and not from a listing
  (listed 2026-10-18; environment/storage.md).
- That bound, ‖r_t/σ_t‖²/4, did not track which pairs blend: every pair failing 8 of 8 seeds fails
  whether its bound is 100 or 3,000 (the report, rungs 1 and 2).
- From step 10 the bound separates two bands, roughly 100 to 400 and 1,300 to 3,200, by how far apart
  the two experts are (the report, rungs 1 and 2).
- The residual is w(ε_J − ε_A − ε_B + ε_∅) at w = 7.5, layer 8's term times the guidance weight
  (step 7).
- The LoRA is rank-8 cross-attention on that residual, and a checkpoint is lora_step_<N>.pt
  (runbook/running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md).
- The repo renders a checkpoint with lambda_boundary_probe.py: each held-out seed's pinned latent, 50
  steps adding λ times the adapter's correction inside a window, then scored (the same runbook
  recipe).
- A compose rate is the fraction of a pair's seeds where grounding-dino-tiny finds at least two
  distinct animal instances; it cannot tell a cat from a second dog (context/world/compose-rate.md).
- The 94% headline at full correction is an upper bound on a true rate around 87% to 94%
  (context/world/compose-rate.md).
- Training seeds are 1 to 8 and 17 to 19, and held-out seeds 9 to 16 (environment/known-failures.md,
  entry poe-data-001).
- 12 cells in 4 pairs sit in both split folders and resolve held-out first unless split="train" is
  passed (environment/known-failures.md, entry poe-data-001).
- MALA's accept step needs an energy, and the cached SDXL branches are noise predictions with no
  energy, so the walk states how "tuned MALA" runs here (an energy parameterisation, as Du et al.
  2023 use) or names the sampler it uses instead (REBUILD_WALK.md, brick "MALA and CHA fix" and its
  linked plan on the accept step; the when-poe idea map, Sources).
- inf_sample.py with anneal_samplers.py at this repo's root composes composable_diffusion CLEVR
  models (its captions name cube, sphere and cylinder), and it does not load SDXL (inf_sample.py).
- A caveat to report beside the result: samplers recover a third to four fifths of the exact A∩B
  mass at 2% to 10% overlap (layer 5).
- A second caveat: trained experts cost more under a better sampler, +0.054 total variation at
  r = 0.05 and +0.232 at r = 0.001 (the when-poe idea map, run 7).
- A third caveat: the interaction bound did not track which of the 25 pairs blend (the report above).
- GPU work runs on a card with under 1 GB in use (environment/hpc/execution-protocol.md).
- biggpu allows one Slurm job per user, so a second job goes by the shared-device path
  (environment/hpc/execution-protocol.md; the transfer plan, Environment Facts).

Rules
- Never edit poe_repair_min's code, plans or cache, or anything under /datasets/mmolefe/poe_repair_min/.
- Never launch on a busy card.
- I type; nothing is pasted into walk12.py.
- No commit.
- Show each of these before moving on: the first piece's four ε shapes and timestep for one cell;
  the interaction piece's 25 rows; the first timed pair's seconds per sample, which sets the GPU
  time; the last piece's table.

What to hand back
- walk12.py, printing one row per held-out pair: compose rate without and with the correction, and
  the interaction term (the mean bound over steps 10 to 40).
- A CSV of those rows beside walk12.py.
- The step file and walk map the --request open writes, with each piece ticked.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk12.py prints the per-pair table and the count of held-out pairs it covers.

What you will see
Along the way: one piece per round, each ending on the number it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk12.py's path
and the CSV's; PASS or MISS against every held-out pair in step 5's set having both compose rates and
its interaction term, each traced to a file, with step 5's threshold applied; the three caveats with
their numbers; what is still open; at most one question.

I judge, from walk12.py's table: done when every held-out pair in step 5's set has both compose rates and its interaction term, each traced to a file, with step 5's threshold applied.
```

added: the seven sections, from the field-labelled version · the three caveats named in the closing report (from the old Settled facts line "Caveats to report beside the result")
dropped: nothing
