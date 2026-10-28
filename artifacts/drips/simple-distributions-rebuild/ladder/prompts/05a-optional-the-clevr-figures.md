# Layer 5a, optional step: the paper's CLEVR figures

⬅️ [Layer 5a, optional: the CLEVR figures](../05a-reading-stop-projective-composition.md#-optional-expensive-the-papers-clevr-figures)

Optional and expensive. Paste into a session opened in
`/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`, after the combined read of
step 2 has closed.

```text
/drip-execute-plan --reproduce 2502.04549

Background
The reading stop of the composition ladder (artifacts/drips/simple-distributions-rebuild/) read
Bradley et al. 2025a, arXiv 2502.04549, against 2D toys and the parts world. The paper's own
evidence is CLEVR figures made with NVIDIA's EDM2. A separate repo,
/home-mscluster/mmolefe/Playground/PhD/projective-composition, already holds two trained EDM2
checkpoints and an untried composition script, which make Figure 3 C reachable without new
training. This walk reproduces only the figures the toys cannot stand in for.

Objectives
1. At the open, print the target table: each of the paper's CLEVR figures with its cost in cluster
   hours, and let me pick which to run.
2. For each target I pick, write its bar before it runs (for Figure 3, the hand count on 100 samples
   per method against the paper's).
3. Reproduce one target per round on mscluster.
4. Per target, quote the paper's value beside ours as ran, with a verdict.
5. Mark every row of the target table ran, disagrees or out of reach, with its reason.

What to read
- In the journey folder
  ~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/, the
  read-ahead sources/mechanisms-of-projective-composition/read-ahead/: the target list.
- The same journey's drips/read-mechanisms-of-projective-composition/READ_MAP.md: where the read
  stopped, the object section and the slips.
- /home-mscluster/mmolefe/Playground/PhD/projective-composition, its README.md and
  runbook/training-edm2-on-clevr.md: the repo, the data, the checkpoints and how training ran.
- artifacts/results/joint-against-poe-by-kind-of-composition/README.md: the CLEVR grids already in
  this repo.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The authors released no code; their CLEVR models were built on NVIDIA's EDM2, cloned at
  reference/edm2 in the projective-composition repo at commit 4bf8162 (READ_MAP.md, Position; the
  repo's README.md).
- The paper trained edm2-img64-xs (124M weights) at batch 2048 on 128 × 2^20 images, 16 hours on 32
  A100s, without classifier-free guidance, sampling with a 32-step Heun sampler (runbook, quoting the
  paper's appendix).
- On mscluster, CLEVR v1.0 is at /datasets/mmolefe/projective-composition/data/CLEVR_v1.0, with two
  20,000-image subsets at 64 × 64, clevr-color-20k-64 (one random object labelled by colour) and
  clevr-none-20k-64 (unlabelled), and EDM2-xs checkpoints clevr-color-20k-64-xs-8Mi (conditional)
  and clevr-none-20k-64-xs-8Mi (background) under
  /datasets/mmolefe/projective-composition/checkpoints/, last snapshot network-snapshot-0008388
  (runbook steps 2 and 3; the checkpoint listing).
- scripts/compose_clevr.py composes two conditional denoisers minus the background with EDM2's Heun
  sampler, and it has run only on untrained step-0 snapshots (runbook step 4).
- Training runs in co3_bw on a Blackwell node (mscluster110), the cluster's EDM2 copy needs the
  InfiniteSampler fix on torch 2.11, and two jobs together run at 6.43 s per thousand images each
  (runbook steps 1 and 3).
- The official CLEVR holds 3 to 10 objects per scene (70,000 training images, 17.7 GiB); a subset
  reaches the cluttered composition (Figure 3 C, Table 1's cluttered rows), and the single-object
  and empty-background experiments need renders nobody released (READ_MAP.md, the object section).
- The paper's measure is a hand count of 100 samples per method, two "Single-object Bayes" rows of
  Table 1 sum to 109 and 43, and Appendix B's 65 network calls in 32 Heun steps is 63 in EDM2's
  sampler (runbook step 5; READ_MAP.md, Disagreements and slips).
- The CLEVR grids in artifacts/results/joint-against-poe-by-kind-of-composition/ come from Liu et
  al.'s clevr_pos and clevr_rel checkpoints, seed 0, with a 100-step ancestral sampler, so they
  stand in for no EDM2 figure (that folder's README.md).

Rules
- Spend cluster hours only on targets I pick, per figure, at the open.
- Write nothing into the read-ahead or READ_MAP.md.
- No commit.
- The open prints the target table with each target's cost and waits for my pick; each round prints
  one target's two values and its verdict.

What to hand back
- Per target, the paper's value quoted beside ours as ran, with a verdict, under
  artifacts/results/<the question the target answers>/.
- The target table, marking each figure ran, disagrees or out of reach with its reason.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: the target table, every chosen row carrying a verdict or an out-of-reach reason.

What you will see
Along the way: the target table with costs at the open, a yes asked per figure, then one target per
round with its two values and its verdict.
At the end, a closing report: what was asked, in one sentence; what was done, with each results
folder's path; PASS or MISS per chosen target against the bar written for it at the open; what is
still open, including targets out of reach; at most one question.

I judge, from the target table: done when each chosen target has met or missed the bar written for it before it ran, and every other row carries its reason.
```

added: the seven sections, from the field-labelled version · the trained checkpoints and the untried composition script as Background (from the repo's README and training runbook)
dropped: nothing
