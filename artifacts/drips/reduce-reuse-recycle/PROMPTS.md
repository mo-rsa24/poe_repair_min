# Reduce, Reuse, Recycle: the prompt sequence

The prompts that take Du et al. 2023 (arXiv 2302.11552) from running code to experiments of our own,
in the order to paste them, with the machine each one runs on. This file is the single copy: it
travels to mscluster with the rest of `poe_repair_min` through GitHub, so both machines read the
same prompts.

## What this is for

**Understand when composing two diffusion experts works.** Three conditions have to hold: each
expert's data covers the place where both are true, the product of the experts equals the joint
distribution, and the sampler actually reaches the product. The first is your support intuition,
the second comes from the ChatGPT "Diffusion Sampling Basics" chat, and the third is Du et al.'s
argument.

**Reproduce the paper**, the 2D product, mixture and negation and then CLEVR, so the sampler
condition is something you have watched run.

**Test the conditions in 2D**, with bars written before anything runs: how much of one expert's
data has to sit inside the overlap, and whether negation works when the thing being kept never
appears without the thing being removed.

**Carry the answer to chest x-rays**: draw pneumothorax with the drain negated, train on those
images, and check whether the classifier's reliance on the drain shrinks. That work lives in
`~/PhD/xray/shortcut-companion`, whose drain-negation idea map rests on its claim 2: that
RoentGen-v2 draws pneumothorax with no drain when "chest tube" is negated.

**Learn only the maths each step needs**: Markov chains, Metropolis-Hastings, Langevin (ULA),
MALA, HMC, and the energy form with its accept step. No Fokker-Planck.

## Where each step runs

| Step | Machine | What it does | Needs first |
|---|---|---|---|
| 1 | mscluster88 | gets the authors' notebook and CLEVR script running | nothing |
| 2 | laptop | brings step 1's facts, walk file and pictures here | 1 |
| 3 | laptop | continues the paper read, running cells in the cluster notebook | 2 for the code-bearing sections |
| 4 | laptop | tests the three conditions as an idea, ending in experiments | 3 has reached section 4 |
| 5 | mscluster88 | reproduces the paper's figures and tables against written bars | 1 |
| 6 | mscluster88 | runs step 4's 2D experiments | 4 and 5 |
| 7 | laptop | merges every session from both machines into one record | 6 |

**Why the paper read stays on the laptop.** It writes notes into the Obsidian vault, and the vault
lives only on the laptop. The code runs on the cluster, so the read names a cell and you run it in
your cluster notebook.

**How results cross between machines.** Facts, recipes, walk files and plans travel as commits in
`poe_repair_min`. Pictures go under `artifacts/results/`, which git ignores, and come back with
`scripts/cluster.sh pull <path>`. Session transcripts do not travel at all, so step 7 copies them
by hand.

**The cluster checkout already has uncommitted files of its own** (26 on 2026-10-13). Every
cluster step commits only the files it names, and asks before it pushes.

## 1. Get the code running on mscluster88

Paste on mscluster88, in the VS Code session opened on the node.

```
/drip-repo ~/PhD/reduce-reuse-recycle

Resume the walk that gets the authors' code for Du et al. 2023, "Reduce, Reuse, Recycle" (arXiv 2302.11552,
github.com/yilundu/reduce_reuse_recycle), running on the cluster so I can follow along in the code: the 2D
notebook first, then CLEVR sampling. This session runs on **mscluster88** (bigbatch, Slurm job 67704, VS Code
remote opened here), so commands run on the node directly: no sbatch, no ssh hop. The walk began on my laptop
and continued on mscluster112 on 2026-10-13. Everything below was learned there, so do not rediscover it.

## Where things stand

- Clone: `/home-mscluster/mmolefe/PhD/reduce-reuse-recycle`, branch `walk` at `81177d9`. It exists. Do not
  re-clone.
- The four notebook fixes (drop `!pip install` cells, drop tensorflow and pandas imports, `a_min=` to `min=`,
  and a chex/jax shim cell after `import math`) are already applied to `notebooks/simple_distributions.ipynb`:
  47 cells, `git diff --stat` reads 15 insertions, 95 deletions. Do not run the script again.
- Walk file: `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/drips/reduce-reuse-recycle-mscluster112/REPO_WALK.md`.
  Read it first. It holds the probes and the run map with marks.
- Fact folders: `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/environment/` and `runbook/`. Read
  `environment/overview.md`, `environment/hpc/nodes.md` and `environment/storage.md`. The laptop's findings are
  in `environment/reduce-reuse-recycle-code.md` and `runbook/setting-up-the-reduce-reuse-recycle-code.md`:
  extend them with a section for the cluster (mscluster88), never a second file.
- Environments: mamba at `/home-mscluster/mmolefe/miniforge3/envs/`, uv at `~/.local/bin/uv` (0.12.7) for new
  ones. Never install into `co3` or `co3_bw`.

## What was probed on 2026-10-13 (already recorded in `environment/hpc/nodes.md`)

- mscluster112's card is faulted: `nvidia-smi` prints `Unable to determine the device handle for GPU0:
  0000:01:00.0: Unknown Error`. That is why the walk moved here.
- mscluster88: RTX 3090, 24 GB, driver 595.71.05. Under `co3_bw` (torch 2.11.0+cu128),
  `torch.cuda.is_available()` is True and a 1000×1000 ones matmul sums to 1e9. Still run `nvidia-smi -L` once at
  the start, because bigbatch cards fault between checks.
- The `jax` mamba env has no jax. `jax115` has jax 0.4.30 with a CPU-only jaxlib and no haiku. Neither is
  reusable, so the 2D notebook gets a new uv env.
- `co3_bw` already has matplotlib 3.10.8, tqdm 4.67.1 and numpy 2.2.6, so CLEVR (`inf_sample.py`) runs there
  with nothing installed.
- PyPI from 88: 13.5 MB/s (30 MB of a CUDA wheel in 2.3 s), so the 2.3 GB CUDA jax install is about 3 minutes.
- Disk: `/home-mscluster` 58% used, `/datasets` 41%. Large things still go under
  `/datasets/mmolefe/reduce-reuse-recycle/` with a symlink from the clone.

## Next step: build the 2D notebook env (I say yes to this now)

    R=/datasets/mmolefe/reduce-reuse-recycle
    mkdir -p $R/uv-cache $R/venvs
    export UV_CACHE_DIR=$R/uv-cache
    uv venv --python 3.12 $R/venvs/rrr-jax
    uv pip install --python $R/venvs/rrr-jax/bin/python "jax[cuda12]==0.11.2" dm-haiku optax distrax chex matplotlib ipykernel
    ln -s $R/venvs/rrr-jax ~/PhD/reduce-reuse-recycle/.venv

Success is `jax.devices()` printing a CUDA device, not a CPU one. Then tell me how to pick `.venv` as the
kernel in VS Code, and start the notebook rounds.

## What the laptop proved about the 2D notebook

It is JAX and Haiku. The failures the applied fixes prevent, in the order they hit: `TypeError: clip() got an
unexpected keyword argument 'a_min'` (raised at the first `hk.multi_transform`, not in the trainer cell);
`[Chex] Assertion assert_type failed ... float64 but expected float32; int32 but expected int64`; `module 'jax'
has no attribute 'tree_leaves'`.

Sections and what each should show: definitions (no output); trainer; seven toy datasets; product = ring of 8
blobs (`toy_gmm(std=.03)`) times a vertical bar (`bar(scale=0.2)`), so only the top and bottom blobs should
survive; mixture = `bar()` plus `right_bar()`; negation = `toy_box(0.7)` minus `toy_box(0.3)`, a square with a
hole. All three sample with `AnnealedMUHASampler` (adjusted HMC), not MALA.

Laptop timing, the number to beat: one expert (15,001 steps, a 4-layer MLP of width 128, batch 1000) took
13 min 7 s on the laptop's CPU, of which about 105 s was training (7.1 ms per step), about 40 s the 16
evaluation pauses (1.9 s per likelihood, 0.1 s per heatmap), and the rest JAX compiling and a second kernel
sharing the cores. The bar expert (the cell that ends `bar_params = ema_params`) then took 629.4 s on the same
CPU with the machine nearly idle, at a mean 12.1 ms per step. Time that same cell here on the 3090, and say
whether the GPU actually helped.

## What the laptop proved about CLEVR (`inf_sample.py`)

- It needs only torch, numpy, matplotlib and tqdm, and hard-codes `.to("cuda")`. Run it in `co3_bw`.
- Checkpoint `ebm-49x1874.pt`: 500,704,677 bytes, sha256 starting `9e9aa1326eecdcff`, 542 tensors, 125.1 M
  parameters. Put it under `/datasets/mmolefe/reduce-reuse-recycle/` with a symlink from the clone. Google
  Drive serves a virus-scan page, so fetch it with:
  `curl -sSL "https://drive.google.com/uc?export=download&id=19ay4bLVCQho8_aXtODWbDCpu6L2l58YB" -o /tmp/gd.html`,
  `UUID=$(grep -oP 'name="uuid" value="\K[^"]+' /tmp/gd.html)`, then
  `curl -sSL -C - --retry 5 -o <dest>/ebm-49x1874.pt "https://drive.usercontent.google.com/download?id=19ay4bLVCQho8_aXtODWbDCpu6L2l58YB&export=download&confirm=t&uuid=$UUID"`.
- It cannot run as released: the energy U-Net's `forward` in `composable_diffusion/unet.py` never assigns `h`
  before `h = module(h, emb)` at line 1386, so every call raises `UnboundLocalError`. The upstream
  guided-diffusion line `h = x.type(self.dtype)` is missing. Show me the one-line patch and wait for my yes
  before applying it.
- Run `--sampler Rev_Diff` first (plain reverse diffusion, the baseline), then `--sampler MALA`. The script
  writes `Energy_Object_A Sphere And A Cylinder_<sampler>_4.png` to the current folder.
- `dataset_train_larger.npz` is read only by training. Skip it unless I ask to retrain.

## Where the code differs from the paper (read before trusting any picture)

From a 60-row audit of this commit against the paper (on my laptop, under
`~/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/drips/read-reduce-reuse-recycle/routes/code-vs-paper.md`;
read it if the learning tree is checked out here, otherwise rely on these rows): R07 the 2D beta schedule is
cosine where the paper says linear; R13 the mixture multiplies energies by 3.5 inside the log-sum-exp; R14
negation is `1.3·E₁ − 0.3·E₂`, where the paper has α = 0.5; R20 the JAX MALA accept step matches the paper;
R21 the PyTorch MALA proposal divides by 2h² where it should divide by 4h, so at the audit's test point it
accepts with probability 1 instead of 0.978; R40 the 2D cells take 10 HMC steps per noise level where the paper
says 3; R59 the unbound `h` above. Say which rows a picture inherits when you show it.

## How I want it walked

I follow along in my own VS Code notebook on this node, running each cell myself. Each round: which cells to
run, named by their first line (VS Code shows no cell numbers); the one or two lines worth reading first; what
the output should be. When something fails, fix it in the files for me and tell me which cells to rerun. Write
each failure as an environment fact and each working step as a runbook recipe in the round that proves it. Ask,
do not infer, for anything about the cluster that `environment/` does not cover.

## Closing the walk, so the laptop's paper read can use it

- Save every picture the walk produces (product, mixture, negation, the two CLEVR samples) under
  `artifacts/results/what-the-authors-code-draws/` in the poe_repair_min checkout, with a `README.md` entry
  per picture saying which cell or command made it and which audit rows it inherits.
- Add a section to the walk file headed "Handoff to the paper read": one row per notebook section and CLEVR
  command, giving the cell's first line, what it showed, the picture's path, and the time it took.
- The clone's `walk` branch is pushed nowhere, so write its changes to
  `artifacts/drips/reduce-reuse-recycle-mscluster112/clone-changes.patch` with
  `git -C ~/PhD/reduce-reuse-recycle diff 81177d9 > <that path>`. The notebook fixes and the U-Net patch
  then survive the clone being deleted.
- Commit only the walk file, the patch, and the environment and runbook files this walk touched. Show me the
  branch and `git status` first, because this checkout has uncommitted files from other work, and push only
  after my yes.
```

## 2. Bring the cluster's results to the laptop

On the laptop, after step 1 has pushed.

```
/reconcile-machines poe_repair_min
```

Then fetch the pictures, which git ignores:

```bash
cd /home/molef/PhD/poe_repair_min
scripts/cluster.sh pull artifacts/results/what-the-authors-code-draws
```

What you want: `artifacts/drips/reduce-reuse-recycle-mscluster112/REPO_WALK.md` present here with its
handoff table, and the pictures opening locally. If reconcile reports a two-sided conflict, it stops
and shows it rather than picking a side.

## 3. Continue the paper read

On the laptop. It can start before step 1 finishes, since sections 2 and 3 need no code; the
code-bearing sections wait for step 2.

```
/drip-read --lean /home/molef/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/drips/read-reduce-reuse-recycle/READ_MAP.md
Resume from the map's Next step.
The authors' code runs on mscluster88, not here. Its handoff table is in /home/molef/PhD/poe_repair_min/artifacts/drips/reduce-reuse-recycle-mscluster112/REPO_WALK.md and its pictures in artifacts/results/what-the-authors-code-draws/. When a chunk is about something the notebook or inf_sample.py computes (section 4.2's samplers, 5.1's 2D results, 5.2's CLEVR, appendix D, appendix E's negation), name the cell by its first line and what to look at; I run it in my cluster notebook. Read any output you need with scripts/cluster.sh sh. Never rerun the authors' notebook on this laptop: one expert takes 13 minutes on its CPU. The small objects I build by hand stay in the playground here.
At a section close where several concepts stopped me, give me the /drip-execute-plan --from-read line rather than teaching them inline.
```

## 4. Test the three conditions as an idea

On the laptop, once the read has reached section 4.

```
/drip-idea Composing two diffusion experts succeeds only when each expert's data covers the place where both are true, the product equals the joint (broken at noisy t by r_t), and the sampler reaches the product. A label is learned as a contrast against unlabelled images, so a confound like a drain rides inside the pneumothorax expert, and negating it needs both a drain label and pneumothorax-without-drain images in the expert's data.
First save a clean markdown copy of /home/molef/PhD/poe_repair_min/.playwright-mcp/share.txt (the ChatGPT "Diffusion Sampling Basics" chat) to /home/molef/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/sources/.
Must cover from the chat: the product equals the joint only under conditional independence; r_t as the gradient of that log ratio; labels that are deterministic at t=0 becoming dependent at noisy t; the multi-condition CFG form s₀ + w(s_A + s_B − 2s₀); the mismatch between curated LoRA experts and SDXL's own unconditional branch.
Use what the authors' code drew (artifacts/results/what-the-authors-code-draws/ in poe_repair_min) as the first evidence.
The verdict feeds claim 2 of /home/molef/PhD/xray/shortcut-companion/artifacts/ideas/drain-negation-shortcut/IDEA_MAP.md: does RoentGen-v2 hold pneumothorax-without-drain mass for negation to reach?
Want out: 2D experiments with bars written first. The share of expert B's training points inside the overlap swept over 0, 1, 10 and 50%, one curve per sampler. Negation with and without support. The prediction that HMC raises the hybrid rate when the product's peak is itself a hybrid.
```

## 5. Reproduce the paper's figures and tables

On mscluster88, using the environments step 1 built.

```
/drip-execute-plan --reproduce 2302.11552
Run on this node with the environments the reduce-reuse-recycle walk built (walk file: artifacts/drips/reduce-reuse-recycle-mscluster112/REPO_WALK.md). Targets in this order: Table 1 and Figure 2 (2D product and mixture), Figure A1 and appendix E's negation, then Table 2 (CLEVR, 1 to 5 cubes). Write each bar before running. Name the audit rows each target inherits (they are listed in step 1 of artifacts/drips/reduce-reuse-recycle/PROMPTS.md); mark a target out of reach, with the reason, when an audit row or a missing released piece (such as the CLEVR position classifier Table 2 scores with) makes it unreproducible.
Close the same way as the repo walk: pictures under artifacts/results/, commit only the files this walk touched, show me the branch and git status, push after my yes.
```

Then repeat step 2 on the laptop.

## 6. Run the 2D experiments of our own

On mscluster88.

```
/drip-execute-plan --request <paste the experiment list drip-idea's compile produced in step 4>
2D only, run with the reduce-reuse-recycle walk's environments on this node. The x-ray work waits until the overlap and negation sweeps have a verdict, and when it starts it goes through ~/PhD/xray/shortcut-companion's plans, not this repo.
Close as in step 5.
```

Then repeat step 2 on the laptop.

## 7. Merge both machines' sessions into one record

`/merge-sessions` finds a session by its file under `~/.claude/projects/`, so cluster sessions merge
once their transcripts are copied to the laptop. On mscluster88, note each cluster session's id
(steps 1, 5 and 6) before closing it. Then on the laptop:

```bash
mkdir -p ~/.claude/projects/from-mscluster
scp 'mscluster84:~/.claude/projects/*/<cluster-session-id>.jsonl' ~/.claude/projects/from-mscluster/
```

```
/merge-sessions <laptop ids from steps 3 and 4> <cluster ids from steps 1, 5 and 6>
```

It ends by recommending the drip skill to continue in. Whether `scp` through mscluster84 reaches
your cluster home is not recorded in `environment/`; `scripts/cluster.sh` reaches the cluster the
same way, through `ssh mscluster84`, so it is expected to work.
