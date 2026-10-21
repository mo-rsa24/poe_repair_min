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

```bash
R=/datasets/mmolefe/reduce-reuse-recycle
mkdir -p $R/uv-cache $R/venvs
export UV_CACHE_DIR=$R/uv-cache
uv venv --python 3.12 $R/venvs/rrr-jax
uv pip install --python $R/venvs/rrr-jax/bin/python "jax[cuda12]==0.11.2" dm-haiku optax distrax chex matplotlib ipykernel
ln -s $R/venvs/rrr-jax ~/PhD/reduce-reuse-recycle/.venv
```

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
