/drip-repo ~/PhD/reduce-reuse-recycle

Get the authors' code for Du et al. 2023, "Reduce, Reuse, Recycle" (arXiv 2302.11552,
github.com/yilundu/reduce_reuse_recycle), running on mscluster112 so I can follow along in the
code: the 2D notebook first, then CLEVR sampling. This session runs on mscluster112 itself (I
opened VS Code remote there and have the node allocated), so commands run on the node directly:
no sbatch, no ssh hop. The same walk was started on my laptop on 2026-10-13. Everything below
was learned there, so do not rediscover it.

## The open, already answered

- Clone: `~/PhD/reduce-reuse-recycle` (that is `/home-mscluster/mmolefe/PhD/reduce-reuse-recycle`).
  If it does not exist, clone it and `git checkout -b walk 81177d9`. That commit is the one every
  fact below refers to.
- Fact folders: the cluster checkout of poe_repair_min,
  `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/environment/` and `runbook/`. Read
  `environment/overview.md`, `environment/hpc/nodes.md` and `environment/storage.md` first. If
  `environment/reduce-reuse-recycle-code.md` and `runbook/setting-up-the-reduce-reuse-recycle-code.md`
  exist there, they are the laptop's findings: extend them with a section for mscluster112,
  never a second file.
- Walk file: `artifacts/drips/reduce-reuse-recycle-mscluster112/REPO_WALK.md` in that checkout.
- Environments: I use mamba (`/home-mscluster/mmolefe/miniforge3/envs/`), but uv is fine for new
  ones. Never install into `co3` or `co3_bw`; they are the project's environments.

## Probe before building anything (read-only, show me the output)

1. `nvidia-smi`, then `/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
   The node's card is an RTX PRO 6000 Blackwell (about 96 GB). `co3_bw` measured torch
   2.11.0+cu128 with sm_120 kernels on this node on 2026-09-03. `co3` has no Blackwell kernels: a
   CUDA op there produces nothing, with no error. mscluster111's card once read
   `[GPU requires reset]` and torch silently fell back to CPU; assert `cuda.is_available()` before
   trusting any timing.
2. The existing `jax` and `jax115` mamba envs: print `jax.__version__` and `jax.devices()` in each.
   If one has a CUDA jaxlib that sees the Blackwell card, use it (cloned, not modified) rather than
   downloading about 2.3 GB of CUDA wheels.
3. Network speed: time one PyPI wheel download. The laptop's line ran at 69 KB/s to 292 KB/s,
   which made a 2.3 GB CUDA install take over an hour. Say the number before any big download.
4. `df -h /home-mscluster /datasets`. Anything large (the 500 MB checkpoint, outputs, a uv cache
   holding CUDA wheels) goes under `/datasets/mmolefe/reduce-reuse-recycle/` with a symlink from
   the clone, because `/home-mscluster` filled to 100% once and silently killed a run.

## What the laptop proved about the 2D notebook (`notebooks/simple_distributions.ipynb`)

It is JAX and Haiku. With jax 0.11.2, dm-haiku, optax, distrax and chex (latest from PyPI), it
needs four edits, all proven on 2026-10-13. Apply them with this script, once, on a fresh clone,
then show me the notebook's diff stat:

```python
import json, sys
p = sys.argv[1]; nb = json.load(open(p)); cells = nb["cells"]
src = lambda c: "".join(c["source"])
cells[:] = [c for c in cells if not src(c).lstrip().startswith("!pip install")]
for c in cells:
    lines = [l for l in c["source"] if not ("import tensorflow" in l or "import pandas" in l)]
    c["source"] = [l.replace("a_min=", "min=") for l in lines]
i = next(k for k, c in enumerate(cells) if src(c).lstrip().startswith("import math"))
cells.insert(i + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "import chex\n",
    "chex.assert_type = lambda *args, **kwargs: None  # dtype checks predate JAX's 32-bit defaults\n",
    "jax.tree_leaves = jax.tree_util.tree_leaves      # removed from jax; lives on in jax.tree_util\n",
    "jax.tree_map = jax.tree_util.tree_map"]})
open(p, "w").write(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
print(len(cells), "cells; fixes applied")
```

The failures each edit prevents, in the order they hit: `TypeError: clip() got an unexpected
keyword argument 'a_min'` (raised at the first `hk.multi_transform`, not in the trainer cell);
`[Chex] Assertion assert_type failed ... float64 but expected float32; int32 but expected int64`;
`module 'jax' has no attribute 'tree_leaves'`. The TensorFlow, tfds and pandas imports have 0
uses. Of the 60 jax, haiku, optax and distrax names the notebook uses, only `jax.tree_leaves` and
`jax.tree_map` are gone.

Sections and what each should show: definitions (no output); trainer; seven toy datasets;
product = ring of 8 blobs (`toy_gmm(std=.03)`) times a vertical bar (`bar(scale=0.2)`), so only
the top and bottom blobs should survive; mixture = `bar()` plus `right_bar()`; negation =
`toy_box(0.7)` minus `toy_box(0.3)`, a square with a hole. All three sample with
`AnnealedMUHASampler` (adjusted HMC), not MALA.

Laptop timing, the number to beat: one expert (15,001 steps, a 4-layer MLP of width 128, batch
1000) took 13 min 7 s on the laptop's CPU, of which about 105 s was training (7.1 ms per step),
about 40 s the 16 evaluation pauses (1.9 s per likelihood, 0.1 s per heatmap), and the rest JAX
compiling and a second kernel sharing the cores. The bar expert (the cell that ends
`bar_params = ema_params`) then took 629.4 s on the same CPU with the machine nearly idle, at a
mean 12.1 ms per step. Time that same cell here, on the GPU, and say whether the GPU actually
helped.

## What the laptop proved about CLEVR (`inf_sample.py`)

- It needs only torch, numpy, matplotlib and tqdm (the import chain reaches 12 repo files). It
  hard-codes `.to("cuda")`. Check whether `co3_bw` already has matplotlib and tqdm; if so it runs
  there with nothing installed.
- Checkpoint `ebm-49x1874.pt`: 500,704,677 bytes, sha256 starting `9e9aa1326eecdcff`, 542
  tensors, 125.1 M parameters. Google Drive serves a virus-scan page, so fetch it with:
  `curl -sSL "https://drive.google.com/uc?export=download&id=19ay4bLVCQho8_aXtODWbDCpu6L2l58YB" -o /tmp/gd.html`,
  `UUID=$(grep -oP 'name="uuid" value="\K[^"]+' /tmp/gd.html)`, then
  `curl -sSL -C - --retry 5 -o <dest>/ebm-49x1874.pt "https://drive.usercontent.google.com/download?id=19ay4bLVCQho8_aXtODWbDCpu6L2l58YB&export=download&confirm=t&uuid=$UUID"`.
  Or copy it from the laptop clone with `scp` if the cluster line is slow.
- It cannot run as released: the energy U-Net's `forward` in `composable_diffusion/unet.py` never
  assigns `h` before `h = module(h, emb)` at line 1386, so every call raises `UnboundLocalError`.
  The upstream guided-diffusion line `h = x.type(self.dtype)` is missing. Show me the one-line
  patch and wait for my yes before applying it.
- Run `--sampler Rev_Diff` first (plain reverse diffusion, the baseline), then `--sampler MALA`.
  The script writes `Energy_Object_A Sphere And A Cylinder_<sampler>_4.png` to the current folder.
- `dataset_train_larger.npz` (Dropbox, README line 42) is read only by training, never by
  `inf_sample.py`. Skip it unless I ask to retrain.

## Where the code differs from the paper (read before trusting any picture)

From a 60-row audit of this commit against the paper (on my laptop at
`~/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/drips/read-reduce-reuse-recycle/routes/code-vs-paper.md`;
read it if the learning tree is checked out here, otherwise rely on these rows):
R07 the 2D beta schedule is cosine where the paper says linear; R13 the mixture multiplies energies
by 3.5 inside the log-sum-exp; R14 negation is `1.3·E₁ − 0.3·E₂`, where the paper has α = 0.5;
R20 the JAX MALA accept step matches the paper; R21 the PyTorch MALA proposal divides by 2h² where
it should divide by 4h, so at the audit's test point it accepts with probability 1 instead of
0.978; R40 the 2D cells take 10 HMC steps per noise level where the paper says 3; R59 the unbound
`h` above. Say which rows a picture inherits when you show it.

## How I want it walked

I follow along in my own VS Code notebook on this node, running each cell myself. Each round:
which cells to run, named by their first line (VS Code shows no cell numbers); the one or two
lines worth reading first; what the output should be. When something fails, fix it in the files
for me (I asked for that on the laptop) and tell me which cells to rerun. Write each failure as an
environment fact and each working step as a runbook recipe in the round that proves it. Ask, do
not infer, for anything about the cluster that `environment/` does not cover.
