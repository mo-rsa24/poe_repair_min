# 🗺️ The Reduce, Reuse, Recycle reference code

Navigation: 📋 [Index](00-INDEX.md) | [Overview](overview.md)

What is true about running the authors' code for Du et al. 2023 (arXiv 2302.11552) on the laptop.
The clone is `/home/molef/PhD/reduce-reuse-recycle`, local branch `walk` at upstream commit
`81177d9`. It has two halves with separate environments: a JAX notebook for the 2D results, and a
PyTorch script for the CLEVR results. Where the code differs from the paper is not an environment
fact; that audit is
[the code-against-paper audit](/home/molef/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/drips/read-reduce-reuse-recycle/routes/code-vs-paper.md).

## The 2D notebook

**`notebooks/simple_distributions.ipynb` runs on jax 0.11.2 only after one rename.** The trainer
cell (the one starting `# Simple diffusion model training`) calls
`jnp.clip(self._posterior_variance, a_min=...)`. jax 0.11 accepts only `min=` and `max=`, so
building any model fails with `TypeError: clip() got an unexpected keyword argument 'a_min'`, raised
at the first `hk.multi_transform(...)` call, not in the trainer cell itself. Change `a_min=` to
`min=` and rerun the trainer cell. It is the only `a_min` or `a_max` in the notebook.
(verified 2026-10-13)

**Its environment is `/home/molef/PhD/reduce-reuse-recycle/.venv-2d`**, Python 3.12, with jax
0.11.2 on CPU, dm-haiku, optax, distrax and chex from PyPI unpinned. jax prints "a CUDA-enabled
jaxlib is not installed. Falling back to cpu" on first use; that is expected, and the notebook's
models are small enough for CPU. (verified 2026-10-13)

**The notebook's `tensorflow`, `tensorflow_datasets` and `pandas` imports are dead.** No cell uses
`tf.`, `tfds.` or `pd.`. Deleting those import lines saves installing TensorFlow. The `!pip
install` cell is also unneeded with the venv. (verified 2026-10-13)

## The CLEVR script

**`inf_sample.py` needs a GPU, with no fallback.** It calls `model1.to("cuda")` unconditionally, so
it cannot run on a CPU-only machine. (verified 2026-10-13, by reading)

**Its interpreter is borrowed, read-only.** The laptop's only working CUDA torch is in
`/home/molef/playground/deep-learning/my-eye-my-brain/.venv` (torch `2.14.1+cu130`,
`torch.cuda.is_available()` True on the RTX 4070 Laptop, 8,188 MiB). It lacks matplotlib, which
sits in `<clone>/.clevr-extras` and is reached with `PYTHONPATH=<clone>/.clevr-extras`. That folder
also carries numpy 2.5.4, which shadows the venv's own. If the playground venv is deleted, CLEVR
has no interpreter until a CUDA torch is downloaded again (about 2.5 GB). The authors pin torch
2.0.0 with CUDA 11.7; Docker with the matching image would need the NVIDIA Container Toolkit,
which is not installed (`docker info` lists only the `runc` runtime). (verified 2026-10-13)

**The released energy U-Net cannot run a forward pass.** `composable_diffusion/unet.py` never
assigns `h` before `h = module(h, emb)` at line 1386, so any call raises `UnboundLocalError`. The
upstream guided-diffusion line `h = x.type(self.dtype)` is missing. (inferred 2026-10-13, from
reading the code; audit row R59 says the same; not yet run)

**The checkpoint `ebm-49x1874.pt` is 500,704,677 bytes** (sha256 starting `9e9aa1326eecdcff`),
542 tensors and 125.1 M parameters, from the Google Drive link at README line 59. Google serves a
virus-scan page for a file this size, so a plain `curl` fetches HTML, not the weights; the download
needs the page's `confirm` and `uuid` values. (verified 2026-10-13)

## On the cluster (mscluster88)

**The cluster clone is `/home-mscluster/mmolefe/PhD/reduce-reuse-recycle`**, branch `walk` at
`81177d9`, with the same notebook fixes as the laptop. The walk runs on `mscluster88` (`bigbatch`,
RTX 3090, 24 GB) because `mscluster112`'s card is faulted; see the
[node table](hpc/nodes.md). (verified 2026-10-13)

**The 2D notebook's environment is a uv venv on `/datasets`**, at
`/datasets/mmolefe/reduce-reuse-recycle/venvs/rrr-jax`, linked into the clone as `.venv`. Python
3.12.10, jax 0.11.2 with the `cuda12` plugin, dm-haiku 0.0.17, optax 0.2.8, distrax 0.1.9, chex
0.1.92. Its uv cache is `/datasets/mmolefe/reduce-reuse-recycle/uv-cache`, so the 2.3 GB of CUDA
wheels never land under `/home-mscluster`. `jax.devices()` prints `[CudaDevice(id=0)]` and a
1000×1000 ones matmul sums to 1e9. The install took 5 min 15 s from PyPI. (verified 2026-10-13)

**No existing mamba env holds a usable jax.** `jax` has none installed; `jax115` has jax 0.4.30
with a CPU-only jaxlib and no haiku. (verified 2026-10-13)

## Cross-references

- The mention of **2.3 GB CUDA install**, timed from the PyPI download rate measured on `mscluster88`, in [the HPC nodes fact, under Python environments](hpc/nodes.md#python-environments).
