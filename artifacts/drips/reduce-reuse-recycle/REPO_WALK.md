# Repo walk: reduce_reuse_recycle

The working file for the `/drip-repo` walk that gets the authors' code for Du et al. 2023
(arXiv 2302.11552) running from a permanent clone. Re-invoking `/drip-repo` resumes from here.

## The open

| Field | Value |
|---|---|
| Clone | `/home/molef/PhD/reduce-reuse-recycle`, local branch `walk` at `81177d9` |
| Remote | `https://github.com/yilundu/reduce_reuse_recycle` |
| Start commit | `81177d9` (the same commit the R3 read's `reference/` copy is at) |
| Fact folders | `/home/molef/PhD/poe_repair_min/environment/` and `/home/molef/PhD/poe_repair_min/runbook/` |
| Playground copy | `/home/molef/playground/deep-learning/reduce-reuse-recycle/reference` stays untouched: the R3 read map names it |
| Must read before trusting a sampler | `/home/molef/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/drips/read-reduce-reuse-recycle/routes/code-vs-paper.md` |
| Cluster launch for a second repo | not covered by `environment/`; ask before any cluster step |

## Run map

| # | Runnable thing | Command | Needs first | Success looks like | Mark |
|---|---|---|---|---|---|
| 1 | clone | `git clone ... ~/PhD/reduce-reuse-recycle && git checkout 81177d9` | network | `git log -1` shows `81177d9` | done 2026-10-13: `81177d9 Update README.md`, 40 files |
| 2 | 2D environment (JAX, CPU) | `uv venv .venv-2d --python 3.12` + jax, dm-haiku, optax, distrax, chex, matplotlib, jupyter | 1 | `python -c "import jax, haiku, distrax; print(jax.devices())"` prints a CPU device | done 2026-10-13: jax 0.11.2 `[CpuDevice(id=0)]`, haiku, optax, distrax, chex import |
| 3a | notebook: model and composition definitions (cells 2-11) | kernel, cells run in order | 2 | no error; the classes defined | done 2026-10-13: cells 1-10 ran (cell 0 `!pip` skipped, cell 3 without its unused pandas and tensorflow imports); `EBMDiffusionModel`, `ProductEBMDiffusionModel`, `MixtureEBMDiffusionModel`, `NegationEBMDiffusionModel`, `ResnetDiffusionModel` defined, beta runs 0.000631 to 0.999 |
| 3b | notebook: trainer, datasets and samplers (cells 12-30) | same kernel | 3a | no error | done 2026-10-13: cell 12 done 2026-10-13, `PortableDiffusionModel` defined (loss, sample, p_sample, logpx, energy_scale, p_gradient); cells 14-20 done 2026-10-13, seven datasets drawn in `figures/the-seven-toy-datasets.png` |
| 3c | notebook: product (cells 32-36) | same kernel | 3b | a scatter of samples from spiral x bar | open |
| 3d | notebook: mixture (cells 37-41) | same kernel | 3b | a scatter covering both bars | open |
| 3e | notebook: negation (cells 42-46) | same kernel | 3b | a scatter of the first set with the second carved out | open |
| 4 | PyTorch environment (CUDA) | venv with torch built for the 4070's CUDA, numpy, matplotlib, tqdm | 1 | `torch.cuda.is_available()` is True | done 2026-10-13 differs: no Docker and no new venv; the interpreter is `/home/molef/playground/deep-learning/my-eye-my-brain/.venv/bin/python` (torch 2.14.1+cu130, CUDA True) with `PYTHONPATH=<clone>/.clevr-extras` (matplotlib 3.11.2, numpy 2.5.4) |
| 5 | checkpoint `ebm-49x1874.pt` | Google Drive link, README line 59 | 1 | file on disk, `torch.load` opens it | done 2026-10-13: 500,704,677 bytes, sha256 starts `9e9aa1326eecdcff`, 542 tensors, 125.1 M parameters |
| 6 | `inf_sample.py --sampler Rev_Diff` | `python inf_sample.py --sampler Rev_Diff --ckpt_path ebm-49x1874.pt` | 4, 5 | `Energy_Object_A Sphere And A Cylinder_Rev_Diff_4.png` in cwd | open |
| 7 | `inf_sample.py --sampler MALA` | same, `--sampler MALA` | 6 | `Energy_Object_..._MALA_4.png` | open |
| 8 | dataset `dataset_train_larger.npz` | Dropbox link, README line 42 | 1 | file on disk, `np.load` lists its arrays | open |

Rows 4 to 8 are walked after the notebook. Row 8 is only read by training (`train_ddp.py`,
`ebm_finetune/`), never by `inf_sample.py`.

## Facts found on the way

- One PyPI download on 2026-10-13 ran at 69,006 B/s (a 3.3 MB wheel took 48 s), yet the JAX stack
  and the 500 MB checkpoint then finished within about 25 minutes, so the line speed varies a lot.
  The Docker route for CLEVR was dropped on the slow reading.
- The uv cache held only package metadata for jax and jaxlib, not the wheels.
- `/home/molef/playground/deep-learning/my-eye-my-brain/.venv` has torch `2.14.1+cu130`, and
  `torch.cuda.is_available()` is True on the RTX 4070. It lacks matplotlib.
- `inf_sample.py` reaches 12 repo files and needs only torch, numpy, matplotlib and tqdm.
- Docker runs natively in WSL (29.8.1) with only the `runc` runtime: no NVIDIA runtime registered.

## Position

Rows 1, 2, 3a, 4 and 5 done. The notebook kernel is alive (`kernel.json`, driven by `kernel.py`).
The reader now runs the cells in their own VS Code kernel on the clone's notebook (they deleted the
`!pip` cell, so their cells sit one lower); `kernel.py` reads cell numbers from git at `81177d9`.
Cell 22 failed on `jnp.clip(..., a_min=)` under jax 0.11.2; fixed by renaming to `min=` in the
trainer cell; fact written to `environment/reduce-reuse-recycle-code.md`.
Cells 22 to 27 done 2026-10-13 in the walk kernel: `forward_product`, `forward_mixture`, `forward_negation` built; 18 jitted `dual_*` functions. Cells 29 and 30 done 2026-10-13: AnnealedULASampler, AnnealedUHASampler, AnnealedMALASampler, AnnealedMUHASampler defined. Row 3b complete. Next: row 3c, the product section, cells 32 to 35 (trains two experts, then samples).
