# Repo walk: reduce_reuse_recycle on mscluster112

**Clone:** `/home-mscluster/mmolefe/PhD/reduce-reuse-recycle`, remote `github.com/yilundu/reduce_reuse_recycle`, branch `walk` at `81177d9`.

**Fact folders:** `poe_repair_min/environment/` and `poe_repair_min/runbook/` (cluster checkout). Laptop findings live in `environment/reduce-reuse-recycle-code.md` and `runbook/setting-up-the-reduce-reuse-recycle-code.md`; this walk adds a mscluster112 section to each.

## Probes (2026-10-13)

| Probe | Result |
|---|---|
| `nvidia-smi` | ❌ `Unable to determine the device handle for GPU0: 0000:01:00.0: Unknown Error`, `No devices were found` |
| `co3_bw` torch | torch imports; `get_device_name(0)` raises `RuntimeError: CUDA unknown error` |
| `co3_bw` extras | matplotlib 3.10.8, tqdm 4.67.1, numpy 2.2.6: CLEVR needs nothing installed once a GPU is available |
| `jax` env | no jax installed |
| `jax115` env | jax 0.4.30, CPU jaxlib only, no haiku: not usable as is |
| PyPI | 347 KB/s on a 403 KB wheel (short sample); GitHub 3.4 MB/s |
| disk | `/home-mscluster` 58% used (32 TB free); `/datasets` 41% (224 TB free) |

## Run map

| Row | Mark | Evidence |
|---|---|---|
| clone + branch `walk` at 81177d9 | done | `git log --oneline -1` → `81177d9 Update README.md` |
| notebook fixes applied | done | 47 cells; `git diff --stat`: 15 insertions, 95 deletions |
| GPU on mscluster112 | blocked | card faulted, see `environment/hpc/nodes.md` |
| GPU on mscluster88 (job 67704) | done | RTX 3090, torch under `co3_bw` computes on it; PyPI 13.5 MB/s |
| 2D notebook env (`rrr-jax`, linked as `.venv`) | done | `jax.devices()` → `[CudaDevice(id=0)]`; install 5 min 15 s |
| 2D notebook: moving average compiled into the training step (6 cells) | done | speed-only, no maths changed; 1000-step benchmark on the 3090: 18.68 → 1.95 ms per step |
| 2D notebook sections | open | definitions and datasets run by the user |
| CLEVR checkpoint | open | |
| CLEVR `h` patch | open, needs the user's yes | |
| CLEVR Rev_Diff, MALA | open | |

## Position

On mscluster88 (job 67704). The `rrr-jax` env works on the GPU. Next: the notebook rounds, starting with the definitions section.
