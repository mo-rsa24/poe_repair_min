# 🧭 Setting up the Reduce, Reuse, Recycle code on the laptop

How to get the authors' code for Du et al. 2023 (arXiv 2302.11552) from an empty folder to running:
the 2D notebook on the CPU (or the GPU), and the CLEVR sampling script on the GPU. What is true
about this code and why each step exists is in
[the reduce-reuse-recycle code facts](../environment/reduce-reuse-recycle-code.md); this file only
holds the commands.

## Table of contents

The recipes are numbered so the index can point at them. Follow 1 to 3 for the notebook, 4 and 5
for CLEVR.

- [Before any of these](#before-any-of-these)
- [1. Clone at the commit the walk uses](#1-clone-at-the-commit-the-walk-uses)
- [2. Build the 2D notebook's environment](#2-build-the-2d-notebooks-environment)
- [3. Make the notebook run on current jax](#3-make-the-notebook-run-on-current-jax)
- [4. Get the CLEVR checkpoint](#4-get-the-clevr-checkpoint)
- [5. Give the CLEVR script a CUDA torch](#5-give-the-clevr-script-a-cuda-torch)
- [6. Run the notebook on the GPU instead](#6-run-the-notebook-on-the-gpu-instead)
- [7. Build the 2D notebook's environment on the cluster](#7-build-the-2d-notebooks-environment-on-the-cluster)
- [Where this came from](#where-this-came-from)

## Before any of these

Navigation: 📋 [TOC](#table-of-contents) | [Next](#1-clone-at-the-commit-the-walk-uses) ➡️

- [ ] 🖥️ **uv is installed**
  ```bash
  uv --version
  ```
  ✅ **a version number**
  ❌ **`command not found`**: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new terminal

## 1. Clone at the commit the walk uses   `ran 2026-10-13`

Navigation: ⬅️ [Before any of these](#before-any-of-these) | 📋 [TOC](#table-of-contents) | [Next](#2-build-the-2d-notebooks-environment) ➡️

```bash
cd ~/PhD
git clone https://github.com/yilundu/reduce_reuse_recycle reduce-reuse-recycle
cd reduce-reuse-recycle
git checkout -b walk 81177d9
```

✅ **`git log -1 --oneline` prints `81177d9 Update README.md`** and `git ls-files | wc -l` prints 40.

## 2. Build the 2D notebook's environment   `ran 2026-10-13`

Navigation: ⬅️ [1. Clone](#1-clone-at-the-commit-the-walk-uses) | 📋 [TOC](#table-of-contents) | [Next](#3-make-the-notebook-run-on-current-jax) ➡️

```bash
cd ~/PhD/reduce-reuse-recycle
uv venv .venv-2d --python 3.12
UV_HTTP_TIMEOUT=600 uv pip install --python .venv-2d/bin/python \
    jax dm-haiku optax distrax chex matplotlib ipykernel
.venv-2d/bin/python -c "import jax, haiku, optax, distrax, chex; print(jax.__version__, jax.devices())"
```

✅ **`0.11.2 [CpuDevice(id=0)]`**, after a warning that a CUDA-enabled jaxlib is not installed.
The download is about 200 MB; on a slow line it takes most of an hour.
❌ **the install sits for minutes with no output**: it is downloading, not stuck. Check with
`du -sm ~/.cache/uv/.tmp*`, which grows while files arrive.

## 3. Make the notebook run on current jax   `ran 2026-10-13`

Navigation: ⬅️ [2. The 2D environment](#2-build-the-2d-notebooks-environment) | 📋 [TOC](#table-of-contents) | [Next](#4-get-the-clevr-checkpoint) ➡️

```bash
python3 ~/PhD/poe_repair_min/scripts/fix_reduce_reuse_recycle_notebook.py \
    ~/PhD/reduce-reuse-recycle/notebooks/simple_distributions.ipynb
```

✅ **`47 cells; fixes applied`**. It drops the `!pip` cell and the unused TensorFlow and pandas
imports, renames `jnp.clip(..., a_min=)` to `min=`, and adds a compatibility cell after the
imports. Run it once only, on a fresh clone.

Then in VS Code: open `notebooks/simple_distributions.ipynb`, click **Select Kernel** (top right),
**Python Environments**, and pick `.venv-2d`. Run the cells from the top. Each training cell
prints a loss every 100 steps and a `TEST <step> logpx <value>` line every 1,000.

❌ **`TypeError: clip() got an unexpected keyword argument 'a_min'`** or **`[Chex] Assertion
assert_type failed`** or **`module 'jax' has no attribute 'tree_leaves'`**: the script was not
run, or the compatibility cell was not run in this kernel.

## 4. Get the CLEVR checkpoint   `ran 2026-10-13`

Navigation: ⬅️ [3. Notebook fixes](#3-make-the-notebook-run-on-current-jax) | 📋 [TOC](#table-of-contents) | [Next](#5-give-the-clevr-script-a-cuda-torch) ➡️

Google Drive answers a file this size with a virus-scan page, so the download has to pass that
page's confirmation.

```bash
cd ~/PhD/reduce-reuse-recycle
curl -sSL "https://drive.google.com/uc?export=download&id=19ay4bLVCQho8_aXtODWbDCpu6L2l58YB" -o /tmp/gd.html
UUID=$(grep -oP 'name="uuid" value="\K[^"]+' /tmp/gd.html)
curl -sSL -C - --retry 5 -o ebm-49x1874.pt \
  "https://drive.usercontent.google.com/download?id=19ay4bLVCQho8_aXtODWbDCpu6L2l58YB&export=download&confirm=t&uuid=$UUID"
ls -l ebm-49x1874.pt
```

✅ **500,704,677 bytes**, and `file ebm-49x1874.pt` says `Zip archive data`.
❌ **a file of a few KB**: that is the HTML page, so `UUID` came back empty. Open the link in a
browser instead and save the file into the clone.

## 5. Give the CLEVR script a CUDA torch   `ran 2026-10-13`

Navigation: ⬅️ [4. The checkpoint](#4-get-the-clevr-checkpoint) | 📋 [TOC](#table-of-contents) | [Next](#6-run-the-notebook-on-the-gpu-instead) ➡️

This borrows the CUDA torch already installed for `my-eye-my-brain`, read-only, and adds only
matplotlib inside the clone.

```bash
cd ~/PhD/reduce-reuse-recycle
uv pip install --target .clevr-extras matplotlib
PYTHONPATH=$PWD/.clevr-extras ~/playground/deep-learning/my-eye-my-brain/.venv/bin/python \
  -c "import torch, matplotlib; print(torch.__version__, torch.cuda.is_available())"
```

✅ **`2.14.1+cu130 True`**.
❌ **`No such file or directory`** for the borrowed interpreter: that playground venv is gone.
Build a CUDA torch in the clone instead (about 2.5 GB):
`uv venv .venv-clevr --python 3.12 && uv pip install --python .venv-clevr/bin/python torch numpy matplotlib tqdm --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match` (`unverified`).

`inf_sample.py` itself still fails as released; see the U-Net fact in
[the reduce-reuse-recycle code facts](../environment/reduce-reuse-recycle-code.md#the-clevr-script).

## 6. Run the notebook on the GPU instead   `unverified`

Navigation: ⬅️ [5. CLEVR torch](#5-give-the-clevr-script-a-cuda-torch) | 📋 [TOC](#table-of-contents) | [Next](#7-build-the-2d-notebooks-environment-on-the-cluster) ➡️

```bash
cd ~/PhD/reduce-reuse-recycle
UV_HTTP_TIMEOUT=600 uv pip install --python .venv-2d/bin/python "jax[cuda13]==0.11.2"
.venv-2d/bin/python -c "import jax; print(jax.devices())"
```

✅ **a `CudaDevice`** in the list. The download is about 2.3 GB of CUDA libraries. Then restart
the notebook's kernel and rerun from the top.

## 7. Build the 2D notebook's environment on the cluster   `ran 2026-10-13`

Navigation: ⬅️ [6. GPU notebook](#6-run-the-notebook-on-the-gpu-instead) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

On a GPU node (the walk used `mscluster88`), with the clone from recipe 1 at
`~/PhD/reduce-reuse-recycle`. Check the card first, because `bigbatch` cards fault between checks:

```bash
nvidia-smi -L
```

✅ **one `GPU 0:` line.** ❌ **`Unable to determine the device handle`**: the card is faulted;
move to another node from [the node table](../environment/hpc/nodes.md).

```bash
R=/datasets/mmolefe/reduce-reuse-recycle
mkdir -p $R/uv-cache $R/venvs
export UV_CACHE_DIR=$R/uv-cache
~/.local/bin/uv venv --python 3.12 $R/venvs/rrr-jax
~/.local/bin/uv pip install --python $R/venvs/rrr-jax/bin/python "jax[cuda12]==0.11.2" dm-haiku optax distrax chex matplotlib ipykernel
ln -sfn $R/venvs/rrr-jax ~/PhD/reduce-reuse-recycle/.venv
~/PhD/reduce-reuse-recycle/.venv/bin/python -c "import jax; print(jax.devices())"
```

✅ **`[CudaDevice(id=0)]`.** The install takes about 5 minutes. ❌ **`[CpuDevice(id=0)]`**: the
CUDA plugin did not install or the node has no visible card; rerun `nvidia-smi -L`.

In VS Code, open the notebook, click the kernel name at the top right, choose **Select Another
Kernel → Python Environments → .venv (Python 3.12.10)**. If `.venv` is not listed, choose **Enter
interpreter path** and paste `/home-mscluster/mmolefe/PhD/reduce-reuse-recycle/.venv/bin/python`.

## Where this came from

Navigation: ⬅️ [7. Cluster environment](#7-build-the-2d-notebooks-environment-on-the-cluster) | 📋 [TOC](#table-of-contents)

**The drip-repo walk of 2026-10-13**, whose working file is
`artifacts/drips/reduce-reuse-recycle/REPO_WALK.md`. Every recipe marked `ran` was run that day
on the laptop, except recipe 7, run on `mscluster88` from the walk file
`artifacts/drips/reduce-reuse-recycle-mscluster112/REPO_WALK.md`.
