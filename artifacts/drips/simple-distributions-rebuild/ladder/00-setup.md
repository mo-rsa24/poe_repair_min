# Setup: once, before layer 1

🏠 [the guide](../PROMPTS.md) · [Layer 1: one expert](01-one-diffusion-expert.md) ➡️

**What this does.** It makes the folder you type into, points Python at the reference code, and gets the first three quizzes ready. Then you sit quiz 0, the college maths every later layer stands on. About 1 to 2 hours of your time in all.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
                                               │  1. typing folder and PYTHONPATH
                                               │       ▼
 quiz 0, quizzes 1.1 and 1.2 ✅ built ────────▶ │  2. (done: the three quiz pages exist)
                                               │       ▼
                                               │  3. quiz 0: the college maths under all
                                               │       ▼
                                               │  4. layer 1
```

## The steps

### 1. You: the typing folder and the environment

You make the folder your walk files live in, and check that the cluster's `co3` Python can import torch and the reference code. Nothing is installed: `co3` already has torch 2.5.1 (see [the environments list](/home/molef/.claude/ENVIRONMENTS.md)).

In a terminal on the cluster:

```bash
mkdir -p ~/Playground/PhD/simple-distributions-walk
cd ~/Playground/PhD/simple-distributions-walk
export PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -c "import torch, rebuild.model; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

⏱ you 5 min · stop: it prints `torch 2.5.1+cu118 cuda True` on a GPU node (`cuda False` on a login node, which the 2D toys can live with).

**If it fails:** `ModuleNotFoundError: No module named 'rebuild'` means the `export` line is missing. The step files for layers 2 to 5 import `rebuild`, so run it in every new shell or add it to `~/.bashrc`.

### 2. AI, in the background: quiz 0 and layer 1's quizzes ✅

Done. The three pages are built: [quiz 0](https://claude.ai/artifact/F4X856bzwavayqUpQUPG9f), [quiz 1.1](https://claude.ai/artifact/LsK1CitQkbAKZLNC3aB8u8) and [quiz 1.2](https://claude.ai/artifact/HMsmKatwrxvhUL2Q72uTWa). Their coverage files are in `.quiz-prep/`.

To rebuild them: [the quiz-building prompt](prompts/00-2-build-the-first-three-quizzes.md).

### 3. You: quiz 0

[Quiz 0](https://claude.ai/artifact/F4X856bzwavayqUpQUPG9f) covers logs and exponents, the chain rule, the Gaussian density and completing the square, mean and variance under scaling and sums, Bayes' rule, and the gradient of a log density. Read the read-first list at the top, then answer.

⏱ you 45 to 90 min · stop: every question marked *needed now* right, and at least 80% of the rest. Below that, read the read-first items your missed questions point at and retake.

---

🏠 [the guide](../PROMPTS.md) · [Layer 1: one expert](01-one-diffusion-expert.md) ➡️
