# Setup: once, before layer 1

🏠 [the guide](../PROMPTS.md) · [Layer 1: one expert](01-one-diffusion-expert.md) ➡️

**What this does.** It makes the folder you type into, points Python at the reference code, and starts quiz 0 and layer 1's two quizzes building in the background. You then sit quiz 0, the college maths every later layer stands on. About 1 to 2 hours of your time in all.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
                                               │  1. typing folder and PYTHONPATH
                                               │       ▼
 quiz 0, quizzes 1.1 and 1.2 ◀──────────────── │  2. paste the quiz prompt
   (prepared, then built) ───────────────────▶ │  3. quiz 0: the college maths under all
                                               │       ▼
                                               │  4. layer 1
```

## The steps

### 1. You: the typing folder and the environment

In a terminal on the cluster:

```bash
mkdir -p ~/Playground/PhD/simple-distributions-walk
cd ~/Playground/PhD/simple-distributions-walk
export PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -c "import torch, rebuild.model; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

⏱ you 5 min · stop: it prints `torch 2.5.1+cu118 cuda True` on a GPU node (`cuda False` on a login node, which the 2D toys can live with). `ModuleNotFoundError: No module named 'rebuild'` means the `export` line is missing: the step files for layers 2 to 5 import `rebuild`, so run it in every new shell or add it to `~/.bashrc`.

### 2. AI, in the background: quiz 0 and layer 1's quizzes

```text
In the background, prepare and build three quizzes for the composition ladder in
artifacts/drips/simple-distributions-rebuild/: quiz 0, quiz 1.1 and quiz 1.2.

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal and minutes
below, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain). The goal for all three is "derive the ε-score
identity and train one 2D expert".
- Quiz 0, --minutes 90: the college maths under the whole ladder. Logs and exponents, the chain
  rule, the Gaussian density and completing the square, mean and variance under scaling and under
  sums of independent variables, Bayes' rule, the gradient of a log density.
- Quiz 1.1, --minutes 40: the college maths under layer 1. √ᾱ x_0 + √(1 − ᾱ) ε as a Gaussian with
  its mean and variance, the score of a Gaussian −(x − μ)/σ², least-squares regression.
- Quiz 1.2, --minutes 90: layer 1 itself. The forward process; the sampled-noise identity
  ε = −√(1 − ᾱ_t)·∇log q(x_t | x_0) against the marginal one,
  E[ε | x_t] = −√(1 − ᾱ_t)·∇log q_t(x_t), which the trained network learns; the denoising loss.
Starts from: nothing before it. Quizzes 0 and 1.1 take their material from
artifacts/drips/simple-distributions-rebuild/ladder/01-one-diffusion-expert.md and its Study first
line; quiz 1.2 takes it from
artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first for
  each quiz (quiz-cadence, "The preparation gate").
- The step file 01-one-expert-from-scratch.md already links a textbook page beside each fundamental
  it uses and carries tables of timestamped video segments; reuse them as nodes before searching
  again (the guide, "When a step stalls on the maths").
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment (checked on mscluster109: import pymupdf works in co3, version 1.28.2, and
  import pypdf fails everywhere).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The notation is the step file's: q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I) over T = 100 steps of a
  cosine schedule, and every worked number prints from python -m rebuild.walk_numbers as the 41 p1.
  keys in rebuild/out/walk-numbers.json (step file 01, steps 4 and 5).
Outputs:
- three coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- rows in the worked-examples index for every banked problem;
- three published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on any page carries the verdict blocked; quizzes 1.1 and 1.2 carry banked
problems, about 4 to 6 per concept at the own depth and 6 to 10 at the build depth; every number in
an answer traces to walk_numbers, the step file or a cited textbook page. I judge, from the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts; quiz-cadence
reports each URL as it publishes.
Proof: the three coverage files exist under .quiz-prep/ and list no question as blocked; the three
URLs open.
Limits: change nothing in the step files' prose or in rebuild/; in the ladder files, write only the
three URLs.
Hand-off: write quiz 0's URL into step 3 of ladder/00-setup.md, and the URLs of quizzes 1.1 and 1.2
into steps 1 and 3 of ladder/01-one-diffusion-expert.md, each replacing "(URL lands here)".
```

added: the preparation gate (from the quiz-cadence agent definition) · the step file's textbook and video links as ready nodes (from the textbook and video passes) · the missing `pdftotext` and the PyMuPDF route (from the textbook pass's report) · the URL write-back
dropped: nothing

⏱ you 2 min to paste · AI ~2 h for three quizzes (estimate: neither quiz agent has a measured run), beside quiz 0's wait and layer 1's maths drip

### 3. You: quiz 0

Quiz 0: https://claude.ai/artifact/F4X856bzwavayqUpQUPG9f

Open it when step 2 writes its URL here. Read the read-first list at the top, then answer. ⏱ you 45 to 90 min · stop: every question marked *needed now* right, and at least 80% of the rest. Below that, read the read-first items your missed questions point at and retake.

---

🏠 [the guide](../PROMPTS.md) · [Layer 1: one expert](01-one-diffusion-expert.md) ➡️
