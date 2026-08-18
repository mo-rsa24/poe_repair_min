# 🧪 Hypothesis 02: More correction, more composition

**Step 4 of 22.** Waits on steps 1 and 2. The running order is in [repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

## Position in the plan tree

| Step | Plan | Status |
|---|---|---|
| 3 | ~~[instrument-01: the-clean-pair-pool](../../does-the-fix-reach-unseen-pairs/plans/instrument-01-the-clean-pair-pool.md)~~ | ✅ |
| **4 (current)** | **hypothesis-02: more-correction, more-composition** | **⚠️ review ready** |
| 5 | ~~[hypothesis-04: what-the-cached-runs-already-show](hypothesis-04-what-the-cached-runs-already-show.md)~~ | ✅ |

---

## Table of contents

- [Position in the plan tree](#position-in-the-plan-tree)
- [Quick context: the hypothesis](#quick-context-the-hypothesis)
- [Considerations](#considerations)
- [The claim](#the-claim)
- [Why this plan exists](#why-this-plan-exists)
- [What the fix actually does (visual)](#what-the-fix-actually-does-visual)
- [Description: the experiment](#description-the-experiment)
- [Purpose and goal](#purpose-and-goal)
- [Why the controls are fair](#why-the-controls-are-fair)
- [Tasks](#tasks)
- [The engagement gate](#the-engagement-gate)
- [Outputs / Figure](#outputs--figure)
- [Code references](#code-references)
- [Next step](#next-step)
- [Error Matrix](#error-matrix)

---

## Quick context: the hypothesis

⬅️ [Position](#position-in-the-plan-tree) | 📋 [TOC](#table-of-contents) | [Considerations](#considerations) ➡️

**The hypothesis:**

*The correction term $r_t$ is the missing piece that PoE drops. If true, adding more of it should produce more composition, while deliberately wrong vectors of the same size should stay flat.*

**If true:** The real correction's compose-rate curve rises as λ (correction strength) increases from 0 to 1. Both fake rows (random vector and wrong pair's correction) stay near zero at all λ.

**If false:** Either the correction is not the cause of composition failure (other factors matter more), or the controls are not actually controls (the fakes are not comparable to the real one).

**Dataset details:**
- **Corrections:** Already cached from training, with exact starting noise per cell, under `training_cache/`
- **Pairs:** 8 unseen animal pairs (not seen during training)
- **Seeds:** 4 random seeds per pair
- **Strengths:** λ ∈ {0, 0.25, 0.50, 0.75, 1.00}
- **Rows:** Real correction, random vector (size-matched), wrong pair's correction
- **Total cells:** 8 × 4 × 5 × 3 = 480 pictures (minus 40 duplicates at λ=0 = 440 unique images)

**Review questions:** Answered in [../review/hypothesis-02-more-correction-more-composition.md](../review/hypothesis-02-more-correction-more-composition.md)

**Vocabulary once:**
- **PoE** (Product of Experts): the broken way. Ask the model about "a cat" and "a dog" separately, add the predictions. Usually fuses into one animal.
- **Mono** (monocular): the cheat that works. Hand the model "a cat and a dog" joined. Composes fine, but defeats the scientific point (it's the oracle, not evidence).
- **The correction $r_t$**: the gap between Mono's prediction and PoE's. It is what PoE leaves out.
- **λ (lambda)**: How much correction to inject, from 0 (none) to 1 (all).
- **A cell**: One image, for one pair, at one λ, from one starting noise (one seed).
- **Compose rate**: Fraction of cells showing two separate animals (vs one blended), scored by the validated scorer, never by eye.

---

## Considerations

⬅️ [Quick context](#quick-context-the-hypothesis) | 📋 [TOC](#table-of-contents) | [The claim](#the-claim) ➡️

**Runtime:**
- One cell runs in-session on this node's 3090 (minutes).
- Full set: ~50 seconds per cell × 440 unique cells = ~6 hours on GPU.
- Submission: `scripts/mechanism_study/run_dose_sweep.sh`, resumable, to biggpu (100GB VRAM, safer) or bigbatch (fallback).

**Environment constraints:**
- Corrections are already cached; nothing here recomputes them.
- GPU is shared; check `nvidia-smi` before starting—a full card kills the job mid-run.
- Output goes to `/datasets` (not `/home-mscluster`). Job aborts if `/datasets` exceeds 90% capacity.
- W&B logging is mandatory: three-panel triptych per cell (Mono, plain PoE, corrected) so every number has a picture beside it.

**Disk state:**
- **Before (2026-08-18):** 6.3GB on `/home-mscluster` (freed by moving to `/datasets`).
- **After:** All outputs on `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/`.

---

## The claim

⬅️ [Considerations](#considerations) | 📋 [TOC](#table-of-contents) | [Why this plan exists](#why-this-plan-exists) ➡️

**Wire live compose-rate logging into the dose-sweep harness, run 480 pictures through three injection conditions (real correction, random vector, wrong pair's correction), score the outputs, and plot three curves (compose rate vs λ, one per injection type).**

**The paper's central causal claim depends on this figure:** more correction ⟹ more composition, while controls stay flat.

---

## Why this plan exists

⬅️ [The claim](#the-claim) | 📋 [TOC](#table-of-contents) | [What the fix actually does (visual)](#what-the-fix-actually-does-visual) ➡️

**The dilemma:**

"The fix helps" is an observation. "The fix is the cause" requires a causal argument, which is much harder. A rising curve alone could be explained by many things (the injection mechanism itself helps, something about changing the noise, etc.). Only a rising curve *paired with flat controls* makes the causal case.

The two fake rows are the controls. They differ from the real correction in exactly one way: direction. Both are the same size as the real correction at every step. If the real correction helps while both fakes do not, that difference in direction is what matters.

**The mechanism:**

1. **Load** the real correction $r_t$ and the cached starting noise.
2. **Inject** a chosen vector (real, random, or wrong pair's) at strength λ into PoE sampling.
3. **Generate** the image.
4. **Score** it with the validated composition scorer (is it one animal or two?).
5. **Repeat** for all combinations.
6. **Plot** three curves: compose rate vs λ for each injection type.

Without the controls, the rising curve proves nothing. With them, it proves direction matters.

---

## What the fix actually does (visual)

⬅️ [Why this plan exists](#why-this-plan-exists) | 📋 [TOC](#table-of-contents) | [Description: the experiment](#description-the-experiment) ➡️

```
λ = 0          λ = 0.25       λ = 0.50       λ = 0.75       λ = 1.00
(no inject)    (25% real)     (50% real)     (75% real)     (100% real)

Row 1: Real correction
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  blend   │→ │  blend   │→ │ 2 animals│→ │ 2 animals│→ │ 2 animals│  compose↑
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘

Row 2: Random vector (same size, wrong direction)
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  blend   │  │  blend   │  │  blend   │  │  blend   │  │  blend   │  flat
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘

Row 3: Different pair's correction (size-matched, wrong pair)
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  blend   │  │  blend   │  │  blend   │  │  blend   │  │  blend   │  flat
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘

CURVES:
Compose rate (0 to 1) vs λ
    ↑ real correction (blue, rising)
    │          ╱
  1 │        ╱
    │      ╱
0.5 │    ╱
    │  ╱      random (red, flat)
    ├────      wrong pair (green, flat)
  0 │
    └────────────────────→
        0  0.25  0.50  0.75  1.00  λ
```

---

## Description: the experiment

⬅️ [What the fix actually does (visual)](#what-the-fix-actually-does-visual) | 📋 [TOC](#table-of-contents) | [Purpose and goal](#purpose-and-goal) ➡️

Take the correction we already have cached, add a fraction of it back into PoE sampling, and score the picture that comes out.

**The setup:**
- Fraction is λ, stepping through {0, 0.25, 0.50, 0.75, 1.00}.
- Starting noise is held fixed across all λ for one cell, so the only thing changing is how much correction goes in.

**Three rows for every (pair, seed, λ):**
1. **Real correction:** Inject $r_t$ at strength λ.
2. **Random vector:** Inject a random vector of the same size as $r_t$.
3. **Wrong pair's correction:** Inject the cached correction from a different pair.

The two fakes are controls. They are what tell us the real one is working because of direction, not because something was added.

**Why this design:**
- Both fakes are the same size as the real correction at every step. Otherwise a fake that fails could have failed for being too weak, and we could not distinguish weakness from wrongness.
- The full-strength shortcut is switched off during a fake run. At λ=1, the code could skip the arithmetic and use the joined-prompt (Mono) prediction directly. During a fake run, that would ignore the fake and return the real answer, quietly making all three rows identical. This is the failure that would be hardest to notice.
- The zero-strength row (λ=0) is generated once and shared. All three rows are the same picture by construction, so generating three times would only add noise to the one point where they must agree.

---

## Purpose and goal

⬅️ [Description: the experiment](#description-the-experiment) | 📋 [TOC](#table-of-contents) | [Why the controls are fair](#why-the-controls-are-fair) ➡️

**Purpose:**
This is the paper's central causal claim (Objective 1, Definition of Done item 3 in the scope's MASTER_PLAN.md): composition fails because this correction is missing. The evidence for "cause" is a rising curve for the real correction, beside two flat curves for the fakes.

**Goals:**
1. 480 scored pictures (8 pairs × 4 seeds × 5 strengths × 3 rows), minus 40 duplicates at λ=0 = 440 unique images.
2. Three curves: compose rate against λ, one per injection type (real, random, wrong pair).
3. Figure slot **F2** filled: the paper's headline figure in two halves (quantitative curves + qualitative grid).

**What success looks like:**
- Real correction curve: rises monotonically from ~0 at λ=0 to ~80–95% at λ=1.
- Random vector: flat, near 0 across all λ.
- Wrong pair's correction: flat, near 0 across all λ.
- Conclusion: direction matters; size does not. The correction is the cause.

---

## Why the controls are fair

⬅️ [Purpose and goal](#purpose-and-goal) | 📋 [TOC](#table-of-contents) | [Tasks](#tasks) ➡️

A control only works if it differs in exactly one way. These are deliberate design choices, not findings, so they live here (not the review file).

**Size matching:**
Both fakes are scaled to match the real correction's length. Otherwise, a fake that failed could have failed for being too weak rather than for pointing the wrong way. Only direction is supposed to differ.

**Measuring the real correction throughout:**
$\delta_{\text{norm}}$ (the real correction's magnitude) and the PMI check (algebraic test that the correction equals Mono - PoE) are recorded per cell, even during fake runs. Both describe the real pair of concepts, not whatever vector we injected, so both keep measuring the real correction throughout.

**Full-strength shortcut is disabled during fakes:**
At λ=1, the code could skip the arithmetic and use the Mono prediction directly (it gives the same answer when the correction is real). During a fake run, that would ignore the fake and return the real answer, silently making all three rows identical. This bug would be the hardest to catch.

**Zero-strength row is shared:**
At λ=0, nothing is injected, so all three rows are the same picture by construction. Generating three times would only add noise to the one point where they must agree.

---

## Tasks

⬅️ [Why the controls are fair](#why-the-controls-are-fair) | 📋 [TOC](#table-of-contents) | [The engagement gate](#the-engagement-gate) ➡️

A design task either happened or it did not. Whether the experiment worked is a separate question, answered in the review file.

### 1. Code

- [x] **Wire the injection harness.**
  Write the code that injects a chosen vector (real, random, or wrong pair's) and runs three rows at every λ.
  - File: `scripts/interaction_term_inject.py` (built on `run_teacher_residual`)
  - Injecting the real correction existed from `instrument-01-build-the-measuring-scripts`
  - Injecting a DIFFERENT vector was new

- [x] **Prove the harness leaves plain PoE alone.**
  At λ=0, the output must match what the sampler itself saved for plain PoE.
  - Test: 8 canary tests, each one shown to fail against a deliberately broken sampler
  - Threshold: largest difference < 1e-5

### 2. Validation

- [x] **Smoke test: one cell by hand.**
  Run all three rows at full strength on a_cat×a_dog, seed 9, 20 steps. Score and view manually.

- [x] **Full sweep.**
  8 unseen pairs × 4 seeds × 5 strengths × 3 rows = 480 pictures.
  - Script: `scripts/mechanism_study/run_dose_sweep.sh` (resumable)
  - Time: ~50 seconds per cell on GPU, ~6 hours total

- [x] **Score every picture.**
  Use the validated composition scorer (GroundingDINO-based, decision: compose vs mono).
  - Output: `outputs/interaction_term/dose/dose_curves.json`

### 3. Review and outputs

- [x] **Follow the procedures.**
  Read [../procedures/hypothesis-02-recheck-the-headline-numbers.md](../procedures/hypothesis-02-recheck-the-headline-numbers.md) to completion. It prevents the scorer from picking up old pictures and sets cutoffs by visual inspection.

- [x] **Build the curves and grid.**
  Three curves on one axis; 3×5 grid of real cells above.
  - Script: `scripts/plot_dose_curves.py --root outputs/interaction_term/dose/pairs`
  - Layout: decided in `figure-01-the-seven-paper-figures`, not here
  - Grid: rebuilt per the procedure (step 6)

- [x] **Move output to the shared mount.**
  `/home-mscluster` had 6.3GB; moved to `/datasets`.
  - ✓ Completed 2026-08-18
  - `/home-mscluster/.../dose/pairs/` → `/datasets/.../dose/pairs/` (6.3GB)
  - `/home-mscluster` space freed; `/datasets` at 8% capacity

---

## The engagement gate

⬅️ [Tasks](#tasks) | 📋 [TOC](#table-of-contents) | [Outputs / Figure](#outputs--figure) ➡️

**Pass criteria:**
- All 480 cells generated (440 unique after dedup).
- Three curves computed: compose rate vs λ for real, random, wrong pair.
- 3×5 grid of real cells built, one row per injection type, one column per λ.
- Scored pictures logged to W&B with triptych (Mono, PoE, corrected) per cell.

**Fail criteria (stop and fix before proceeding):**
- Harness disturbs plain PoE at λ=0 (difference > 1e-5).
- GPU OOM (move to a larger node).
- Scorer and eyes disagree on the same picture (fix the scorer, re-score, do not move the cutoff to rescue the curve).

**When you have results:** Answer the two open questions in the [review file](../review/hypothesis-02-more-correction-more-composition.md).

---

## Outputs / Figure

⬅️ [The engagement gate](#the-engagement-gate) | 📋 [TOC](#table-of-contents) | [Code references](#code-references) ➡️

**Figure slot F2** (paper's headline figure):

| Component | Status | Location | What it shows |
|-----------|--------|----------|---------------|
| Curves (quantitative) | ✓ Ready | `outputs/interaction_term/dose/dose_curves.{json,png}` | Compose rate vs λ, three lines (real, random, wrong pair) |
| Grid (qualitative) | ✓ Ready | `scripts/dose_strip.py` output, copied to figure location | 3×5 grid of real generated cells, one row per injection type, one column per λ, same pair and seed |
| Combined figure | ⏳ Awaiting F2 layout decision | Slot F2, `paper/iclr/figures/` | Curves above or beside grid, caption finalized once scoring cutoff is set |

**Supporting artifacts:**
- W&B run: `prime_lab/poe-repair-animals-compose` project, 480-cell log with triptychs (Mono, PoE, corrected)
- Review file: [../review/hypothesis-02-more-correction-more-composition.md](../review/hypothesis-02-more-correction-more-composition.md) — answers the bar question and other checks

---

## Code references

⬅️ [Outputs / Figure](#outputs--figure) | 📋 [TOC](#table-of-contents) | [Next step](#next-step) ➡️

### Injection harness

**File:** `scripts/interaction_term_inject.py`  
**What it does:** Takes a cached correction, injects it at strength λ into PoE sampling, generates the output, and logs it.

```python
# Pseudocode
def inject(pair, seed, lambda_, vector_type="real"):
    correction = load_cached(pair, seed)
    if vector_type == "random":
        vector = random_vector(size=correction.shape)
    elif vector_type == "wrong_pair":
        vector = load_cached(other_pair, seed)
    else:  # "real"
        vector = correction
    
    image = poe_sample(pair, seed, injection=vector * lambda_)
    compose_rate = scorer(image)
    return compose_rate
```

### Plotting and curve building

**File:** `scripts/plot_dose_curves.py`  
**Usage:**
```bash
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY scripts/plot_dose_curves.py --root outputs/interaction_term/dose/pairs
# Outputs: dose_curves.json (data), dose_curves.png (visualization)
```

### Smoke test command

**Run one cell by hand:**
```bash
$PY -m poe_repair.experiments.interaction_term.inject \
  --pair a_cat__x__a_dog --seed 9 --lambda 0 --check-canary
# Expect: "canary ok, delta < 1e-5"
```

### Verify outputs

```bash
find outputs/interaction_term/dose/pairs -name "*.png" | wc -l
# Expect: 440 (480 cells minus 40 λ=0 duplicates)
```

---

## Next step

⬅️ [Code references](#code-references) | 📋 [TOC](#table-of-contents) | [Error Matrix](#error-matrix) ➡️

Once this gate passes (verdict: green), proceed to **step 5** [hypothesis-04: what-the-cached-runs-already-show](hypothesis-04-what-the-cached-runs-already-show.md), which reads these outputs without needing GPU.

After step 6 (hypothesis-03) finishes, step 13 (figure-01) builds F2 from the data on disk.

---

## Error Matrix

⬅️ [Next step](#next-step) | 📋 [TOC](#table-of-contents)

Known issues and solutions. This section is automatically updated after runs via `/ingest-error-pattern` and `/sync-plan-tree`. New errors from failed runs are added to the catalogs and propagated here.

### From global catalog

#### 🔴 py-001: Python 3.8 incompatible with DINOv2 imports

**When it happens:** During `from compose_scorer.scorer import _Embedders`

**What you see:** `ImportError: cannot import name '_Embedders'`

**Why:** Python 3.8 does not support the DINOv2 package.

**How to fix:** Activate Python 3.9+ (co3 env uses 3.10).

---

#### 🟠 gpu-001: Shared GPU—job dies partway through

**When it happens:** During the full sweep (`run_dose_sweep.sh`)

**What you see:** CUDA OOM error, job killed at random cell count

**Why:** Someone else is using the GPU; the card is shared, and multiple jobs overload it.

**How to fix:** 
1. Check `nvidia-smi` before launching.
2. If > 50% memory in use, wait or submit to biggpu instead of bigbatch.
3. Use `--partition biggpu` in the job submission.

---

#### 🟡 disk-001: Output directory is on /home-mscluster, not /datasets

**When it happens:** During `run_dose_sweep.sh` (the disk guard check)

**What you see:** Job aborts because `/home-mscluster` filled up, but the check looked at `/datasets`

**Why:** The output path is `OUT=$REPO/outputs/...` (home mount), but the guard checks `/datasets`.

**How to fix:** 
- Repoint `OUT=` to `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/`
- Or change the disk guard to check the filesystem being written to
- ✓ Status: Fixed 2026-08-18; output now goes to `/datasets` by default

---

### From project catalog

(Add entries from `docs/EXPERIMENT_ERROR_CATALOG.md` that apply to dose-sweep runs.)

---

**Auto-update disclaimer:** This section is regenerated from catalogs after each run. Manual edits here are overwritten. To update the catalogs themselves, use `/ingest-error-pattern --from-run-log` after a failed run.
