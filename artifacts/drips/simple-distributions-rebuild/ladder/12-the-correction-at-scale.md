# ⭐ Layer 12: the correction at scale, and the chest X-ray question

⬅️ [Layer 11: the testbed](11-the-text-conditioned-testbed.md) · 🏠 [the guide](../PROMPTS.md)

**Why it is critical.** It is the goal. Three facts from the medical sessions set its order:

- **Drainless pneumothorax is not rare in NIH.** 1,906 of the 3,543 NEATX pneumothorax films have no drain, 54%, so any overlap dial on real films is deliberate subsampling.
- **The drain shortcut is not yet shown in this project.** TorchXRayVision's DenseNet moves a median of `+0.001` between drain and drain-free films of the same patient over 237 pairs, and it scores the drain film higher in 134 of them.
- **RadEdit removes the pneumothorax and keeps the drain.** Only FastDiME removes and adds drains.

So the drain-gap test is a gate: a gap under 5 points, or reversed, ends the drain case. Composition's own medical job may then be a shortcut that cannot be masked, such as AP against PA view or the scanner.

**Depth.** Build: the drain-gap measurement and the correction on the real model. Own: [LoRA](../../../../../../Obsidian/Artificial%20Intelligence/Large%20Language%20Models/LoRA.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/ai-engineering/ai-engineering.pdf#page=649 "AI engineering, chapter 7, LoRA, Works the parameter saving with numbers (a 9 × 9 matrix of 81 parameters against a 9 × 1 and a 1 × 9 factor of 18), gives th..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/natural-language-processing/hands-on-large-language-models.pdf#page=497 "Hands-on large language models, chapter 12, LoRA, Rebuilds a 10 × 10 matrix from two smaller ones, 20 weights in place of 100, and why language models' low i..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/natural-language-processing/hands-on-large-language-models.pdf#page=503 "Hands-on large language models, chapter 12, Instruction tuning with QLoRA, Fine-tunes TinyLlama in code with a 4-bit base and LoRA adapters, explains the ran..."), [▶1](https://www.youtube.com/watch?v=_kREM2UAiJ8&t=3738s "CS229 lecture 12, 1:02:18 to 1:08:24, Works the count with numbers: W₀ + AB with inner rank r cuts the free parameters from d² to 2dr, at d = 1000 and r = 10..."), [▶2](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=5868s "CME295 lecture 4, 1:37:48 to 1:44:56, Freeze W₀ and learn BA with rank about 4 to 10, one pair per task; it helps most in the feed-forward layers, wants abou..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/plans/08-lora-update.md "The low-rank update, B times A, deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt, plan 08. Derives the parameter count r · (in + out) for..."), [◈](https://claude.ai/artifact/1GayZthQkRL4FmukdCV9WW#nb-lora "The SDXL notebook in ten parts, LoRA repair, Low-rank adapters on the cross-attention projections, trained against the single-prompt teacher.")</sub> as a low-rank update. The plans below set their own depth beyond that. You 8.3 to 16.2 h for the parts that run whatever the gate decides.

**Study first** for quizzes 12.1 and 12.2. [The LoRA update worked on paper](../../../../../../goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/plans/08-lora-update.md) and [the SVD of the real matrix](../../../../../../goal-setting/learning/deep-learning/diffusion-models/08-spectral-structure-of-the-correction/plans/06-svd-of-the-real-matrix.md). Textbooks: Anton and Rorres, *Elementary Linear Algebra* 4.8 ([rank](../../../../../../goal-setting/learning/mathematics/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=263), p262) and 9.4 to 9.5 (the [SVD](../../../../../../Obsidian/Mathematics/Linear%20Algebra/Singular%20Value%20Decomposition.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=125 "Mathematics for machine learning, section 4.5, Works an SVD by hand (Example 4.13, page 131) after its geometric picture (Example 4.12) and its construction ..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=528 "Linear algebra textbook, section 9.4, Singular values, then the SVD in brief and expanded form, with Example 2 for a matrix that is not square."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=535 "Linear algebra textbook, section 9.5, The reduced SVD, and compressing an image by keeping only the largest singular values."), [▶1](https://www.youtube.com/watch?v=HZl32cDhWEM&t=1007s "tilestats, PCA and SVD: the math explained simply, The singular value decomposition, 16:47 to 22:59, The SVD as the second way to compute PCA, after the eige..."), [▶2](https://www.youtube.com/watch?v=FgakZw6K1QQ&t=728s "StatQuest, Principal Component Analysis (PCA), Step-by-Step, Singular vector/value, Eigenvector/value and loading scores defined, 12:08 to 12:56, Names the s..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/08-spectral-structure-of-the-correction/plans/06-svd-of-the-real-matrix.md "SVD of your own 440 x 65,536 matrix, deep-learning/diffusion-models/08-spectral-structure-of-the-correction, plan 06. Stacks cached correction rows and runs ...")</sub> and compression, p528 to p535); Deisenroth et al. 4.5 to 4.6 (the SVD and [matrix](../../../../../../Obsidian/Mathematics/Linear%20Algebra/Matrix.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=39 "Linear algebra textbook, section 1.3, Matrix notation, then the row-by-column product worked on a 2 by 3 times a 3 by 4 matrix (Example 5, page 43)."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=28 "Mathematics for machine learning, section 2.2, Matrix addition and multiplication, inverse and transpose, and a system of linear equations written compactly ..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=50 "Deep learning textbook, section 2.2, The matrix product, the element-wise product and its properties, as deep learning uses them."), [▶1](https://www.youtube.com/watch?v=QtAZsWseIKk&t=511s "tilestats, Matrices and matrix operations for beginners 1/2, Matrix multiplication, 8:31 to 12:27, Matrix multiplication for beginners, followed in the next ..."), [▶2](https://www.youtube.com/watch?v=kYB8IZa5AuE&t=212s "3Blue1Brown, Linear transformations and matrices (Essence of linear algebra, chapter 3), Basis vectors and matrices, 3:32 to 6:18, A matrix read as where the..."), [◆](../../../../../../goal-setting/learning/deep-learning/transformers/textbook/03-multiply-two-matrices-row-by-column.md "Multiply two matrices row by column and say why the order matters, deep-learning/transformers, chapter 03. The same two matrices multiplied in both orders, t...")</sub> approximation, p125 to p135). Video: [Strang's MIT 18.065 lecture 6](https://www.youtube.com/watch?v=rYz83XPxiZo&t=468s).

**Ideas to tie in.** From [the ideas that tie the ladder together](../../../ideas/ideas-that-tie-the-ladder-together.md): [the interaction score as the object a correction can learn](../../../ideas/ideas-that-tie-the-ladder-together.md#5-learn-the-identifiable-object-which-is-the-interaction-score) (needed, 2 to 3 h, no training); [the interaction's own diffusion-time trajectory](../../../ideas/ideas-that-tie-the-ladder-together.md#6-give-the-interaction-its-own-diffusion-time-trajectory) (needed, 2 to 4 h, no training); [whether the chest target is observational or counterfactual](../../../ideas/ideas-that-tie-the-ladder-together.md#8-specify-whether-the-chest-target-is-observational-or-counterfactual) (needed before the chest experiment, 2 to 4 h). Each comes with an exact toy check.

**Achieve.** The drain gap with its interval · a transfer plan whose terms are defined · the LoRA's effect on compose rate on held-out pairs beside the interaction term per pair · on the whiteboard: a rank-1 update to a 2 × 2 weight matrix · quizzes 12.1 and 12.2 passed. This is the guide's last Achieve line.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 1. quizzes 12.1 and 12.2, prepared ─────────▶ │  2. the drain-gap measurement
   (paste during layer 9's build)              │       ▼
                                               │  3. /drip-idea, claim 8: which shortcut
                                               │       ▼
                                               │  4. only if the gap passes: /frame-hypothesis
                                               │       ▼
                                               │  5. /drip-refine-plan: the transfer plan
                                               │       ▼
                                               │  6. quiz 12.1, 7. LoRA maths, 8. quiz 12.2
                                               │       ▼
                                               │  9. build: the LoRA on held-out pairs
 layer 9's correction on the toys ───────────▶ │ 10. compare: the direction it predicts
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │ 11. /drip-retro, and the ladder is done
```

## The steps

### 1. AI, in the background: layer 12's quizzes

**AI, one layer ahead.** Paste this when you start layer 9's build.

```text
In the background, prepare and build layer 12's two quizzes, bottom up, the way the setup section of
artifacts/drips/simple-distributions-rebuild/PROMPTS.md does for layer 1. Goal: "read the LoRA as a
learned interaction term". Quiz 12.1, --minutes 40: matrix rank, a low-rank factorisation and its
parameter count, the singular value decomposition at gloss depth. Quiz 12.2, plan PROMPTS.md (layer
12), --minutes 40: LoRA as a low-rank weight update, why its residual is the ε-space interaction
term from layer 8, compose rate as the measure.

Task: for each quiz, first dispatch prep-cadence (quiz-prep-pulse --build) with the goal, minutes
and topics above, then dispatch quiz-cadence (mode build, repo
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain). Quiz 12.1 first, so quiz 12.2 can rest on it.
Starts from: nothing before it in this layer. Quiz 12.1 takes its material from the Study first
line of artifacts/drips/simple-distributions-rebuild/ladder/12-the-correction-at-scale.md; quiz 12.2
from that file's steps 7 and 9 and from layer 8's file,
artifacts/drips/simple-distributions-rebuild/ladder/08-the-interaction-term.md.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- quiz-cadence stops when a quiz has no coverage file in .quiz-prep/, so prep-cadence runs first
  for each quiz (quiz-cadence, "The preparation gate").
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails; open the PDFs with PyMuPDF from
  the co3 environment (checked on mscluster109: import pymupdf works in co3, version 1.28.2, and
  import pypdf fails everywhere).
- The shelf index is ~/Obsidian/Mathematics/references/textbooks/Textbooks.md and the
  worked-examples index is ~/Obsidian/Mathematics/references/exercises/Worked Examples.md
  (quiz-prep-pulse, "The bank").
- The textbook passages are Anton and Rorres 4.8 (rank, p262) and 9.4 to 9.5 (the SVD and
  compression, p528 to p535), Deisenroth et al. 4.5 to 4.6 (p125 to p135), and Strang's MIT 18.065
  lecture 6 from 7:48 (this file's Study first line).
- A LoRA keeps a projection W frozen and trains a rank-r update BA, giving (W + BA)x with r·(in +
  out) parameters: 384 for the journey's toy (rank 2, 3 blocks) and 4,956,160 for the real case
  (rank 8, 210 modules) (the journey plan 08-lora-update.md, Description and Goal).
- poe_repair_min's correction is a rank-8 cross-attention LoRA trained on the cached guided
  residual r_t = ε̃_J − ε̃_PoE (Mission, in
  ~/Playground/PhD/poe_repair_min/plans/04-does-the-fix-reach-unseen-pairs/MASTER_PLAN.md).
- That residual is w·(ε_J − ε_A − ε_B + ε_∅) at guidance w = 7.5, written in the cache as
  gs*(eps_j − eps_a − eps_b + eps_uncond); its bracket is layer 8's ε-space interaction term
  ε_joint − ε_A − ε_B + ε_∅ (the definition field of
  /datasets/mmolefe/poe_repair_min/outputs/showcase/interaction_bound/interaction_bound.json;
  REBUILD_WALK.md, brick "The interaction term, exact").
- The journey's real matrix is 440 × 65,536 cached correction rows, whose first singular value is
  133.3 (the journey plan 06-svd-of-the-real-matrix.md, Goal).
- A compose rate is the fraction of a pair's seeds where grounding-dino-tiny finds at least two
  distinct "animal" instances (NMS IoU under 0.5, confidence at least 0.30); it counts animals and
  does not identify them, so the 94% headline at full correction is an upper bound, the true rate
  sitting around 87% to 94% (~/Playground/PhD/poe_repair_min/context/world/compose-rate.md).
Outputs:
- two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each with a
  verdict per question: ready, blocked, or unprepared by choice;
- rows in the worked-examples index for every banked problem;
- two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
Target: no question on either page carries the verdict blocked; quiz 12.2 carries about 4 to 6
banked problems for its concept at the own depth; every number in an answer traces to a source
above or a cited textbook page. I judge, from the pages.
Progress: prep-cadence reports, per quiz, its question count, node count and verdicts;
quiz-cadence reports each URL as it publishes.
Proof: the two coverage files exist under .quiz-prep/ and list no question as blocked; the two URLs
open.
Limits: change nothing in poe_repair_min, the journey plans or rebuild/; in this ladder file, write
only the two URLs.
Hand-off: write quiz 12.1's URL into step 6 and quiz 12.2's into step 8 of
artifacts/drips/simple-distributions-rebuild/ladder/12-the-correction-at-scale.md, each replacing
"(URL lands here)".
```

added: the standalone task, with prep-cadence first and quiz-cadence second, the preparation gate, the missing `pdftotext` and the PyMuPDF route, the two indexes (from the setup's quiz prompt) · the [LoRA](../../../../../../Obsidian/Artificial%20Intelligence/Large%20Language%20Models/LoRA.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/ai-engineering/ai-engineering.pdf#page=649 "AI engineering, chapter 7, LoRA, Works the parameter saving with numbers (a 9 × 9 matrix of 81 parameters against a 9 × 1 and a 1 × 9 factor of 18), gives th..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/natural-language-processing/hands-on-large-language-models.pdf#page=497 "Hands-on large language models, chapter 12, LoRA, Rebuilds a 10 × 10 matrix from two smaller ones, 20 weights in place of 100, and why language models' low i..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/natural-language-processing/hands-on-large-language-models.pdf#page=503 "Hands-on large language models, chapter 12, Instruction tuning with QLoRA, Fine-tunes TinyLlama in code with a 4-bit base and LoRA adapters, explains the ran..."), [▶1](https://www.youtube.com/watch?v=_kREM2UAiJ8&t=3738s "CS229 lecture 12, 1:02:18 to 1:08:24, Works the count with numbers: W₀ + AB with inner rank r cuts the free parameters from d² to 2dr, at d = 1000 and r = 10..."), [▶2](https://www.youtube.com/watch?v=VlA_jt_3Qc4&t=5868s "CME295 lecture 4, 1:37:48 to 1:44:56, Freeze W₀ and learn BA with rank about 4 to 10, one pair per task; it helps most in the feed-forward layers, wants abou..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt/plans/08-lora-update.md "The low-rank update, B times A, deep-learning/diffusion-models/latent-diffusion-architecture-rebuilt, plan 08. Derives the parameter count r · (in + out) for..."), [◈](https://claude.ai/artifact/1GayZthQkRL4FmukdCV9WW#nb-lora "The SDXL notebook in ten parts, LoRA repair, Low-rank adapters on the cross-attention projections, trained against the single-prompt teacher.")</sub>'s parameter counts (from the journey plan on the LoRA update) · the correction's definition and why its bracket is layer 8's term (from poe_repair_min's scope 04 master plan and `interaction_bound.json`) · the real [matrix](../../../../../../Obsidian/Mathematics/Linear%20Algebra/Matrix.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=39 "Linear algebra textbook, section 1.3, Matrix notation, then the row-by-column product worked on a 2 by 3 times a 3 by 4 matrix (Example 5, page 43)."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=28 "Mathematics for machine learning, section 2.2, Matrix addition and multiplication, inverse and transpose, and a system of linear equations written compactly ..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=50 "Deep learning textbook, section 2.2, The matrix product, the element-wise product and its properties, as deep learning uses them."), [▶1](https://www.youtube.com/watch?v=QtAZsWseIKk&t=511s "tilestats, Matrices and matrix operations for beginners 1/2, Matrix multiplication, 8:31 to 12:27, Matrix multiplication for beginners, followed in the next ..."), [▶2](https://www.youtube.com/watch?v=kYB8IZa5AuE&t=212s "3Blue1Brown, Linear transformations and matrices (Essence of linear algebra, chapter 3), Basis vectors and matrices, 3:32 to 6:18, A matrix read as where the..."), [◆](../../../../../../goal-setting/learning/deep-learning/transformers/textbook/03-multiply-two-matrices-row-by-column.md "Multiply two matrices row by column and say why the order matters, deep-learning/transformers, chapter 03. The same two matrices multiplied in both orders, t...")</sub>'s size (from the journey plan on the [SVD](../../../../../../Obsidian/Mathematics/Linear%20Algebra/Singular%20Value%20Decomposition.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/mathematics-for-machine-learning.pdf#page=125 "Mathematics for machine learning, section 4.5, Works an SVD by hand (Example 4.13, page 131) after its geometric picture (Example 4.12) and its construction ..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=528 "Linear algebra textbook, section 9.4, Singular values, then the SVD in brief and expanded form, with Example 2 for a matrix that is not square."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/linear-algebra/linear-algebra-textbook.pdf#page=535 "Linear algebra textbook, section 9.5, The reduced SVD, and compressing an image by keeping only the largest singular values."), [▶1](https://www.youtube.com/watch?v=HZl32cDhWEM&t=1007s "tilestats, PCA and SVD: the math explained simply, The singular value decomposition, 16:47 to 22:59, The SVD as the second way to compute PCA, after the eige..."), [▶2](https://www.youtube.com/watch?v=FgakZw6K1QQ&t=728s "StatQuest, Principal Component Analysis (PCA), Step-by-Step, Singular vector/value, Eigenvector/value and loading scores defined, 12:08 to 12:56, Names the s..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/08-spectral-structure-of-the-correction/plans/06-svd-of-the-real-matrix.md "SVD of your own 440 x 65,536 matrix, deep-learning/diffusion-models/08-spectral-structure-of-the-correction, plan 06. Stacks cached correction rows and runs ...")</sub>) · the compose rate's rule and its bound (from poe_repair_min's compose-rate context file) · the URL write-back
dropped: nothing

⏱ AI ~1.5 h (estimate), beside layer 9's build

### 2. You: the gate first, the drain-gap measurement

```text
/drip-execute-plan
~/Playground/PhD/medical-imaging/compositional_diffusion/b-shortcut-through-a-device/b1-drain-shortcut-in-pneumothorax/plans/01-measure-the-drain-gap.md
--depth "to 10.1"

Task: resume the paused walk on this plan from its walk map, in the plan's order up to task 10.1.
B1's folder below means
~/Playground/PhD/medical-imaging/compositional_diffusion/b-shortcut-through-a-device/b1-drain-shortcut-in-pneumothorax/
and every relative path is inside it.
Starts from: the walk map artifacts/drips/drain-gap-walk/EXECUTE_WALK.md and the pick-up page
plans/00-where-we-left-off.md, both in B1's folder.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The walk stands at round 2 of 5: tasks 1.1, 1.2, 3.1 to 3.3, 5.1 and 7.1 are ticked, instruction
  2.1 is confirmed and 4.2 skipped; it waits on instructions 4.1 and 6.1 (my count checks) and my
  prediction of how the folds come out (the walk map; the pick-up page, "Walk 1").
- The label table data/pneumothorax_drain_table.csv holds 112,120 NIH films of 30,805 patients;
  all 3,543 NEATX rows join, 1,011 patients, 0 dropped; groups drain 1,637, no_drain 1,906,
  ptx_unlabelled 1,759, negative 106,818 (plan, task 3.2).
- data/folds.csv puts the 30,805 patients in 5 folds of 6,161, with 0 patients in two folds; the
  112,120 image files at /datasets/mmolefe/data/nih/images/ have 0 missing and 0 empty (plan, tasks
  5.1 and 7.1; the pick-up page).
- NEATX Drain_a1 and Drain_a2 are confidence grades (0, 0.25, 0.75, 1), not yes or no; two
  drain-less gallery films look like missed drains, 00026451_014 and 00012752_005 (the walk map).
- The classifier is DenseNet-121 from ImageNet weights at 224 px on all 14 NIH labels, only the
  pneumothorax output scored, early stopping on 10% of the training patients, one model per fold;
  the cxr environment has torch 2.2.2 and torchvision 0.17.2 and no timm (plan, Description and
  Prerequisites; task 8.1).
- Held-out scores go to /datasets/mmolefe/shortcut_companion/runs/fold<k>/scores.csv; the runs
  folder there does not exist yet (listed 2026-10-18).
- The plan names mscluster107, which has been down since 2026-10-11 15:41 and takes no run;
  mscluster106 was idle under my Slurm job 67958 on 2026-10-14. Read the node table and nvidia-smi
  before choosing (the node table:
  ~/Playground/PhD/medical-imaging/compositional_diffusion/environment/cluster/nodes.md).
- Every launch pins CUDA_VISIBLE_DEVICES, uses nohup and absolute paths, and its script checks df
  /datasets and aborts if the chosen card holds more than 1 GB (plan, Considerations and task
  8.1; environment/cluster/launch-rules.md in the same project).
- Training time is not measured: task 8.2 times one epoch, and instruction 9.1 is my choice of 2 or
  5 folds on that number (plan, Expected runtime).
- Task 10.1 fixes the threshold at 90% specificity on each fold's held-out non-pneumothorax films,
  pools, and takes a 95% interval from 1,000 patient resamples; it prints each group's AUROC beside
  Oakden-Rayner's 0.940 (drain) and 0.770 (drain-less), and the age, view and drain side of both
  groups (plan, task 10.1).
- The bar, fixed before any run: a gap is at least 5 points with the interval excluding 0; under 5
  points, or an interval including 0 with an estimate under 5, or a reversed difference, is no gap
  and the drain case ends; 5 or more with an interval including 0 is inconclusive and does not
  continue; above 15 points is checked for a label or split error first (plan, "What has to pass").
- Two estimates of the interval disagree: about ±2.6 points on the difference (plan, "Why folds")
  and about ±1 point per group (assumed, session b268bbd8); the first full run settles it (the
  pick-up page).
- A gap is a necessary sign of a shortcut, not proof: drains go into large pneumothoraces (plan,
  Considerations).
- The public TorchXRayVision model is not this measurement: median +0.001 over 237 same-patient
  pairs, drain film higher in 134; 180 of 1,637 drain films against 514 of 1,906 drain-free films
  below its threshold, on films it may have trained on (the results card
  ~/Playground/PhD/medical-imaging/compositional_diffusion/b-shortcut-through-a-device/artifacts/results/does-the-classifier-look-at-the-drain/README.md).
Outputs: src/train_fold.py and src/measure_gap.py in B1's folder; five scores.csv files;
artifacts/results/does-the-classifier-use-the-drain/sensitivity-by-drain__nih-neatx.png with its
sidecar CSV and group-mix__age-view-side.csv; the walk map and the plan's ticks current.
Target: task 10.1 prints each group's hits over group size, the difference in points and its 95%
interval; the verdict follows the bar above. I judge, in instruction 11.1.
Progress: task 8.1's 2-minute smoke run on 2,000 images writes one scores.csv row per held-out
image; task 8.2 prints images per second and the projected hours; task 8.3 counts five scores.csv
files on the node that ran them; task 10.1 prints the two sensitivities.
Proof: src/measure_gap.py prints both groups' hits over group size and the difference with its
interval, and the figure and both CSVs exist.
Limits: the bar never changes after a run; never launch on a card with more than 1 GB in use or
another user's process; instruction 11.1 is mine; no commit.
Hand-off: the verdict instruction 11.1 records (gap, no gap or inconclusive, with the difference
and its interval), which step 3's walk and step 4's gate read.
```

added: the walk map and pick-up page as the resume point, with what is ticked and what waits on you (from `EXECUTE_WALK.md` and `00-where-we-left-off.md`) · the label, fold and image counts (from tasks 3.2, 5.1 and 7.1) · the grade meaning of the NEATX columns and the two suspect films (from the walk map) · the [classifier](../../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Classifier.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/natural-language-processing/hands-on-large-language-models.pdf#page=160 "Hands-on large language models, chapter 4, Works a classifier end to end on real data: 5,331 positive and 5,331 negative Rotten Tomatoes reviews labelled by ..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-an-introduction.pdf#page=30 "Probabilistic machine learning: an introduction, section 1.2.1, Classification on the iris data: 150 flowers with four measurements each and one of three spe..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=78 "Understanding deep learning, sections 5.4 to 5.5, A network becomes a binary classifier by predicting a Bernoulli probability through the logistic sigmoid, a..."), [▶1](https://www.youtube.com/watch?v=pdqofxJeBN8&t=995s "CS231n lecture 2, 16:35 to 20:12, Builds the nearest neighbour classifier: training memorises the data, prediction returns the label of the closest training ..."), [▶2](https://www.youtube.com/watch?v=DATnpGoGhM8&t=759s "CS229 lecture 1, 12:39 to 14:39, Separates regression (continuous y) from classification (discrete y) with a house against townhouse boundary, and frames nex..."), [◆](../../../../../../goal-setting/learning/machine-learning/classical-ml-from-scratch/MASTER_PLAN.md "Classical ML from Scratch, machine-learning/classical-ml-from-scratch. Goal 1 classifies a toy dataset with logistic regression written by hand, and goal 7 b..."), [◈](https://claude.ai/artifact/DBmJtn6xy75sPtQwPEvQZU#nb-sec4 "Kinds of composition notebook, A classifier learns 'not B', A classifier trained on the labels the data carries learns the hidden-B points as 'not B'.")</sub> and environment (from the plan) · the missing runs folder (listed) · mscluster107 down and the node to check (from the compositional_diffusion node table) · the launch guards, the unmeasured runtime, task 10.1's method, the full three-way bar and the surprise line (from the plan) · the two [interval estimates](../../../../../../Obsidian/Mathematics/Statistics/Hypothesis%20Testing/Confidence%20Interval.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-statistics.pdf#page=99 "Introduction to statistics (Wits lecture notes), section 5.3, Works with numbers: three cases for an interval on a population mean (σ known, a large sample, ..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=559 "Introduction to probability for data science, section 9.1, The estimator as a random variable, the definition with a confidence level of 1 − α (page 563), wh..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-an-introduction.pdf#page=177 "Probabilistic machine learning: an introduction, sections 4.7.4 and 4.7.5, The frequentist definition over hypothetical repeated datasets, and why a 95% conf..."), [▶1](https://www.youtube.com/watch?v=kv8u7yl6VQk&t=1235s "Professor Leonard, Statistics lecture 7.3, Confidence Interval for the Sample Mean, Population Std Dev Known, 20:35 to 23:40, The lecture's worked example, i..."), [▶2](https://www.youtube.com/watch?v=TqOeMYtOc1w&t=175s "StatQuest, Confidence Intervals, Clearly Explained, 2:55 to 3:41, Defines a confidence interval, right after the chapter that calculates bootstrapped means. ..."), [◆](../../../../../../goal-setting/learning/mathematics/statistics/sampling/textbook/05-what-an-interval-claims.md "What a 95% interval claims, on one side or two, mathematics/statistics/sampling, chapter 05. The 95% belongs to the procedure over repeated runs and never to...")</sub> (from the pick-up page) · the public model's numbers as context, not as the gate (from its results card)
dropped: nothing

⏱ you ~4 h (task groups 0 to 10, about 8 at 30 minutes each) · cluster: five fold models, hours unknown until task 8.2 times one [epoch](../../../../../../Obsidian/Artificial%20Intelligence/Machine%20Learning/Epoch.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/ai-engineering/ai-engineering.pdf#page=686 "AI engineering, chapter 7, Number of epochs, Puts numbers on the choice: 1 to 2 epochs may be enough for millions of examples, 4 to 10 can still help for tho..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/understanding-deep-learning.pdf#page=99 "Understanding deep learning, section 6.2.1, Batches are drawn without replacement until the data is used up, and a single pass through the entire training se..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=262 "Deep learning textbook, section 7.8, Learning curves over 250 epochs of a network on MNIST, where training loss keeps falling and validation loss turns back ..."), [▶1](https://www.youtube.com/watch?v=Q0TtSM9aRew&t=1080s "TileStats, Stochastic gradient descent (SGD) vs mini-batch GD, iterations vs epochs, Epochs vs iterations, 18:00 to 20:00, The closing segment that separates..."), [▶2](https://www.youtube.com/watch?v=dyNGd06MWn4&t=2189s "CS231n lecture 3, 36:29 to 43:01, Defines stochastic gradient descent on random minibatches of 256 examples and the epoch, then its failure modes. Caption-read.")</sub>, as the plan itself says · box 8 h of your time · stop: task 10.1 prints the gap with its interval, and instruction 11.1 records the verdict

**Outcome:** the drain gap with its interval, which decides whether the drain case continues. Its answer keys are that plan and the README of `pneumothorax-score-per-film.csv` in the same project.

### 3. You: then the open questions, in one walk

```text
/drip-idea resume artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md at claim 8, with the
drain-gap result: which shortcut leads (the drain, or one that cannot be masked such as view or
scanner); whether NIH's drainless pneumothoraces are mostly large (measured with the SIIM-ACR masks
at /datasets/mmolefe/siim_acr); whether FastDiME trains classifiers on its edits; and how many of
the 4,312 follow-up films with no pneumothorax label still carry a drain (estimated at about 1,100)

Task: walk claim 8 with the four questions above, one per round, and mark whether the drain case
stands, in repo /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.
Starts from: step 2's verdict in instruction 11.1 of the drain-gap plan, and the sidecar CSV beside
sensitivity-by-drain__nih-neatx.png in B1's
artifacts/results/does-the-classifier-use-the-drain/.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- Claim 8 is wrong as stated; its repair: on the drain shortcut, composition is one arm against
  upweighting the real drain-free films and against masked editing, run only after B1's drain gap
  shows the shortcut exists; composition's own job is a shortcut that cannot be masked, such as AP
  against PA view or scanner (the map, claim 8).
- 1,906 of the 3,543 NEATX pneumothorax films have no drain, so the drainless cell is common, not
  missing (claims 6 and 8).
- Oakden-Rayner et al. report AUC 0.940 with a drain and 0.770 without on the NIH test set; Thian
  et al. 2021 find 0.95 against 0.94 at six outside hospitals (claim 8).
- TorchXRayVision's DenseNet: median +0.001 over 237 same-patient pairs, drain film higher in 134;
  180 of 1,637 drain films against 514 of 1,906 drain-free films below its threshold; the split
  holds within each view, PA 131 of 1,030 against 407 of 1,281 and AP 49 of 607 against 107 of
  625, which may come from which patients get a drain rather than from the tube (claim 8; B1's
  results card does-the-classifier-look-at-the-drain/README.md).
- RadEdit removes the pneumothorax and keeps the drain; a classifier trained on CANDID-PTX falls
  from 93.3 to 17.9 accuracy on 629 edited films; it tests classifiers and never trains on its
  edits (claim 8; the map's Sources).
- Route 6's return answers the FastDiME question: FastDiME trains ResNet-18 at 100, 75 and 50%
  drain association and reports AUROC 0.98 with the association and 0.65 without, detection only;
  its discussion names training on counterfactuals as an untested next step. The return is not yet
  integrated: the map's Sources still say "unchecked" and its Routes row says "emitted"
  (routes/06-composition-against-masked-editing.return.md).
- Route 6's return also names two missing arms, a jointly conditioned model p(x | pneumothorax, no
  drain) and prevalence recalibration (Kina and Petersen: AUROC 0.226 to 0.726 on films where the
  drain points the wrong way), the bounded positive label "no drain" in place of negation, and test
  sizes of about ±0.14 at 25 films against 25 and ±0.07 at 100 against 100 (same return).
- The follow-up pool is 4,312 later NIH films of NEATX drain patients with no pneumothorax label,
  385 patients; about 1,100 probably still carry a drain, range 900 to 1,400, three counted group
  sizes times estimated shares, none looked at; 907 of them sit between two drain films, 397 of
  those labelled "No Finding" (B1's pick-up page, route 9's answer; claim 6).
- /datasets/mmolefe/siim_acr holds positive_imgs.npy, positive_masks.npy and
  positive_imgs_names.npy (listed 2026-10-18).
- Data_Entry_2017.csv carries View Position, image size and pixel spacing per film and no scanner
  or site column (its header line, in B1's data/).
- B1 never settled which shortcut leads: a session proposed keeping the drain as the lead and adding
  a non-localised shortcut (AP against PA view, or scanner and site) only if the drain result is a
  null or ties with RadEdit (B1's pick-up page, "Proposals never answered").
Outputs: claim 8's row and section in the map, with the shortcut to target and the drain case
marked standing or ended; an answer or a "needs a check" entry per question, counted ones with their
counts; route 6 integrated if the walk takes it in.
Target: the shortcut to target is named and the drain case is marked; both counted questions carry
a number; I judge.
Progress: each round closes one question with a mark.
Proof: grep -n "^| 8\." artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md prints claim 8's new
mark.
Limits: write only in that idea map and its routes/ and runs/ folders; read B1's files and never
edit them; no commit.
Hand-off: claim 8's row and section, which step 4 reads if the gate passed, and step 11's
cross-links.
```

added: the step 2 verdict as the starting point · claim 8's repair, the counts, the published and public-model numbers with the view split, RadEdit's numbers (from the idea map and B1's results card) · route 6's unintegrated answer to the FastDiME question and its missing arms and test sizes (from the route 6 return) · the follow-up pool's counts (from B1's pick-up page) · the SIIM-ACR files (listed) · the NIH metadata's columns (from its header) · B1's unanswered proposal on which shortcut leads (from the pick-up page)
dropped: "how many of the about 1,100 drain films have no pneumothorax label" (why: the 1,100 are already films with no pneumothorax label, estimated to carry a drain, so the question is now how many of the 4,312 really carry one)

⏱ you ~1.5 h (four questions, two of them counted from data) · AI ~30 min · box 3 h · stop: the shortcut to target is named, and the drain case is marked as standing or ended. This walk runs whatever the gate's verdict is.

**Outcome:** the shortcut composition should target, and whether the drain case stands.

### 4. You: then the experiment, only if the gate passes

```text
/frame-hypothesis the four training arms in artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md
claim 8 (including upweighting the real drainless films), with stress-testing before augmentation,
and the dial, memorisation and test-size checks from
artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md,
with the claim that the toy's overlap-dial curve predicts the share of real films below which
composition recovers the cell and joint conditioning does not

Task: sharpen that claim into a falsifiable, ordered plan over refinement rounds, then compile. Run
it only if step 2 recorded a gap.
Starts from: step 2's verdict (a gap: at least 5 points, interval excluding 0); step 3's claim 8
entry in the idea map, which names the shortcut to target.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- Claim 8's check names four arms, one pneumothorax classifier trained on: the original NIH films;
  the real drain-free films upweighted; plus drain-removed edits of positive films; plus composed
  pneumothorax-without-drain films. All four are tested on real drain-free pneumothorax films
  (NEATX labels) beside the drain films, and on the same films with only the drain removed, scored
  against a VAE round trip of the original (the map, Checks outstanding, claim 8).
- What each outcome means there: upweighting closes the gap, so no generated data is needed; edits
  close it and composition adds nothing; composition closes it further, so the untreated look
  matters; none closes it, so data of no kind fixes the shortcut (same row).
- Route 6's return proposes six arms: ERM, prevalence recalibration or DFR, drain-removed edits,
  edits that remove the drain and grow the pneumothorax, the jointly conditioned model, and
  composed samples (construction 1 with a "no drain" label, annealed MALA or UHA graded on the toy
  first); the dial is the share of real drainless films the generators see, 100, 25, 5, 1 and 0%;
  the null is composition tracking joint conditioning at every share (the route 6 return).
- The toy's prediction: construction (1) holds 0.989 of the product in the rare corner down to a
  0.001 share on the four-quadrant toy, while separately fine-tuned experts reach at most 0.5 when
  each one's leftover density sits in the two adjacent quadrants (the answer key's case; leftover on
  the top-right alone gives 0.9997, ladder layer 5's corrected answer key) (the map, claim 2, run
  1).
- Before any classifier trains: pneumothorax size in each arm's films against real drainless films
  (SIIM-ACR masks or a segmenter), and nearest-neighbour distance to the training films for
  memorisation (the route 6 return).
- Test size: about ±0.14 at 25 films against 25 and ±0.07 at 100 against 100 for AUROC 0.75
  (the route 6 return).
- Negation with "contains" labels is unbounded, p(drain|x)^(−w) growing as p(drain|x) → 0; use the
  positive label "no drain", whose factor is at most 1 (the route 6 return).
- Baselines to beat: FastDiME AUROC 0.98 on films that keep the drain association and 0.65 without
  it; Kina and Petersen's recalibration 0.226 to 0.726 (the route 6 return).
- Scoop risk: Weng et al. name training on counterfactuals as their next step, and Petersen is on
  both the FastDiME and the recalibration paper (the route 6 return).
- Stress-testing comes before augmentation: the treated-looking lung without a drain is the
  counterfactual a stress test wants, and a training risk only where the air rim vanishes (the map,
  claim 2b and route 5).
- 1,906 drainless films exist, so every share below 100% is deliberate subsampling (claims 6 and
  8).
- Samplers recover a third to four fifths of the exact A∩B mass at 2% to 10% overlap (the
  overlap-dial map, claim 4), and trained experts cost more under a better sampler: interaction
  +0.054 total variation at r = 0.05 and +0.232 at r = 0.001 (the map, run 7).
Outputs: the five-part compile, saved after compile to
artifacts/ideas/when-poe-reaches-a-and-b/HYPOTHESIS.md (frame-hypothesis compiles inline, so
saving it is asked for here).
Target: a three-way rule with a number in each branch (supports, null, inconclusive) on the
drainless AUROC per arm at each dial share, with the films per cell counted; I judge.
Progress: each round states what is solid, what needs sharpening and what is not yet falsifiable,
with the routing verdict.
Proof: HYPOTHESIS.md exists and its falsification rule carries a number in each of the three
branches; I read it.
Limits: trains and generates nothing; gives no novelty verdict (route 6 holds one); writes only
HYPOTHESIS.md; no commit.
Hand-off: HYPOTHESIS.md, which hypothesis-to-scope or experiment-planner reads next.
```

added: the gate condition and the step 3 entry as the starting point · claim 8's four arms, their test films and outcome meanings (from the idea map's Checks outstanding) · route 6's six arms, dial shares, null, size and memorisation checks, test sizes, the bounded label, the baselines and the scoop risk (from the route 6 return) · the toy's prediction, the stress-testing finding, the sampler caveats (from both idea maps) · where the compile is saved
dropped: nothing

⏱ untimed: it runs only if the gate passes, and its size depends on which shortcut the walk above picks.

**Outcome:** a falsifiable plan for the thesis chapter. The baselines to beat:

- FastDiME: AUROC `0.98` on films that keep the drain association and `0.65` without it.
- Kina and Petersen's prevalence recalibration: `0.226` to `0.726`.

The interval at 100 films against 100 is about ±0.07.

### 5. You: then the transfer plan, repaired before it runs

```text
/drip-refine-plan
~/Playground/PhD/poe_repair_min/plans/04-does-the-fix-reach-unseen-pairs/plans/hypothesis/04-transfer-as-a-rate-over-fifteen-pairs.md:
settle whether compose rate is binary per pair or a fraction over evaluation seeds (the plan and its
master plan disagree); which 15 pairs it means, since pair_pool.yaml lists 11 trained and 6 held
out, 17 in all; what its x-axis becomes, since leave-one-pair-out holds out exactly 1 of 15 in every
run, so the planned curve from 2/15 to 15/15 cannot be drawn; and what count out of 15 "most of the
15" means

Task: walk the four questions one per piece, writing each answer into the plan, in repo
~/Playground/PhD/poe_repair_min.
Starts from: nothing earlier in this layer; the plan, its scope's MASTER_PLAN.md and
artifacts/results/does-the-fix-reach-unseen-pairs/pair_pool.yaml.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The scope's master plan defines compose rate over the eval seeds and calls support "held-out pairs
  compose above the do-no-harm baseline across most of the 15, direction intact"; the plan's task
  3 collects "compose-rate (binary: yes/no)" per pair ("Reading a result" and Goals in
  plans/04-does-the-fix-reach-unseen-pairs/MASTER_PLAN.md; the plan, task 3).
- A compose rate is the fraction of a pair's seeds where grounding-dino-tiny finds at least two
  distinct animal instances; fail rates are reported as, for example, 1.00 (8/8)
  (context/world/compose-rate.md).
- pair_pool.yaml lists 11 train pairs and 8 heldout entries: 6 marked transfer test (leopard×jaguar,
  frog×toad, eagle×hawk, seal×walrus, goose×swan, cow×buffalo), cat×dog as the known-failure
  reference and elephant×penguin as a control. Its header comment says 4 held-out blend pairs, 3
  controls and "the full 15", which the list does not match: the list holds 17 blend pairs and 1
  control (artifacts/results/does-the-fix-reach-unseen-pairs/pair_pool.yaml).
- A second file of the same name, artifacts/_shared/cross_pair_pool_configs/pair_pool.yaml, is a
  different pool of 5 train and 5 held-out pairs across groups; the pool this plan means is the one
  under artifacts/results/does-the-fix-reach-unseen-pairs/ (both files read; the clean-pair-pool
  plan, plans/04-does-the-fix-reach-unseen-pairs/plans/tools/01-the-clean-pair-pool.md).
- The plan's x axis runs 1/15 to 15/15 "fraction held out", while every one of its 15 runs holds
  out exactly one pair (the plan, Description step 4 and Figure Catalog).
- Pass needs at least 13 of 15 runs; a run whose distance-reached is zero is delivery-null and left
  out of the curve (the plan, "What has to pass here").
- Runtime is about 8 to 12 hours per run, 5 to 7 days for all 15; biggpu allows one Slurm job per
  user, so the 15 run one after another or across idle second GPUs by the shared-device path;
  checkpoints go to /datasets (the plan, Considerations and Environment Facts).
- Training seeds are 1 to 8 and 17 to 19 and held-out seeds 9 to 16; 12 cache cells in 4 pairs sit
  in both split folders and resolve held-out first unless split="train" is passed
  (environment/known-failures.md, entry poe-data-001).
Outputs: the plan file with the four answers written into its Goals, Description and "What has to
pass here"; the walk's REFINE_WALK.md under artifacts/ideas/<slug>/ in poe_repair_min; a
CHANGELOG.md entry if the direction text changes.
Target: each of the four questions has a written answer in the plan, and the curve the plan names
can be drawn from 15 one-pair-out runs; I judge.
Progress: each piece ends with the plan's new sentence for one question.
Proof: grep -n -i "compose.rate\|pair_pool\|x axis\|out of 15" on the plan prints the four new
lines.
Limits: edit only this plan; a change the master plan needs is proposed for my yes, not written;
train nothing; no commit.
Hand-off: the edited plan, whose compose-rate unit and pair set step 9's build reads.
```

added: the master plan's and plan's wording on compose rate (from both files) · the compose-rate rule (from poe_repair_min's context) · what `pair_pool.yaml` actually lists, and that its header disagrees with its list (from the file) · the second pool file of the same name (both files read) · the axis, the 13-of-15 pass line, runtime, the Slurm limit and the seed split with the held-out-first failure (from the plan and `known-failures.md`)
dropped: nothing

⏱ you ~1 h (the one measured `drip-refine-plan` took 66 minutes, too few runs to trust) · AI ~30 min · box 1.5 h · stop: each of the four questions has a written answer in the plan

**Outcome:** a transfer plan whose compose rate, pair set and curve are defined before any run, so the last bar of this layer measures something.

### 6. You: quiz 12.1

**Then the correction on the real model.** Take quiz 12.1 first.

Quiz 12.1: (URL lands here)

Built by step 1. ⏱ you 20 to 40 min · stop: every *needed now* question right and at least 80% of the rest; below that, read the items your misses point at and retake.

### 7. You: the maths drip

```text
/drip --math LoRA as a low-rank weight update, and why its residual is the ε-space interaction term
from layer 8

Starts from: quiz 12.1 passed, its misses as first rungs; the layer 8 --math drip folder under
artifacts/drips/, whose dispatch.md holds my notation for r(x) and ε_joint − ε_A − ε_B + ε_∅.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- A LoRA keeps W frozen and trains a rank-r update BA, (W + BA)x, with r·(in + out) parameters:
  384 for the journey's toy (rank 2, 3 blocks) and 4,956,160 for the real case (rank 8, 210
  modules) (the journey plan 08-lora-update.md).
- poe_repair_min's LoRA is rank-8 cross-attention, trained on the cached guided residual r_t = ε̃_J
  − ε̃_PoE (its scope 04 master plan, Mission).
- With ε̃_PoE = ε_∅ + w(ε_A − ε_∅) + w(ε_B − ε_∅) and ε̃_J = ε_∅ + w(ε_J − ε_∅), the residual is
  w(ε_J − ε_A − ε_B + ε_∅), the cache's gs*(eps_j − eps_a − eps_b + eps_uncond) at w = 7.5 (the
  testbed idea map, claim 1; interaction_bound.json, definition).
- The bracket is layer 8's ε-space interaction term (REBUILD_WALK.md, brick "The interaction term,
  exact").
- A rank-1 update to a 2 × 2 weight matrix is the whiteboard instance (this file's Achieve line).
Outputs: the drip's thread folder under artifacts/drips/, with dispatch.md recording my notation,
the worked numbers I used and what clicked.
Target: I derive w(ε_J − ε_A − ε_B + ε_∅) from the two guided predictions without notes, and count
a LoRA's parameters for one projection by hand; I judge.
Hand-off: the thread folder's path, which quiz 12.2 and step 9's build read.
```

added: the parameter counts (from the journey plan on the LoRA update) · the LoRA's target and the algebra from the two guided predictions to layer 8's term (from poe_repair_min's scope 04 master plan, the testbed idea map and `interaction_bound.json`) · the whiteboard instance (from this file's Achieve line) · quiz 12.1's misses as first rungs
dropped: nothing

⏱ you 10 to 20 min, AI ~10 min

**Outcome:** the LoRA understood as a learned interaction term.

### 8. You: quiz 12.2

Quiz 12.2: (URL lands here)

Built by step 1. ⏱ you 20 to 40 min · stop: every *needed now* question right and at least 80% of the rest. Below that, the misses become the first rungs of a second `/drip --math`, and you retake before building.

### 9. You: the build

```text
/drip-execute-plan --request in ~/Playground/PhD/poe_repair_min: read the cached ε_joint, ε_A, ε_B
and ε_∅ for the 25 pairs as the layer 8 quantity, with the layer 9 correction as the reference
behaviour, sampled with tuned MALA; bar: the correction's effect on compose rate on held-out pairs,
reported beside the interaction term per pair

Task: cut this into 4 to 6 pieces and build, teach and test one per round. poe_repair_min is read,
never written: I type every block myself into
~/Playground/PhD/simple-distributions-walk/walk12.py and run it with the python the node table in
~/Playground/PhD/poe_repair_min/environment/hpc/nodes.md names.
Starts from: step 5's refined transfer plan (the compose-rate unit and the held-out pair set); the
layer 12 --math drip folder (step 7); walk08.py's ε-space term and walk09.py with
rebuild/phase9_correction.py (from layers 8 and 9).
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The cache is /datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/; one cell is one
  pair and seed, with residuals/step_NNN.pt per step holding eps_a_raw, eps_b_raw, eps_j_raw,
  eps_uncond and timestep, for all 50 DDIM steps of the uncorrected run at guidance 7.5
  (poe_repair_min report does-interaction-strength-predict-which-pairs-blend.md, "Where this came
  from"; its figure explainer).
- The 25 pairs are the keys of pairs in
  /datasets/mmolefe/poe_repair_min/outputs/showcase/interaction_bound/interaction_bound.json, each
  with n_seeds, fail_rate and median_bound_by_step over steps 0 to 49 (the file).
- The cache folder now holds 1,165 pair folders under train/ and 58 under heldout/, far more than
  storage.md's "18 train + 58 heldout", so the 25 are named from that file, not from a listing
  (listed 2026-10-18; environment/storage.md).
- That bound, ‖r_t/σ_t‖²/4, did not track which pairs blend: every pair failing 8 of 8 seeds fails
  whether its bound is 100 or 3,000; it separates two bands from step 10, roughly 100 to 400 and
  1,300 to 3,200, by how far apart the two experts are (the report, rungs 1 and 2).
- The residual is w(ε_J − ε_A − ε_B + ε_∅) at w = 7.5, layer 8's term times the guidance weight
  (from step 7).
- The LoRA is rank-8 cross-attention on that residual; a checkpoint is lora_step_<N>.pt, and the
  repo renders one by lambda_boundary_probe.py: each held-out seed's pinned latent, 50 steps adding
  λ times the adapter's correction inside a window, then scored
  (runbook/running-things-on-the-cluster/probing-a-pooled-lora-checkpoint.md).
- A compose rate is the fraction of a pair's seeds where grounding-dino-tiny finds at least two
  distinct animal instances; it cannot tell a cat from a second dog, and the 94% headline at full
  correction is an upper bound on a true rate around 87% to 94% (context/world/compose-rate.md).
- Training seeds are 1 to 8 and 17 to 19, held-out seeds 9 to 16; 12 cells in 4 pairs sit in both
  split folders and resolve held-out first unless split="train" is passed
  (environment/known-failures.md, entry poe-data-001).
- MALA's accept step needs an energy, and the cached SDXL branches are noise predictions with no
  energy, so the walk states how "tuned MALA" runs here (an energy parameterisation, as Du et al.
  2023 use) or names the sampler it uses instead (REBUILD_WALK.md, brick "MALA and CHA fix" and its
  linked plan on the accept step; the when-poe idea map, Sources).
- inf_sample.py with anneal_samplers.py at this repo's root composes composable_diffusion CLEVR
  models (its captions name cube, sphere and cylinder); it does not load SDXL (inf_sample.py).
- Caveats to report beside the result: samplers recover a third to four fifths of the exact A∩B
  mass at 2% to 10% overlap (layer 5); trained experts cost more under a better sampler, +0.054
  total variation at r = 0.05 and +0.232 at r = 0.001 (the when-poe idea map, run 7); the
  interaction bound did not track which of the 25 pairs blend.
- GPU work runs on a card with under 1 GB in use; biggpu allows one Slurm job per user, so a second
  job goes by the shared-device path (environment/hpc/execution-protocol.md; the transfer plan,
  Environment Facts).
Outputs:
- walk12.py, printing one row per held-out pair: compose rate without and with the correction, and
  the interaction term (the mean bound over steps 10 to 40);
- a CSV of those rows beside walk12.py;
- the step file and walk map the --request open writes, with each piece ticked.
Target: every held-out pair in step 5's set has both compose rates and its interaction term, each
traced to a file, with the threshold step 5 wrote into the transfer plan applied; I judge.
Progress: the first piece prints one cell's four ε shapes and its timestep; the interaction piece
prints 25 rows; the first timed pair prints seconds per sample, which sets the GPU time; the last
prints the table.
Proof: python walk12.py prints the per-pair table and the count of held-out pairs it covers.
Limits: never edit poe_repair_min's code, plans or cache, or anything under
/datasets/mmolefe/poe_repair_min/; never launch on a busy card; I type, I do not paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: where you type, keeping poe_repair_min read-only (from the guide's one file per layer) · the [cache](../../../../../../Obsidian/Information%20Technology/System%20Design/Caching/Cache.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/system-design/computer-networking-a-top-down-approach.pdf#page=140 "Computer networking: a top-down approach, section 2.2.5, Web caching, A web cache, or proxy server, in front of an institution's 15 Mbps access link worked w..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/system-design/system-design-interview.pdf#page=15 "System design interview, chapter 1, Cache, Walks one web request through a cache tier (check the cache, on a miss query the database and store the answer), t..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/system-design/grokking-the-system-design-interview.pdf#page=15 "Grokking the system design interview, section 13, Caching, The three invalidation schemes (write-through, write-around and write-back), each with what it cos..."), [▶1](https://www.youtube.com/watch?v=dGAgxozNWFE&t=239s "ByteByteGo, Cache Systems Every Developer Should Know, Distributed and backend cache, 3:59 to 4:42, The shared cache that sits between services and the datab..."), [▶2](https://www.youtube.com/watch?v=RCdAGvtTYI0&t=97s "Hayk Simonyan, The Ultimate Guide to Caching and CDNs, Cache HIT, MISS, and Ratio, 1:37 to 2:09, Names a hit, a miss and the hit ratio that measures a cache...."), [◆](../../../../../../goal-setting/learning/system-design/foundations/plans/10-the-nginx-cache.md "The nginx cache: MISS, then HIT, system-design/foundations, plan 10. An nginx cache answers the corner shop's catalogue page by itself on the second request,...")</sub>'s path and fields, the 25 pairs' file and fields, the cache's current size against `storage.md`, the bound's null and its two bands (from poe_repair_min's report, figure explainer and the listing) · the residual as layer 8's term times w (from step 7) · the checkpoint and how the repo renders it (from the runbook) · the compose-rate rule and its bound, the seed split and the held-out-first failure (from poe_repair_min's context and known failures) · that MALA needs an energy the SDXL branches lack, and that `inf_sample.py` is CLEVR code (from `REBUILD_WALK.md`, the idea map and `inf_sample.py`) · the caveats with their numbers · the GPU guard
dropped: nothing

⏱ build (rough): you 1 to 2 h, AI ~35 min plus GPU [sampling](../../../../../../Obsidian/Mathematics/Statistics/Sampling%20From%20A%20Distribution.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=539 "Pattern recognition and machine learning, chapter 11, Why expectations need samples, ancestral sampling of a directed graph (page 541), standard distribution..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/statistics/introduction-to-probability-for-data-science.pdf#page=245 "Introduction to probability for data science, section 4.8, Generates random numbers for any PDF by running a uniform through the inverse CDF, in Python, with..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=606 "Deep learning textbook, section 17.1, Why machine learning samples at all (Monte Carlo estimates of sums and integrals, and sampling as the goal of a generat..."), [▶1](https://www.youtube.com/watch?v=9ixzzPQWuAY&t=474s "ritvikmath, Inverse Transform Sampling, 7:54 to 10:04, Inverts the exponential CDF and turns a uniform draw into an exponential one. Chapter-timed, from the ..."), [▶2](https://www.youtube.com/watch?v=OXDqjdVVePY&t=232s "ritvikmath, Accept-Reject Sampling, 3:52 to 7:45, An easy proposal g scaled by M into an envelope above f, then each draw accepted with probability f/(M g). ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/13-drawing-samples-by-hand.md "Drawing samples by hand: Monte Carlo, inverse transform and accept-reject, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 13. Thr...")</sub> time that the walk's first timed pair sets · stop: the build's bar

**Outcome:** the LoRA understood as a learned interaction term, measured on held-out animal pairs. `inf_sample.py` with `anneal_samplers.py` is the repo's image-scale [MCMC](../../../../../../Obsidian/Mathematics/Statistics/Markov%20Chain%20Monte%20Carlo/Markov%20Chain%20Monte%20Carlo.md#where-to-learn-this)<sub>[1](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/pattern-recognition-and-machine-learning.pdf#page=553 "Pattern recognition and machine learning, section 11.2, The Metropolis algorithm with a symmetric proposal, drawn as accepted and rejected steps on a 2D Gaus..."), [2](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/probabilistic-machine-learning-advanced-topics.pdf#page=526 "Probabilistic machine learning: advanced topics, sections 12.1 to 12.2, The Metropolis-Hastings algorithm (Algorithm 12.1), why it works, and the choice of p..."), [3](../../../../../../Obsidian/Mathematics/references/textbooks/machine-learning/deep-learning-textbook.pdf#page=611 "Deep learning textbook, section 17.3, MCMC for energy-based models: running many chains in parallel, the distribution q(t) over their states converging to th..."), [▶1](https://www.youtube.com/watch?v=KmqTrm-bn8k&t=141s "ritvikmath, Coding MCMC, 2:21 to 5:50, One chain written in Python, at its accept or reject step. Chapter-timed, from the video lessons index."), [▶2](https://www.youtube.com/watch?v=yApmR-c_hKU&t=247s "ritvikmath, Markov Chain Monte Carlo, 4:07 to 8:24, Drawing samples by running a Markov chain whose long-run distribution is the target. Chapter-timed, from ..."), [◆](../../../../../../goal-setting/learning/deep-learning/diffusion-models/00-from-scratch-to-sdxl/textbook/l3-stochastic-processes/03-metropolis-hastings-and-why-z-cancels.md "3. Metropolis-Hastings, and why Z cancels, deep-learning/diffusion-models/00-from-scratch-to-sdxl, volume L3, section 3. Why a product of densities can be sa...")</sub> composition code for this step.

### 10. You: compare with layer 9's correction

**Compare.** The layer 9 correction on the toys is the reference behaviour: the per-pair effect should move in the direction the toy's fair test predicts for the pair's interaction term.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.walk_numbers | grep 'p9\.'
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -c "import json, numpy as np; p = json.load(open('/datasets/mmolefe/poe_repair_min/outputs/showcase/interaction_bound/interaction_bound.json'))['pairs']; [print(k, v['fail_rate'], round(float(np.mean([v['median_bound_by_step'][str(s)] for s in range(10, 41)])))) for k, v in sorted(p.items())]"
cd ~/Playground/PhD/simple-distributions-walk
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python walk12.py
```

⏱ you ~15 min · stop: the `p9.` lines (they exist once layer 9's background prompt has built the reference) say how composition plus a correction did against composition alone at each overlap; the 25 bound lines fall in the two bands, roughly 100 to 400 and 1,300 to 3,200; your table's interaction column matches them pair for pair. Then read the direction: write down, before looking at your table, which way the `p9.` lines say a pair's compose rate should move as its interaction term grows, and check each held-out pair against it. A pair that moves the other way is the finding to write down.

**Caveats carried from earlier layers:**

- Samplers recover only a third to four fifths of the exact mass at 2% to 10% overlap (layer 5).
- Trained experts plus a better sampler cost more at low overlap (layer 9).
- The interaction term's lower bound did not track which of the 25 pairs blend in `poe_repair_min`.

### 11. You: close the layer

Run the `/drip-retro` command the build printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 12 of the composition ladder wrote:
the layer 12 --math drip folder under artifacts/drips/, claim 8 of
artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, its HYPOTHESIS.md if step 4 ran, and the
retrospective the layer 12 walk closed on.

Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference sets to link against are the two journey plans on the Study first line of
  artifacts/drips/simple-distributions-rebuild/ladder/12-the-correction-at-scale.md (the LoRA
  update and the SVD of the real matrix), the "The correction at scale" row of the Bricks table in
  artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, claims 6 and 8 of
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md with its route 6 return, the drain-gap plan
  ~/Playground/PhD/medical-imaging/compositional_diffusion/b-shortcut-through-a-device/b1-drain-shortcut-in-pneumothorax/plans/01-measure-the-drain-gap.md
  and the transfer plan
  ~/Playground/PhD/poe_repair_min/plans/04-does-the-fix-reach-unseen-pairs/plans/hypothesis/04-transfer-as-a-rate-over-fifteen-pairs.md.
- The drain-gap plan and the transfer plan belong to their own projects; link toward them and write
  nothing inside either.
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between this repo's files and the reference sets inside it, and
one-way links toward the two outside plans.
Hand-off: the agent's report, listing every link it wrote. This is the ladder's last layer.
```

added: the reference sets, by path (from this layer's Study first line, `REBUILD_WALK.md`, the idea map and the two plans this layer walks)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 11: the testbed](11-the-text-conditioned-testbed.md) · 🏠 [the guide](../PROMPTS.md)
