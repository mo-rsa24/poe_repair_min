# Layer 5, step 0: build layer 5's quizzes

⬅️ [Layer 5, step 0](../05-the-overlap-dial.md#0-ai-in-the-background-quizzes-51-and-52)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`
when you start layer 4's build.

```text
In the background, prepare and build layer 5's two quizzes for the composition ladder, bottom up:
quiz 5.1 and quiz 5.2.

Background
The composition ladder in artifacts/drips/simple-distributions-rebuild/ walks from one 2D diffusion
expert to a correction for product-of-experts composition. Layer 5, the overlap dial, answers how
much A-and-B data a product needs, and whether pretraining supplies it. Each layer opens on two
quizzes, one on the college maths underneath it and one on the layer itself, built during the
previous layer's build. Every path below is from the repo root,
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain, except code paths, which are
under toy/.

Objectives
1. For each quiz, dispatch prep-cadence (quiz-prep-pulse --build) with the goal "say how much
   A-and-B data a product needs, with the numbers" and the minutes and topics given below.
2. For each quiz, once its coverage file exists, dispatch quiz-cadence in build mode on the repo
   above.
3. Quiz 5.1, 40 minutes, covers integrating a density over a region, the mass share of a region,
   and ratios of normalising constants.
4. Quiz 5.2, 60 minutes, covers the plan 05-the-overlap-dial.md: why 5% overlap puts 0.30 of the
   product's mass deep in A∩B, the three answer-key numbers, what fine-tuning does to inherited
   support (at gloss depth), and how Z moves with expert B's blob.
5. Write each published URL into its ladder step.

What to read
- The miss ledger (quiz-ledger) of every earlier quiz on this ladder.
- artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md and its Study first
  line: the material for quiz 5.1.
- artifacts/drips/simple-distributions-rebuild/05-the-overlap-dial.md, steps 0 to 8, with step 4 of
  the ladder file (the answer key) and its optional Z step: the material for quiz 5.2.
- The diffusion models quiz named under What is already known, before writing.
- ~/Obsidian/Mathematics/references/textbooks/Textbooks.md: the shelf index.
- ~/Obsidian/Mathematics/references/exercises/Worked Examples.md: the worked-examples index.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- quiz-cadence stops when a quiz has no coverage file in
  artifacts/drips/simple-distributions-rebuild/.quiz-prep/, so prep-cadence runs first for each
  quiz (quiz-cadence, "The preparation gate").
- The step file links a textbook page beside each fundamental it uses: p. 230 of
  introduction-to-probability-for-data-science.pdf for the normal CDF and p. 42 of
  matrix-cookbook.pdf for the constant of a product of Gaussians (step file 05, steps 1, 2, 5 and
  6).
- The step file carries three tables of timestamped video segments (the kernel density estimate,
  annealed Langevin, the product of two mixtures), which are ready nodes to reuse before searching
  again (step file 05, steps 1, 2, 5 and 6).
- pdftotext is not installed on mscluster, so
  ~/.claude/skills/index-source/scripts/find-in-textbooks.py fails (checked on mscluster109,
  2026-10-18).
- pypdf does not import in any environment there either; open the PDFs with PyMuPDF (import
  pymupdf, version 1.28.2) from /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python (checked on
  mscluster109, 2026-10-18).
- The shelf index and the worked-examples index are the two files under What to read
  (quiz-prep-pulse, "The bank").
- Two toys carry this layer, in two notations (step file 05, steps 1 and 3; answer-key-quadrants
  README, in artifacts/ideas/when-poe-reaches-a-and-b/runs/).
- The overlap dial's notation: A is the left half, B the top half, A∩B the top-left quadrant, each
  expert 800 points smoothed by σ = 0.05, a share f of them in A∩B, and "deep" means x1 < −0.15 and
  x2 > 0.15 (step file 05, steps 1 and 3).
- The answer key's notation: one Gaussian blob of std 0.15 per quadrant at (±0.5, ±0.5), A is
  x1 > 0, B is x2 > 0, A∩B the top-right quadrant, shares counted within 3 std of a blob's centre
  (answer-key-quadrants README, in artifacts/ideas/when-poe-reaches-a-and-b/runs/).
- Overlap-dial numbers over 3 data seeds: deep A∩B mass 0.00 at 0% (rounded: it is below 1.1e-6),
  0.30 ± 0.10 at 5% (about 40 shared points per set on average, since each point is shared with
  probability f), 0.59 at 20% and 0.76 at 100% (python -m rebuild.phase5_tables; step file 05,
  steps 2 to 5).
- At 0% the whole quadrant reads 0.20 ± 0.02, all of it the sliver around the corner (python -m
  rebuild.phase5_tables; step file 05, steps 2 to 5).
- Tuned MALA reaches 0.17 at 5% (python -m rebuild.phase5_tables; step file 05, step 5).
- One conditional model with "contains" labels keeps 0.989 in A∩B for every A∩B share r from 0.25
  down to 0.001 (answer_key.json; when-poe-reaches-a-and-b idea map, run 4; ladder file 05, step
  4).
- Separately fine-tuned experts whose leftover sits only on the two quadrants next to their own
  hold 0.5, fixed by that construction's x → −x symmetry, and this is not a ceiling for every pair
  of separately fine-tuned experts (answer_key.json; the three-ways scene's README; ladder file 05,
  step 4).
- Experts over separate parts reach 0.9999 with no A∩B data (answer_key.json; ladder file 05, step
  4).
- 0.989 is 1 − e^(−9/2), the mass of one blob within 3 std, and 0.9999 is 1 − e^(−9), the same
  radius around a blob squared to variance σ²/2 (p5.mass_within_3sd_2d and
  p5.mass_within_3sd_2d_half_variance; ladder file 05, step 4).
- Fine-tuning: pretrained support falls from 0.159 deep mass to 0.001 within 30 fine-tune steps on
  the exact grid (step file 05, step 7; toy/rebuild/out/phase5/finetune.json).
- Tuned MALA and UHA still read 0.056 and 0.068 at step 30, 0.004 and 0.0055 at step 100, and at
  most 0.001 from step 300, and experts trained from scratch give 0.000 (step file 05, step 7;
  toy/rebuild/out/phase5/finetune.json).
- Z with expert A's blob in the bottom-right corner and B's blob moved is 7.9 × 10⁻¹⁰ opposite,
  5.3 × 10⁻⁵ next door and 3.54 on top of A (scene_key.json in
  artifacts/scenes/three-ways-a-product-reaches-a-and-b/, keys support.TL.Z, support.TR.Z and
  support.BR.Z).
- By hand, two normalised 2D Gaussians of std σ with centres d apart multiply to a total mass
  Z = exp(−d²/(4σ²)) / (4πσ²), which with σ = 0.15 and d² = 2, 1, 0 gives the same three values
  (p5.Z_prefactor_sd0.15, p5.Z_next_door_by_hand and p5.Z_opposite_by_hand; ladder file 05, step
  5b).
- The diffusion models quiz, https://claude.ai/artifact/MyTdAaVfZW3wiqhbLKg4wu, already holds
  sections "When neither training set holds A and B" and "The four-quadrant answer key", 24
  questions on the answer key and the three rescues, so the new pages extend them
  (when-poe-reaches-a-and-b idea map, run 3).
- The bar a learner meets is every question marked needed now right and at least 80% of the rest,
  and a missed question returns in the next quiz (PROMPTS.md, "The bar for every quiz").

Rules
- Change nothing in the step file's prose, in toy/rebuild/ or in the answer-key and scene folders.
- In the ladder file, write only the two URLs.
- Read the diffusion models quiz's two sections before writing a question.
- Report each quiz's question count, node count and verdicts from prep-cadence before its page is
  built.

What to hand back
- Two coverage files under artifacts/drips/simple-distributions-rebuild/.quiz-prep/, each giving
  every question a verdict: ready, blocked, or unprepared by choice.
- A row in the worked-examples index for every banked problem.
- Two published quiz pages, each opening on a read-first list with minutes per item, a short path
  and a full path.
- Quiz 5.1's URL in step 1 and quiz 5.2's URL in step 3 of
  artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md, each replacing "(URL
  lands here)".
- The proof: the two coverage files exist under .quiz-prep/ and list no question as blocked, and
  the two URLs open.

What you will see
Along the way: per quiz, prep-cadence's question count, node count and verdicts, then each URL as
quiz-cadence publishes it.
At the end, a closing report: what was asked, in one sentence; what was done, with each coverage
file's path and each URL; PASS or MISS against no question blocked, about 6 to 10 banked problems
per concept on quiz 5.2 for its two build-depth concepts (why the product weights shared places
most, measuring A∩B in the right region) and 4 to 6 for the own-depth Z; what is still open; at
most one question for me.

I judge, from the two pages: done when no question is blocked and every number in an answer traces to phase5_tables, answer_key.json, scene_key.json, the step file or a cited textbook page.
```

added: the seven sections, from the field-labelled version · a report of coverage counts before each page is built (from the old Progress line) · reading the existing quiz's two sections as a rule (from the settled fact that said "read them before writing")
dropped: nothing (the earlier version's corrections from the Codex mathematics review stay visible: the 40 shared points as an average, the 0% deep mass as rounded, and the 0.5 as a property of the adjacent-leftover construction rather than a ceiling)
