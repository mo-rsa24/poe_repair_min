I want a polished teaching notebook that runs one small experiment and explains it to me step by step, with a labelled picture at every step and a maths markdown cell wherever a number is claimed. I learn by seeing, so each step is: one short markdown cell saying what happens and why, one code cell, one labelled figure, one sentence reading the figure.

## Where this comes from

Repo: /home-mscluster/mmolefe/PhD/reduce-reuse-recycle. I am walking an idea in artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md: when does a product of two diffusion experts (composition by adding their scores) generate samples that are both A and B? The medical reason: a pneumothorax classifier leans on chest drains, and the films I want to generate are pneumothorax without a drain, which are rare in NIH data.

What the walk has settled so far:
- An expert trained on A's data learns the density of its training set; when experts are composed separately (score_A + score_B), the label never enters the loss.
- Adding scores multiplies densities, p_A(x)·p_B(x). Anything times zero is zero, so the product can only put samples where both experts already give probability. If no training point is both A and B, the product has (almost) nothing there. A vault sandbox on the notebook's ring × bar toy measured it: with the overlap, `0.996` of the product sits in A∩B; without it, the normaliser falls from `0.313` to `1.6e-07`, and blurring the experts by `h = 0.1` brings back 31% of the mass but only `0.005` of it in A∩B.
- So composition can only produce the missing combination if something gives each expert probability there. Rescue (a), tested here, is support kept from pretraining: both experts are fine-tuned from one model that saw every combination.

## The words, defined in the notebook's first markdown cell

- **A** is the property `x1 > 0` (right half). **B** is the property `x2 > 0` (top half).
- **A∩B**, "A and B", is the top-right quadrant. "**A, not B**" is bottom-right: right half, bottom half. "B, not A" is top-left. "Neither" is bottom-left.
- The medical reading, as a labelled table: A = pneumothorax, B = no chest drain. Then top-right = pneumothorax without a drain (the rare films I want), bottom-right = pneumothorax with a drain (most positive films), top-left = no pneumothorax and no drain (most negative films), bottom-left = drain without pneumothorax.
- **Expert**: a small diffusion model with an energy parameterisation (it outputs a scalar energy E(x, t); its score is −∇E; at the lowest noise level its density is proportional to exp(−E)).
- **Pretraining**: training one model on all four quadrants. **Fine-tuning**: continuing to train a copy on one quadrant only.
- **Product**: the composed density proportional to exp(−(E_A + E_B)), read on a grid, so no sampler is involved and sampler error cannot enter.

## The steps

1. **The four quadrants.** One Gaussian blob per quadrant at `(±0.5, ±0.5)`, std `0.15`, equal weights. Maths cell: p(x) = ¼ Σ_q N(x; μ_q, σ²I). Figure: 2,000 samples, each quadrant shaded and labelled with its A/B membership and its medical name.
2. **Pretrain one expert on all four.** PyTorch only (no JAX). An MLP diffusion model like the notebook's `ResnetDiffusionModel` (4 layers, 128 hidden, time embedding), with the energy parameterisation from `notebooks/simple_distributions.ipynb` (`EBMDiffusionModel`), cosine schedule, 100 noise levels, denoising score matching. Figure: true density beside the pretrained expert's exp(−E) on a `241 × 241` grid over `[−1.1, 1.1]²`, with the mass in each quadrant written on that quadrant.
3. **Fine-tune two copies.** Copy the pretrained weights twice. Fine-tune expert A on bottom-right samples only ("A, not B") and expert B on top-left samples only ("B, not A"), for N = `0, 100, 300, 1000, 3000, 10000` steps, saving a snapshot at each N. Figure: two rows (expert A, expert B), one column per N, each panel with its four quadrant masses written on it, so I can watch the leftover probability drain out of the other quadrants.
4. **Multiply them.** At each N, the product exp(−(E_A + E_B)) on the grid, normalised. Measure shares within 3 std of each blob centre and within `3·0.15/√2` of the origin, never by quadrant: two far-apart experts multiply into a blend at the origin, which a quadrant count would split evenly and misread as `0.25`. Figure: one panel per N with each blob share and the origin share written on it, the top-right blob outlined.
5. **Why the starting share is 0.25: maths cell.** At N = 0 both experts are the same four-blob density p, so the product is p², normalised. The blobs sit `1.0` apart with std `0.15` (6.7 std), so cross terms vanish and p² is four equal squared blobs: each quadrant gets exactly ¼. Then the general case: if each fine-tuned expert keeps `1 − 3ε` on its own quadrant and ε on each other one, the quadrant masses of the product are ε(1−3ε) in bottom-right and top-left, ε² in top-right and bottom-left, so the top-right share is ε/(2 − 4ε). At ε = ¼ this gives ¼, and for small ε it is about ε/2. If instead the leftover sits only on the two quadrants next to each expert's own, top-right ties with bottom-left and the share approaches ½. Say in plain words why: an expert fine-tuned on "A, not B" also learns "not B", so the product of "A, not B" and "B, not A" asks for something contradictory, and the corner where both are weak wins.
6. **The measured curve.** Top-right blob share against fine-tune steps N (log x-axis), 3 seeds as mean ± sd. Overlay the ε/(2 − 4ε) prediction computed from each expert's measured leftover ε at that N. Draw the from-scratch baseline (two experts trained only on their fine-tune quadrant) as a horizontal line. Label the curves on the plot, not in a legend.
7. **The same quadrants under construction (1).** One conditional energy model trained on all data with "contains" labels (a point carries A if `x1 > 0`, B if `x2 > 0`, and the null condition is dropped 10% of the time), composed as E(x|A) + E(x|B) − E(x). Maths cell: this targets p(x)·p(A|x)·p(B|x) up to a constant, so the base density p(x) supplies the support and the label ratios pick the quadrant. Dial the top-right share of the training data r in `{0.25, 0.05, 0}`. Expected: near all of the product in top-right while r > 0, and nothing when r = 0. Figure: one product panel per r with blob and origin shares written on it. Expected from the answer key: `0.989` in the top-right blob for every r from `0.25` down to `0.001`, and the origin blend (`0.970`) at r = 0.
8. **What this says.** A short markdown cell: the measured numbers, which of the step 5 cases they match, and what that means for rescue (a) and for the chest X-ray plan.

Every number in a markdown cell is printed by a code cell above it, and the maths cells say where each formula's inputs come from. Cache trained weights so re-running the notebook does not retrain.

## Where it runs and lands

- Conda environment `co3` (`co3_bw` on the Blackwell GPU nodes). One GPU is plenty; CPU also works at this size.
- Notebook: artifacts/ideas/when-poe-reaches-a-and-b/pretraining-support-on-four-quadrants.ipynb, executed top to bottom with outputs saved.
- Figures also saved as PNGs to artifacts/results/does-pretraining-support-reach-a-and-b/, with a README.md card saying what each figure shows and how it was made, and a results.json whose every row carries N, seed, construction, r and the quadrant shares.

## Return

Close with a fenced return package, and also write it to artifacts/ideas/when-poe-reaches-a-and-b/routes/02-pretraining-support-notebook.return.md:
  origin: artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md · claim 2, child 2a · route 2
  ask: the pretraining-support check as a labelled teaching notebook
  deliverable: the notebook path, the results folder, the top-right share at N = 0, 300 and 10000 (mean ± sd), and construction (1)'s share at r = 0.05 and r = 0
  changes for the walk: one line on whether pretraining support alone can choose A∩B
Then I paste it into the walk and say `integrate`.
