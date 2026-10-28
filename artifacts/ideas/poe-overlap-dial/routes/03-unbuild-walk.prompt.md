/unbuild --walk ~/PhD/reduce-reuse-recycle/notebooks/simple_distributions.ipynb

Rebuild the JAX/Haiku reference notebook in PyTorch, layer by layer, then extend it past the notebook into the composition testbed this project's idea walk settled on. Python, NumPy and PyTorch only, no JAX. Run Python with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python. GPU work follows the GPU guard in /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/environment/hpc/execution-protocol.md (use a card with under 1 GB in use, else CPU).

Context (cold start). Read first, in this order:
1. artifacts/ideas/poe-overlap-dial/IDEA_MAP.md: nine claims with marks; the Runs table holds every result so far.
2. artifacts/ideas/poe-overlap-dial/composition-terms.md: the kinds of composition (co-occurrence, intersection, interaction, contention, plus the [field] kinds and route 1's proposed changes).
3. artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb: the 2D suite (Figure 1: p1, p2, exact, perfect sampler, reverse diffusion, annealed ULA, MALA, UHA per kind; Figure 2: dial curves; runs 1 and 2: tuned MALA and the steep points).
4. artifacts/ideas/poe-overlap-dial/example-9-parts-world/ (the .tex, parts_world.py, numbers.json): the 16-pixel Gaussian-mixture parts world where cat x dog products are exact.

Settled facts the rebuild must keep:
- A density expert learns its training set's density; labels matter only where something reads them (claim 1).
- Ideal experts (training points smoothed by N(0, σ²I)) give exact noised densities at every step; trained 2D MLPs match them within 0.04 nats (run 2), so 2D toys use ideal experts as ground truth.
- Reverse diffusion with summed scores misses the product on every kind except disjunction; annealed UHA and tuned MALA (about 0.6 acceptance) reach the perfect-sampler floor on most kinds; dominance (unequal mode weights) defeats every sampler (runs 4 to 6).
- anneal_samplers.py: ULA and UHA are reused unchanged. Its MALA (and CHA) has two bugs: the proposal log-density divides by 2*ss**2 where the proposal variance is 2*ss (the divisor should be 4*ss), and it accepts or rejects the whole batch with one random number. inf_sample.py uses that MALA for the image pipeline, so fix it in a way that keeps inf_sample.py working, with a test that shows per-sample acceptance.

How every phase is written: as markdown the learner walks with /drip-code or /drip --code, step by step with no step skipped: the mathematical definition, its derivation, a worked example with real numbers, then the code in small steps using NumPy, matplotlib, scikit-learn, seaborn and PyTorch.

Phases, each its own layer with a check before the next. Each phase points the learner at the matching scene, textbook section and journey plan in /home-mscluster/mmolefe/goal-setting/learning/deep-learning/diffusion-models/ (01-poe-derivation-foundations, 02-sampler-correctors-for-composition, 05-poe-composition-diffusion, 06-diffusion-speciation-dynamics) and the Obsidian notes Diffusion Models.md and Multivariate Normal Distribution.md. Use /drip --code for code layers, /drip-code where the learner types the code, /drip --math for derivations.
1. The notebook's datasets, the ResNet score model, the EBM parameterisation, the cosine schedule, training: in PyTorch, reproducing the notebook's 2D toys. Check: the trained model's samples match the dataset plot datasets-1000-samples-each.png.
2. Product, mixture and negation wrappers and the annealed samplers, reusing anneal_samplers.py with the MALA fix. Check: the paper's ring-times-bar product and left-plus-right mixture reproduced, and MALA acceptance reported per sample.
3. Ideal experts and exact targets: the 2D suite of seven kinds with one dial each, as in the kinds notebook. Check: the kinds notebook's Figure 1 and Figure 2 numbers reproduced from the rebuilt code.
4. The parts world: Gaussian-mixture experts on 16 pixels, exact products, mosaic, blend and co-occurrence, then the samplers on it against the exact product (component-weight error and TV). Check: numbers.json reproduced; each sampler's error per setting reported against exact draws.
5. The overlap dial: the share of A-and-B examples in each expert's training set from 0% to 100%, on the co-occurrence toy and on the parts world, with a pretrained-then-fine-tuned variant (pretrain on all placements, fine-tune at 0% overlap, product mass in A∩B against fine-tune steps). Check: the curve of A∩B mass (and TV to exact) against overlap, per sampler.

Finish line: phase 5's curves exist for the exact composition and for tuned MALA and UHA, and each phase's check passed by hand.

Return: write artifacts/ideas/poe-overlap-dial/routes/03-unbuild-walk.return.md with (1) where the PyTorch code lives, (2) each phase's check and whether it passed, (3) the MALA fix and its test, (4) phase 5's curves with their numbers. Then say `integrate` in the idea walk. Parent: the drip-idea walk in session 0481b097-3bd5-4858-85f2-333cedbbe5b4, map artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, routes row 3.
