# Route 3 runs: the PyTorch rebuild and the overlap dial

These are the figures and numbers from the PyTorch rebuild of `notebooks/simple_distributions.ipynb`, which the [route 3 return](../../routes/03-unbuild-walk.return.md) reports to the idea walk. Each file named below lives in `rebuild/out/` at the repo root, in the folder for its phase (`rebuild/out/phase1/` to `rebuild/out/phase5/`). It is produced by that phase's script, `python -m rebuild.phase<N>_...`, run from the repo root.

## Phase 1: the trained models against the datasets

**`datasets-against-trained-models.png`**

Ten panels laid out like `datasets-1000-samples-each.png`. Rows 1 and 3 are 1000 samples from each dataset, and rows 2 and 4 are 1000 reverse-diffusion samples from the PyTorch energy model trained on it (15,001 steps, the notebook's settings). Each model panel's title gives its TV to the data against the TV between two data draws. Made by `python -m rebuild.phase1_check`. Numbers in `phase1.json`.

## Phase 2: product, mixture and negation of trained models

**`product-mixture-negation-every-sampler.png`**

One row per composition (ring × bar, left + right, box(0.7) not box(0.3)). The columns are the two training sets, the trained experts' composition at t = 0 on the grid, then 2000 samples each from the perfect sampler, reverse diffusion, ULA, MALA at the notebook's step 0.001, tuned MALA, UHA and the notebook's Metropolis-adjusted UHA. Made by `python -m rebuild.phase2_compose`, with numbers in `phase2.json`.

**`mala-acceptance-per-sample.png`**

A histogram of each sample's acceptance over the last 10 levels (100 moves), for MALA at step 0.001 and for tuned MALA. It is the evidence that acceptance is now decided per sample.

## Phase 3: the seven kinds, rerun from the rebuilt modules

**`figure1-every-sampler-on-every-kind.png`**

The kinds notebook's Figure 1, redrawn from `rebuild/phase3_kinds.py`. `phase3.json` holds every number beside the notebook's printed one.

## Phase 4: the parts world

**`parts-world-samplers-against-exact.png`**

For the mosaic, the blend and co-occurrence, it plots each sample's two half-means (head and body, or left and right) for 100,000 exact draws and for 2000 samples from each sampler. Made by `python -m rebuild.phase4_parts_world`, with numbers in `phase4.json`. `numbers.json` is example 9's file recomputed.

## Phase 5: the overlap dial

**`overlap-dial-curves.png`**

Three panels. The first two are the quadrant toy: A∩B mass on y (at least 0.15 inside the top-left quadrant in the first panel, the whole quadrant in the second) against the share of A-and-B points in each training set on x. They show the exact composition, tuned MALA, UHA and reverse diffusion, each the mean of 3 data seeds with bars at 1 sd. The third is the parts world: the share of samples with a cat half and a dog half on y, against the share of cat-and-dog images on x, for an empty background (variance 0.01, solid) and a loose one (variance 1.0, dashed). Made by `python -m rebuild.phase5_dial`, with numbers in `overlap-dial.json`.

**`overlap-dial-quadrant-pictures.png`**

The quadrant toy at 0%, 2%, 20% and 100% overlap, seed 0. Each row shows the two training sets, the exact composition, then tuned MALA, UHA and reverse diffusion.

**`finetune-curves.png` and `finetune-pictures.png`**

One expert pretrained on all four quadrants, then two copies fine-tuned at 0% overlap (A on the bottom-left square, B on the top-right). The curves plot A∩B mass on y against fine-tune steps on x, with experts trained from scratch dotted. The pictures show the exact composition and both samplers at steps 0, 300 and 10,000 and from scratch. Made by `python -m rebuild.phase5_finetune`, with numbers in `finetune.json`.
