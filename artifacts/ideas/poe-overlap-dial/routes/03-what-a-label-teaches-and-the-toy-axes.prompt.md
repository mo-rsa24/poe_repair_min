/drip-idea Add a claim to the overlap-dial walk (artifacts/ideas/poe-overlap-dial/IDEA_MAP.md) and walk it.

Claim 8 (proposed): what a product of experts produces is decided by what each expert's label meant during training, not by the operator. If "cat" meant "contains a cat" (images with a cat and a dog count as cat images), p(x|cat) already holds cat-and-dog scenes and p(x|cat)·p(x|dog) lands on co-occurrence. If "cat" meant "only a cat" (single-label classifier-free guidance, or a separate model trained on cat-only images), the two densities share no support where both appear, and the product returns a compromise: the fused cat-dog seen in every Stable Diffusion grid.

Why it is proposed now: on SD3.5 the PoE fails the same way in every interaction grid (fusion or a dropped component; artifacts/results/joint-against-poe-by-kind-of-composition/interaction-grids/), so a large model cannot separate the interactions. The joint-only atlas in the same folder shows what each interaction should look like.

Evidence already in hand: the worked-examples notebook (what-an-expert-learns-and-kinds-of-composition.ipynb) puts 1.00 of the composed mass in A∩B under "contains A" and 0.08 under "exactly A" (claim 1.2). Bradley 2025b (arXiv 2509.16447, Fig. 7, p.22) shows the same on CLEVR: a model whose label lists every object's location length-generalizes; one whose label marks a single random object does not.

Axes to vary one at a time on the 2D toy (then a small image toy such as CLEVR or coloured MNIST):
1. labelling scheme: contains A / exactly A
2. one conditional model with labels {A, B, none} / two separately trained models
3. overlap dial: share of A-and-B samples in each training set (0, 5, 20, 50, 100%)
4. kind: separate regions (product over a partition) / same variables (product over the same variables)
Sampler held fixed and tested separately against exact analytic scores (claim 4), so sampler error and score error stay apart, as in Soiffer et al. 2026 (arXiv 2606.23920, Table 1).

Prediction to write before any run: co-occurrence appears only under "contains" labels or a partition product; fusion appears under "exactly" labels with a same-variable product; the overlap dial moves the "exactly" case toward co-occurrence as it rises.

Return line: write artifacts/ideas/poe-overlap-dial/routes/03-what-a-label-teaches-and-the-toy-axes.return.md with the claim's mark, the repaired wording, and the first check to run.
