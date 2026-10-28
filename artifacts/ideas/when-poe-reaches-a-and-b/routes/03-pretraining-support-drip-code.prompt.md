/drip --code the pretraining-support check on the four-quadrant toy: whether two diffusion experts fine-tuned from one pretrained model still compose into A-and-B, built as a Jupyter notebook one cell per message

Repo: /home-mscluster/mmolefe/PhD/reduce-reuse-recycle. This walk comes from the idea map at artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 2, child 2a. The full experiment is specified step by step in artifacts/ideas/when-poe-reaches-a-and-b/routes/02-pretraining-support-notebook.prompt.md: read it first and use its eight steps as the ladder's top level. The exact numbers every cell should reproduce are in artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md and answer_key.json.

How I want it delivered, overriding the mode's plain-.py default:
- The file I type is a notebook, artifacts/ideas/when-poe-reaches-a-and-b/pretraining-support-on-four-quadrants.ipynb. Each message gives me one cell to type: a markdown cell (with LaTeX where a formula is claimed) or a code cell, never both in one message. I copy it, run it, and can ask questions before `next`.
- Setup: machine mscluster, the existing conda environment `co3` (`co3_bw` on the Blackwell GPU nodes). Install nothing into it; if a package is missing, say so and stop.
- PyTorch, NumPy and matplotlib only, no JAX. Reuse the energy parameterisation from notebooks/simple_distributions.ipynb (`EBMDiffusionModel`): the network outputs a scalar energy, its score is −∇E.
- Every figure is labelled on the thing: each quadrant named with its A/B status and its medical name, the mass or share written on the region it belongs to, curves named on the plot. The last rung ends in an interactive Plotly view: the product at a slider for fine-tune steps N, with the top-right blob share beside it.
- Measure shares within 3 std of each blob's centre and near the origin, never by quadrant, because two far-apart experts multiply into a blend at the origin that a quadrant count misreads as `0.25`.
- Keep the walk's words: A is `x1 > 0` (pneumothorax), B is `x2 > 0` (no chest drain), top-right is A and B (pneumothorax without a drain, the rare film), "A, not B" is bottom-right, "B, not A" is top-left, pretraining is training on all four, fine-tuning is continuing on one, the product is exp(−(E_A + E_B)), "support" is where a density is not zero.
- Before each number appears, ask me to predict it (`--quiz`), then show the answer key's value beside the measured one.

The rungs must reach these numbers: the pretrained pair gives `0.25` in every blob; evenly spread leftover ε gives a top-right share of ε/(2 − 4ε) (`0.0625` at ε = 0.1); leftover only next door gives `0.50` top-right and `0.50` bottom-left; experts trained from scratch put `0.989` at the origin; construction (1) with "contains" labels puts `0.989` in the top-right blob for any data share r from `0.25` down to `0.001`, and falls to the origin at r = 0.

Return. When the walk closes, write a fenced return package to artifacts/ideas/when-poe-reaches-a-and-b/routes/03-pretraining-support-drip-code.return.md:
  origin: artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md · claim 2, child 2a · route 3
  ask: the pretraining-support check, built by hand as a notebook
  deliverable: the notebook path, the measured top-right blob share at N = 0, 300 and 10000 (mean ± sd over 3 seeds), the fitted leftover ε at each N, construction (1)'s share at r = 0.05 and 0
  changes for the walk: one line on whether trained fine-tuned experts behave like the even-leftover case, the next-door case, or neither
Then I paste it into the idea walk and say `integrate`.
