/drip --code --quiz rebuild, by hand and one notebook cell at a time, the four experiments behind my idea walk on combining diffusion models, so that I understand every number it produced

I got lost in the walk because too many terms and numbers arrived at once. I want to rebuild the evidence myself, slowly, so each number lands before the next one appears.

Repo: /home-mscluster/mmolefe/PhD/reduce-reuse-recycle. The walk's map is artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md; read its "The idea, as it stands" paragraph and the Runs table first. Each run has a folder under artifacts/ideas/when-poe-reaches-a-and-b/runs/ with a README.md, the script that produced it and its printed output. Those outputs are the answer key: every number I produce must match them.

How I want it delivered:
- One notebook, artifacts/ideas/when-poe-reaches-a-and-b/rebuild-the-walks-numbers.ipynb, written by me one cell per message: a markdown cell in plain words (and LaTeX where there is a formula), or a code cell, never both in one message.
- Before any number appears, ask me to predict it, then show my result beside the answer key's.
- Every new word gets one plain sentence the first time it appears, and a "Words so far" cell grows as we go. No word is used before it is explained.
- Every picture is labelled on the thing itself, with the number written where it belongs.
- Conda environment `co3` on the cluster; NumPy, PyTorch and matplotlib; install nothing.

The four experiments, in this order (each one is a top-level rung):
1. **Four blobs and what "combining" does.** Four Gaussian blobs at (±0.5, ±0.5), std 0.15. A is "right half" (pneumothorax), B is "top half" (no chest drain), so top-right is the rare film I want. Multiply two experts' densities on a grid and see where the result sits: from scratch (it sits at the origin, `0.989` there), after pretraining and fine-tuning (top-right share ε/(2 − 4ε), `0.25` at the start, `0.5` at most), and with one model trained on all the data with "contains" labels (`0.989` in top-right even when only 1 in 1,000 training points is top-right). Answer key: runs/answer-key-quadrants/.
2. **Does it matter which space you combine in.** Warp the plane with a fixed invertible map and combine before and after it: one recipe is unchanged, the plain product shifts by `0.035` to `0.126`. Answer key: runs/compose-before-or-after-a-warp/.
3. **Samplers with perfect experts.** Draw samples with reverse diffusion and with annealed MALA, ULA and UHA, and measure how far each lands from the exact answer (total variation, with `0.133` as the floor that perfect samples score by chance). Reverse diffusion misses by `0.049`; the others sit at the floor. Answer key: runs/samplers-on-exact-experts/.
4. **Samplers with trained experts.** Train small experts, repeat step 3, and see the better sampler make the trained experts' mistakes cost more (`+0.054` at a 1-in-20 rare cell, `+0.232` at 1 in 1,000). Answer key: runs/sampler-by-expert-interaction/.

The walk ends when every number in those four README files has been reproduced and I can say in one sentence what each one means. Then write a fenced return package to artifacts/ideas/when-poe-reaches-a-and-b/routes/09-rebuild-the-walks-numbers-by-hand.return.md:
  origin: artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md · runs 1, 5, 6, 7 · route 9
  ask: rebuild the walk's numbers by hand
  deliverable: the notebook path, and any number that did not match its answer key
  changes for the walk: the sentences I wrote for each number, which become the plain-words version of the map
Then I paste it into the idea walk and say `integrate`.
