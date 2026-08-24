# F3: the two-stage normalization behind size follows noise

Captured from `/drip --math` on the F3 explanation (`paper/iclr/figures/correction-size-over-the-denoising-run-across-17-pairs.png`, built by `scripts/correction_size_vs_run_position.py`), ongoing.

Each rung carries its chain status: taught (in chat by /drip), filed (captured here), scene built (by /math-scene). No mark means not taught yet.

## The six-piece ladder
1. The two axes (taught): the real measure is relative correction size, $\lVert r_t \rVert / \lVert \varepsilon_{\mathrm{PoE}} \rVert$ per denoising step, then each pair's own curve divides again by its own median across steps before pooling into the population bands.
2. Not taught yet. Titles for pieces 2 through 6 are not known to this capture. Run `/drip --math` to continue the ladder, then `/polish` each piece as it lands.
3. Not taught yet.
4. Not taught yet.
5. Not taught yet.
6. Not taught yet.

## Side documents
Documents that support the ladder without being one of its six numbered pieces.

- [A worked example: the two-stage normalization behind F3](worked-example-toy-numbers.md) (filed): four invented toy pairs, three toy steps, the norm-ratio-median-normalize arithmetic run by hand, tied back to piece 1's rule.
