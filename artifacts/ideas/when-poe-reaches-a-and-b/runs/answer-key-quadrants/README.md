# Answer key: where the composed density sits on the four-quadrant toy

**What it is**

Exact shares of the composed density on the four-quadrant toy (one Gaussian blob per quadrant at `(±0.5, ±0.5)`, std `0.15`; A is `x1 > 0`, B is `x2 > 0`). Each row is one case, with the share of the composed density within 3 std of each blob's centre and within `3·0.15/√2` of the origin. Numbers are in `answer_key.json`; the printout is `run.log`.

**How it was made**

`answer_key.py` builds ideal experts as blob mixtures with chosen weights and multiplies them on a `441 × 441` grid over `[−1.1, 1.1]²`. No training and no sampler, so the shares are the target every trained model and sampler is measured against. It ran in 2 s with the `co3` environment's Python on 2026-10-17.

**The cases**

Construction (2), two separately fine-tuned experts. Expert A keeps `1 − 3ε` on bottom-right and ε on each other quadrant; expert B keeps `1 − 3ε` on top-left. Top-right share matches ε/(2 − 4ε) to four digits (`0.25` at ε = ¼, `0.0625` at ε = 0.1). At ε = 0 (experts trained from scratch on one quadrant each) the product sits at the origin: `0.989` there, `0.006` in the top-right blob, normaliser `7.9e-10`.

Construction (2) with leftover only on the two quadrants next to each expert's own: top-right and bottom-left each get `0.50`.

Construction (1), one conditional model with "contains" labels, composed as p(x|A)·p(x|B)/p(x), with top-right holding a share r of the data: `0.989` in the top-right blob for every r from `0.25` down to `0.001`, normaliser equal to r. At r = 0 it falls to the origin (`0.970` there).

Rescue 2b, experts that model only their own part. Expert A models x1 (the "lung" part) from bottom-right data only and takes x2 from the background marginal of all four quadrants; expert B does the mirror image. No training point is in top-right. Their plain product puts `0.9999` in the top-right blob, and the background-divided product p_A·p_B/p_bg (projective composition) puts `0.989` there.
