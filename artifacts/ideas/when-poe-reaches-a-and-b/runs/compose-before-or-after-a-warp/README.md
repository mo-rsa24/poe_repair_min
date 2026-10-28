# Composing before or after an invertible warp

**What it is**

A check of claim 10 in the idea map: whether composing two experts in a latent space and decoding gives the same density as composing the decoded experts in pixel space. Numbers are in `warp_check.json`; the printout is `run.log`.

**How it was made**

`warp_check.py` puts the four-quadrant toy (A-and-B share r = 0.05; expert A on the right half's data, expert B on the top half's) in a latent space z and decodes it through a fixed invertible warp x1 = z1 + a·tanh(2 z1), x2 = z2 + a·tanh(2 z2) + 0.4 z1, at warp strengths a = 0, 0.3, 1 and 3. Pixel-space experts are the exact pushforwards p_Z(z)/|det J|. Both routes are compared as probability over the same cells by total variation on an `801 × 801` grid. It ran in seconds with the `co3` environment's Python on 2026-10-17.

**What it found**

Construction (1), p_A·p_B/p, and negation p_A^1.3/p_B^0.3 agree across the warp to `4e-17` and `7e-17` total variation, which is rounding. The plain product p_A·p_B differs by `0.035`, `0.080` and `0.126` at warps whose largest area stretch is `2.6`, `9` and `49` times. Compositions whose exponents sum to 1 do not depend on the space they are computed in; the plain product does.
