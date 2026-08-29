# SPEC: noise-shell-and-basins

**Location** `artifacts/scenes/noise-shell-and-basins/`

**Ran under** any browser; open `index.html` directly. No Node, no Python, no GPU. Math
rendering pulls KaTeX from a CDN, so equations need internet; everything else is inline.

**Built from** no model trace. This is the canonical-fallback rung: textbook Gaussian
concentration (the chi-square norm of a standard normal in d dimensions) plus two measured
numbers read from `context/world/interaction-term.md`, which surfaces
`report/experiments-log.md` EXP-01 and EXP-04.

**Why** corrects the contour-map intuition about diffusion seeds before the paper argues about
early-window timing: in d = 65,536 a seed cannot sit at an unusual radius, so a seed is a
direction on one shell, and the timing numbers should be read against a flow of directions,
not a density map.

run: open `index.html` in a browser (over SSH, `scp` it down or use the editor's preview; it is
one file with no server).

## Tree

2 panels plus a header, one page, no routes. No architecture is traced, so no node-type
templates apply; both panels are bespoke SVG, named here as untemplated by design.

| Node | Treatment | Provenance |
|---|---|---|
| header | title claim, one line, setup strip (SDXL, epsilon-prediction, latent 4x128x128 = 65,536 dims, DDIM 50 steps, guidance 7.5) | pipeline configuration, stated |
| panel 1 `#panel-cartoon` | the cartoon that fails: concentric rings, seeds at several radii, NOT SUPPORTED stamp; KaTeX block; chi-density plot with a dimension slider (d = 2 to 65,536) and live readouts | canonical (textbook); computed in-page by `chiLogPdf()` |
| panel 2 `#panel-shell` | one shell at radius 256, four seeds as directions, basin arcs, caustic marker, sketched manifold with two modes, DDIM trajectories with the 0-10 window overlaid, step ruler with both real bands, step scrubber | window 0-10 and commitment 18-36 tagged real (EXP-01/EXP-04); seeds, modes, manifold, basin geometry tagged illustrative or canonical |

Equation-bearing nodes (carry the drip cue): the concentration block in `#panel-cartoon`.
Route for math-scene to link back to: `index.html#panel-cartoon`.

Floor: no repo model code exists for a geometry fact, so the floor is the in-page
`chiLogPdf()` (the function that draws the curve) plus a two-line PyTorch check in panel 1's
disclosure (`torch.randn(1000, 4*128*128).norm(dim=1)` giving mean about 256 and sd about 0.71).

## Gate note

Built unattended. The `/deep-learning-scene` arguments specified both panels, every number,
the provenance tags, and the finish condition, and were taken as the confirmed spec; no
interactive confirmation was possible in this session.
