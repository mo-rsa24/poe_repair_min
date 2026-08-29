# When the correction arrives

Candidate images for section 4's timing paragraph. Register slots F4a, F4g, F4h. Each shows
generated pictures with green borders, and the border rule differs by figure, quoted from the
figure code into each caption. The window map and the sliding strip frame cells the detector
scored as composed. The two coverage grids (F4g, F4h) frame decodes taken while the correction
was active, the `contains()` rule in their scripts, and say nothing about the detector.

## Which composition rule made the picture

The question this folder asks is when in the run the missing term has to arrive. The rule that
supplies the term is the variable inside it, so each rule gets a subfolder and the folder keeps
its question.

**`poe/`** holds every figure where the term is the cached `r_t = eps_J - eps_PoE` injected into
plain product-of-experts. Everything below is one of these.

**`mcmc/`** will hold the same layouts with a Langevin corrector in place of the injected `r_t`,
built by
[the sampler-against-model scope](../../../../plans/is-the-gap-the-samplers-or-the-models/MASTER_PLAN.md).
Empty until that scope's gate at step 26 has fired and its window sweep at step 27 has run.

**`superdiff/`** will hold the SuperDiff arm from the same scope. Empty until step 28 wires the
pipeline at 50 steps.

**poe/samples-over-the-window-map.png** (with `.pdf` and `.json` siblings). The manuscript's
timing figure, F4a and F4f combined in one render by `scripts/window_samples_over_map.py` so
the two halves share one x axis. Top: cat and dog at seeds 9 to 12, the final image per window
position, a green frame where the scored verdict in `window_curves.json` is composed. Bottom:
seeds composed out of 4 for all eight pairs, rows ordered strongest-first at the earliest
window, daggers on the two frog-and-toad error cells. It reads the scored grid only, so this
figure and the window curves cannot disagree; counts and the green-frame rule are in the
sidecar. It is the injected-`r_t` arm, hence `poe/`.

**poe/samples-as-a-ten-step-window-slides.png** (slot F4a). Final picture for cat and dog at seeds
9 to 12, one column per position of a ten-step correction window sliding across the run. Two
animals survive only when the window sits early.

**poe/samples-as-the-window-extends-from-the-start.png** (slot F4g). One cell decoded at steps 10
to 50, one row per window that starts at 0 and extends later (0-10 up to 0-50). Ten early
steps already give two animals, and longer coverage adds nothing.

**poe/samples-as-the-window-starts-later.png** (slot F4h). Same layout, one row per window that
ends at 50 and starts later (10-50 down to 40-50). Starting at 10 still composes, starting at
20 or later never does, whatever the coverage.

**poe/seeds-composed-per-pair-as-the-window-start-moves.png** (slot F4f). The population view:
eight pairs as rows, the first step of the ten-step window as columns (0 to 40 in steps of 5),
each cell the number of seeds out of 4 that composed. Composition concentrates in the two
earliest columns for every pair, and from a window start of 15 onward the map is zeros except
two single-seed cells for frog and toad at starts 35 and 40. Those two cells were inspected on
2026-08-27 and both are detector error, confirmed at detector level in
[the miscount check](../../../../artifacts/results/did-the-detector-miscount-the-late-frog-toad-cells/README.md):
the second kept box sits on the animal's own rear haunch, passing the confidence floor at
exactly 0.30 and escaping NMS because a nested box's overlap is only its area ratio. Both are seed 10 (`window_curves.json` rows: windows
[35,45] and [40,50], n_instances 2, compose 1). The two images, at
`outputs/interaction_term/window/pairs/a_frog__x__a_toad/seed_10/teacher_residual_const_lam100_w35-45/`
and `.../w40-50/`, each show one illustrated toad-like animal with a bulbous lump at its rear
that the detector boxed as a second animal. They are also nearly identical to each other,
which is itself the late-window result, the correction arriving at step 35 or 40 changed
almost nothing. The caption carries one sentence naming the two cells as instrument error.
Effect size varies by pair, from 4 of 4 (seal and walrus, cow and buffalo at start 0) down to
1 of 4 (leopard and jaguar).

**Where they came from.** Dropped in during the section 4 walk from the window-sweep runs.
The two coverage grids are `scripts/longer_correction_grid.py` and `scripts/later_start_grid.py`
at their default cat-and-dog seed 12, confirmed by matching their couch scene to the window
map's seed 12 row. Which script rendered the sliding strip and the per-pair population map is
still to be confirmed before a caption cites a path.
