# Three ways a product reaches A-and-B

An interactive page on the four-quadrant toy for the drip-idea walk "When a product of two diffusion experts reaches A-and-B", claim 2 and its children 2a, 2b and 2c. It is published at [the live scene](https://claude.ai/artifact/9eFnG2rWEciPsceMwdem4u), and its five frames run over 48 presses. Each press reveals one thing (a region, a label, a number or a formula); what was taught earlier stays on screen in grey, and nothing not yet taught is drawn. A prediction always comes on its own press before the press that reveals its answer. The first frame shows two from-scratch experts whose product lands at the origin, a blend that is none of the four kinds of film, then moves expert B's blob from the opposite corner to next door to on top of A's, so the product's total mass Z climbs from 7.9 × 10⁻¹⁰ to 5.3 × 10⁻⁵ to 3.54. The next three are the rescues: support kept from pretraining (fine-tuned experts, then one conditional model with "contains" labels), and experts that learn parts. The last shows the toy as an exact target for testing samplers.

**What the reader comes away with**

What support means, what "A, not B" and "B, not A" are on the picture, and what 0.25, 0.5, 0.05 and 0 stand for. A ledger panel, "What each label and number means", gains one row per press, newest on top in colour and earlier rows in grey. Unticking "Grey what is already taught" in the bottom bar shows everything in full colour.

**How it was made**

Every number on the page comes from `scene_key.json`, written by `scene_key.py`. That script first reproduces every row of the walk's answer key (`artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/answer_key.json`) with a relative gap of 0, then adds the slider stops between the key's values, four closed forms the page explains with, and the three positions of expert B's blob, which it checks against the same values in the [Where the Product Lands key](../when-a-product-reaches-a-and-b/checks/scene_key.json). It ran with the `co3` environment's Python; its printout is `scene_key.out`. `check_numbers.mjs` runs the page's own compute block against every slider stop (123 of 123 pass, `check_numbers.out`). Every number in the narration and the ledger is formatted from `scene_key.json` at load time rather than typed in.

**Files**

| File | What it is |
|---|---|
| `page.src.html` | the page source, with the key as a placeholder |
| `build.py` | fills the key in and writes `page.html`, the published file |
| `scene_key.py`, `scene_key.json`, `scene_key.out` | the numbers, and the printout that shows they match the walk's key |
| `check_numbers.mjs`, `check_numbers.out` | the page's compute block checked against the key |
| `screens/` | phone (dark) and desktop (light) screenshots of presses 10, 16, 22 and 44, and `check-scene.json` from the browser walk |

**How it was checked**

The page was stepped press by press with `codex-author`'s `check_scene.mjs --region` at 1440 px light and 390 px dark (Playwright's Chromium 1243, passed as `CHROME`): no must-fix items across all 48 presses, and the screenshots were looked at. Presses 10, 20 and 42 at both sizes are kept in `screens/` with the checker's report. A Codex mathematics review read every press against the key and found 7 errors, all fixed before publishing. A Gaussian is never exactly zero, so support is drawn as each expert's blob. The origin blend is called none of the four films, since "neither" names the bottom-left quadrant. ε/(2 − 4ε) and the next-door 0.5 are same-corner approximations, with 0.5 also shown to be an exact ceiling by the x → −x symmetry. Z ≈ r is the integral of a named q_r. "0.989 for every r" is limited to the stops the key computed, down to 1 film in 100,000.

**What it does not show yet**

No sampler has been run on any target here. The sampler frame draws the target and leaves the histogram as an empty slot until the walk's claim 4 check produces one.
