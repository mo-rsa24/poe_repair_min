# Where the product lands

One interactive scene for claim 2 of the walk in the [idea map for when a product reaches A and B](../../ideas/when-poe-reaches-a-and-b/IDEA_MAP.md): the three ways a product of two diffusion experts can reach a combination neither expert's fine-tuning data contains, on the four-quadrant toy.

The same material is taught one concept per press, with this page's drag of expert B's blob and its fourth answer to the first prediction, in [Three Ways a Product Reaches A-and-B](../three-ways-a-product-reaches-a-and-b/README.md). That scene is the one kept current; this page stays published as it stands.

| File | What it is |
|---|---|
| `where-the-product-lands.src.html` | the authored page, with `__KEY_JSON__` where the key goes |
| `where-the-product-lands.html` | the published page, built by `checks/build.py` |
| `checks/scene_key.py` | every number the page shows; reproduces all 18 rows of the walk's answer key, then adds the slider stops |
| `checks/scene_key.out`, `checks/scene_key.json` | its printout and its data |
| `checks/check_page.mjs`, `checks/check_page.out` | runs the page's own key and compute blocks against the key: one PASS or FAIL line per number |
| `checks/screens/` | the browser pass's screenshots, phone dark and desktop light, one per press |

**Published at**

[Where the Product Lands](https://claude.ai/artifact/2A7YDnfKe8EPsS7yFDVcNT), version 1, private to its owner until shared from the page's Share menu. Each press opens on its own anchor, `#p1` to `#p14`.

**What it shows, press by press**

Support (presses 1 to 3): two experts trained from scratch on one quadrant each put 0.989 of their rescaled product at the origin and 0.006 in the top-right blob; dragging expert B's blob shows the raw mass Z fall from 3.54 to 5.3 × 10⁻⁵ to 7.9 × 10⁻¹⁰.

Rescue 2a, separately fine-tuned experts (presses 4 to 8): 0.25 in every blob at fine-tune step 0; a slider for the leftover ε with the draggable formula ε/(2 − 4ε); a toggle for leftover next door only (0.50 top-right, 0.50 bottom-left).

Rescue 2a, one labelled model (presses 9 and 10): 0.989 in the top-right blob for every r from 0.25 down to 0.001, and 0.970 at the origin at r = 0.

Rescue 2b, experts that learn one part (presses 11 and 12): 0.9999 plain, 0.989 background-divided, with no A-and-B film; the chest X-ray catch as a caption; "Divide Out the Wall" linked as a second view of the background division on the two-pixel toy.

Rescue 2c (press 13): the exact product beside an empty sampler slot, to be filled by the sampler runs of claim 4.

Explore (press 14): every construction and slider, narration computed from the state, and a hand check at ε = 0.1.

**How it was made and checked**

The page was built on 2026-10-17 by the scene-cadence agent from the idea map's claim 2 (children 2a, 2b, 2c), the walk's answer key in `artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/`, and the vault note "Diffusion Models", section "When neither training set holds A and B". There was no drip dispatch and no plan capstone, so the notation is the learner's own words as given in the request. Every number on the face is read from the embedded scene key; the pictures are drawn from the same formulas on a coarser grid. `node checks/check_page.mjs` passes 124 of 124. A Codex mathematics review was run directly with `codex exec`, because the installed codex-cli 0.147 rejects the bridge's `--no-daemon` flag. It returned 11 must-fix wording errors, all from treating Gaussian tails as true zeros or overstating construction (1)'s raw mass, plus three missing derivations. Each was checked against the key and fixed in the same round, and the derivations went into the disclosure. The browser pass (`check_scene.mjs --region`, Playwright's Chromium 1243) walked all 14 presses at 390 px dark and 1440 px light with no must-fix items, and a second script exercised the support pins and drag, the option previews, the ε and r sliders and every Explore construction with no console errors; the screenshots in `checks/screens/` were looked at.

To rebuild after a change: `python checks/scene_key.py && python checks/build.py && node checks/check_page.mjs` with the co3 environment's Python.
