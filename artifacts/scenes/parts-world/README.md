# The parts world

One interactive scene for claim 9 of the [overlap-dial idea map](../../ideas/poe-overlap-dial/IDEA_MAP.md): a toy where every image is 16 pixels drawn as a 4 × 4 tile and every expert is a Gaussian (or a mixture of two), so the cat-dog product is exact. The same product gives a mosaic, a grey blend or two animals side by side, depending only on which pixels each expert is sure of. It serves the vault section "Mosaic, blend or two animals from one product" in `Artificial Intelligence/Generative Modeling/Diffusion Models/Composition/Kinds Of Composition.md`, with "Multivariate Normal Distribution", "Products of Gaussian mixtures", as its pair-grid panel.

| File | What it is |
|---|---|
| `parts-world.src.html` | the authored page, with `__KEY_JSON__` where the key goes |
| `parts-world.html` | the published page, built by `checks/build.py` |
| `checks/scene_key.py` | every number the page shows; reproduces all 18 numbers of the walk's `numbers.json` at a relative gap of 0, then adds every slider stop |
| `checks/scene_key.out`, `checks/scene_key.json` | its printout and its data |
| `checks/check_page.mjs`, `checks/check_page.out` | runs the page's own key and compute blocks against the key: one PASS or FAIL line per number |
| `checks/interact.out` | a browser script's record of every control exercised: option previews, pins, sliders, the formula drag, the position toggle, the noise slider, the key table |
| `checks/math_review_brief.md`, `checks/math_review.json` | the Codex mathematics review: what it was given and what it returned |
| `checks/page_text.txt` | the page's visible text, press by press, as the review read it |
| `checks/screens/` | the browser pass's screenshots, phone dark and desktop light, one per press |

**Published at**

[The Parts World](https://claude.ai/artifact/6XzsUMYdEb2eopRyGkpYkh), version 1, private to its owner until shared from the page's Share menu. Each press opens on its own anchor, `#p1` to `#p10`; the pair grid is `#p6`.

**What it shows, press by press**

Blend (press 1): both experts sure on every pixel give a grey product, 10 of each expert's standard deviations from its mean, log Z −1583.4. It opens here because this is the midpoint chimera a 2D toy can show, and the one the learner found "very fuzzy".

Mosaic (presses 2 to 5): loosening the cat's body and the dog's head gives a cat head on a dog body, head +0.980 and body −0.980, log Z −46.5. A one-pixel panel draws the two bells and the precision-weighted mean, 99/101, and the precisions in its formula are draggable. Making the dog sure on the head too drops log Z to −814.9. From press 5 the four variances are sliders, with pins for the mosaic, the blend and the half-and-half case.

Two animals (presses 6 and 7): with each animal in the left or right half, the product is a 2 × 2 grid of component pairs. The side-by-side pairs take about 0.5 each (unnormalised log weight ℓ = −24.1); the same-half pairs get e⁻⁷⁷⁹ of that (ℓ = −803.2).

Noise (presses 8 and 9): at ᾱ = 0.5 summed scores put the head at 0.233 where the noised product, the correct target, has 0.693; a slider runs ᾱ from 1 to 0.01.

Explore (press 10): every control, narration computed from the state, a hand check (cat head 0.02, dog head 0.5 gives (50 − 2)/52 = 0.923) and a code-from-memory box whose expected output is `numbers.json`'s.

**The design sheet**

The one idea: on every pixel the product's mean is the precision-weighted mean, so where each expert is sure decides whether a cat-dog product is a mosaic, a blend, or (with position varied) two animals side by side.

The learner, in their own words: they wanted "a toy domain where we leverage exact parameterizations and solutions to replicate the cat-dog hybrid problem", and the walk until now felt "very fuzzy". They already have the precision-weighted product of two Gaussians from the vault's "Products and quotients of diagonal Gaussians".

The felt problem: ask for a cat-dog and get either a cat head on a dog body or a grey mush that is neither; a 2D toy can only show the mush.

The stage picture and its mapping:

| On the picture | In the maths |
|---|---|
| the blue tile | the cat expert's mean μ_a = +1 |
| the red tile | the dog expert's mean μ_b = −1 |
| "head · var 0.01" on a tile, and the four sliders | the diagonal of Σ_a or Σ_b on the head or body pixels |
| the violet-captioned product tile | μ = Σ(Σ_a⁻¹μ_a + Σ_b⁻¹μ_b), per pixel |
| the one-pixel panel's bells and formula | one coordinate of the same identity |
| the log Z gauge | Z = N(μ_a; μ_b, Σ_a + Σ_b), summed in logs over 16 pixels |
| the 2 × 2 pair grid, its w and ℓ | ℓ_ij = log π_i + log ρ_j + log N(μ_i; ν_j, Σ_i + Λ_j), w_ij = e^ℓ_ij / Σ e^ℓ_kl |
| the noise slider and the two right-hand tiles | N(μ, v) → N(√ᾱ μ, ᾱ v + 1 − ᾱ), applied before or after the product |
| the three draws under a tile | which pixels vary for that expert; illustrative only |

The break: the blend (both sure everywhere) and the noise chapter (summed scores dragging the mosaic's head toward grey).

The questions: the tile after loosening (q1), where log Z goes when the dog's head turns sure (q2), which pairs carry the weight (q3), and where summed scores put the head at ᾱ = 0.5 (q4). Every option draws its consequence on the stage when pointed at.

What stays out: trained networks, any sampler run, the 2D toys' own numbers (linked as a second view instead), the mixture case under noise, and how much of the mosaic a full reverse run keeps at ᾱ = 1, which no key computes.

**How it was made and checked**

Built on 2026-10-17 by the scene-cadence agent from claim 9 and Runs row 7 of the idea map, the walk's answer key in `artifacts/ideas/poe-overlap-dial/example-9-parts-world/` (`parts_world.py`, `numbers.json`, the .tex and PDF), and the two vault sections above. There was no drip dispatch and no plan capstone, so the notation is the vault section's and the learner's words are the ones quoted above. The shell (header band, question cards, press engine, key table) is copied from the sibling scene in `../when-a-product-reaches-a-and-b/`; only the stage is new.

`node checks/check_page.mjs` passes 77 of 77. The browser pass (`check_scene.mjs --region`, Playwright's Chromium 1243) walked all 10 presses at 390 px dark and 1440 px light with no must-fix items, and every screenshot in `checks/screens/` was looked at. `checks/interact.out` records the controls exercised, with no console errors.

A Codex mathematics review (`codex exec`, read-only, with `codex-author`'s math-review brief; the brief and the reply are in `checks/`) returned 6 errors and 4 smaller findings, all checked against the key and all fixed in one round. The errors: press 1 called the blend's pixel 0 where its mean is 0 and its sd 0.071; "nearer 0 is more overlap" was stated as a general rule for log Z, which holds only because the means stay at +1 and −1; sums were written with rounded operands (8 × (−99.0) + 8 × (−2.9) gives −815.2, not −814.9), on press 4 and in the disclosure; the same-half weights were shown as exactly 0 and the side-by-side as exactly 0.5, where they are r/[2(1 + r)] and 1/[2(1 + r)] with r = e⁻⁷⁷⁹; and summed scores were described as sampling the product of noised experts, where the page only computes that fixed-level density's mean. The others added the per-pixel log Z substitutions and the weight normalisation to the disclosure, the shortcut (v_dog − v_cat)/(v_dog + v_cat) for the mean, and ℓ for the unnormalised log weight in place of "log w".

To rebuild after a change: `python3 checks/scene_key.py && python3 checks/build.py && node checks/check_page.mjs`.
