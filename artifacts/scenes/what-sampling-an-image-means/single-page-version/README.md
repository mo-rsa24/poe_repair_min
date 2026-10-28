# What sampling an image means

One interactive scene that starts from the heights of ten people, yours at 173 cm, and ends with a sampler turning pure noise into a clean dog photo. It answers the question the [forward-process scene on the dog photo](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9) raised and could not answer: how x₀ is "a dog", and how drawing from a distribution gives a clean image. The notation and colours come from Step 1 (the Gaussian density) and Step 4 (the forward process) of [the one-expert walk](../../drips/simple-distributions-rebuild/01-one-expert-from-scratch.md).

| File | What it is |
|---|---|
| `what-sampling-an-image-means.src.html` | the authored page, with `%%KATEX_CSS%%`, `%%DOG_B64%%` and `%%DOGS_B64%%` where the stylesheet and photos go |
| `what-sampling-an-image-means.html` | the published page, built by `checks/build.py` |
| `assets/labrador-128px.png` | the 128 × 128 Labrador used for the pixel probe and the forward process |
| `assets/four-training-dogs-48px.png` | the four training photos side by side, 48 × 48 each |
| `checks/build.py` | fills the three placeholders; its output is byte-identical to the published version 1 |
| `checks/node_check.mjs`, `checks/node_check.out` | the node walk with the real `katex` 0.16.11 package set to throw on any error |
| `checks/browser_presets.mjs`, `checks/browser_presets.out` | every preset and switch in Chromium with the real photos, and the status line each one prints |
| `checks/check-scene.json`, `checks/screens/` | `check_scene.mjs` at a 30-press budget: phone dark and desktop light, one screenshot per frame |

**Published at**

[From 173 cm to a Dog](https://claude.ai/artifact/NYeZh73Sg18F2Ui4ZvNxrV), version 1. It is private to its owner until it is shared from the page's Share menu. Each frame opens on its own anchor, `#f1` to `#f25`.

**The scene plan that was agreed**

Seven tabs in the order the ideas build: ten heights, density, fitting, sampling, an image, a model samples, your questions. Fitting comes before sampling so the sampling tab draws from the best-fit bell. 25 frames, one press each, against a budget of 30. Purple is the random number fed in (u, z, ε) on every tab. The image reverse chain runs the exact best guess over four CC-licensed dog photos at 48 × 48, so no trained network appears; the page says so.

**Each question in the request, and the frame that answers it**

| Item | From the request | Frame |
|---|---|---|
| S1 | "x in p(x) is just a sample; my height is 173 cm" | 1 |
| S2 | "from the 10 I create a distribution, get mean and variance" | 2, then 7 |
| S4 | "likelihood: change μ, σ so the density at the sample is highest" (corrected: at all ten) | 3, 6, 7, 8 |
| | what a density is | 4, 5 |
| S3 | "sampling: evaluate the inverse CDF, get a height" | 9, 10, 11 |
| S5 | "images with labels; an image is H × W × C" | 12, 13 |
| | why the inverse-CDF recipe fails for an image | 14, 15 |
| Q1 | "how does a forward prop equate to sampling?" | 16 |
| Q2 | "x₀ appears to be the mean; how does sampling work?" | 17, 18 |
| Q3 | "how does sampling give a clean image?" | 19, 20 |
| Q4 | "how much does the training data matter?" | 21 |
| Q5 | "how much does Bayes' theorem play a role?" | 22 |
| Q6 | "how many assumptions are required?" | 23 |
| Q7 | "do convnets or transformers carry probabilistic assumptions?" | 24 |
| Q8 | "how do we scale teaching this from the fundamentals?" | 25 |

**Where the numbers come from**

Every number is computed in the page. The ten heights (158, 163, 165, 168, 170, 172, 173, 176, 179, 184 cm) are example values, and the page says so on frame 1. The forward process uses Step 4's cosine schedule (T = 100, offset 0.008, β clipped at 0.999), recomputed in the page. Three numbers are also checked against journey 03's `artifacts/substrate/facts.json` in `generative-models/03-sampling-from-a-graph/`: the 8 × 8 table size 1.845 × 10¹⁹, the stripes toy's 0.125 against 1.0, and Φ⁻¹(0.975) = 1.96. The page names that file in each frame's closed disclosure. Two claims come from outside the page and are marked that way in their disclosures: that trained networks produce new images because they fall short of the exact denoiser (Kadkhodaie et al. 2024), and classifier guidance (Dhariwal and Nichol 2021).

**How it was made**

The page started from the dog scene's template (`~/.claude/skills/picture-speak/assets/example-consequence-forward-process.src.html`) and kept its look, its KaTeX drag handles and its colour meanings. One walk runs across all seven tabs with a `k / 25` counter, so the right-arrow key reaches every frame. The four training photos came from Wikimedia Commons, with licences read from the Commons API: "Golde33443.jpg" by Golden Trvs Gol twister (CC BY-SA 3.0), "Border Collie liver portrait.jpg" by John Haslam (CC BY 2.0), "Siberian-husky.jpg" by Utopialand (CC BY-SA 4.0), and the Labrador "YellowLabradorLooking new.jpg" by Djmirko (CC BY-SA 3.0). Each was cropped square and resized.

**What the checks found on version 1**

The node walk: 25 frames, 59 presets, 7 switches toggled both ways, 33 handles driven to their minimum, maximum and middle, 197 actions, 0 errors. The browser preset pass: 91 actions, 0 KaTeX errors, 0 page errors. `check_scene.mjs`: 25 frames within the 30-press budget, 0 must-fix items, 0 should-check items. Earlier builds failed on a frame taller than the 860 px window, an axis tick labelled 0.05 that sat at 0.055, equations running out of their boxes, and 1 cm bins that split one height's chains across two bars. Each was fixed before publishing.

**What stays out**

No trained network appears. The exact best guess over a small training set stands in for one, so every chain returns a training example. Frame 21 shows that with the exact guess and contrasts it with a fitted bell, which returns new heights.
