# What sampling an image means

Three connected scenes that start from the heights of ten people, yours at 173 cm, and end with a sampler turning pure noise into a clean dog photo. A concept map runs along the top of every frame, with the operation this frame teaches lit, so each idea is shown as something built from the one before. Where two concepts are one object read two ways, a flip frame shows both readings on one figure with a single switch. The mathematics behind every frame is written out in [the mathematics of sampling an image](../../notes/the-mathematics-of-sampling-an-image/README.md).

| File | What it is |
|---|---|
| `index.html` | scene 1, One number (13 frames), the artifact's main page |
| `many-numbers.html` | scene 2, Many numbers (9 frames) |
| `going-back.html` | scene 3, Going back (12 frames) |
| `source/shell.html` | the page shell all three share: layout, colours, the map strip, the walk controls |
| `source/lib.js` | the shared library: the ten heights, bells, Step 4's schedule, the best guess and both reverse chains, the KaTeX drag handles, the map and the walk |
| `source/scene1.js`, `source/scene2.js`, `source/scene3.js` | each scene's frames |
| `source/build.py` | builds the three pages; its output is byte-identical to the published version 2 |
| `assets/` | the 128 × 128 Labrador and the strip of four 48 × 48 training dogs |
| `checks/node_check.mjs`, `checks/node_check.out` | the node walk of each page with the real `katex` 0.16.11 package set to throw on any error |
| `checks/browser_presets.mjs`, `checks/browser_presets-*.out` | every preset and switch in Chromium with the real photos, each status line printed, and every equation measured against its card |
| `checks/check-scene-*.json`, `checks/screens/` | `check_scene.mjs` on each page at its press budget, with phone dark and desktop light screenshots of every frame |
| `single-page-version/` | version 1, the 25-frame single page, with its own card and checks |

**Published at**

[From 173 cm to a Dog](https://claude.ai/artifact/NYeZh73Sg18F2Ui4ZvNxrV), version 2. It is private to its owner until it is shared from the page's Share menu. Scenes 2 and 3 are the sibling pages `many-numbers.html` and `going-back.html` at the same link. Every frame opens on its own anchor, `#f1` and up.

**The three scenes**

Scene 1, One number: one height is one sample; the pile's mean 170.8 cm and variance 54.16 cm²; a bell scored by log-likelihood; density per cm and chance as area; a flip frame showing one formula as a density over x and as a likelihood over μ; the product of ten densities; the slope in μ that is zero at the mean and the slope in σ that is zero at the spread; one height alone shrinking σ to zero; F as the running area of p; a flip frame showing F read upwards and F⁻¹ read from the side; many random u's rebuilding the bell.

Scene 2, Many numbers: a 2 × 2 grey image as a point in ℝ⁴; a density over whole images on a hand-built toy, two real images a = (1, 1, −1, −1) and b = −a with spread 0.1 (normaliser 253.3, p(a) = 126.65); a flip frame showing the joint of two pixels and the marginal of one (peaks 1.995); four pixels drawn by F⁻¹ four times in order, u = (0.7, 0.4, 0.6, 0.5) giving (0.975, 0.975, −0.975, −1.000); drawing pixels alone giving a real image 2 times in 16; a network's output becoming a density through the Gaussian formula's normaliser; a flip frame putting scene 1's F⁻¹ beside a transport G from five Gaussian numbers to the toy; fitting G's scale by likelihood on eight training images; adding noise to one photo.

Scene 3, Going back: a flip frame showing one table of training heights by noisy values, read along a row as adding noise and down a column as Bayes' theorem; prior times likelihood divided by the total; the best guess as a weighted average, equal to scene 1's mean at step 99; one step back with x₀ known derived as the product of two Gaussians, checked on a grid; the same step with the best guess, beside the exact ten-bell mix; the chain on heights; the chain on four dog photos; a label entering by Bayes; what the training data decides; five assumptions; the architecture as a fixed box; the whole map.

**Where the numbers come from**

Every number is computed in the page. The ten heights (158, 163, 165, 168, 170, 172, 173, 176, 179, 184 cm) and the 2 × 2 toy are example values chosen for this series, and the pages say so. The forward process uses Step 4's cosine schedule (T = 100, offset 0.008, β clipped at 0.999), recomputed in the page. Four numbers are also checked against journey 03's `artifacts/substrate/facts.json` in `generative-models/03-sampling-from-a-graph/`: one-sample fitted variance 0, the 8 × 8 table size 1.845 × 10¹⁹, the stripes share 0.125, and Φ⁻¹(0.975) = 1.96. Claims from outside the pages are marked in each frame's disclosure: generalisation from an inexact denoiser (Kadkhodaie et al. 2024), classifier guidance (Dhariwal and Nichol 2021), the reverse-step posterior (Ho, Jain and Abbeel 2020), and the deterministic sampler (Song, Meng and Ermon 2021).

**How it was made**

The design was agreed in conversation: three connected scenes instead of one long page, a shared concept map with flip frames, and scene 2 rebuilt around a 2 × 2 toy so that an image's density, its normaliser, the order of drawing and transport by a network are each exact numbers. The pages started from the version 1 page and its dog-scene template, and the layout rules from Codex's polish of version 1 (job 20261021-061533-d107db, closed as superseded) were carried into the shared shell. The four training photos come from Wikimedia Commons, with licences read from the Commons API: "Golde33443.jpg" by Golden Trvs Gol twister (CC BY-SA 3.0), "Border Collie liver portrait.jpg" by John Haslam (CC BY 2.0), "Siberian-husky.jpg" by Utopialand (CC BY-SA 4.0), and the Labrador "YellowLabradorLooking new.jpg" by Djmirko (CC BY-SA 3.0).

**What the checks found on version 2**

Node walk: 34 frames, 80 presets, 12 switches both ways, 47 handles at their minimum, maximum and middle, 0 errors. Browser preset pass: 126 actions, 0 KaTeX errors, 0 page errors, no equation wider than its card at 1440 px. `check_scene.mjs`: budgets 14, 10 and 13 presses, 0 must-fix and 0 should-check items on each page. On a 390 px phone, wide figures and the widest equations scroll inside their own boxes; the page itself never scrolls sideways.

**What stays out**

No trained network and no trained classifier appear. The exact best guess over a small training set stands in for a network, so every chain returns a training example; scene 3 frame 9 contrasts it with a fitted bell. G in scene 2 and p(tall | x) in scene 3 are written by hand and labelled that way.
