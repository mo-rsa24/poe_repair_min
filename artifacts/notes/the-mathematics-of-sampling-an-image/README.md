# The mathematics of sampling an image

The written-out mathematics behind the three scenes of [From 173 cm to a Dog](../../scenes/what-sampling-an-image-means/README.md), kept so the understanding outlives the pages. One chapter, in the order the scenes build it, with every number recomputed by a script that shares no code with the scenes.

| File | What it is |
|---|---|
| `the-mathematics-of-sampling-an-image.md` | the chapter: 3 parts, 24 sections and a closing count of the assumptions, each equation taken apart with the scenes' colour key, each section linked to the frame that shows it |
| `check_numbers.py`, `check_numbers.out` | recomputes all 74 numbers the chapter states with NumPy and SciPy and prints PASS or FAIL for each; the chapter cites each number by its label here |
| `katex_md_check.mjs`, `katex_md_check.out` | parses every equation in the chapter with KaTeX 0.16.11, the engine VS Code's markdown preview uses (needs the `katex` npm package installed beside it) |

**What it covers**

Part 1, one number: a sample and a dataset; a density and where its normaliser 1/(σ√2π) comes from, by the polar-coordinates Gaussian integral; a chance as an area; likelihood as the same formula read with the data fixed, and why it is not a density over the parameters; the maximum-likelihood derivation that returns the mean and the variance; one sample sending σ to zero; the CDF and its slope; the one-line proof that F⁻¹(U) samples the density.

Part 2, many numbers: an image as a point; a density over ℝ⁴ and why its normaliser grows to 253.3; marginals; the chain rule and drawing in order (the Rosenblatt transform), with the toy's Bayes-updated conditionals; why skipping the conditioning breaks images, and why a table cannot hold the conditionals; three ways a network's output vector becomes a density (parameters of a fixed family, transport, the score); the pushforward and the change-of-variables formula; learning the map by likelihood; the forward process.

Part 3, going back: Bayes on the joint table of training values and noisy values; the best guess as the posterior mean, with Tweedie's formula tying it to the score; the reverse step with x₀ known, derived as a product of two Gaussians (Ho, Jain and Abbeel 2020, equations 6 and 7); the exact step back as a mix of bells and why small steps make one bell enough; the chain and its deterministic form; what the training data decides; a label entering by Bayes. It ends with the five assumptions counted, and which one is a free choice.

**How it was made**

Written after the scenes were built and checked, from their source and the derivations behind each frame. Each number was first written into the chapter and then recomputed independently in `check_numbers.py`. The first run caught seven faults, all in the checking or in the chapter's stated values, none in the scenes: SciPy's integrator missing a narrow bell over an infinite range, an importance sampler that covered the two bumps badly, three stated values written before they were computed, and a log-density that counted one bump where the checkerboard image sits equally far from both. All 74 now pass. Outside results the chapter relies on are named where they are used: Ho, Jain and Abbeel (2020), Song, Meng and Ermon (2021), Dhariwal and Nichol (2021), Kadkhodaie et al. (2024).
