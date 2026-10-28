# Video sweep over the name sections of phases 2 to 5, 2026-10-20

Every segment this sweep found for the 🔤 name sections of [phase 2](../02-composing-and-sampling.md), [phase 3](../03-ideal-experts-and-the-seven-kinds.md), [phase 4](../04-the-parts-world.md) and [phase 5](../05-the-overlap-dial.md), placed or not, so a later session reads this instead of searching again. Each placed table has its source in the [video lessons index](/home-mscluster/mmolefe/Obsidian/Mathematics/references/videos/Video%20Lessons.md) and a copy inside its name section, after the section's last paragraph and directly above its 🔨 Build line, or at the end of the section where there is none. The earlier records are [the phase 1 sweep](pulse-2026-10-20.md) and [the 2026-10-18 sweep](pulse-2026-10-18.md).

**What was excluded before searching.** Every video id and start second already linked anywhere in the walk folder or its `ladder/` (93 distinct links before this sweep) was left out. A concept that already had a table in the same file was skipped: product of experts, Langevin sampling, Metropolis-Hastings and annealed Langevin in phase 2; kernel density estimate, sum of independent Gaussians, total variation and product of experts in phase 3; product of two Gaussians, product of two mixtures and total variation in phase 4; kernel density estimate, annealed Langevin and product of two mixtures in phase 5.

**How times were read.** Captions were requested one video at a time with a minute between requests. The first, Yilun Du's thesis defense (`7jvWDF6ZXPc`, human captions), downloaded at 12:41. YouTube refused the second (CS236 lecture 11) with HTTP 429 at 12:42:59, the same block the phase 1 sweep met at 12:02. Every later video was timed from its chapter list, which is good to the chapter and does not check the sentence inside. Captions already cached by earlier sweeps were reused: CME296 lectures 1 and 2 (`tr-CUpw--ck`, `_WaR2fjZpEQ`) and ritvikmath's Gaussian mixture video (`EWd1xRkyEog`).

## Phase 2: composing experts, and the samplers

### The normaliser (`normaliser`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Stanford CS236 | Lecture 11, Energy Based Models | m61KiAMCJ5Q | 2024-05-06 | 6:45 to 15:31 ("Normalization constraints") | Intuition | chapter-timed (captions refused, HTTP 429) | yes, row 1. The same row sits in the index's `energy-based-models` source, whose 15:31 row the walk already links, so that block could not be copied whole |
| The Organic Chemistry Tutor | Probability Density Functions | QKA4HNEw3aY | see the index | 0:52 to 3:39 | Worked example | chapter-timed (from the index's `density-integrates-to-one` source) | yes, row 2: checking that a function's integral equals 1 |
| Yilun Du (open web) | PhD thesis defense, Generalizing Outside The Data Distribution through Compositional Generation | 7jvWDF6ZXPc | 2024-08-28 | 15:40 to 16:08 | Intuition | caption-read | no: maximum likelihood shrinking the partition function is about training, which this section does not do |

### Noised experts against the noised product (`summed-noised-scores-miss-the-product`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Yilun Du (open web) | PhD thesis defense | 7jvWDF6ZXPc | 2024-08-28 | 27:10 to 28:43 | Intuition | caption-read | yes, row 1: a diffusion model learns one energy per noise level; adding two models' denoiser outputs and running the ordinary reverse sampler does not draw from the composed distribution, and treating them as energy-based models with MCMC does |
| Yilun Du (open web) | same | 7jvWDF6ZXPc | 2024-08-28 | 28:43 to 29:49 | Intuition | caption-read | no: the fix (several MCMC steps per noise level), which is the annealing step the file's `annealed-langevin` table already covers |

No listed creator covers why summed noised scores miss the product; it stays on the gap list in `VIDEO_CREATORS.md` ("product-of-experts score composition"). Du is the first author of the paper this step cites.

### Temperature (`temperature`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Yilun Du (open web) | PhD thesis defense | 7jvWDF6ZXPc | 2024-08-28 | 21:54 to 22:15 | Intuition | caption-read | yes, row 1: scaling an energy's magnitude sets how much that factor shapes the result |
| Vizuara | Top-k sampling in Large Language Models | EhU32O7DkA4 | 2024-11-10 | 0:00 to 5:39 ("Temperature Scaling recap") | Intuition | chapter-timed | yes, row 2 |
| MarbleScience (open web) | Softmax, What is the Temperature of an AI?? | YjVuJjmgclU | 2023-02-02 | 3:45 to 6:09 ("Boltzmann and Softmax", "Effect of temperature") | Intuition | chapter-timed | yes, row 3: the physics the word is borrowed from, which Vizuara does not cover |
| Kapil Sachdeva (open web) | Softmax (with Temperature) | SFsc2P240rw | 2022-05-16 | 10:51 to 13:30 ("Temperature scaling") | Intuition | chapter-timed | no: the same skill as Vizuara's row from a creator not on the list |

### The soft minimum (`soft-minimum`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Yilun Du (open web) | PhD thesis defense | 7jvWDF6ZXPc | 2024-08-28 | 22:42 to 23:21 | Intuition | caption-read | yes, row 1: energies as sets; adding two gives low energy only in the intersection, the soft minimum gives low energy in either set |
| mathematicalmonk (open web) | (ML 14.10) Underflow and the log-sum-exp trick | -RVM21Voo7Q | 2011-07-07 | none | none | no chapters, captions not reached after the 429 | no: kept for phase 4's underflow section, and untimed |

### Negation (`negation-by-dividing-densities`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Yilun Du (open web) | PhD thesis defense | 7jvWDF6ZXPc | 2024-08-28 | 25:22 to 25:37 | Intuition | caption-read | yes, row 1: subtracting one energy from the other samples p1(x)/p2(x) |
| Yilun Du (open web) | same | 7jvWDF6ZXPc | 2024-08-28 | 24:13 to 24:41 | Worked example | caption-read | yes, row 2: a gray cylinder behind a yellow cube and not a gray cylinder below a brown cube |
| Yilun Du (open web) | same | 7jvWDF6ZXPc | 2024-08-28 | 23:21 to 23:36 | Intuition | caption-read | no here: placed in phase 3's negation section, so each segment is linked once |

### Unadjusted Langevin (no table)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Stanford CME296 | Lecture 2, Score matching | _WaR2fjZpEQ | 2026-04-14 | 14:28 to 17:05 | Intuition | caption-read | no: following the score alone collapses to the densest point and Langevin adds noise, the same skill as the file's `langevin-sampling` table; nothing on the step-size bias. It is the index's `langevin-dynamics` source |
| Deepia (open web) | Unadjusted Langevin Algorithm | m0OTso2Dc2U | 2024-12-27 | 12:39 to 14:47 ("Langevin Algorithm"), 17:50 to 19:18 ("Limitations") | Intuition | chapter-timed | no: neither chapter title says the stationary law is biased at a finite step, and captions were not reached after the 429 |

### Metropolis-adjusted Langevin (no table)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Abdul Fatir Ansari (open web) | A Visualization of Langevin Monte Carlo (Metropolis-adjusted Langevin Algorithm) | cVn0kru3hL8 | 2020-04-09 | none | none | no chapters, no captions | no: nothing could be timed |
| Simons Institute (open web) | Minimax Mixing Time of the Metropolis-Adjusted Langevin Algorithm | TIhk23fCMJg | not read | none | none | not read | no: a research talk on mixing-time bounds |

No listed creator has a MALA video. The file's `metropolis-hastings` table covers the accept step itself.

### The chain (`markov-chain`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| ritvikmath | Markov Chains, Data Science Basics | prZMpThbU3E | 2020-09-09 | 0:37 to 2:50 ("Example Markov chain", "State space", "The Markov assumption") | Intuition | chapter-timed | yes, row 1 |
| ritvikmath | Markov Chain Monte Carlo (MCMC), Data Science Concepts | yApmR-c_hKU | 2021-01-20 | 4:07 to 8:24 ("Markov Chain Monte Carlo") | Intuition | chapter-timed | yes, row 2. The same row sits in the index's `markov-chain-monte-carlo` source |
| ritvikmath | Coding MCMC, Data Science Code | KmqTrm-bn8k | 2021-02-15 | 2:21 to 5:50 ("Accept Reject") | Code walk | chapter-timed | yes, row 3 |
| ritvikmath | Markov Chain Stationary Distribution | 4sXiCxZDrTU | 2021-01-18 | 3:06 to 13:00 | Proof | chapter-timed | no: the stationary distribution, which the section does not use |

### Hamiltonian (`hamiltonian-monte-carlo`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Meerkat Statistics (open web) | MCMC 5, Hamiltonian Monte Carlo (HMC) | 2zr7cMjltAI | 2025-08-27 | 1:26 to 2:22 ("Hamiltonian physics mapping") | Intuition | chapter-timed | yes, row 1 |
| Meerkat Statistics (open web) | same | 2zr7cMjltAI | 2025-08-27 | 2:22 to 3:08 ("Sampling momentum") | Intuition | chapter-timed | yes, row 2 |
| Ben Lambert (open web) | The intuition behind the Hamiltonian Monte Carlo algorithm | a-wydhEuAm0 | 2018-05-15 | 10:17 to 16:58 ("Joint distribution definition") | Intuition | chapter-timed | yes, row 3 |
| Ben Lambert (open web) | same | a-wydhEuAm0 | 2018-05-15 | 4:30 to 10:17 ("Statistical mechanics basis") | Intuition | chapter-timed | no: Meerkat's row 1 covers the physics mapping in a minute |
| stats 4 good (open web) | How Hamiltonian Monte Carlo Works | yRymPs9si6g | 2026-09-06 | none | none | no chapters, captions not requested | no: nothing could be timed |

No listed creator has a Hamiltonian Monte Carlo video (ritvikmath's channel search found none).

### Leapfrog (`leapfrog`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Meerkat Statistics (open web) | MCMC 5, Hamiltonian Monte Carlo (HMC) | 2zr7cMjltAI | 2025-08-27 | 3:08 to 6:38 ("Simulating system dynamics") | Intuition | chapter-timed | yes, row 1 |
| Meerkat Statistics (open web) | same | 2zr7cMjltAI | 2025-08-27 | 6:38 to 7:16 ("Numerical integration") | Intuition | chapter-timed | yes, row 2 |
| Meerkat Statistics (open web) | same | 2zr7cMjltAI | 2025-08-27 | 8:49 to 10:33 ("Numerical error and acceptance") | Intuition | chapter-timed | yes, row 3 |
| Erik Lindahl (open web) | Lecture 06, concept 09: Integration using the leap-frog algorithm updates x & v | pVudb6-_FaM | not read | none | none | not read | no: a molecular-dynamics lecture, not checked |

### Skipped because the file already has the concept's table

Product of experts (`product-of-experts` at Step 1), acceptance probability (row 1 of `metropolis-hastings` at Step 6) and annealing (`annealed-langevin` at Step 9).

## Phase 3: ideal experts, exact targets, and the seven kinds

A second caption request (Serrano's mixture video) was refused with HTTP 429 at 12:51:07, so phase 3's new rows are chapter-timed except those from captions already cached.

### The grid (`grid-of-cells`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Professor Leonard | Calculus 3 Lecture 14.1, Introduction to Double Integrals | lv_awaaT6gY | 2016-04-12 | 10:43 to 30:15 ("Concept of double integrals") | Intuition | chapter-timed | yes, row 1. The same row sits in the index's `double-and-iterated-integrals` source; that block's other four rows (iterated integrals, swapping the order, a substitution) do not serve this section, so it was not copied |
| Professor Leonard | same | lv_awaaT6gY | 2016-04-12 | 35:35 to 47:03 ("Numerical approximation example") | Worked example | chapter-timed | yes, row 2 |
| Professor Leonard | Calculus 1 Lecture 4.3, Area Under a Curve, Limit Approach, Riemann Sums | F0uuW-I6icY | 2012 (not read) | not read | Worked example | not read | no: one dimension, where Leonard's 14.1 does the two-dimensional grid this section uses |

### Responsibility (`responsibility`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Luis Serrano | Gaussian Mixture Models | q71Niz856KE | 2020-12-28 | 1:56 to 3:36 ("Hard clustering - soft clustering") | Intuition | chapter-timed (captions refused, HTTP 429) | yes, row 1: a point belonging partly to each cluster |
| Luis Serrano | same | q71Niz856KE | 2020-12-28 | 3:36 to 6:10 ("Step 1: Colouring points") | Intuition | chapter-timed | yes, row 2 |
| ritvikmath | Gaussian Mixture Model | EWd1xRkyEog | 2019-02-04 | 8:47 to 10:13 | Proof | caption-read (cached from an earlier session) | yes, row 3: γ, the probability that one observation came from class k, salmon or tuna, written with Bayes' theorem as π_k times the class-k normal over the same sum for every class |
| ritvikmath | same | EWd1xRkyEog | 2019-02-04 | 12:29 to 13:19 | Intuition | caption-read | no: the E-step loop, which fitting does and this section does not |
| Stanford CS229 | Spring 2026, Lecture 9, K-Means and GMM (non-EM) | bSmIGBCoffA | 2026-07-31 | none | none | no chapters, captions not requested | no: nothing could be timed |

### The perfect sampler (copy of `inverse-transform-sampling`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| ritvikmath | Inverse Transform Sampling, Data Science Concepts | 9ixzzPQWuAY | 2019-10-06 | 0:07 to 10:04, four rows | Intuition, definition, worked example | chapter-timed | yes: the index's `inverse-transform-sampling` source copied whole, since none of its rows was linked in the walk. It teaches drawing exact samples from a known distribution, which is what the perfect sampler does from the grid table |

### Disjunction (`disjunction-as-a-mixture`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Yilun Du (open web) | PhD thesis defense | 7jvWDF6ZXPc | 2024-08-28 | 25:11 to 25:22 | Intuition | caption-read | yes, row 1: the log-sum-exp of two energies samples p1(x) plus p2(x) |

### Negation (`negation-as-set-difference`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Yilun Du (open web) | PhD thesis defense | 7jvWDF6ZXPc | 2024-08-28 | 23:21 to 23:36 | Intuition | caption-read | yes, row 1: subtracting one energy from another leaves low energy in the first set and high energy elsewhere |

### No segment found

Ideal expert, exact, sampling-noise baseline, co-occurrence, contention, dominance and interaction. The first two are the walk's own names for a kernel density estimate and its closed-form noising, whose tables (`kernel-density-estimate`, `sum-of-independent-gaussians`) already sit in this file. The sampling-noise baseline is on the phase 1 gap list. The four composition kinds are the paper's taxonomy, and no video names them; dominance is a product of two Gaussians, but every row of the walk's `product-of-two-gaussians` source is already linked in phases 2 and 4, so it could not be repeated here.

### Skipped because the file already has the concept's table

Kernel density estimate (Step 1) and intersection (`product-of-experts` at Step 5).

## Phase 4: the parts world

No caption request was made for phase 4, since the block was still in place at 12:51. The CME296 lecture 1 rows are read from captions cached by the phase 1 sweep.

### The pixel (`pixel`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Stanford CME296 | Lecture 1, Diffusion | tr-CUpw--ck | 2026-04-10 | 27:09 to 27:59 | Intuition | caption-read | yes, row 1: an image is height by width pixels, each holding red, green and blue values |
| Stanford CME296 | same | tr-CUpw--ck | 2026-04-10 | 32:58 to 33:23 | Intuition | caption-read | yes, row 2: values 0 to 255 rescaled to between −1 and 1 and treated as continuous |
| Stanford CME296 | Lecture 4, Latent Space and Guidance | WUUq6TVAu8U | 2026-04-28 | 7:05 to 12:39 ("Pixel space") | Intuition | chapter-timed | no: lecture 1's caption-read rows cover the same skill |

### The image as a vector (`image-as-a-vector`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Stanford CME296 | Lecture 1, Diffusion | tr-CUpw--ck | 2026-04-10 | 28:04 to 28:52 | Intuition | caption-read | yes, row 1: the image becomes one vector of length pixels times 3, and x_0 and x_t are such vectors |

### The diagonal Gaussian (`diagonal-gaussian`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Stanford CS229 | Autumn 2018, Lecture 5, GDA and Naive Bayes | nt63k3bfXS0 | 2020-04-17 | 8:11 to 18:57 ("Multivariate Gaussian") | Intuition | chapter-timed | yes, row 1 (also row 1 of the index's `multivariate-normal` source) |
| ritvikmath | The Covariance Matrix, Data Science Basics | 152tSYtiQbw | 2019-09-20 | 2:40 to 5:23 ("The Covariance Matrix") | Intuition | chapter-timed | yes, row 2 (also row 2 of `multivariate-normal`) |
| Stanford CME296 | Lecture 1, Diffusion | tr-CUpw--ck | 2026-04-10 | 29:30 to 30:46 | Intuition | caption-read | yes, row 3: a Gaussian over vectors, its covariance usually simplified to σ² times the identity, a diagonal matrix |
| ritvikmath | same as row 2 | 152tSYtiQbw | 2019-09-20 | 5:23 to 10:59 ("Calculating Covariance") | Worked example | chapter-timed | no: computing a covariance from data, which this section does not do. This is why the `multivariate-normal` source was not copied whole |
| mathematicalmonk (open web) | (PP 6.2) Multivariate Gaussian, examples and independence | j2RYo0-fMTM | 2011-06-29 | none | none | no chapters, captions not requested | no: nothing could be timed, though its title names the independence a diagonal covariance gives |
| Herman Kamper (open web) | Multivariate Gaussian distribution | azrTdjrA2bU | 2021-03-16 | 3:39 to 5:14 ("Special Gaussian cases") | Intuition | chapter-timed | no: CME296's caption-read row covers the diagonal case from a listed creator |

### Precision (`precision`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Machine Learning & Simulation (open web) | Precision vs Variance/Standard Deviation for the Normal/Gaussian distribution | 2M9VO_52NpQ | 2021-04-30 | 1:15 to 2:04 ("Normal by Precision", "Defining the Precision") | Intuition | chapter-timed | yes, row 1 |
| same | same | 2M9VO_52NpQ | 2021-04-30 | 2:17 to 2:47 ("Why the precision?") | Intuition | chapter-timed | yes, row 2 |
| same | same | 2M9VO_52NpQ | 2021-04-30 | 2:47 to 4:37 (three Gaussians, wide and narrow, mean unchanged) | Worked example | chapter-timed | yes, row 3 |
| Robert Ghrist (open web) | CalcBLUE 3, Ch. 12.6, BONUS! Precision & Information | UE09AtcVYgw | 2019-01-04 | none | none | no chapters, captions not requested | no: nothing could be timed |

No listed creator has a video on precision.

### Underflow and the log-sum-exp (`underflow-and-log-sum-exp`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Bert Huang (open web) | Avoid Numerical EXPLOSIONS with the Log Sum Exp Trick | pZbkVup7fYE | 2021-02-27 | 6:55 to 9:16 ("Numerical stability issues") | Intuition | chapter-timed | yes, row 1 |
| same | same | pZbkVup7fYE | 2021-02-27 | 9:16 to 12:06 ("The log-sum-exp trick") | Proof | chapter-timed | yes, row 2 |
| same | same | pZbkVup7fYE | 2021-02-27 | 17:21 to 20:34 ("Numerical example") | Worked example | chapter-timed | yes, row 3 |
| mathematicalmonk (open web) | (ML 14.10) Underflow and the log-sum-exp trick | -RVM21Voo7Q | 2011-07-07 | none | none | no chapters, captions not reached after the 429 | no: nothing could be timed |
| Mathematics behind Data Science (open web) | Log-Probabilities for numerical stability, Naive Bayes, Lec 8 | zobQRQSrrlg | 2020-12-11 | 2:24 to 8:01 ("Numerical stability issues") | Intuition | chapter-timed | no: underflow of a product of probabilities, which Huang's row 1 covers beside the trick itself |
| Mayank Khanna (open web) | Log Probabilities | dIn2EknlpKU | 2021-09-10 | 0:20 to 3:15 ("Numerical Underflow") | Intuition | chapter-timed | no: the same skill as Huang's row 1 |

No listed creator covers the log-sum-exp trick.

### No segment found

The nat (searches for nats against bits found only a short lecture and a text-to-speech reading of an encyclopedia entry), mosaic, blend, summed-score error, projected total variation and component-weight error. Mosaic, blend and component-weight error are the walk's own names. The summed-score error is phase 2's noised-order error on the mosaic, and Du's caption-read segment for it is already placed in phase 2, so it could not be repeated here. Du's University of Washington talk (`kFchGKIrGzY`, chapter "Compositional generative modeling", 14:13 to 21:20) may cover it, but its chapter title does not say so and its captions were not requested. Projected total variation has no video that reduces samples to a lower-dimensional summary before comparing distributions.

### Skipped because the file already has the concept's table

Chimera (`product-of-two-mixtures` at Step 5).

## Phase 5: the overlap dial

A third caption request (CME295 lecture 4) was refused with HTTP 429 at 12:57:07, so phase 5's rows are all chapter-timed.

### Expectation (`expectation-as-a-weighted-sum`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| StatQuest | Expected Values, Main Ideas!!! | KLs_7b7SKi4 | 2021-03-29 | 5:09 to 8:02 ("Calculating an Expected Value") | Worked example | chapter-timed | yes, row 1 |
| StatQuest | same | KLs_7b7SKi4 | 2021-03-29 | 8:02 to 10:30 ("Expected Value notation") | Intuition | chapter-timed | yes, row 2 |
| StatQuest | same | KLs_7b7SKi4 | 2021-03-29 | 10:30 to 13:39 ("A more complicated example") | Worked example | chapter-timed | no: a second worked example of the same skill |
| StatQuest | Expected Values for Continuous Variables!!! | OSPr6G6Ka-U | 2021-06-22 | 7:03 to 12:18 | Proof | chapter-timed | no: the continuous case, which this step's count does not need |

The index's `expectation-and-marginals` source could not be copied: both of its CS229 rows are already linked in the walk.

### Tuned (copy of `tuning-a-metropolis-chain`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| DataMListic | Metropolis-Hastings, Explained | 7IpvsfRgTTY | 2026-01-26 | 2:00 to 2:34, two rows | Intuition | chapter boundaries checked against captions by the sweep that wrote the source | yes: the index's `tuning-a-metropolis-chain` source copied whole, since neither row was linked in the walk |
| Zamrik Quant Research (open web) | Reject Three Out of Four: The 23.4% Rule of MCMC | HZNi2wn2qSY | 2026-10-07 | none | none | no chapters, captions not requested | no: nothing could be timed |

### Pretrain (`pretraining`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Stanford CME295 | Autumn 2025, Lecture 4, LLM Training | VlA_jt_3Qc4 | 2025-10-21 | 7:19 to 13:26 ("Pretraining") | Intuition | chapter-timed (captions refused, HTTP 429) | yes, row 1 |
| Stanford CS224N | 2023, Lecture 9, Pretraining | DGfCRXuNA2w | 2023-09-20 | 14:56 to 19:39 ("Pre-training motivation") | Intuition | chapter-timed | yes, row 2 |

### Fine-tune (`fine-tuning`)

| Creator | Title | Id | Published | Start to end | Kind | Timing | Placed? |
|---|---|---|---|---|---|---|---|
| Stanford CS224N | 2023, Lecture 9, Pretraining | DGfCRXuNA2w | 2023-09-20 | 29:58 to 35:53 ("Pre-training and fine-tuning") | Intuition | chapter-timed | yes, row 1 |
| Stanford CME295 | Autumn 2025, Lecture 4, LLM Training | VlA_jt_3Qc4 | 2025-10-21 | 1:02:31 to 1:09:21 ("Supervised finetuning") | Intuition | chapter-timed | yes, row 2 |
| Stanford CS224N | same as row 1 | DGfCRXuNA2w | 2023-09-20 | 55:01 to 58:39 ("Parameter efficient fine-tuning") | Intuition | chapter-timed | no: this step fine-tunes every parameter |
| Stanford CME295 | Autumn 2025, Lecture 5, LLM tuning | PmW_TMQ3l0I | 2025-11-14 | 4:50 to 11:31 ("Preference tuning") | Intuition | chapter-timed | no: preference tuning, which this step does not do |

### No segment found

Overlap share, corner, sliver, interior mass, whole-quadrant mass, ghost and the nat. All but the nat are the walk's own names for regions and counts on its toy; the nat has no video, as in phase 4.

### Skipped because the file already has the concept's table

None of phase 5's name sections had a table for its own concept; the file's three older tables (`kernel-density-estimate`, `annealed-langevin`, `product-of-two-mixtures`) sit at step level.

## Counts

Phase 2: 8 tables, 18 rows. Phase 3: 5 tables, 11 rows. Phase 4: 5 tables, 12 rows. Phase 5: 4 tables, 8 rows. 49 rows in 22 tables. 12 rows are caption-read: 7 from Du's thesis defense, 4 from CME296 lecture 1 and 1 from ritvikmath's mixture video. The other 37 are chapter-timed, including DataMListic's two copied rows, whose chapter boundaries the sweep that wrote that source checked against captions.
