# Two toy walks, reconciled

This file lines up what two walks claim about the same five ideas, so either walk can be corrected from the cited line rather than from memory. Side A is this repo's six-checkpoint toy: red circles and blue squares on 8 × 16 RGB pictures, each word's model a Gaussian mixture around its training pictures with σ = 0.1 on every pixel. Side B is the PyTorch rebuild of Du et al.'s 2D notebook, extended with the quadrant toy and a 16-pixel parts world, brought in from `reduce-reuse-recycle`. Each claim below is marked **agrees** (same conclusion, with a note on whether the numbers can be compared across the two toys), **disagrees** (both numbers quoted), **A only** or **B only**. A ⚑ marks a claim that one side states as settled while the other shows it depends on a setting.

## Words this uses

The evidence column cites files by these short names, with a line number after the colon. A JSON file is cited by its key.

| Short name | File, from the repo root |
|---|---|
| `checkpoints` | `plans/10-what-poe-loses-when-it-composes/today-the-toy-failure-in-six-checkpoints.md` |
| `write-up` | `plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md` |
| `route` | `plans/10-what-poe-loses-when-it-composes/route-structural-can-a-toy-reproduce-the-hybrid.md` |
| `dispatch` | `artifacts/drips/toy-failure-six-checkpoints/dispatch.md` |
| `first-run.json` | `artifacts/notes/what-poe-loses-when-it-composes/toy-sampler-hybrids.json` (and `-B-strong-mala.json` beside it) |
| `first-run.py` | `artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py` |
| `ddim.json` | `artifacts/notes/what-poe-loses-when-it-composes/fig-ddim-guided.json` |
| `rebuild` | `artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md` |
| `walk05` | `artifacts/drips/simple-distributions-rebuild/05-the-overlap-dial.md` |
| `ladder02`, `ladder04`, `ladder10`, `ladder11` | `artifacts/drips/simple-distributions-rebuild/ladder/02-composing-two-experts.md`, `04-the-parts-world.md`, `10-latent-space-and-vaes.md`, `11-the-text-conditioned-testbed.md` |
| `overlap-map`, `overlap-compiled` | `artifacts/ideas/poe-overlap-dial/IDEA_MAP.md`, `COMPILED.md` |
| `a-and-b-map` | `artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md` |
| `testbed-map` | `artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md` |
| `out/` | `toy/rebuild/out/` |

**Construction (1) and construction (2)** are B's names and are used below because A has none. Construction (1) is one conditional model run as score(x|A) + score(x|B) − score(x), which targets p(x|A)·p(x|B)/p(x); at guidance w it is A's guided sum. Construction (2) is two separate experts multiplied, p_A·p_B, which is A's exact product in checkpoints 3 to 5.

## Same thing, two names

| Side A says | Side B says | What it is |
|---|---|---|
| caption dial | overlap dial | how many of each word's training pictures already show both objects |
| s, the pair's share of each word's pictures | f (overlap), r (A∩B data share, construction (1)), ε (leftover, construction (2)) | the setting of that dial, 0 to 1 |
| "pair captioned under each name" / "singles only" | "contains" labels / "exactly" labels | whether a both-objects picture trains each single word's model |
| the model of a word | an expert; an ideal expert; a Gaussian-mixture expert | the density learned from one word's pictures |
| exact product | exact composition, exact target, answer key | the product computed in closed form, with no sampler |
| overlay, purple overlay, hybrid | blend; chimera (2D) | a same-place average of the two objects. B's blend encodes cat +1 and dog −1, so its average is 0, the background value, where A's is a visible purple |
| "pale, apart" (a different-place average) | ghost, half-means (0.5, −0.5) | each object averaged with the other expert's background, in different places. B's "ghost" is A's "pale, apart", not A's overlay |
| "apart", the two shapes apart, the pair | A∩B, two animals, co-occurrence, side by side | both objects at full strength, separately |
| sum, OR | mixture, disjunction | pick one expert, then sample it |
| guided sum, "as SDXL runs PoE" | construction (1) tilted by w, p(x)·p(A\|x)^w·p(B\|x)^w | ε∅ + w(ε_A − ε∅) + w(ε_B − ε∅) |
| error A (the sampler) / error B (the target) | summed-score error, sampler failure / missing-data failure, data failure | the two sources of a wrong picture |
| Du's reverse diffusion, ancestral steps | reverse diffusion with summed scores | DDPM steps on the sum of the noised experts' scores |
| MALA "tuned until it accepts about half" | tuned MALA (adapted to 0.6 acceptance) | Langevin with an accept step whose step size adapts |
| nearest-of-16 library readout, dot map | interior or deep A∩B mass, share, total variation (TV) | the number a result is read by |
| σ = 0.1 on every pixel | object variance 0.01 (σ = 0.1) and a separate background variance, 0.01 or 1.0 | how tight each expert is; B splits object pixels from background pixels |

## The structural route's "already exists on the cluster" table, checked

`route:315-322` claims six correspondences. Each is checked against side B.

| Route row | What B's files say | Mark | Evidence |
|---|---|---|---|
| Du's 2D product and sum, rebuilt (`compose.py`, `phase2_compose.py`) | ring × bar, left + right and box(0.7) not box(0.3), matched by picture and measure | holds | `rebuild:50` |
| parts world, "the hybrid reproduced exactly (claim 9)" | claim 9 says mosaic, blend or two animals; the same-place fusion appears only in the single-placement blend, and with placements varying the tight-background product picks the different-halves ghost (A's "pale, apart") | holds with a correction ⚑ | `overlap-map:17`, `walk05:836`, `ladder04:155-157` |
| ideal experts and samplers, "ULA, MALA, UHA with their tests" | the 6 tests cover the MALA divisor and the CHA fix only; ULA and UHA are checked by matching the notebook, not by tests | overclaims slightly | `rebuild:49`, `rebuild:51` |
| "contains" against "exactly" labels, claim 1.2, the caption dial's first evidence | 1.00 of the composed mass in A∩B under "contains" against 0.08 under "exactly" | holds | `overlap-map:71` |
| chimera mass by position, run 1, "centred shapes fuse more than shapes at varied places" | 0.66 centred against 0.21 position-varied, in 2D | holds for B's 2D toy, fails on A's own toy (row 1) ⚑ | `overlap-map:163`, `checkpoints:188` |
| the text-conditioned testbed, layer 11, "the cluster home for phase 2" | layer 11 is "not built" and "to be designed in that walk" | overclaims: it is a design, not a thing that exists | `rebuild:61`, `ladder11:5` |

Two provenance slips sit outside the table. `route:335-336` and `write-up:1064` credit "the cluster's layer 10" with finding that a plain product changes under an invertible warp, but layer 10 is not built (`rebuild:60`). The finding is run 6 of `a-and-b-map:169`. And `route:322` names the cluster as phase 2's home while `route:165` and `route:176` train phase 2 on the laptop's 8 GB RTX 4070.

## Row 1: the exact product of mixture experts

Side A: checkpoints 1 to 3 and the one-pixel section. Side B: layer 4, the parts world.

| Claim | Side A | Side B | Mark | Evidence |
|---|---|---|---|---|
| Two equal bells multiply into one bell at the average, narrower by √2 | one pixel: peak (0.5, 0, 0.5), sd σ/√2 = 0.0707 | blend: mean 0, sd 0.0707, 10 sds from each expert | agrees, numbers comparable (both σ = 0.1) | `write-up:894-904`, `dispatch:31`; `ladder04:78`, `out/phase4/numbers.json` b.product_sd |
| With unequal certainty the mean leans to the surer expert | derived (precision-weighted v*); checkpoint 8 planned, not run | computed: mosaic head 99/101 = 0.9802 with precisions 100 and 1 | agrees, qualitative; only B has a number | `write-up:914-919`; `ladder04:78`, `ladder04:143-144` |
| A product of mixtures is a mixture over every pair, weighted by how alike the pair is | w_jk = shares × e^(−d²/4σ²) | w_ij ∝ π_i ρ_j N(μ_i; ν_j, Σ_i + Λ_j) | agrees; A's form is B's with equal diagonal σ | `write-up:821-824`, `dispatch:36`; `ladder04:150-152` |
| With no pair in training, the exact product fuses the two objects | 200 of 200 same-place overlays; d² 48 same place, 80 elsewhere; others smaller by e^−800 | tight background, 0% overlap: A∩B 0.000, the main component a ghost at half-means (0.5, −0.5), which is two half-strength objects in different halves | **disagrees** on which picture wins: A's product picks the same-place average, B's picks the different-place average (A's "pale, apart"). Same mechanism, opposite winner, set by how the two objects are encoded in pixels ⚑ | `checkpoints:160`, `checkpoints:188`, `dispatch:37`; `walk05:836`, `walk05:965`, `out/phase5/overlap-dial.json` parts_world background_variance 0.01 overlap 0.0 |
| Why that pairing wins | overlapping red and blue share a zero green channel, so same-place d² (48) is below different-place (80) | cat +1 and dog −1 on the same pixels mismatch by 2, so same-half pairs cost 8 × 4 against different-half 16 × 1 | B only for the encoding dependence; A never varies the encoding ⚑ | `checkpoints:160`; `ladder04:155-157` |
| A loose background turns the product into two objects with no pair in training | not tested; σ = 0.1 on white pixels throughout | background variance 1.0: A∩B weight 1.000 at 0% overlap; side-by-side pairs 0.5 each, same-half pairs underflow to 0 | B only; contradicts A's general wording "the product draws a purple overlay" ⚑ | `checkpoints:84`, `write-up:826-828`; `overlap-map:10`, `walk05:889`, `out/phase4/numbers.json` c |
| Varying where each object appears reduces fusion | three positions per shape still give 200 of 200 overlays, because the product selects the same-place pairings | chimera mass 0.66 centred against 0.21 position-varied (2D) | **disagrees** ⚑ (B's 2D slot has one position per sample; A's mixture can pick same-place pairs among nine) | `checkpoints:188`, `route:321`; `overlap-map:163`, `overlap-compiled:15` |
| The product needs noise to exist at all | at σ = 0 the product is zero everywhere and cannot be normalised | Gaussian experts are never exactly zero, so "no support" means too small to matter | agrees, qualitative | `write-up:928-933`; `a-and-b-map:167` |
| Sharper experts do not fix the fusion | shrinking σ narrows the overlay bell | not stated | A only | `write-up:924` |
| The sum (OR) never averages | exact sum: 88 circles, 112 squares; pixel [4, 8] red 26, blue 34, white 140, purple 0 | mixture left + right: right bar's share 0.4223 trained exact, reverse diffusion 0.4875, perfect sampler 0.4315 | agrees, qualitative | `checkpoints:255-256`; `out/phase2/phase2.json` mixture |
| Samplers reproduce a fused exact answer | first run, singles only: reverse diffusion 298 of 300 overlays, MALA 300 of 300 | blend: reverse diffusion TV 0.016, tuned MALA 0.008, exact-draw floor 0.007 | agrees, qualitative | `first-run.json` A; `out/phase4/phase4.json` b blend |
| The toy's exact rule matches what a small trained network learns | "the cluster's run 2 found small trained models match it closely" | trained 2D MLPs within 0.04 nats of ideal; contention chimera 0.652 trained against 0.663 ideal | agrees; A borrows B's number | `write-up:812-815`; `overlap-map:164` |

## Row 2: the caption dial and the overlap dial

Side A: checkpoints 4 and 5, write-up §9.2. Side B: layer 5.

| Claim | Side A | Side B | Mark | Evidence |
|---|---|---|---|---|
| With the pair in both words, any small share flips the exact product to the objects apart | 200 of 200 apart at 1 in 100, 1 in 20, 1 in 5, half; pair × pair leads the best overlay by 9s²/(1−s)² e^1200 | empty background: 0.000 at 0%, 1.000 from 0.1%; the matching pair wins by 386.2 nats | agrees, numbers comparable in kind (both discrete mixtures at σ = 0.1; the lead differs by geometry, 1200 against 386 nats) | `checkpoints:266`, `checkpoints:279-281`; `walk05:862`, `walk05:889` |
| The dial is a switch | "at σ = 0.1 the dial is a switch"; softens as σ grows (pair share at s = 0.01: 1.0 at σ 0.5, 0.97 at σ 1, 0.005 at σ 2) | parts world: a switch. Quadrant toy: a ramp, deep A∩B 0.00 at 0%, 0.05 at 1%, 0.30 ± 0.10 at 5%, 0.76 at 100%, rising as f² | agrees on the parts world; the quadrant ramp shows the switch depends on the pair being one discrete picture rather than a region ⚑ | `checkpoints:281`, `dispatch:51`, `write-up:1177-1181`; `walk05:704`, `walk05:635-700` |
| Crossover share where the pair stops winning | s* ≈ e^(−48/8σ²)/√3: 1.4 × 10⁻³ at σ 1, about 0.15 at σ 2 | construction (1) keeps 0.989 down to r = 0.001, then 0.9855 at 10⁻⁴, 0.7364 at 10⁻⁶, 0.0935 at 10⁻⁸ | agrees in kind; not comparable (A's plain product, B's construction (1) in 2D) | `dispatch:51`; `a-and-b-map:165` |
| The pair must be in both words | pair under "circle" only: apart share equals s, 15 of 200 at 1 in 20, 112 of 200 at half | not tested one-sided | A only | `checkpoints:282-284` |
| Labels decide it | the dial is the label: s = 0 gives 0 apart, s > 0 gives 200 of 200 | "contains" 1.00 against "exactly" 0.08 | agrees, qualitative | `checkpoints:279-281`, `write-up:1163-1164`; `overlap-map:71` |
| Construction (1) and construction (2) reach A∩B differently | the guided target "has no exact sampler", so it is read against the exact plain product | computed both exactly in 2D: construction (1) 0.989 in the A∩B blob; separately trained experts cap at 0.5 by symmetry | B only for the exact construction (1) answer; A's comparison is mixed by its own admission | `write-up:1031-1032`, `checkpoints:330-332`; `a-and-b-map:8-9`, `a-and-b-map:165` |
| Support inherited from pretraining survives fine-tuning | not tested; §9.2 plans to fine-tune with circle × square held out and hopes it generalises | exact A∩B mass falls from 0.159 to 0.0009 within 30 fine-tune steps at 0% overlap; samplers under 0.01 by 100 | B only; bears on A's held-out-pair remedy ⚑ | `write-up:1190-1199`, `route:213-217`; `overlap-map:10`, `walk05:1074`, `out/phase5/finetune.json` |
| The remedy can win in a way that teaches nothing | the pair put under both names is found because it was put there | not stated | A only | `write-up:1172-1175` |
| What the 0% product holds in the quadrant | (no quadrant toy) | three different numbers in B's own files: 0.12 "in the crossing", 0.20 ± 0.02 in the whole quadrant, under 1.1 × 10⁻⁶ deep | B only, and B disagrees with itself (different regions, not labelled as such in `overlap-compiled`) | `overlap-compiled:29`; `walk05:293`; `overlap-map:10` |

## Row 3: samplers against the exact answer

Side A: checkpoint 6, write-up §3.3 and §8.5. Side B: layer 2 and step 5 of walk 05.

| Claim | Side A | Side B | Mark | Evidence |
|---|---|---|---|---|
| Summed noisy scores are not the score of the noisy product | one pixel at ᾱ 0.5: same centre 0.354, variance 0.5025 exact against 0.2525 summed | parts world at ᾱ 0.5: head mean 0.6931 exact against 0.2326 summed | agrees; A's equal-width case moves the variance, B's unequal-width case moves the centre. Both use noised variance 0.505 for a σ = 0.1 bump | `checkpoints:292`, `write-up:222-230`; `ladder04:82`, `out/phase4/numbers.json` d, `out/walk-numbers.json` p2.noised_A_var_abar0.5 |
| Error A softens the dial at high noise | at s = 1 in 20, summed scores put 1.000 on apart at ᾱ 0.9, 0.997 at 0.5, 0.015 at 0.1, 0.003 at 0.01 | the product is one blob at high noise and its piece weights only become clear at low noise, when local moves cannot carry mass | agrees, qualitative; only A has a closed form | `checkpoints:292`; `walk05:708` |
| Reverse diffusion on the plain product, pair included | first run (16 × 32 pictures, s = 0.1): 14 of 300 apart, 281 overlays; "the failure is in the sampler" | parts world: 0.734 at 0.1%, 0.885 at 1%, 0.969 at 5%, 0.996 at 20%. Quadrant: 0.02 against exact 0.30 at 5% | **disagrees** with the parts world (about 0.05 against 0.97 at comparable shares), agrees with the quadrant ⚑ depends on dimension (1,536 against 16 numbers) and how far the fused picture sits from the pair | `first-run.json` B, `write-up:1046-1048`; `walk05:889`, `out/phase5/overlap-dial.json`, `overlap-map:12` |
| Reverse diffusion on construction (1), pair included | guided-sum DDIM at s = 1 in 20, w 1: 17 of 200 apart, 115 overlay, 68 pale apart | reverse diffusion at r = 0.05: top-right share 0.983 against exact 0.989; at r = 0.001 it fails, 0.501 | **disagrees** (0.085 against 0.983 at a 5% share); mixed comparison: deterministic DDIM against ancestral steps, 384 numbers against 2 ⚑ | `checkpoints:321-326`, `ddim.json`; `a-and-b-map:168`, `a-and-b-map:170` |
| Where the exact answer is fused, every sampler is fused too | the check expects it, but the guided sum with the pair excluded draws 122 overlays and 78 pale apart against 200 overlays exact | blend: every sampler within 0.03 TV of the floor | A disagrees with its own check; B agrees for the plain product | `checkpoints:314`, `checkpoints:323-324`; `out/phase4/phase4.json` b blend |
| Tuned MALA and UHA reach the exact answer | not measured yet; first run's MALA was "weak and untuned" (acceptance 0.377; 0.0026 at the strong setting), 14 and 16 of 300 apart | parts world 0.1%: tuned MALA 0.995, UHA 1.000. Quadrant: lag at 5% (0.17 and 0.18 against 0.30), reach from 20% up (0.55 against 0.59) | B only for tuned numbers ⚑ settled on the parts world, not on the quadrant | `write-up:1052-1053`, `first-run.json` mala_acceptance; `walk05:736`, `walk05:889` |
| A fixed MALA step of 0.05 × the noised variance | first run: acceptance 0.377 at this step, in 1,536 dimensions | "accepts nearly every move and behaves like ULA", in 2 dimensions | **disagrees** (0.377 against nearly 1), explained by dimension ⚑ | `first-run.json` mala_step_fraction, mala_acceptance, `first-run.py:129`; `ladder02:244`, `ladder02:454` |
| The acceptance to tune to | about half | 0.6 | disagrees, small | `checkpoints:304`; `ladder02:370` |
| MALA tells whether a failure is the sampler's | "the tool for reaching the exact product, so it tells whether a failure is the sampler's" | only where the exact answer is known; tuned samplers lag on separated modes at low overlap; dominance (unequal mode weights) defeats every sampler, 0.30 against a 0.11 floor | B qualifies what A states as settled ⚑ | `checkpoints:304-305`, `write-up:1028-1030`; `overlap-map:12`, `overlap-map:167`, `overlap-compiled:31` |
| The step count matters at high guidance | 50 DDIM steps at w 7.5 overshoot to single shapes (123 circles, 77 squares); 200 and 1,000 settle | SDXL itself runs 50 DDIM steps at 7.5 | A only for the overshoot; ⚑ A's broken setting is SDXL's own | `checkpoints:333-336`, `ddim.json`; `ladder11:40` |
| One number is not enough to say it worked | readout is a count of library pictures | share, TV to the exact density and Z together; a share of 0.983 against 0.989 hid a TV 0.049 above the floor | B only; A reports counts alone | `checkpoints:88`, `route:187`; `a-and-b-map:11` |
| The MALA accept step is computed per sample with the right divisor | first run's MALA: per-sample accept, divisor 4 × step | the repository's original had the wrong divisor (0.5 against 2.0); fixed, 6 tests pass | agrees | `first-run.py:134-137`; `rebuild:49` |
| The sum is sampled correctly by summed scores | first run row C: reverse diffusion 140 and 160 against exact 145 and 155 | "reverse diffusion misses the product, except for mixtures" | agrees, qualitative | `first-run.json` C; `overlap-compiled:20` |
| The trained experts change the sampler verdict | not tested | MALA's gain over reverse diffusion shrinks with trained experts: an interaction of +0.054 TV at r = 0.05 and +0.232 at r = 0.001 | B only; bears on A's phase 2 | `a-and-b-map:170` |

## Row 4: latent space

Side A: the structural route's "Later" section and write-up §8.6. Side B: layer 10 and claim 10 of the A-and-B map.

| Claim | Side A | Side B | Mark | Evidence |
|---|---|---|---|---|
| The plain product changes under an invertible change of space | cites it from "layer 10" | TV 0.035, 0.080, 0.126 at largest area stretches 2.6, 9, 49 | agrees; A's provenance is wrong (layer 10 not built; it is run 6) | `route:335-336`, `write-up:1064`; `a-and-b-map:16`, `a-and-b-map:169`, `rebuild:60` |
| A composition whose exponents sum to 1 is unchanged by an invertible change of space | not stated | construction (1) and negation agree across the warp to 4 × 10⁻¹⁷ and 7 × 10⁻¹⁷ TV | B only. A's guided sum has exponents (1 − 2w) + w + w = 1 at every w, so it falls under this ⚑ | `a-and-b-map:16`, `ladder10:5`, `ladder10:70-71` |
| The latent's organisation decides the picture | pixel-like latent gives a ghost overlay, disentangled gives one in-between object, one slot per object gives two objects | the latent matters only for the plain product, or through a non-invertible decoder; "optional" for guided compositions | **disagrees** on importance ⚑ (A's prediction can hold for the plain product, or through what the network learns, not through the guided target under an invertible encoder) | `write-up:1083-1090`, `route:332-334`; `ladder10:5` |
| A VAE decoder is not invertible | not stated | its volume factor is √det(JᵀJ), marked "likely, verify" | B only | `a-and-b-map:113`, `ladder10:73` |
| A disentangled latent can be built | planned: shape, colour and position as separate directions | unsupervised disentanglement is not identifiable without supervision (Locatello et al. 2019), so a β-VAE may not separate A and B | B only, a warning to A's plan | `route:333-334`; `a-and-b-map:138` |
| SDXL's fused animal comes from the denoiser's density, not from the VAE | averaging two SDXL latents should decode as a ghost; tested by the VAE-averaging walk | not stated | A only | `write-up:979`, `checkpoints:395-396` |

## Row 5: the text-conditioned testbed

Side A: the structural route's phase 2 and write-up §8.6 to §8.8. Side B: layer 11 and the SDXL-shaped testbed map.

| Claim | Side A | Side B | Mark | Evidence |
|---|---|---|---|---|
| One small caption-conditioned denoiser, like SDXL | a U-Net on 8 × 16 pictures, caption through a learned word embedding, condition dropout | a UNet on 32 × 32 shapes, text encoder over a small vocabulary, cross-attention at two resolutions | agrees in kind; sizes differ | `route:163-165`, `write-up:1077`; `testbed-map:55` |
| The pooled text embedding added to the time embedding is a route | switch 1, global conditioning | listed among SDXL's parts | agrees | `write-up:1125`; `testbed-map:55` |
| How word similarity enters | a learned word embedding | a jointly trained encoder cannot give SDXL's cat-dog closeness; use a frozen table with the cosine set by hand | **disagrees** | `route:164`; `testbed-map:10`, `testbed-map:77` |
| An attention reading alone shows the cause | locality test reads Δ_circle and then edits the square's region; switches turn routes off | every probe must be paired with an intervention | agrees (A's check 2 and switches are interventions) | `write-up:1103-1109`, `write-up:1123-1130`; `testbed-map:11` |
| How outputs are scored | nearest of 16 library pictures and the dot map | a parser back to drawing parameters, trusted above 99% on 1,000 held-out draws and 200 hand-built hybrids | agrees in kind; B only for a bar on the scorer | `checkpoints:88`; `ladder11:128` |
| An exact answer exists for the trained model | yes: trained on 7 fixed pictures, so the exact rule is known | no exact density in pixel space; score in drawing parameters | not a conflict: A's fixed picture set keeps an exact target, B's procedural shapes do not | `route:166-167`, `write-up:812-815`; `testbed-map:8`, `testbed-map:69` |
| The toy's hybrid has SDXL's cause | "Yes, the same mechanism, in a different space"; "the toy shows the mechanism exactly", with the caveat that SDXL mixes in sampler error | wrong as stated: SDXL's cause is not established; the interaction lower bound did not track which of 25 pairs blend | **disagrees** on how settled this is ⚑ | `write-up:975-987`; `testbed-map:7`, `testbed-map:63` |
| Guidance weight 7.5 has been measured on a toy | yes: guided-sum DDIM at w 1 and 7.5 | "the 2D walk measured construction (1) only at w = 1"; layer 6 not built | A only (A has gone past B here) | `checkpoints:321-326`; `testbed-map:63`, `rebuild:56` |
| Candidate causes switched one at a time | global conditioning, cross-attention masking, self-attention, Bradley's patch-local model, position words, merge point | single-object captions, guidance weight, word similarity, shared attention pixels, the sampler | agrees on method; the lists overlap on cross-attention only | `write-up:1123-1130`; `testbed-map:55`, `rebuild:61` |
| A gate before any switch | the base network must reproduce checkpoints 3 and 5 and draw both shapes for the joint caption | the smallest model must reach 95% parser-correct single-concept samples in under 30 minutes | agrees in kind; different gates | `route:188-189`; `ladder11:62-64` |
| A cause counts only if SDXL's cache carries its signature | not stated | required for every candidate cause | B only | `testbed-map:7`, `testbed-map:65` |
| Where it runs | laptop RTX 4070 (`route:176`) but "the cluster home" (`route:322`) | `co3` on the cluster | disagrees (A disagrees with itself) | `route:165`, `route:176`, `route:322`; `ladder11:62` |

## Disagreements to settle

Ordered by how much each would change either walk.

1. **Which picture the exact product fuses into depends on how the two objects are encoded and how tight the background is.** A's product picks the same-place overlay (d² 48 against 80, 200 of 200 overlays, `checkpoints:160`, `checkpoints:188`) and the write-up states it generally (`write-up:826-828`). B's parts world, at the same σ = 0.1, picks the different-halves ghost at half-means (0.5, −0.5) with a tight background (`walk05:836`) and two full animals with a loose one (A∩B 1.000 at 0% overlap, `walk05:889`). B's "ghost" is A's "pale, apart", so B's claim 9 that the parts world "replicates the cat-dog hybrid exactly" (`overlap-map:17`) holds only for its single-placement blend. To settle: rerun checkpoint 3 with the background at σ 1.0, and the parts world with the cat and dog encoded so they do not cancel.
2. **Whether reverse diffusion finds the pair once it is in training differs by a factor of about 20 between the toys.** A's first run draws 14 of 300 apart at s = 0.1 (`first-run.json` B) and its guided-sum DDIM 17 of 200 at s = 0.05 (`checkpoints:325`). B's parts world reverse diffusion reaches 0.969 at 5% (`walk05:889`) and its construction (1) reverse diffusion 0.983 at r = 0.05 (`a-and-b-map:168`). A concludes "here the failure is in the sampler" (`write-up:1046-1048`); B's layer 12 carries "samplers recover a third to four fifths of the exact mass" into Stable Diffusion (`rebuild:62`). To settle: run B's samplers on A's 8 × 16 toy at checkpoint 5's 1 in 20, and A's error-A closed form on the parts world, so dimension is the only thing that changes.
3. **How settled it is that the toy's hybrid shares SDXL's cause.** A writes "Yes, the same mechanism" (`write-up:975`); B's testbed map marks the same claim wrong as stated because SDXL's cause is not established (`testbed-map:7`, `testbed-map:63`). To settle: A's write-up states it as a hypothesis with B's signature-matching test as its check.
4. **Whether varying object position reduces fusion.** B: chimera mass 0.66 centred against 0.21 position-varied (`overlap-map:163`), cited by A as given (`route:321`). A's own toy with three positions per shape gives 200 of 200 overlays (`checkpoints:188`). To settle: the route table's row is reworded as B's 2D finding, with A's count beside it.
5. **The guided target does not depend on the latent's organisation under an invertible encoder.** Its exponents sum to 1 at every w, and B measured that such a composition is unchanged by a warp to 4 × 10⁻¹⁷ TV (`a-and-b-map:16`). A's latent stage predicts a ghost, an in-between object or two objects by organisation (`write-up:1086-1090`, `route:332-334`). To settle: A's table names which operator each prediction is for (the plain product, or the guided sum through a non-invertible decoder).
6. **What MALA can and cannot tell.** A treats it as the arbiter of "is this the sampler's failure" (`checkpoints:304-305`). B shows tuned MALA lags at low overlap (0.17 against 0.30 at 5%, `walk05:736`) and that dominance defeats every sampler (`overlap-compiled:31`). The fixed step that B says accepts nearly every move (`ladder02:244`) accepted 0.377 in A's 1,536 dimensions (`first-run.json`). A tunes to about half acceptance (`checkpoints:304`), B to 0.6 (`ladder02:370`). To settle: checkpoint 6 tunes to one stated acceptance and reports TV to the exact draws beside the counts.
7. **Inherited support erodes within 30 fine-tune steps.** B: 0.159 to 0.0009 at 0% overlap (`walk05:1074`). A's caption fine-tuning remedy holds circle × square out and hopes PoE still composes it (`write-up:1190-1199`). To settle: A's phase 2c pass mark names B's erosion as the null it must beat.
8. **Whether the dial is a switch or a ramp, and which one SDXL's measured share should be placed on.** A and B's parts world: a switch at σ 0.1 (`checkpoints:281`, `walk05:889`). B's quadrant: a ramp rising as f², 0.30 at 5% (`walk05:704`). The route's step 7 places SDXL's s_cat on the toy's dial (`route:303`). To settle: step 7 places it on both curves and says which one a photo of a cat beside a dog resembles.
9. **Word similarity.** A learns the word embedding (`route:164`); B says a learned one cannot give SDXL's cat-dog closeness and fixes the cosine by hand (`testbed-map:77`). To settle before phase 2 is written.
10. **SDXL's own step count is A's broken setting.** At 50 DDIM steps and w 7.5, A's toy overshoots to single shapes (`checkpoints:333-336`); SDXL runs 50 steps at 7.5 (`ladder11:40`). To settle: say whether the overshoot is the toy's sharp σ or a property of 50 steps at 7.5.
11. **Counts alone against share, TV and Z.** A reports counts (`checkpoints:88`); B shows a share can hide a TV 0.049 above the floor (`a-and-b-map:11`). To settle: checkpoint 6 adds TV to the exact draws.
12. **Provenance in A's route table.** "Layer 10 found" is run 6 of the A-and-B map (`route:335`, `rebuild:60`); "the testbed" is a design, not built (`route:322`, `rebuild:61`); "ULA, MALA, UHA with their tests" are tests of MALA and the CHA fix only (`route:319`, `rebuild:49`); phase 2's home is the laptop in one line and the cluster in another (`route:176`, `route:322`).
