# Route 3 return: the PyTorch rebuild and the overlap dial

This is route 3's return to the [overlap-dial idea map](../IDEA_MAP.md), routes row 3, asked by [the unbuild route prompt](03-unbuild-walk.prompt.md). The JAX notebook is rebuilt in PyTorch and extended through all five phases, and every phase's check ran in code and passed. The overlap dial answers claim 2 for ideal experts. With no A-and-B training points, the product puts no mass deep inside A∩B, either on the quadrant toy or in the parts world with an empty background. Pretraining on all placements does not rescue it either, because 100 fine-tune steps erase what the pretraining left.

## 1. Where the code lives

**The rebuild**

`rebuild/` at the repo root, on branch `unbuild-walk` (worktree `.claude/worktrees/unbuild-walk/`). It is Python, NumPy and PyTorch only. Each phase is one command from the repo root with `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m rebuild.<script>`:

| Phase | Modules | Check script |
|---|---|---|
| 1 | `datasets.py`, `model.py`, `train.py`, `launch_phase1.sh` | `phase1_check.py` |
| 2 | `compose.py`, `samplers.py`, the repository's `anneal_samplers.py` | `phase2_compose.py` |
| 3 | `ideal.py`, `kinds.py` | `phase3_kinds.py` |
| 4 | `parts_world.py` | `phase4_parts_world.py` |
| 5 | `phase5_dial.py`, `phase5_finetune.py` | `phase5_tables.py` prints the tables below |

**The walk file and the outputs**

The walk file with one row per brick is [the rebuild walk](../../../drips/simple-distributions-rebuild/REBUILD_WALK.md). Figures and numbers are in [the unbuild-walk runs folder](../runs/unbuild-walk/README.md), whose card says what each file shows and which script made it.

**Bringing it into `walk`**

This file, the walk file and the runs folder are committed on `unbuild-walk`, because the session was isolated in a worktree and could not write into the shared checkout. `git merge unbuild-walk` from the `walk` checkout brings everything over. None of the new paths exist there, so nothing is overwritten.

## 2. Each phase's check

| Phase | Check | Result | Passed |
|---|---|---|---|
| 1 | Each of the 10 trained models' samples against its dataset: TV on 40 × 40 bins of [−1.5, 1.5]² between 5000 model samples and 5000 data samples, within 0.10 of the TV between two data draws | 9 of 10 within 0.05 of the floor. The ring (std 0.03) reads 0.179 against a floor of 0.049 and 0.075 without the last step's noise | yes, the ring once the notebook's t = 0 noise is removed |
| 2 | Ring × bar and left + right against the notebook's own output pictures, and MALA acceptance per sample | Metropolis-adjusted UHA puts 0.90 of the mass on the two in-bar modes, split 0.46/0.54. Reverse diffusion keeps the six outer modes pulled inward, with 0.58 in-bar, which is the notebook's picture. Left + right: right bar's share 0.47 to 0.49 for every sampler | yes |
| 3 | The kinds notebook's Figure 1 and Figure 2 from the rebuilt modules: exact numbers to printed precision, sampler numbers within 0.03 | Figure 1 and every Figure 2 dial match to printed precision, MALA column included. Runs 1 and 2 match within 0.03 (largest: contention 5% tuned MALA 0.56 against 0.53) | yes |
| 4 | `numbers.json` recomputed, then each sampler against 100,000 exact draws | largest relative difference 2.9e-16. Sampler errors in the table under phase 4 below | yes |
| 5 | Curves of A∩B mass and TV against overlap, exact and per sampler, plus the fine-tune variant | the curves exist for the exact composition, tuned MALA and UHA (section 4) | yes |

"Passed" means the script's own comparison passed. You have not looked at any of these pictures yet; the walk file's Instructions list what to open and what each picture should show.

**Phase 4, each sampler against exact draws**

Each sampler drew 2000 samples, against 100,000 exact draws. Projected TV is computed on the two half-means. Weight error is the TV between the sampled and exact product-component weights. The perfect sampler is 2000 exact draws, so its row is the floor.

| Setting | Perfect | Reverse diffusion | ULA | MALA | Tuned MALA | UHA |
|---|---|---|---|---|---|---|
| (a) mosaic: projected TV, head mean (exact 0.980) | 0.013, 0.981 | 0.732, 0.923 | 0.200, 0.966 | 0.191, 0.964 | 0.036, 0.979 | 0.036, 0.981 |
| (b) blend: projected TV | 0.007 | 0.016 | 0.024 | 0.034 | 0.008 | 0.017 |
| (c) co-occurrence: weight error, projected TV | 0.026, 0.033 | 0.010, 0.378 | 0.009, 0.156 | 0.014, 0.129 | 0.009, 0.030 | 0.010, 0.040 |

## 3. The MALA fix and its test

**What changed in `anneal_samplers.py`**

`AnnealedMALASampler` now divides the proposal log-density by `4*ss`, the proposal variance being `2*ss`, where it used to divide by `2*ss**2`. It also accepts each sample on its own whenever the gradient function returns one log-density per sample. A scalar log-density makes the batch one joint chain with one decision per move, which is how `inf_sample.py`'s `gradient_cha` uses it (it sums the energy over the batch). The constructor signature is unchanged, so `inf_sample.py` runs as before. After each `sample_step`, `last_accept` holds each sample's acceptance fraction.

**CHA**

`AnnealedCHASampler` gets the same per-sample accept. It also had a third bug: `leapfrog_step_c` updates its inputs in place, so the refreshed momentum `v_prime` was overwritten by the leapfrog's output. The kinetic terms then cancelled exactly, and the accept test read the energy change alone. The leapfrog now gets copies. ULA and UHA are unchanged.

**The test**

`tests/test_anneal_samplers.py`, 6 tests, all pass (`python -m pytest tests/test_anneal_samplers.py -s`):

- the proposal log-density differs from `torch.distributions.Normal(x + ss·g, √(2ss))` by the same constant on every sample
- per-sample acceptance over 10 moves at step 1.5 on N(0, I): mean 0.656, sd 0.153 across samples, histogram over 0.0 to 1.0 `[0, 2, 18, 80, 279, 557, 947, 1001, 732, 312, 72]`. One shared random number would make every entry equal.
- fixed MALA at step 0.9 on N(0, 1): variance 0.989
- the same chain with the old divisor: variance 4.03, so the variance check fails on the old code
- a `gradient_cha`-shaped scalar energy over a (3, 3, 8, 8) batch: runs, keeps its shape, one decision per move
- fixed CHA on N(0, 1): variance 1.010

In phase 2, MALA at the notebook's step 0.001 accepts 0.96 to 0.98 of moves, with sd 0.01 to 0.02 across samples, and tuned MALA settles at 0.59 to 0.60 with sd 0.06.

## 4. Phase 5's curves

The curves are in [the overlap-dial curves](../../../../toy/rebuild/out/phase5/overlap-dial-curves.png), the quadrant pictures in [the overlap-dial pictures](../../../../toy/rebuild/out/phase5/overlap-dial-quadrant-pictures.png), and the fine-tune in [the fine-tune curves](../../../../toy/rebuild/out/phase5/finetune-curves.png) and [pictures](../../../../toy/rebuild/out/phase5/finetune-pictures.png).

**The quadrant toy**

The setup is the worked-examples notebook's quadrant sets with ideal experts: σ = 0.05, 800 points per expert, A the left half, B the top half, overlap f of each set in the top-left quadrant. The two training squares touch at the origin. So the main number is A∩B mass at least 0.15 (3σ) inside the top-left quadrant; the whole-quadrant mass also counts a sliver of mass sitting on the corner. Each cell is the mean ± sd over 3 data seeds, with 2000 samples per sampler.

| Overlap | Exact, 0.15 inside | Tuned MALA | UHA | Reverse diffusion | Exact, whole quadrant | TV tuned MALA / UHA / reverse diffusion (floor) |
|---|---|---|---|---|---|---|
| 0% | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.20 ± 0.02 | 0.12 / 0.13 / 0.31 (0.09) |
| 1% | 0.05 ± 0.05 | 0.02 ± 0.02 | 0.02 ± 0.02 | 0.01 ± 0.01 | 0.37 ± 0.06 | 0.22 / 0.20 / 0.43 (0.12) |
| 2% | 0.09 ± 0.01 | 0.03 ± 0.01 | 0.03 ± 0.01 | 0.00 ± 0.00 | 0.39 ± 0.06 | 0.28 / 0.25 / 0.60 (0.13) |
| 5% | 0.30 ± 0.10 | 0.17 ± 0.04 | 0.18 ± 0.05 | 0.02 ± 0.00 | 0.66 ± 0.02 | 0.24 / 0.23 / 0.55 (0.15) |
| 10% | 0.51 ± 0.04 | 0.40 ± 0.04 | 0.40 ± 0.04 | 0.10 ± 0.02 | 0.82 ± 0.04 | 0.22 / 0.22 / 0.58 (0.18) |
| 20% | 0.59 ± 0.02 | 0.55 ± 0.02 | 0.55 ± 0.03 | 0.24 ± 0.04 | 0.87 ± 0.02 | 0.18 / 0.19 / 0.44 (0.18) |
| 50% | 0.71 ± 0.03 | 0.72 ± 0.03 | 0.72 ± 0.03 | 0.69 ± 0.04 | 0.96 ± 0.01 | 0.18 / 0.18 / 0.27 (0.17) |
| 100% | 0.76 ± 0.00 | 0.76 ± 0.01 | 0.76 ± 0.01 | 0.91 ± 0.01 | 0.98 ± 0.00 | 0.18 / 0.18 / 0.31 (0.17) |

At 0% every method puts its mass on the corner where the squares touch, and none of it reaches 0.15 inside A∩B. The exact curve rises faster than the overlap share: at 5% overlap, 0.30 of the product's mass is deep in A∩B, because the product weights most heavily the places both experts share.

Tuned MALA and UHA agree with each other within 0.02 at every setting. They reach the exact curve from 20% up, but at 2% to 10% they recover only a third to four fifths of it (5%: 0.17 against 0.30). The pictures show the missing mass staying at the corner and along the x₂ = 0 edge, which is the 0% answer: the samplers do not move enough mass onto the new, separated modes. This is the mode-weight failure runs 4 to 6 found for dominance.

Reverse diffusion is far lower at low overlap (5%: 0.02, 20%: 0.24) and overshoots at 100% (0.91 against 0.76, pulled toward the quadrant's centre). That is claim 4's sampler error, at 100% overlap, with zero model error.

**The parts world**

A is "contains a cat" and B is "contains a dog". At overlap f, each expert's set is f cat-and-dog images and 1 − f single-animal images, with the animal on either half and the other half background. A sample counts as A∩B when one half's mean is above 0.75 and the other's below −0.75. Each sampler drew 2000 samples, with TV on the two half-means against 100,000 exact draws.

| Background variance | Overlap | Exact A∩B | Tuned MALA | UHA | Reverse diffusion | TV tuned MALA / UHA / reverse diffusion (floor) |
|---|---|---|---|---|---|---|
| 0.01 (empty half is definite) | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.03 / 0.02 / 0.22 (0.03) |
| 0.01 | 0.001 | 1.000 | 0.995 | 1.000 | 0.734 | 0.04 / 0.04 / 0.39 (0.03) |
| 0.01 | 0.01 | 1.000 | 1.000 | 1.000 | 0.885 | 0.04 / 0.04 / 0.17 (0.03) |
| 0.01 | 0.05 | 1.000 | 1.000 | 1.000 | 0.969 | 0.03 / 0.04 / 0.18 (0.03) |
| 0.01 | 0.2 to 1 | 1.000 | 1.000 | 1.000 | 0.996 to 1.000 | 0.02 to 0.03 / 0.03 / 0.22 to 0.25 (0.02 to 0.04) |
| 1.0 (expert says nothing about the other half) | 0 to 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.02 to 0.05 / 0.02 to 0.04 / 0.22 to 0.36 (0.02 to 0.04) |

With an empty background at 0% overlap, the product is a ghost: its main component has half-means (0.5, −0.5), each animal blended with the background. One cat-and-dog image in a thousand flips it to 1.000. The matching cat-and-dog pair beats the ghost by about 386 nats: the ghost's 16 mismatched pixels cost 16 × 1²/(2 × 0.02) = 400, and the small overlap weights give back 2·ln(0.001) ≈ −14. A loose background gives A∩B at 1.000 with no overlap at all, which is example 9(c). So in the parts world, whether a single-animal image pins down the other half decides claim 2, more than the overlap share does.

**The fine-tune variant (quadrant toy, trained experts)**

One energy MLP with the notebook's architecture was pretrained on all four quadrants for 10,000 steps. Two copies were then fine-tuned at 0% overlap (A on the bottom-left square, B on the top-right) with Adam at 1e-4, batch 1000, evaluated on raw weights. The baseline is two experts trained from scratch on the same data for 10,000 steps. "Exact" here is the trained experts' summed t = 0 energy, normalised on the grid. There is one pretraining seed.

| Fine-tune steps | Exact, 0.15 inside | Tuned MALA | UHA | Exact, whole quadrant | Tuned MALA | UHA | TV tuned MALA / UHA |
|---|---|---|---|---|---|---|---|
| 0 | 0.159 | 0.158 | 0.163 | 0.228 | 0.228 | 0.226 | 0.34 / 0.34 |
| 10 | 0.089 | 0.104 | 0.122 | 0.145 | 0.162 | 0.193 | 0.33 / 0.32 |
| 30 | 0.001 | 0.056 | 0.068 | 0.002 | 0.080 | 0.101 | 0.33 / 0.42 |
| 100 | 0.005 | 0.004 | 0.005 | 0.172 | 0.186 | 0.198 | 0.13 / 0.14 |
| 300 | 0.000 | 0.000 | 0.001 | 0.118 | 0.133 | 0.143 | 0.10 / 0.10 |
| 1000 | 0.000 | 0.000 | 0.000 | 0.190 | 0.239 | 0.235 | 0.09 / 0.09 |
| 3000 | 0.000 | 0.000 | 0.000 | 0.158 | 0.220 | 0.230 | 0.11 / 0.10 |
| 10000 | 0.000 | 0.000 | 0.000 | 0.021 | 0.291 | 0.287 | 0.90 / 0.90 |
| from scratch, 10000 | 0.000 | 0.000 | 0.000 | 0.203 | 0.251 | 0.255 | 0.10 / 0.09 |

The support inherited from pretraining falls from 0.16 to under 0.01 within 30 steps on the exact composition and within 100 steps on both samplers, the from-scratch value. In this 2D toy, fine-tuning erases it almost at once. Fine-tune length works as a dial only over the first 100 steps at this learning rate, and a smaller rate would stretch that axis.

The step-10,000 row's whole-quadrant 0.021 and TV 0.90 come from a spurious low-energy blob near (0.75, 0.9) in the trained t = 0 energy, far from both training squares. The annealed samplers never reach it. The A∩B number does not depend on it.

## What this changes in the idea map

**Claim 2**

The check ran, and with ideal experts the product does not put mass in A∩B when no training sample is there. On the quadrant toy the deep A∩B mass is 0.00 at 0%. In the parts world with an empty background it is 0.000, a ghost blend instead. Pretraining support survives under 100 fine-tune steps. The one case that produces A∩B with no overlap is an expert that is loose on the region its concept does not occupy (parts world, background variance 1.0).

**Claim 4**

Tuned MALA and UHA reach the exact curve at 20% overlap and above. Below that they lag (5%: 0.17 to 0.18 against 0.30), so low-overlap image results read with these samplers will understate A∩B by up to about half. Reverse diffusion understates it far more at low overlap and overstates it at 100%.

**Checks outstanding**

The sampler-code row is resolved by section 3. The claim 2 overlap check and the claim 2 fine-tune check have both run.

**Three findings outside the five phases**

- The notebook's reverse sampler adds noise at t = 0, with sd √β₀ = 0.025 on the cosine schedule. That widens modes as narrow as the ring's 0.03 to about 0.043.
- A trained energy at t = 0 is not a trustworthy ground truth for mass between separated regions. For ring × bar, the trained experts' t = 0 composition splits the in-bar mass 0.10/0.90, while the data's own product splits it 0.54/0.46 and every annealed sampler gives 0.46 to 0.54. The step-10,000 blob is the same effect. 2D ground truth should keep coming from ideal experts, as the idea map already does.
- CHA's in-place leapfrog cancelled its kinetic term (section 3).

## Caveats

- The quadrant overlap dial used 3 data seeds; the parts-world dial and the fine-tune used one seed each.
- The fine-tune step axis is in Adam steps at learning rate 1e-4; only that one rate was run.
- The fine-tune variant ran on the 2D quadrant toy only. A parts-world fine-tune (16-pixel trained experts) was not run.
- The 0.15 margin sets the ceiling of the deep A∩B number (0.76 at 100% overlap).
- Every pass above is the script's own comparison. You have not looked at the pictures yet.
