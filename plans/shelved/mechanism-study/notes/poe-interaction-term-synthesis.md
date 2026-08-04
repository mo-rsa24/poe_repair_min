# The mechanism, reconciled across 7 papers + your data

## THE ONE SENTENCE
PoE composition adds two concept scores as if the concepts were independent; the true
composition differs by an INTERACTION TERM that PoE drops; your residual r_t = ε_joint − ε_PoE
is that term made explicit, and your LoRA is a learned, pair-generic estimate of it.

## THE CENTRAL IDENTITY (typeset in MathML)
At every noise level t:
   ∇log p_t(x|c1,c2)  =  [∇log p_t(x|c1) + ∇log p_t(x|c2) − ∇log p_t(x)]  +  I_t(x)
                          └─────────── PoE / additive estimate ──────────┘    └ interaction ┘
In ε (noise-prediction) terms, ε = −σ_t ∇log p_t, so:
   ε_joint  =  ε_PoE  −  σ_t I_t     ⟹     r_t := ε_joint − ε_PoE  =  −σ_t I_t
YOUR r_t IS THE INTERACTION TERM (scaled). This is the spine of the whole artifact.

Exact at t=0 (product of densities ⟺ sum of scores). Nonzero for t>0 because NOISING A
PRODUCT ≠ PRODUCT OF NOISED MARGINALS (Du 2023; PoE-for-Visual-Gen). That non-commutativity is
the mechanical origin of the term.

## WHEN IS I_t = 0? (the condition, from Projective Composition)
Factorized Conditionals: concepts occupy DISJOINT, independent coordinate blocks and copy the
background elsewhere. Then adding scores is exact at every t. Heuristic test (Lemma 8.1):
centered concept means orthogonal, (μ_cat − μ_bg)·(μ_dog − μ_bg) = 0.
- cat & dog are NOT orthogonal (both foreground animals, same coordinate territory) → I_t ≠ 0
  → PoE lands BETWEEN them → CHIMERA. THIS IS YOUR MANIFOLD PLOT: chimera sits mid-axis.

## TWO FAILURE MODES (do not conflate — Projective Composition)
A. TARGET WRONG: FC violated, additive score ≠ any valid composition. Error ≤ Σε_i (per-concept
   independence violation). SYMPTOM: chimera / blend. ← this is cat×dog.
B. PATH UNREACHABLE: target fine but composed path non-smooth in t (Lemma 7.2, Lipschitz Ω(1/τ)).
   SYMPTOM: collapse to background, concept missing. Different disease.

## THE FIVE LEVELS (each a panel, each with a visual)

L1. VELOCITY / SCORE FIELD  [Catastrophic; PoE-Vis-Gen; CO3]
  Two score vectors add; in the overlap they point to the VALLEY between modes → chimera
  attractor. I_t is the vector that bends the sum back onto the two-object configuration.
  Gaussian truth (Catastrophic Thm2): naive gives GEOMETRIC mean of covariances, truth is
  HARMONIC of precisions — provably wrong at intermediate t even with perfect scores.
  VISUAL: live 2D toy — two Gaussians, quiver of ∇logp1+∇logp2, mark the chimera attractor in
  the valley; toggle I_t on to see the field bent to two separated modes. ILLUSTRATIVE (toy).
  REAL anchor: delta_norm(t) from TRACE = ‖r_t‖ = ‖I_t‖·σ_t measured over the 50 steps.

L2. MANIFOLD / GEOMETRY  [Projective Composition; Test-Time Concept Discovery]
  Chimera = off-manifold point in the valley between cat cloud and dog cloud. Composition needs
  the manifold to have support for "cat AND dog". Orthogonality heuristic = whether the two
  concept directions are separable.
  VISUAL: YOUR REAL manifold scatter (manifold_pts.json). cat_x=−0.22, dog_x=+0.29; broken λ=0
  cloud mid-axis (the valley), fixed λ=1 cloud pulled off it; seed-9 sweep = the path chimera→
  fixed as training proceeds. THE MONEY FIGURE, REAL, already computed. 48 pts + 9 sweep.

L3. SDE / ODE — TWO BASINS  [PATHS; CO3]
  Composition basin vs chimera attractor, separated by a barrier. Naive sampler falls into the
  chimera basin during the COMMITMENT WINDOW (your measured 5–25) and can't leave. r_t = grad of
  a log-ratio between joint and PoE dists = the force that diverts the trajectory into the
  composition basin. PATHS reaches it by SEARCH (temperature+swaps, per prompt); your LoRA by a
  LEARNED SHORTCUT (no search, no joint prompt). "Two ways to the same basin."
  VISUAL: energy landscape w/ two basins + trajectory; tie to REAL commit window 5–25 and the
  per-step attention timeseries (mech_timeseries2.json). Basins ILLUSTRATIVE, window REAL.

L4. VAE — the spectator  [honest scoping]
  The whole thing lives in VAE latent space; chimera is a latent that decodes to a blended
  animal. But the FAILURE IS IN THE SCORE FIELD, not the decoder. Say so plainly.
  VISUAL: real vae_act thumbs from TRACE, labelled "where it lives, not where it breaks."

L5. THE INTERACTION TERM — the unifying object  [Catastrophic; CO3; PoE-Vis-Gen]
  L1/L2/L3 are ONE quantity from three angles: I_t is the field (L1), the off-manifold
  correction (L2), the basin-diverting force (L3). CO3 builds it by hand (divide out overlap,
  closeness-gated); you LEARN it. Catastrophic proves it's non-recoverable at inference → must
  be learned → your LoRA.
  VISUAL: the identity lit term-by-term as you hover each level; REAL anchors: attention corr
  96.5%, value-direction 4×, delta_norm curve, commit window 5–25.

## WHAT "JOINT PROMPT UNDERSTANDS COMPOSITION" CASHES OUT TO
ε_joint has I_t baked in for free: the UNet saw "a cat and a dog" as ONE context and its
cross-attention already separates the two token streams spatially (your 96.5% attention corr,
4× value-direction result). PoE never sees the joint context so it cannot manufacture I_t. The
LoRA stores a pair-generic approximation of I_t so PoE can borrow it without the joint prompt.

## THE HONEST LEDGER (what's yours vs the papers')
PROVEN BY PAPERS:
 - r_t is a real named object = interaction/correction term (Catastrophic g_t; PoE-Vis-Gen Δ_t;
   Projective-Composition FC-gap). [all three]
 - It's zero under conditional independence / FC, nonzero under overlap. [Projective Comp; Catastrophic]
 - Non-recoverable at inference from individual scores → must be learned. [Catastrophic]
 - A pair-generic composition correction EXISTS and transfers across pairs & backbones. [CO3]
 - A LoRA CAN absorb a composed target & drop the machinery at inference. [Test-Time Concept Disc]
YOURS TO PROVE (no paper does):
 - That a FIXED rank-8 LoRA CAPTURES that correction and TRANSFERS across held-out pairs.
   Low-rank ⇒ the interaction term lives in a low-dim pair-generic subspace. THIS IS THE CLAIM.
HONEST CAVEATS TO BAKE IN:
 - Don't say "PoE assumes conditional independence" citing PoE-Vis-Gen; that framing is Liu/Du.
 - r_t absorbs BOTH diffusion-time coupling Δ_t AND genuine semantic interaction; paper covers Δ_t.
 - Catastrophe is OOD-specific: need ≥1 genuinely OOD held-out pair to claim you beat it.
 - Projective-Comp Lemma 7.2 is non-smoothness across t; your transfer is across PAIRS at matched
   t — orthogonal, but pre-empt the reviewer.
HONEST TESTS THE PAPERS HANDED ME (future figures, flag as not-yet-done):
 - correlate learned r_t with CO3 closeness contrast on held-out pairs [CO3]
 - is r_t spatially localized at the cat/dog boundary, ~0 inside each animal [FactorDiff]
 - r_t-as-basin-arrow vs PATHS per-prompt search [PATHS]

## PAPER LEDGER (title → arXiv → role)
- Mechanisms of Projective Composition — 2502.04549 (Apple, ICML25) — THE CONDITION (FC), 2 failures, orthogonality heuristic
- Catastrophic Compositional Generation — 2606.23920 (CMU/Valence) — names g_t, non-recoverable, OOD catastrophe, Gaussian geo-vs-harmonic
- Product of Experts for Visual Generation — 2506.08894 (Stanford, ICLR) — PoE formalism, Δ_t non-commutativity (via parent Du 2023 "Reduce Reuse Recycle" 2302.11552)
- Steer Away From Mode Collisions / CO3 — 2509.25940 — closest to chimera, repulsive coupling, pair-generic RULE exists
- Parallel Tempering / PATHS — 2605.30991 — two-basin, search-vs-shortcut
- Test-Time Compositional Gen via Concept Discovery — 2605.07078 (GT/UVA) — LoRA-absorbs-PoE distillation confirmed
- From Global to Factor-Wise / FactorDiff — 2607.11758 (Toronto/Vector) — global scalar PoE too coarse, competence local (discrete)
