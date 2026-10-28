# Ideas that tie the composition ladder together

**Where this comes from.** Codex wrote it on 2026-10-18 from the three idea maps, the ladder's index and `artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md`. Every number it marks "by hand" was rerun in Python and reproduces. Codex checked its citations with 10 web searches; they have not been opened here. One supplied number is out of date: idea 4 quotes `0.891` of the mass off every mode at negation weight 0.3, which is an earlier draw; the ladder's settled figure is `0.863` ([layer 7](../drips/simple-distributions-rebuild/ladder/07-a-without-b.md)). The ranked list at the end orders the ideas by how far each moves the goal per hour.

## 1. Make conditional independence visible as an interaction residual

**What it explains**

From the supplied facts, construction (1) has the unnormalised target

$$
q_{\mathrm{PoE}}(x)=\frac{p(x\mid A)p(x\mid B)}{p(x)}
\propto p(x)\,p(A\mid x)p(B\mid x).
$$

It equals the joint conditional after normalisation when $A\perp B\mid x$. Bayes’ rule gives

$$
r(x)
=
\log\frac{p(A,B\mid x)}{p(A\mid x)p(B\mid x)}
+
\log\frac{p(A)p(B)}{p(A,B)}.
$$

The first term is the report-label interaction. It varies with $x$ when observing one label changes the probability of observing the other even after the image is known. The second term is a global normalisation constant.

This qualifies one settled statement. The supplied map says that the interaction term is exactly zero for clean deterministic labels. With the stated definition of $r$ using normalised conditional densities, it is generally constant rather than literally zero. Its gradient and its centred version are zero. Both statements agree if $r$ is defined modulo constants or if the PoE is normalised before taking the log ratio.

My reasoning is that the thesis should distinguish three objects explicitly. The scalar $r$ includes its normalising constant. The centred interaction $r(x)-\mathbb E[r(X)]$ shows density-shape failure. The score correction $\nabla r(x)$ is what diffusion sampling and a LoRA can learn. This also separates report-label dependence from appearance interaction caused by separately trained experts that are not coherent conditionals of one common distribution.

**Where it attaches (the ladder layer and the idea-map claim)**

It belongs at the reading stop and layer 8. It sharpens overlap-dial claim 7, the parallel map’s claims 3 and 9, and layer 8’s definition of the pixel and label interactions. The report-label term is directly relevant to pneumothorax and drain labels. The appearance residual belongs in the wet-parts-world example, not inside the label term.

**The smallest check (a 2D or 16-pixel toy computation with exact answers where possible, with what result supports the idea, what counts as a null, and what would surprise)**

Use the two-state report-mining toy. Let $p(x_1)=p(x_2)=0.5$. At both states set $p(A\mid x)=p(B\mid x)=0.5$. Set $p(A,B\mid x_1)=0.30$ and $p(A,B\mid x_2)=0.20$. The remaining three label-pair probabilities are respectively $(0.20,0.20,0.30)$ and $(0.30,0.30,0.20)$, so each table is valid.

Globally, $p(A)=p(B)=0.5$ and $p(A,B)=0.25$. The PoE gives masses $(0.5,0.5)$, while the true joint conditional gives $(0.6,0.4)$. Therefore

$$
r(x_1)=\log 1.2=0.1823,\qquad
r(x_2)=\log 0.8=-0.2231
$$

by hand. Multiplying the PoE masses by $\exp r$ and renormalising recovers $(0.6,0.4)$ exactly.

Support is a nonconstant residual with the same values obtained independently from the label log ratio. The null sets $p(A,B\mid x)=0.25$ at both states and gives $r=0$. A surprising result would be a nonzero centred residual under that null. It would indicate inconsistent density normalisation, label bookkeeping, or an expert family that does not represent the claimed conditionals.

The learner should also reproduce the supplied wet-parts-world result. The interaction setting moves mean $x_1$ to $-0.79$, while the zero-overlap construction remains near $-0.48$. They should explain why that appearance residual and the report-label residual have different causes.

**Cost (hours, and whether it needs training)**

Two to three hours. No training is required.

**Prior work (author, year, title; mark each "confident" or "recalled, verify", and never invent a paper)**

[Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B. Tenenbaum, 2022, *Compositional Visual Generation with Composable Diffusion Models*](https://arxiv.org/abs/2206.01714), confident.

[Arwen Bradley, Preetum Nakkiran, David Berthelot, James Thornton, and Joshua M. Susskind, 2025, *Mechanisms of Projective Composition of Diffusion Models*](https://arxiv.org/abs/2502.04549), confident.

**Verdict for the goal (needed, could help, or skip, with the reason)**

Needed. It defines exactly what the correction is, prevents a normalisation constant from being mistaken for learnable interaction, and separates appearance interaction from report-label dependence.

## 2. Use fixed decoders before training a VAE

**What it explains**

From the supplied facts, a weighted product with exponent sum $S=1$ is invariant under an invertible change of variables. Construction (1) and negation have $S=1$. A plain product has $S=2$ and is generally reweighted by decoder volume.

My reasoning is that a deliberately rotated latent is primarily a negative control. An orthogonal rotation has constant Jacobian determinant one. It cannot change an ideal plain product after normalisation. If rotation alone changes the result, it exposes approximation, architecture, or sampler error rather than the change-of-variables effect.

For an injective decoder $x=g(z)$ onto a manifold, let

$$
G(z)=J_g(z)^\top J_g(z),\qquad
v(z)=\sqrt{\det G(z)}.
$$

A latent density pushes forward to manifold density $p(z)/v(z)$. A pixel-manifold product with exponent sum $S$, pulled back to $z$, is therefore proportional to

$$
\prod_i p_i(z)^{\alpha_i}v(z)^{1-S}.
$$

Thus $S=1$ removes the volume factor. A plain product retains $1/v(z)$.

A VAE adds two further failures. A noninjective decoder sums density over several preimages, and a product of sums is not a sum of products. A stochastic decoder integrates over $p(x\mid z)$, which also does not commute with multiplication. These are my deductions from the change-of-variables formula.

**Where it attaches (the ladder layer and the idea-map claim)**

It belongs at layer 10. It refines settled claim 10 and the held VAE claim 11 in the parallel idea map. It should precede any comparison among a standard VAE, $\beta$-VAE, $\beta$-TCVAE, and FactorVAE.

**The smallest check (a 2D or 16-pixel toy computation with exact answers where possible, with what result supports the idea, what counts as a null, and what would surprise)**

Use a fixed one-dimensional latent with two unit-length intervals. Decode the first interval along a line with speed $v_1=1$, and the second along a disjoint line with speed $v_2=2$. Give both intervals equal mass under every latent expert.

The latent plain product assigns masses $(1/2,1/2)$. The pixel-manifold plain product, pulled back through the decoder, assigns weights proportional to $(1,1/2)$, hence masses $(2/3,1/3)$ by hand. Their total variation distance is $1/6=0.1667$ by hand.

Construction (1) assigns $(1/2,1/2)$ in both spaces because $S=1$. An orthogonal $45^\circ$ rotation is the null and should give total variation zero for both constructions. Support is exactly the predicted $1/6$ discrepancy for the plain product and zero for construction (1). It would be surprising if a trained VAE’s change were attributed to disentanglement without first matching this fixed-decoder answer.

Only after this check should one train the VAE variants. The useful measures are reconstruction error, decoder-volume variation, preimage multiplicity, conditional mutual information between each latent coordinate and $A,B$, and composition total variation. Latent axis alignment alone is not sufficient.

**Cost (hours, and whether it needs training)**

Three to five hours for the fixed decoder, with no training. A controlled comparison of four VAE objectives would add roughly eight to sixteen hours and requires training.

**Prior work (author, year, title; mark each "confident" or "recalled, verify", and never invent a paper)**

[Irina Higgins et al., 2017, *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*](https://openreview.net/pdf/9fcd6616f4c4e9985552865eb0d2383e4fd55a26.pdf), confident.

[Hyunjik Kim and Andriy Mnih, 2018, *Disentangling by Factorising*](https://proceedings.mlr.press/v80/kim18b.html), confident.

[Ricky T. Q. Chen, Xuechen Li, Roger Grosse, and David Duvenaud, 2018, *Isolating Sources of Disentanglement in Variational Autoencoders*](https://arxiv.org/abs/1802.04942), confident.

[Francesco Locatello et al., 2019, *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*](https://arxiv.org/abs/1811.12359), confident.

**Verdict for the goal (needed, could help, or skip, with the reason)**

Could help. The fixed-decoder check isolates a real latent-space effect. Training several disentangling VAEs is optional because unsupervised axis alignment is not identifiable and does not by itself explain Stable Diffusion.

## 3. Treat decoder geometry as a known measure correction, not as the interaction

**What it explains**

From the supplied facts, $\sqrt{\det(J^\top J)}$ is the relevant local volume factor for an injective decoder. My reasoning is that it can predict a representation-induced preference of the plain product, but it cannot predict semantic co-occurrence, report-label dependence, or a wet-dog appearance residual.

For exponent sum $S$, the exact latent target corresponding to a product of manifold densities contains

$$
\frac{1-S}{2}\log\det G(z)
$$

in its log density. For the plain product, $S=2$, so the latent log target needs $-\tfrac12\log\det G(z)$. Equivalently, its energy needs $+\tfrac12\log\det G(z)$. Construction (1) needs no such term.

Geodesics describe intrinsic paths and distances. They do not determine product weights. Manifold dimension determines the size of $G$, but dimension alone does not determine its determinant or its correlation with concepts. Scores on a manifold become directly relevant only when the forward diffusion itself is defined on that manifold. Standard latent diffusion adds Euclidean noise in latent coordinates, so Riemannian score formulas are not automatically the model’s sampling equations.

**Where it attaches (the ladder layer and the idea-map claim)**

It belongs after layer 10’s warp experiment. It refines parallel-map claim 10 and should be linked back to layers 8 and 9 so that the geometric correction is not given to the interaction learner as if it were semantic residual data.

**The smallest check (a 2D or 16-pixel toy computation with exact answers where possible, with what result supports the idea, what counts as a null, and what would surprise)**

Use $g(z)=(z,z^2/2)$. Then $G(z)=1+z^2$. Put two equally weighted, increasingly narrow latent modes at $z=0$ and $z=2$. In the narrow-mode limit, a latent plain product keeps weights $(0.5,0.5)$. The corresponding manifold product weights are proportional to $(1,1/\sqrt5)$, giving

$$
\left(\frac{\sqrt5}{\sqrt5+1},
\frac{1}{\sqrt5+1}\right)
=
(0.6910,0.3090)
$$

by hand. The total variation from equal weights is $0.1910$ by hand. Adding the metric correction recovers these weights. Construction (1) remains at $(0.5,0.5)$.

The geodesic length from $z=0$ to $z=2$ is

$$
\int_0^2\sqrt{1+z^2}\,dz
=
\sqrt5+\tfrac12\operatorname{asinh}(2)
=
2.9579
$$

by hand. The ambient chord is $\sqrt8=2.8284$ by hand. Neither number determines the mode weights.

Support is a plain-product weight shift predicted by $G$ and removed by the metric correction. The null is constant $G$, which gives no shift. It would be surprising if $G$ predicted the wet-rule interaction after the representation effect had been removed. That would suggest that the decoder learned the interaction into its geometry, which is a different and testable hypothesis.

**Cost (hours, and whether it needs training)**

Three to five hours for the analytic check. No training is needed. Estimating Jacobian-volume effects for a Stable Diffusion VAE could take eight to sixteen hours of implementation and compute, but still needs no new model training.

**Prior work (author, year, title; mark each "confident" or "recalled, verify", and never invent a paper)**

[Georgios Arvanitidis, Lars Kai Hansen, and Søren Hauberg, 2018, *Latent Space Oddity: on the Curvature of Deep Generative Models*](https://arxiv.org/abs/1710.11379), confident.

[Hang Shao, Abhishek Kumar, and P. Thomas Fletcher, 2018, *The Riemannian Geometry of Deep Generative Models*](https://arxiv.org/abs/1711.08014), confident.

[Valentin De Bortoli et al., 2022, *Riemannian Score-Based Generative Modelling*](https://arxiv.org/abs/2202.02763), confident.

**Verdict for the goal (needed, could help, or skip, with the reason)**

Could help as a diagnostic. It supplies a principled correction for decoder measure distortion. It should not become the main theory of the cat-dog interaction unless its predicted signature appears after semantic and sampler effects are controlled.

## 4. Treat guidance as tempering and test negation for normalisability

**What it explains**

From the supplied facts, composable classifier-free guidance at weight $w$ targets

$$
p(x)\,p(A\mid x)^w p(B\mid x)^w
\propto
p(x\mid A)^w p(x\mid B)^w p(x)^{1-2w}.
$$

Increasing $w$ changes the target distribution. It is not merely a stronger numerical push toward the same target. This matters because the SDXL hybrid is observed at $w=7.5$, while most of the ladder’s exact composition work used $w=1$.

Negation uses a quotient such as $p_A^{1+u}/p_B^u$. Exponents summing to one preserve coordinate invariance, but they do not guarantee integrability. Tail precision can become zero or negative.

Sampler error must be measured against each tempered target. The supplied exact-expert results already show why. At overlap $r=0.05$, reverse diffusion has total variation $0.182$ against an exact-sample floor of $0.133$, while corrected MALA reaches $0.124$. At $r=0.001$, trained MALA attains the desired $0.989$ cell share but still has total variation $0.391$. A success rate alone therefore confounds target choice, model error, and sampler error.

**Where it attaches (the ladder layer and the idea-map claim)**

It belongs in layers 6 and 7, then feeds layers 8, 9, and 12. It also completes overlap-dial claims 4 and 5 and the SDXL testbed’s claim 1 about guidance $7.5$.

**The smallest check (a 2D or 16-pixel toy computation with exact answers where possible, with what result supports the idea, what counts as a null, and what would surprise)**

For guidance, use centred one-dimensional Gaussians $p_A=p_B=\mathcal N(0,1)$ and $p_0=\mathcal N(0,4)$. The guided precision is

$$
w+w+\frac{1-2w}{4}
=
\frac14+\frac32w.
$$

At $w=1$, the variance is $4/7=0.5714$ by hand. At $w=7.5$, it is $2/23=0.0870$ by hand. This is a sixfold reduction in variance. The null sets all three densities equal, in which case the target remains unchanged for every $w$.

For negation, use $p_A=\mathcal N(0,4)$ and $p_B=\mathcal N(0,1)$. The precision of $p_A^{1+u}/p_B^u$ is $0.25-0.75u$. It is normalisable only for $u<1/3$. At $u=0.05$, its variance is $80/17=4.7059$ by hand. At $u=0.30$, its variance is $40$ by hand. At $u=1$, no probability density exists. This does not derive the supplied multimodal result that $0.891$ of the mass lies off every mode at $u=0.3$. It supplies an analytic mechanism consistent with failure near that weight.

Support is agreement between direct Gaussian sampling and the predicted variances. A null is no change when all component densities coincide. It would be surprising if a sampler appeared stable for the nonintegrable $u=1$ target. That would mean finite bounds or discretisation are hiding the invalid target.

**Cost (hours, and whether it needs training)**

Two to four hours. No training is required.

**Prior work (author, year, title; mark each "confident" or "recalled, verify", and never invent a paper)**

[Jonathan Ho and Tim Salimans, 2022, *Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598), confident.

[Nan Liu et al., 2022, *Compositional Visual Generation with Composable Diffusion Models*](https://arxiv.org/abs/2206.01714), confident.

[Yilun Du et al., 2023, *Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC*](https://arxiv.org/abs/2302.11552), confident.

**Verdict for the goal (needed, could help, or skip, with the reason)**

Needed. The scale experiment uses strong guidance and negation. Without identifying the actual tempered target and checking its tails, a failed composition cannot be assigned to interaction learning.

## 5. Learn the identifiable object, which is the interaction score

**What it explains**

My reasoning is that a diffusion model does not identify the additive constant in $r_t(x)$. It identifies the score correction $\nabla_x r_t(x)$, or an energy whose gradient is that correction. A time-dependent scalar added to $r_t$ changes neither the score nor the samples.

This has a practical consequence for the LoRA. Regression against the cached residual should be formulated in score or noise-prediction space. Absolute scalar interaction values require normalisers and should not be used as targets unless those normalisers are estimated separately. Density-level evaluation should use samples, likelihood ratios, or an explicit estimate of $Z$.

**Where it attaches (the ladder layer and the idea-map claim)**

It belongs at the end of layer 8 and the start of layer 9. It connects layer 8’s $r_t$ to the cached residual

$$
\epsilon_{\mathrm{joint}}
-\epsilon_A-\epsilon_B+\epsilon_\varnothing
$$

and clarifies what the layer 12 LoRA can actually recover.

**The smallest check (a 2D or 16-pixel toy computation with exact answers where possible, with what result supports the idea, what counts as a null, and what would surprise)**

Use four two-dimensional unit-covariance Gaussians:

$$
p_0=\mathcal N((0,0),I),\quad
p_A=\mathcal N((-1,0),I),\quad
p_B=\mathcal N((0,-1),I),
$$

and let the desired joint be $p_{AB}=\mathcal N((-1.5,-1),I)$.

The normalised construction (1) target is $\mathcal N((-1,-1),I)$. Its interaction is

$$
r(x)=-0.5x_1-0.625
$$

by hand, so $\nabla r=(-0.5,0)$. The uncorrected KL divergence to the joint is

$$
\tfrac12\lVert(-1,-1)-(-1.5,-1)\rVert^2
=0.125
$$

nats by hand. The exact correction makes the KL zero. A learned slope of $-0.4$ produces mean $(-1.4,-1)$ and KL $0.005$ nats by hand. Adding $100$ to $r$ changes none of these samples or scores.

Support is correct recovery of the slope and the mean shift despite arbitrary constants. The null sets the joint mean to $(-1,-1)$, giving a constant $r$ and zero correction. It would be surprising if changing the target’s additive constant changed training or sampling. That would reveal an improperly centred energy loss.

**Cost (hours, and whether it needs training)**

Two to three hours for the analytic and linear-regression check. No neural training is needed. Applying the lesson to the LoRA changes the training objective but does not add a separate model.

**Prior work (author, year, title; mark each "confident" or "recalled, verify", and never invent a paper)**

Aapo Hyvärinen, 2005, *Estimation of Non-Normalized Statistical Models by Score Matching*, confident.

[Yang Song et al., 2021, *Score-Based Generative Modeling through Stochastic Differential Equations*](https://arxiv.org/abs/2011.13456), confident.

[Yilun Du et al., 2023, *Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC*](https://arxiv.org/abs/2302.11552), confident.

**Verdict for the goal (needed, could help, or skip, with the reason)**

Needed. It states precisely what the correction network can identify and prevents an avoidable mismatch between the mathematical $r$ and the LoRA’s training target.

## 6. Give the interaction its own diffusion-time trajectory

**What it explains**

From the supplied facts, layer 8 needs $r_t$ and $\nabla r_t$, not only the endpoint $r_0$. My reasoning is that noising the joint and noising the three PoE factors do not commute with taking their log-density ratio. Consequently, $r_t$ is not generally a fixed residual multiplied by the signal schedule.

This ties the toy directly to the scale correction. A successful time-conditioned LoRA should learn the residual’s shape and magnitude at every noise level. Endpoint agreement alone is insufficient.

**Where it attaches (the ladder layer and the idea-map claim)**

It belongs in layer 8’s noised interaction derivation, layer 9’s correction regression, and layer 12’s cached 50-step SDXL residuals. It also connects to the sampler finding that score error is largest at the lowest noise level for rare overlap.

**The smallest check (a 2D or 16-pixel toy computation with exact answers where possible, with what result supports the idea, what counts as a null, and what would surprise)**

Use one-dimensional centred Gaussians:

$$
p_0=\mathcal N(0,4),\quad
p_A=p_B=\mathcal N(0,2),\quad
p_{AB}=\mathcal N(0,1).
$$

At $t=0$, the PoE precision is $1/2+1/2-1/4=3/4$. Therefore

$$
r_0(x)=\text{constant}-x^2/8,\qquad
\nabla r_0(x)=-x/4
$$

by hand.

At $\bar\alpha_t=0.5$, forward noising changes the variances to $2.5$, $1.5$, $1.5$, and $1$. The PoE precision becomes

$$
2/1.5-1/2.5=14/15.
$$

Thus

$$
r_t(x)=\text{constant}-x^2/30,\qquad
\nabla r_t(x)=-x/15
$$

by hand. At complete noise, all four variances are one and the interaction score is zero.

Support is recovery of slopes $-1/4$, $-1/15$, and $0$ at the three noise levels. The null makes $p_{AB}$ equal to the PoE at the endpoint, giving zero residual at all times. It would be surprising if the exact Gaussian residual changed sign or grew at complete noise. In a learned non-Gaussian model that pattern would call for checking parameterisation and score error before interpreting it semantically.

**Cost (hours, and whether it needs training)**

Two to four hours for the exact derivation and plots. No training is needed. A small time-conditioned residual network would add four to eight hours of training.

**Prior work (author, year, title; mark each "confident" or "recalled, verify", and never invent a paper)**

[Yang Song et al., 2021, *Score-Based Generative Modeling through Stochastic Differential Equations*](https://arxiv.org/abs/2011.13456), confident.

[Yilun Du et al., 2023, *Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC*](https://arxiv.org/abs/2302.11552), confident.

[Marta Skreta et al., 2025, *Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts*](https://arxiv.org/abs/2503.02819), confident.

**Verdict for the goal (needed, could help, or skip, with the reason)**

Needed. A scale correction is applied throughout denoising, so the object learned at each step must be the correct $r_t$, not a heuristic rescaling of $r_0$.

## 7. Turn the evaluation into an oracle correction cube

**What it explains**

The supplied facts establish that model and sampler errors interact. At overlap $r=0.05$, the interaction is $+0.054$ total variation. At $r=0.001$, it grows to $+0.232$. Soiffer et al.’s supplied result likewise says that a stronger inference-time correction can improve exact scores while amplifying learned-score error.

My reasoning is to add a third experimental axis: no correction, oracle interaction correction, or learned correction. This produces a cube with expert quality, sampler, and correction on its axes. It answers four distinct questions. It tests whether the target is correct, whether the experts represent it, whether the sampler reaches it, and whether the learner approximates the oracle residual.

**Where it attaches (the ladder layer and the idea-map claim)**

It connects layers 3 through 5 to layers 8 and 9. It is the cleanest implementation of idea-map claims 4 and 5 and should become the fair-test protocol in layer 9 before moving to layer 12.

**The smallest check (a 2D or 16-pixel toy computation with exact answers where possible, with what result supports the idea, what counts as a null, and what would surprise)**

Use the Gaussian-shift interaction toy from idea 5. Construct the following cells: exact or trained experts; direct sampling, reverse diffusion, or corrected MALA; no correction, oracle $\nabla r$, or learned $\widehat{\nabla r}$.

Report KL or grid total variation, target-region mass, and $Z$. With exact experts and direct samples, the uncorrected KL is $0.125$ nats, the oracle-corrected KL is zero, and a learned slope of $-0.4$ gives $0.005$ nats, all by hand.

For the trained and sampled cells, my proposed support criterion is that the oracle sampler finishes within $0.02$ total variation of the exact-sample floor and that the learned correction closes at least $80\%$ of the oracle gain. The proposed null is a learned gain no larger than $0.02$. A surprising outcome is an oracle correction that helps exact experts but systematically hurts trained experts. That is evidence that the correction is amplifying expert error rather than failing to represent $r$.

The supplied quadrant results should appear beside this cube as calibration points. They include reverse diffusion $0.182$, MALA $0.124$, exact-sample floor $0.133$ at $r=0.05$, and trained-MALA total variation $0.391$ despite $0.989$ target-cell mass at $r=0.001$.

**Cost (hours, and whether it needs training)**

Four to eight hours if the existing checkpoints and samplers are reused. The oracle cells need no training. The learned-correction cells need a small residual model.

**Prior work (author, year, title; mark each "confident" or "recalled, verify", and never invent a paper)**

[Yilun Du et al., 2023, *Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC*](https://arxiv.org/abs/2302.11552), confident.

[Marta Skreta et al., 2025, *Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts*](https://arxiv.org/abs/2503.02819), confident.

[Duncan Soiffer, Chandler Squires, Yuan Guan, Jason Hartford, and Pradeep Ravikumar, 2026, *Catastrophic Compositional Generation: Why Vanilla Diffusion Models Fail to Extrapolate*](https://arxiv.org/abs/2606.23920), confident.

**Verdict for the goal (needed, could help, or skip, with the reason)**

Needed. Without the oracle axis, a failed learned correction is not identifiable as a target, representation, estimation, or sampling failure.

## 8. Specify whether the chest target is observational or counterfactual

**What it explains**

From the supplied facts, pneumothorax without a drain is not absent from NIH. It accounts for 1,906 of 3,543 NEATX pneumothorax films. The supplied material also notes that a drain changes the lung it treats. Therefore an untreated pneumothorax without a drain and a treated lung with the drain digitally removed are different image distributions.

My reasoning is that the project must name its estimand before composing anything. The observational target is $p(x\mid\mathrm{PTX},\neg\mathrm{drain})$. The paired editing target is the counterfactual image of the same treated patient under drain removal. The stress-test target may intentionally be a treated-looking lung without the device. These targets answer different questions and require different validation.

**Where it attaches (the ladder layer and the idea-map claim)**

It belongs at layer 7 and before layer 12’s drain experiment. It sharpens the parallel map’s repaired claims 6 and 8 and the parts-world warning about a treated-lung chimera.

**The smallest check (a 2D or 16-pixel toy computation with exact answers where possible, with what result supports the idea, what counts as a null, and what would surprise)**

Use a 16-pixel Gaussian toy. Eight pixels represent lung appearance and eight represent the device region. Give every coordinate standard deviation $0.1$.

Let untreated pneumothorax have mean $+1$ on all lung pixels and $0$ on device pixels. Let treated pneumothorax have mean $-1$ on lung pixels and $+1$ on device pixels. Let the paired drain-removal counterfactual retain the treated lung mean $-1$ but set the device mean to $0$.

Both the observational untreated target and the counterfactual target are perfectly drain-free. Their mean distance is

$$
\sqrt{8(1-(-1))^2}=\sqrt{32}=5.657
$$

by hand. A midpoint lung with mean $0$ is $ \sqrt8=2.828 $ from both targets by hand. It is the toy’s ambiguous chimera.

Support is concentration near the target appropriate to the declared use. Training augmentation for untreated disease should approach the $+1$ lung prototype. Paired shortcut stress-testing should approach the $-1$ prototype. The null removes the treatment effect by setting both lung means to $+1$, making the two targets identical. It would be surprising if a drain classifier or a pneumothorax label alone distinguished the two drain-free targets. The required measure is phenotype-conditioned coverage against real held-out films, not only label success.

**Cost (hours, and whether it needs training)**

Two to four hours for the exact toy. No training is needed. Writing the scale evaluation protocol would take another eight to sixteen hours and requires clinical review, but not necessarily new generator training.

**Prior work (author, year, title; mark each "confident" or "recalled, verify", and never invent a paper)**

[Luke Oakden-Rayner, Jared Dunnmon, Gustavo Carneiro, and Christopher Ré, 2019, *Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning for Medical Imaging*](https://arxiv.org/abs/1909.12475), confident.

[Fernando Pérez-García et al., 2024, *RadEdit: Stress-Testing Biomedical Vision Models via Diffusion Image Editing*](https://arxiv.org/abs/2312.12865), confident.

[Nina Weng, Paraskevas Pegios, Eike Petersen, Aasa Feragen, and Siavash Bigdeli, 2024, *Fast Diffusion-Based Counterfactuals for Shortcut Removal and Generation*](https://arxiv.org/abs/2312.14223), confident.

Donald B. Rubin, 1974, *Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies*, confident.

**Verdict for the goal (needed, could help, or skip, with the reason)**

Needed before the chest X-ray experiment. Otherwise a composition can be visually and statistically successful while answering the wrong medical question.

## Ranked by movement toward the goal per hour

1. **Conditional independence and the label interaction.** It fixes the definition of $r$, resolves the additive-constant issue, and costs only two to three hours.

2. **Learn the identifiable interaction score.** It converts the definition directly into the correct LoRA target with a very small exact check.

3. **The oracle correction cube.** It makes the layer 9 result interpretable and prevents sampler or expert error from being credited to the learned correction.

4. **The interaction trajectory across diffusion time.** It tests the actual object used by the scale model rather than only its endpoint.

5. **Guidance tempering and negation integrability.** It is essential for interpreting guidance $7.5$ and for preventing invalid chest-negation targets.

6. **The observational versus counterfactual chest target.** It is indispensable before medical claims, although it becomes most valuable near layer 12.

7. **Fixed decoders before VAEs.** It gives a clean answer about coordinate volume and noninvertibility, but VAE training is not required for the core correction.

8. **Riemannian geometry.** It provides a principled representation correction and useful null tests, but it is unlikely to explain the semantic interaction by itself.