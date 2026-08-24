# CoInD: enabling logical compositions in diffusion models

Gaudi, Sreekumar, Boddeti (Michigan State). ICLR 2025. arXiv 2503.01145.
Code: [github.com/sachit3022/compositional-generation](https://github.com/sachit3022/compositional-generation/).

> [!aside]
> The arXiv listing page shows this as a preprint with no venue. The PDF header says
> "Published as a conference paper at ICLR 2025", so the venue is real.

Read from the full 33-page PDF, every appendix included.

## The problem

Train a diffusion model on colored MNIST digits where every one of the 100 digit-colour pairs
appears equally often. Now ask it for "digit 4 AND cyan" using the standard composition rule,
adding the score for "4" and the score for "cyan" and subtracting the unconditional score. It
gets the attributes right 98.15% of the time, while asking the same model directly for the
pair gets 99.98%. That 1.8 point gap should not exist: the model saw every pair, so the two
routes to the same image should agree.

Now train on partial support, where digit $d$ only ever appears in colours $d$ and $d+1$.
Asking directly gets 33.14%. Composing from marginals gets **7.40%**. The composition rule
does not just fail to help, it destroys what the model knew. The paper exists because that gap
traces to a specific assumption everyone was making silently.

## Solution overview

The composition rule $p(C_1, C_2 \mid X) = p(C_1 \mid X)\,p(C_2 \mid X)$ is an assumption
about the trained model, and nobody had checked whether training produces a model that
satisfies it. CoInD checks (it does not) and then adds a second loss term during training that
forces the model's own joint score to equal its own sum-of-marginal-scores.

**This is a training-time fix, not a sampling-time one.** The model is made to satisfy the
assumption that the sampler was already relying on. Everything else stays the same: same
architecture, same classifier-free guidance, same DDIM sampler, same composition equations at
inference.

## Background

### Score-based models

The network learns $s_\theta(x) = \nabla_x \log p_\theta(x)$, the gradient of log-density with
respect to the image. In a diffusion model it is parameterised through noise prediction:

$$s_\theta(x_t, t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}},
\qquad x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

Everything in this paper is written in score space and implemented in $\epsilon$ space, and the
conversion factor matters later.

### Classifier-free guidance with per-attribute dropout

This is the piece that makes the whole method possible and it is easy to skim past. During
training each attribute $c_k$ is independently replaced by the null token $\varnothing$ with
probability $p_{\text{uncond}}$ (0.2 here). One network therefore learns *every subset* of the
attributes at once. Setting all but attribute $i$ to $\varnothing$ gives the marginal score
$s_\theta(X \mid C_i)$; setting all of them to $\varnothing$ gives the unconditional score.

So a single 8.2M-parameter UNet can produce the joint score, either marginal, and the
unconditional score, just by changing which slots hold $\varnothing$.

### Liu et al.'s composition rules

CoInD keeps these unchanged at sampling time.

<a id="eq-and"></a>**(1) The AND rule**

$$\nabla_X \log p_\theta(X \mid C_1 \wedge C_2)
= \nabla_X \log p_\theta(X \mid C_1) + \nabla_X \log p_\theta(X \mid C_2)
- \nabla_X \log p_\theta(X)$$

The NOT rule, for $C_1 \wedge \neg C_2$:

$$\nabla_X \log p_\theta(X \mid C_1 \wedge \neg C_2)
= \nabla_X \log p_\theta(X) + \nabla_X \log p_\theta(X \mid C_1)
- \nabla_X \log p_\theta(X \mid C_2)$$

and a weighted form swapping the unit coefficients for $\gamma$ to control attribute strength.

Both rules rest on the CI relation $p(C \mid X) = \prod_i p(C_i \mid X)$. The NOT rule
additionally rests on the approximation $p(\neg C_2 \mid X) \propto 1 / p(C_2 \mid X)$, which
CoInD does **not** fix.

### The implicit classifier

You can read a classifier out of a trained diffusion model without training one:

$$p_\theta(C_i = c_i \mid x) \propto
\exp\left\{ -\mathbb{E}_{t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t, c_i)\|^2\right] \right\}$$

The condition that best explains the noise is the most likely condition. This is how the paper
measures whether the model's internal conditionals factorise.

> [!aside]
> From Li et al. 2023, "Your Diffusion Model is Secretly a Zero-Shot Classifier".

### The three training supports

This is the axis every table varies.

**Uniform**

All 100 digit-colour pairs equally likely.

**Non-uniform**

All pairs appear, but digit $d$ with colours $d$, $d+1$ at probability 0.5 total.

**Diagonal partial**

Digit $d$ appears only with colours $d$ and $d+1$, so 80 of the 100 pairs are never seen.

### Fisher divergence

The expected squared difference between two score functions. Its one load-bearing property: it
is zero if and only if the two distributions are equal. This is what lets a score-space penalty
enforce a distribution-space equality.

## Methodology, step by step

The training loop, one operation per step, in their Colored MNIST setting because it is the
smallest place where all four network evaluations are distinguishable.

1. **Draw a training pair.** Image $x_0$ of shape `[1, 3, 28, 28]` and its attribute vector
   $c = [\text{digit}=4,\ \text{colour}=\text{cyan}]$.
2. **Apply guidance dropout.** Each attribute independently becomes $\varnothing$ with
   probability 0.2. Call the result $c$.
3. **Pick two attribute slots at random.** $i \sim \text{Uniform}\{0..N\}$,
   $j \sim \text{Uniform}\{0..N\} \setminus \{i\}$. With only two attributes these are always
   digit and colour; with Shapes3D's six attributes this is where the cost saving happens.
4. **Draw a timestep and noise.** $t \sim \text{Uniform}\{1..1000\}$,
   $\epsilon \sim \mathcal{N}(0, I)$ of shape `[1, 3, 28, 28]`.
5. **Make the noisy image.**
   $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$, shape
   `[1, 3, 28, 28]`.
6. **Build four condition vectors from the same $c$**, differing only in which slots are masked:
   - $c_i = [4, \varnothing]$
   - $c_j = [\varnothing, \text{cyan}]$
   - $c_{i,j} = [4, \text{cyan}]$
   - $c_\varnothing = [\varnothing, \varnothing]$
7. **Run the same UNet four times**, once per condition vector, each returning a noise
   prediction of shape `[1, 3, 28, 28]`.
8. **Form the independence penalty**, equation [(5)](#eq-lci-eps) below.
9. **Form the standard loss:** $\mathcal{L}_{\text{score}} = \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2$.
10. **Step on the sum:**
    $\nabla_\theta[\mathcal{L}_{\text{score}} + \lambda \mathcal{L}_{\text{CI}}]$.

**Reading step 8 in words:** what the model predicts when told both attributes should equal what
it predicts from each attribute separately, once the doubly-counted unconditional part is
subtracted. If it does, the term is zero and the composition rule at sampling time is exactly
valid for this model.

<figure class="slot">
<img src="figures/coind-four-masks.png" alt="One UNet evaluated four times under four attribute mask patterns">
<figcaption>
The same network, four condition vectors, one penalty.
<span class="slot-note">Waiting on <code>figures/coind-four-masks.png</code>: generate from the prompt below (class: annotated-diagram) and save it here.</span>
</figcaption>
</figure>

**Prompt** (class: annotated-diagram, save as `figures/coind-four-masks.png`)

```text
A dark background. One rounded rectangle in the centre representing a single
neural network, drawn plainly with no icon and no shading. Four thin lines
enter it from the left, each carrying a two-slot condition vector drawn as
two small squares side by side. From top to bottom the four vectors are:
first slot filled, second slot empty; first slot empty, second slot filled;
both slots filled; both slots empty. Filled slots in one restrained accent
colour, empty slots drawn as hollow outlines. Four lines leave the network
on the right and converge into a single summation point, with the top two
carrying a plus sign and the bottom two carrying a minus sign in a second
accent colour. Symbols in clean serif mathematical type. No axes, no
gridlines, no cards, no pills, no flowchart boxes, no labels beyond the
plus and minus signs.
```

### A detail worth not skipping

Converting the score-space loss to $\epsilon$ space introduces a factor
$1/(1 - \bar{\alpha}_t)$. Appendix D.1 then weights the term by $(1 - \bar{\alpha}_t)$,
following Ho et al. 2020, which cancels it exactly. That is why Algorithm 1 shows a plain
unweighted squared norm with no timestep factor anywhere.

> [!aside]
> The cancellation is stated in one sentence in the appendix and is easy to read as arbitrary.

### Cost

Roughly five UNet forward passes per training step instead of one. The paper never states the
resulting wall-clock or memory overhead.

## The mathematics

The derivation has four moves. Only the first three change how the method reads.

### Move 1: write the target as a factorisation, then measure distance to it

Under the causal graph plus $C_1 \perp \dots \perp C_n \mid X$, Bayes gives what the paper
calls the JM relation:

<a id="eq-jm"></a>**(2) The JM relation**

$$p(X \mid C) = \frac{p(X)}{p(C)} \prod_i \frac{p(X \mid C_i)\, p(C_i)}{p(X)}$$

The left side is what you sample when you condition directly. The right side is what you sample
when you compose. **The whole paper is the observation that a trained model does not make these
equal.** Their Eq 6 measures the gap with a 2-Wasserstein distance between the true
$p(X \mid C)$ and the model's factorised version.

### Move 2: split it with the triangle inequality

$$\mathcal{L}_{\text{comp}} \le
\underbrace{W_2\big(p(X \mid C),\ p_\theta(X \mid C)\big)}_{\text{distribution matching}}
+ \underbrace{W_2\big(p_\theta(X \mid C),\ p_\theta\text{'s factorisation}\big)}_{\text{conditional independence}}$$

The first term is "did you learn the data", the second is "are you internally consistent". The
first is what vanilla training already optimises. The second is the one nobody was optimising.

<figure class="slot">
<img src="figures/coind-triangle-split.png" alt="One distance split into two legs through the model's own conditional">
<figcaption>
The gap nobody was measuring is the second leg.
<span class="slot-note">Waiting on <code>figures/coind-triangle-split.png</code>: generate from the prompt below (class: geometry) and save it here.</span>
</figcaption>
</figure>

**Prompt** (class: geometry, save as `figures/coind-triangle-split.png`)

```text
A dark background. Three points arranged as a wide triangle in a plain
open space suggesting a space of distributions, with no axes and no
gridlines. A dashed line runs directly between the top-left and the
bottom-right point, drawn in a muted neutral tone. Two solid lines run
from the top-left point up through the top-right point and down to the
bottom-right point, forming the other two sides. The first solid leg is
drawn in one restrained accent colour, the second solid leg in a second
accent colour that reads as brighter or more saturated so the eye lands
on it. The three points are small filled dots with clean serif
mathematical labels beside them. No arrows, no boxes, no legend, no
caption text inside the image.
```

### Move 3: convert both terms to score matching

Kwon et al. 2022 give
$W_2(p_0, q_0) \le K\sqrt{\mathbb{E}\|\nabla_x \log p_0 - \nabla_x \log q_0\|^2}$. Applied to
term one this recovers the standard diffusion loss. Applied to term two it gives the
independence penalty:

<a id="eq-lci-score"></a>**(3) The independence penalty, score space**

$$\mathcal{L}_{\text{CI}} = \mathbb{E}\left\|
\nabla_X \log p_\theta(X \mid C) - \nabla_X \log p_\theta(X)
- \sum_i \big[\nabla_X \log p_\theta(X \mid C_i) - \nabla_X \log p_\theta(X)\big]
\right\|_2^2$$

**Read aloud:** the joint score, minus the unconditional score, should equal the sum of each
marginal's departure from the unconditional score.

> [!aside]
> The $\nabla_X \log p_\theta(C_i)$ and $\nabla_X \log p_\theta(C)$ terms vanish because they do
> not depend on $X$. That is why the prior terms in the JM relation never appear in the loss.

### Move 4, flagged rather than derived

Appendix B.3 assembles these into

$$\mathcal{L}_{\text{comp}} \le K_1 \sqrt{\mathcal{L}_{\text{score}}} + K_2 \sqrt{\mathcal{L}_{\text{CI}}}$$

**The paper does not optimise this.** It optimises

<a id="eq-final"></a>**(4) The objective actually trained**

$$\boxed{\ \mathcal{L}_{\text{final}} = \mathcal{L}_{\text{score}} + \lambda\, \mathcal{L}_{\text{CI}}\ }$$

which drops both square roots and is a different objective, justified by gradient stability and
by reusing existing hyperparameters. They test both, and the results are below.

### Two approximations sit between the derivation and the code

Mutual conditional independence is replaced by pairwise (Hammond & Sun 2006, an equivalence
that holds at large $n$), reducing model evaluations from $O(n)$ to $O(1)$. In $\epsilon$ space
the implemented penalty is:

<a id="eq-lci-eps"></a>**(5) The independence penalty as implemented**

$$\mathcal{L}_{\text{CI}} = \big\|\, \epsilon_\theta(x_t, t, c_i) + \epsilon_\theta(x_t, t, c_j)
- \epsilon_\theta(x_t, t, c_{i,j}) - \epsilon_\theta(x_t, t, c_\varnothing) \,\big\|_2^2$$

Then for Shapes3D specifically, Appendix D.1 says they actually enforced
$C_i \perp C_{-i} \mid X$ rather than pairwise, because it "led to slightly better results".
That is a third variant of the objective, disclosed only in an appendix line.

## Results

### The headline finding needs no method

Under uniform support, where every attribute pair was seen equally often, a standard Composed
GLIDE model still has JSD = 0.16, not zero. The model invented a dependence between digit and
colour that the data did not contain. This is the result that makes the rest of the paper
necessary, and it costs nothing to check.

### Colored MNIST, diagonal partial support

The hardest setting. All numbers are conformity score in percent, the fraction of generated
images whose attributes match the requested logical expression.

| | AND | NOT colour | NOT digit | JSD |
|---|---|---|---|---|
| LACE | 10.85 | 9.03 | 28.24 | not measurable |
| Composed GLIDE | 7.40 | 5.09 | 33.86 | 2.75 |
| CoInD ($\lambda = 1.0$) | **52.38** | **53.28** | **52.59** | **1.17** |

7.1$\times$ on AND, 10.5$\times$ on NOT colour, 1.55$\times$ on NOT digit. Under uniform support
the AND gap is small (98.15 to 99.99) because there was little to fix.

### Shapes3D, orthogonal partial support

Six attributes, so this tests scaling. AND conformity 51.56 to 91.10, pixel-level $R^2$ against
the unique ground-truth image 0.86 to 0.97, JSD 0.503 to 0.287.

### The theory-faithful objective loses to the practical one

Table 3, Colored MNIST partial support, AND: the square-root version gets 23.44 while the
version at [(4)](#eq-final) gets 52.38. It wins on NOT colour (64.84 against 53.28) and loses on
the headline task by 2.2$\times$. Whatever is producing the main result, it is not fidelity to
the derived bound.

### CelebA is where the table caption and the numbers disagree

The caption says CoInD "outperforms the baselines on both CS and FID across various
compositionality tasks." For the "smiling" $\wedge$ "male" composition, conformity score is
LACE 24.20, Composed GLIDE 10.55, **CoInD 8.79**. CoInD is last, by 2.75$\times$ against LACE.

It does win on FID for that task (43.76 against 95.41 and 80.40) and on the joint-sampling
column (2.51 to 8.63). The claim holds for joint sampling and for FID, not for the composition
conformity score, and the caption does not make that distinction.

> [!aside]
> Every CelebA conformity score is under 25%, so no method does this task well.

### Fine-tuning Stable Diffusion v1.5

On CelebA text prompts: AND conformity 14.19 to 49.15, NOT 11.02 to 18.80, FID improving on
both. This is the only result on a pretrained large model.

### λ has an interior optimum

Appendix C.3: as $\lambda$ grows the conformity score rises then falls, because at large
$\lambda$ the model achieves independence by ignoring the conditions and sampling from the
prior. Nothing in the objective prevents that degenerate solution.

### Diversity, unrequested and free

Asked for "digit = 4" under non-uniform support, colour entropy is 2.63 bits for CoInD against
1.71 for Composed GLIDE, ceiling $\log_2(10) = 3.32$.

## Evaluation methodology

### Training procedure

**Objective:** equation [(4)](#eq-final), with the penalty in the $\epsilon$ form at
[(5)](#eq-lci-eps).

**λ values:** 0.2 and 1.0 for Colored MNIST, 1.0 for Shapes3D, **100 for CelebA**. That last is
a 100$\times$ jump with no explanation given at the point of use. Appendix C.3 offers a
principled rule (train a vanilla model, set
$\lambda = \mathcal{L}_{\text{score}} / \mathcal{L}_{\text{CI}}$) but then says only that two
values were tried and gave similar results. **Whether $\lambda = 100$ came from that rule is not
stated.**

**Guidance dropout:** $p_{\text{uncond}} = 0.2$ throughout.

**Data:** Colored MNIST (10 digits $\times$ 10 colours from a fixed 10-colour palette, three
supports with analytical forms given in D.4); Shapes3D (six attributes, orthogonal split from
Schott et al.'s public code); CelebA (two attributes, "male" and "smiling", trained on all
combinations except male $\wedge$ smiling).

**Hyperparameters that matter** (Table 5): AdamW, learning rate $2\times10^{-4}$, DDPM linear
schedule with 1000 noise steps, UNet with 2 layers per block, dropout 0.1. 50,000 training steps
for Colored MNIST, 100,000 for Shapes3D. 8.2M parameters for Colored MNIST, 17.2M for Shapes3D.
CelebA differs: $128 \times 128$ images encoded through the **Stable Diffusion 3 latent
encoder**, block channels [224, 448, 672, 896], learning rate $1\times10^{-4}$, 500,000 steps.

**Hardware and time:** one A6000 GPU for CelebA. **Nothing disclosed for Colored MNIST or
Shapes3D, and no wall-clock or training-time figure anywhere.** Given the roughly 5$\times$
forward passes per step, the cost of the method is the number a practitioner most needs and it
is absent.

**Not mentioned anywhere:** EMA, learning-rate warmup, gradient clipping, batch size. Batch size
in particular is absent from Table 5 despite the table being presented as complete.

**Baseline fairness, one point in CoInD's favour and one against.** LACE gets 2$\times$ the
parameters on Colored MNIST and 6$\times$ on Shapes3D, since it trains one model per attribute,
so it is not starved overall. But for Shapes3D they "reduce the Block Out Channels for each
attribute model to fit these into memory", so each LACE expert is individually smaller than the
CoInD model. LACE also cannot be assigned a JSD at all, since it never models the joint, which
is why that column is empty rather than bad.

### Inference procedure

**Sampler:** DDIM. Table 5 says 150 steps for Colored MNIST and 100 for Shapes3D; the D.3 prose
says "100 steps" without qualification. **Minor inconsistency, unresolved in the paper.**

**Composition happens at sampling time using the unmodified Liu et al. equations**, the AND rule
at [(1)](#eq-and) and its NOT counterpart. Nothing about CoInD changes inference. The Shapes3D
AND composition sums six marginal scores and subtracts five copies of the unconditional; the NOT
composition on a 4-valued shape attribute is expanded as $\neg[0 \vee 1 \vee 3]$, costing three
extra network evaluations.

**Guidance scale:** the composition weights *are* the guidance here, with unit coefficients.
$\gamma$ appears only in the weighted form for the deliberate attribute-strength sweep
($\gamma \in \{0, 1, 2, 6\}$ on CelebA faces), with $\gamma = 1$ for the reported CelebA table.
**No separate classifier-free guidance scale is reported for any main result.** For a same-model
comparison this is internally fair, since all methods use unit coefficients, but it means none of
these numbers are at the guidance settings a practitioner would actually deploy.

**The NOT rule stays approximate.** $p(\neg C_2 \mid X) \propto 1 / p(C_2 \mid X)$ is unchanged
by CoInD, and Appendix G.2 attributes a named class of CoInD's own failures to exactly this.
Worth holding onto: CoInD fixes what the model learned, not what the sampling rule assumes.

**No test-time tricks.** No best-of-N, no reranking, no prompt engineering. The generated sample
is scored as produced.

### Qualitative assessment

**What is shown:** Fig 1d (Colored MNIST, 4 logical expressions $\times$ 3 methods $\times$ 3
supports), Fig 5 (diversity grids for "digit=4"), Fig 6b (Shapes3D against the unique expected
image), Figs 7 and 11 (the $\gamma$ sweep on faces), Fig 13 (fine-tuned SDv1.5), Fig 16 (failure
cases).

**Selection protocol, disclosed once and only once.** Fig 13 states that columns 1, 3, 5 share a
random seed and columns 2, 4 share another. That makes the SDv1.5 comparison a genuine
matched-seed comparison, and the surrounding argument (CoInD's AND-composed image looks like its
own joint-sampled image; Composed GLIDE's drifts toward female-associated features) is
well-supported by it.

**Everywhere else, no seed protocol and no selection criterion.** Fig 1d, Fig 5, Fig 6b and Fig 7
do not say whether samples were random with fixed seeds across methods, hand-picked, or
best-of-N. Treat those four as illustration rather than evidence. Fig 6b is the one where this
bites hardest, because Shapes3D has a unique correct image per prompt, so a matched-seed grid
would have been strong evidence and the paper leaves that on the table.

**Counts are small and mostly unstated.** Fig 5 shows one grid per method; the number of samples
is not given in text. Fig 6b shows three rows.

**Credit where it is due.** Appendix G.2 is a dedicated failure-mode appendix with named
categories per dataset: colour leaking from adjacent seen combinations, wrong attributes traced
to the NOT approximation, and plain unrealistic samples. Appendix G.3 goes further, giving a
per-combination conformity heatmap. This is more self-criticism than most papers in this area
publish, and it should count in the paper's favour.

### Quantitative assessment

**Conformity score (CS), the primary metric.** Generate an image for a logical expression, run a
classifier over it, check whether the inferred attributes satisfy the expression, average. The
classifier is a **single ResNet-18 with one head per attribute, trained on full support**. Its
accuracies (Fig 10): Colored MNIST digit 98.93% and colour 100%; all six Shapes3D attributes
100%; CelebA gender 98.2% and smile 92.1%.

> [!aside]
> CS is ceiling-limited by the classifier. On CelebA the scores (2.51 to 24.20) are so far below
> that ceiling that classifier error is not what is limiting them.

**CS averages over a bimodal distribution, which the paper discloses and which changes how to
read every headline number.** Appendix G.3's heatmap for the 52.38% partial-support result shows
unseen combinations scoring above 90% and others at exactly 0%. That number is not "52% competent
everywhere", it is "some pairs work, some are dead". Colours 2 and 3 fail against most digits
while 4, 5 and 6 succeed against all of them, and the paper offers only a hypothesis for why.

**How many prompts and samples CS averages over is never stated.** Several Shapes3D values land
exactly on multiples of $1/64$ (51.56 = 33/64, 48.43 = 31/64, 92.19 = 59/64, 95.31 = 61/64,
23.44 = 15/64), which suggests an evaluation set of 64 for at least some cells.

> [!aside]
> That is inference from the decimals, not something the paper says. If it is 64, differences of
> a few points are one or two images.

**JSD, the mechanism metric, has an unexplained scale.** Defined as
$\mathbb{E}[D_{\text{JS}}(p_\theta(C \mid X) \,\|\, \prod_i p_\theta(C_i \mid X))]$, with both
distributions read out via the implicit classifier. **A standard Jensen-Shannon divergence is
bounded above by 1 bit (0.693 nats).** Reported values include 2.75, 2.44 and 1.82. So the
reported quantity is not a plain bounded JSD, and the paper never states the log base, the
normalisation, or any summation that would explain values above the bound. The relative ordering
across methods is still informative; the absolute values cannot be interpreted as written.

**Two approximations inside the JSD, both consequential.** First, the implicit-classifier
expectation is estimated from **5 timesteps sampled in [300, 600]** rather than across $[0, T]$,
following Kynkäänniemi et al. The JSD number therefore depends on an arbitrary mid-range window,
and no sensitivity check is shown. Second, for Shapes3D's six attributes the JSD is computed
**only between $C_1$ and $C_2$**. The paper says computing it between other pairs "does not
change our examples' conclusion" and shows no supporting numbers.

**$R^2$ (Shapes3D only):** variance-weighted coefficient of determination at the pixel level
against the unique ground-truth image for that attribute tuple. Reported range 0.61 to 0.98. Only
usable on Shapes3D because only there is the target image unique.

**FID (CelebA only):** pytorch-fid, 10,000 generated samples, computed **against the held-out
smiling-male subset specifically** rather than against CelebA as a whole. That is the right
reference set for the question being asked. The size of that real subset is not stated, and FID is
biased upward on small reference sets, so the absolute values (43.76 to 95.41, all high) should be
compared to each other and not to published CelebA FIDs.

**Baselines:** LACE (Nie et al. 2021, reimplemented with score-based models rather than the
original EBMs, a substitution the paper discloses) and Composed GLIDE (Liu et al. 2022). Both are
the natural comparisons. No comparison against any 2023 to 2024 composition method; Du et al.
2023's MCMC samplers are cited in related work but not run.

**The largest gap: no error bars, no seeds, anywhere.** Every number in Tables 1, 2, 3, 4, 6, 7
and Figs 4a, 6a, 15b is a single run. No variance, no seed count, no confidence intervals, no
repeated training. For the partial-support Colored MNIST result (7.40 to 52.38) the effect is
large enough that run-to-run variance is unlikely to explain it. For the CelebA results, where
CoInD's AND conformity of 8.79 sits against Composed GLIDE's 10.55, and for the comparison
between the two objectives, single runs cannot support the conclusions being drawn either way.

## What this breakdown could not establish from the paper alone

Training wall-clock and the actual overhead of the roughly 5$\times$ forward passes; batch size;
whether $\lambda = 100$ for CelebA came from the stated
$\lambda = \mathcal{L}_{\text{score}} / \mathcal{L}_{\text{CI}}$ rule; the number of prompts and
samples per prompt behind each conformity score; the normalisation that lets the reported JSD
exceed the standard bound; and the seed and selection protocol for every qualitative figure
except Fig 13. The code release is the place to settle the first four.

## Why this paper matters here

The penalty at [(5)](#eq-lci-eps) rearranges to

$$\big\| \underbrace{\big[\epsilon_\theta(c_i) + \epsilon_\theta(c_j) - \epsilon_\theta(c_\varnothing)\big]}_{\epsilon_{\text{PoE}}} - \underbrace{\epsilon_\theta(c_{i,j})}_{\epsilon_{\text{joint}}} \big\|^2$$

which is $\|r_t\|^2$ exactly. CoInD's whole method is driving that residual to zero during
training, which is the opposite of retaining it and asking what structure it has.

The two approximations noted above (pairwise instead of mutual, and the Shapes3D switch to
$C_i \perp C_{-i} \mid X$) are also the closest thing in the literature to a statement about how
the term behaves across attribute pairs.
