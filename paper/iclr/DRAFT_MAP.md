# The draft, piece by piece

## abstract
- [x] sketch drafted, 6 sentences minted below
- [x] sentence 1: A natural image typically depicts several distinct concepts arranged so that they form a single coherent scene, and text-to-image models generate compositional scenes from a description of their parts.
- [x] sentence 2: Recent methods compose pretrained diffusion models at inference time by sampling from a product of experts, one expert per concept.
- [x] sentence 3: Product-of-experts composition can fail catastrophically, producing a single blended chimera instead of a scene containing both concepts, and we trace this failure to a specific, correctable gap.
- [x] sentence 4: This gap is the interaction term: the difference between the sample a model conditioned on the true joint prompt would produce and the sample the product of experts actually produces, and it is exactly what the independence assumption drops.
- [x] sentence 5: We train a low-rank adapter on the cross-attention layers of Stable Diffusion XL to predict this interaction term at every denoising step without ever seeing the joint prompt and add its prediction back in as a correction.
- [x] sentence 6: We show the correction transfers to concept pairs it was never trained on, and that adding more of it raises the compose rate toward a scene that composes both concepts correctly. (numbers deliberately left out, not blocked: this phrasing is qualitative by choice, not waiting on a review file)
- [ ] full paragraph assembled, ready for `compile`   ← current

## introduction

Second walk. The pieces below are re-minted from what is in `iclr2027_conference.tex` today,
which carries three displayed equations the first walk never minted. This walk's question is how
the mathematics enters: whether each symbol is earned before it appears, and whether a reader
meets the failure before the formalism that names it.

- [x] the order decision, settled: introduce product-of-experts, show it fail, then name the
      interaction term and the strength lambda that scales it. Two displayed equations, both in
      score space. Where the interaction term comes from, the derivation through the two
      concepts' conditional dependence, moves to its own later section (title and placement not
      yet decided). Classifier-free guidance, the noise-prediction form r_t, and the w sigma_t
      conversion leave the introduction entirely and belong wherever guidance gets defined.
- [x] paragraph 1, settled, author's wording: Natural images usually depict several objects that
      appear together and interact as part of a coherent scene. A cat may sit beside a dog, or a
      cup may rest on a table. Thus, generating such scenes requires more than producing each
      object correctly. The model must also capture how the objects are arranged and how they
      relate to one another. A cup described as resting on a table should sit naturally on the
      tabletop rather than float above it, appear underneath it, or merge into its surface.
- [x] paragraph 2, settled, author's wording: Recently, text-to-image diffusion models (Saharia
      et al., 2022; Podell et al., 2023) have demonstrated a remarkable ability to generate
      complex scenes with high visual fidelity. Much of this progress has been driven by scaling
      model capacity and training data, enabling these models to represent a broad range of
      objects and their interactions within a scene. Recent work has shown that these models can
      also capture aspects of the physical structure and dynamics of visual scenes (Author et al.,
      YYYY). Despite this progress, they remain unreliable on several forms of compositional
      reasoning, including counting (Author et al., YYYY), negation (Author et al., YYYY), and
      attribute binding (Author et al., YYYY). Many specific concept compositions also remain rare
      or unseen during training, motivating methods that construct them directly at inference
      time. This has led to approaches that compose pretrained diffusion models without additional
      training (Liu et al., 2022).
      FOUR CITATIONS OPEN: physical structure and dynamics, counting, negation, attribute binding.
      Paper-scout prompts written for each. The combinatorial-coverage prompt was dropped: the
      rarity sentence carries no citation slot.
- [ ] the PoE build, replaces the single paragraph 3. Four paragraphs that build from diffusion
      to PoE in the reader's order, sketch drafted in-chat, sub-pieces below. Home notation is the
      noise prediction epsilon_theta; the score appears once as its probabilistic reading.
      Placement (introduction vs. its own background section) deliberately open.
      - [x] ¶A settled, author's wording: Diffusion models generate images by learning to
            reverse a gradual noising process (Ho et al., 2020). In the forward process, Gaussian
            noise is progressively added to a clean image x_0, producing increasingly noisy
            samples x_1, ..., x_T. The reverse process is learned by a neural network that
            predicts the noise eps_theta(x_t, t, c) present in x_t, where t denotes the diffusion
            timestep and c is an optional conditioning signal such as a text prompt. At inference
            time, generation begins from Gaussian noise x_T and repeatedly applies the learned
            denoising process until a clean sample x_0 is obtained. In text-to-image diffusion
            models, the conditioning c guides this reverse process toward images that match the
            input prompt.
            (needs ho2020denoising, NOT IN BIB. Join note: if the build stays in the
            introduction, the opener takes one bridging clause from paragraph 2; as a section
            opener it stands as written. Notation open: comma form eps_theta(x_t,t,c) as written
            vs bar form eps_theta(x_t,t|c); the bar recommended since ¶C/¶D place conditional
            and unconditional side by side.)
      - [ ] ¶B the score reading: p_t(x_t|c), epsilon approximates -sigma_t times its score,
            steps accumulate into sampling (cites song2021scorebased)   ← current
      - [ ] ¶C guidance and sampling: Bayes split of the conditional, CFG amplifies the prompt's
            share by w, display equation for the guided prediction (needs ho2022classifierfree,
            NOT IN BIB)
      - [ ] ¶D PoE from distributions down to sampling: one more factor in the same
            factorization, display equation for the product of densities, composed prediction
            inline, exactness needs conditional independence, natural concepts interact (cites
            liu2022compositional, bradley2025mechanisms)
- [ ] Figure 1 and its paragraph: the cat x dog chimera, what the schematic panel may and may not claim (register row F1)
- [ ] paragraph 4 with Eq. 2: the interaction term as the gap between the joint-prompt score and the product-of-experts score, the strength lambda that scales it, lambda=0 is plain composition and lambda=1 is the joint-prompt score, one forward pointer to the section that derives where the term comes from
- [ ] paragraph 5: the three matched controls (wrong pair, wrong seed, shuffled step order)
- [ ] Figure 2 and its paragraph: the matched correction restores composition, the controls do not (register row F2, bar ✅, caption stays qualitative by register instruction). Eq. 2 sits on the same page so the lambda strip reads against it
- [ ] Contributions block (tex 149-156): three bullets, define / show causal / learn and transfer
- [ ] full introduction assembled, ready for `compile`
- OWED to the later section: the backwards derivation. The ratio of the true joint conditional to
  the product-of-experts surrogate splits into an x_t-dependent factor and a constant, so the
  interaction term is the gradient of the log ratio between the two prompts' joint and separate
  likelihoods given the image. It is zero exactly when the two concepts are conditionally
  independent given x_t. Needs a verbatim check of what Liu et al. assume before it is written as
  a re-derivation rather than a new result.
- PARKED: where the two Skreta papers (2412.17762, 2503.02819) go, likely Related Work, not decided yet, not blocking
- OPEN: the tex includes `F1-two-meanings.png` and `F2-correction-strength.png`; the register names `F1-two-meanings.pdf` and `F2-dose-response.pdf`. `F2-correction-strength.png` is not in `figures/`, so the build is showing a stale or missing file. Resolve before `compile` on the Figure 2 piece.

## related work
- [ ] not yet broken into paragraphs   (blocked: spine exists, no paragraph-level pass yet)

## problem setting and background
- [ ] not yet broken into paragraphs   (blocked: spine exists, no paragraph-level pass yet)

## methodology
- [ ] not yet broken into paragraphs   (blocked: spine exists, no paragraph-level pass yet)

## benchmark design and experiments
- [ ] not yet broken into paragraphs   (blocked: figure register slots F2, F6, F7, F8 not all fillable)

## discussion
- [ ] not yet broken into paragraphs   (blocked: mechanism and limitations plan, writing-06, not started)

## conclusion
- [ ] not yet broken into paragraphs   (blocked: everything above)
