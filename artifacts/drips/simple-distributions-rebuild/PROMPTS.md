# The prompts that walk the composition ladder

This is the one guide for walking from a single 2D diffusion expert to a correction for product-of-experts composition that scales to diffusion models. Each layer is one file in [the ladder folder](ladder/), walked in the order the table below gives. Under each prompt is what you have when it finishes. Paste the prompts one at a time.

**The goal they serve.** Understand product-of-experts composition well enough to define the interaction term, show it, and learn a correction that fixes it at scale (Stable Diffusion, then chest X-rays). You should have mastery of the critical maths and code, and leave the rest.

**Where it ends.** The last Achieve line is layer 12's: the LoRA understood as a learned interaction term and measured on held-out animal pairs, with its core (layers 8 and 9) rebuilt by you from an empty file.

**This is the only entry point.** Three idea walks fed this guide: `artifacts/ideas/poe-overlap-dial/`, `artifacts/ideas/when-poe-reaches-a-and-b/` and `artifacts/ideas/sdxl-shaped-hybrid-testbed/`. Their findings, and those of the sessions that ran them, are folded in here. Where one of them still has an open question, the prompt that resumes it sits at the layer that needs it. Start every new session from this guide's next prompt, never from another session's suggestion.

## How to read the marks

- ⭐ **Critical:** the goal rests on this maths or code. Do not skip it.
- ◇ **Optional:** it deepens understanding but the goal does not depend on it. Skip it the first time through.

Inside a critical layer, the parts you can skip are named under "Skip".

## How every layer runs

**The order inside a layer.** Quiz N.1 on the college maths the layer leans on, then the `--math` drip, then quiz N.2 on the layer itself, then the build, then the compare step, then the scene. The two quizzes are built in the background one layer ahead, so they are waiting when you arrive.

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
 quizzes N.1 and N.2, prepared ──────────────▶ │  1. quiz N.1: the college maths underneath
   (pasted during layer N−1's build)           │       ▼
                                               │  2. /drip --math: the derivation the code needs
                                               │       ▼
                                               │  3. quiz N.2: needed-now right, 80% of the rest
                                               │       ▼
                                               │  4. build: walkNN.py, typed by you
 rebuild/ reference (layers 6 to 9: built) ──▶ │  5. compare: the same numbers from both
                                               │       ▼
 the scene, built beside layer N+1 ◀────────── │  6. scene
                                               │       ▼
 xref-cadence links the layer both ways ◀───── │  7. /drip-retro, then layer N+1
```

**The maths comes first.** `/drip --math` teaches only the derivation the code needs, one rung per message, in LaTeX with worked numbers.

**The build comes next, and you type every line.** Where a step file exists (layers 1 to 5), `/drip-execute-plan <step file>` walks it one step per message, in the file's order. It ticks each step in the file and keeps its place across sittings in a walk map. It also opens on a bar and a time box and closes on a retrospective. Where no step file exists yet (layers 6 to 9), `/drip-execute-plan --request` cuts the layer into 4 to 6 pieces, writes the step file at its open, and builds, teaches and tests one piece per round. `/drip --code` is not used for the builds: it traces its own ladder instead of following a step file, and it insists on a throwaway virtual environment, which cannot use the `co3` PyTorch the toys need.

**When a step stalls on the maths.** The step files for layers 1 to 5 link a textbook page beside each fundamental they use (log rules, completing the square, a convolution, the Metropolis-Hastings acceptance), 42 links in all, so the passage that teaches it opens in one click. They also carry 24 tables of timestamped video segments (a proof, an intuition or a worked example per row) for the ideas their steps teach, from the score and score matching to total variation distance. MALA and the cosine schedule have no timed segment yet. Each layer's **Study first** line lists the journey plans, vault notes, textbooks and videos for the layer as a whole.

**Where you type.** `~/Playground/PhD/simple-distributions-walk/`, one file per layer (`walk01.py`, `walk02.py` and so on), run with `/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python`, set up once by [the setup file](ladder/00-setup.md). Nothing is installed for a walk: `co3` already has torch, and any other interpreter is borrowed from [the environments list](/home/molef/.claude/ENVIRONMENTS.md), which `plumb` reads before it creates anything.

**Two lanes, and where they meet.** Claude's reference code in `toy/rebuild/` is the AI lane: built for layers 1 to 5, and built one layer ahead for layers 6 to 9 by the background prompt at the top of those layers. Your walk files are your lane. They meet in each layer's **Compare** line: your file and the reference print the same numbers from the same seed, and a difference is either a lesson or a bug in one of them. Every worked number for layers 1 to 5 prints from `python -m rebuild.walk_numbers`, run from toy/. The AI lane never waits for your quizzes.

**The quiz ladder.** Quiz 0 holds the college maths under the whole ladder. Each layer then has quiz N.1 on the college maths that layer adds, and quiz N.2 on the layer's own concepts, so every quiz rests on the ones before it. Each quiz is prepared before it is built. `quiz-prep-pulse --build` traces every question back to its prerequisites and links each one to a textbook passage, an Obsidian note and a timestamped video, writing what is missing. It also adds graded problems from your textbooks to the quiz and lists them in the worked-examples index. `quiz-cadence` then publishes the page with a read-first list at the top, minutes per item. Coverage files land in `.quiz-prep/` beside this guide.

**The bar for every quiz.** Every question marked *needed now* right, and at least 80% of the rest (an estimate: change it here if it proves too hard or too easy). Below the bar on N.1, read the read-first items the missed questions point at and retake. Below the bar on N.2, the misses become the first rungs of a second `/drip --math` and you retake before building. The miss ledger (`quiz-ledger`) records every attempt, and a missed question returns in the next quiz.

**The scene comes after the maths.** `/math-scene` on the folder the `--math` drip wrote builds a scene in your own notation. `/concept-scene` is used only where no maths thread exists. A build that ran a script or a notebook also gets `/hld-diagram` in scene mode on the code it ran: one box per function, each opening onto its inputs, what it records and its output, with the lines walkable. That page is the one to screenshot for the layer's record.

**Retention.** Each `/drip-execute-plan` walk closes on a retrospective and prints a `/drip-retro` command. Run it before the next layer: it asks for your prediction on each chunk before showing the result. It takes about 25 minutes (estimate).

**Time.** The ⏱ line under each step gives your attention as a range and the AI's time, and "beside" names what you work on while the AI's part runs. Each range comes from the depth of the concepts the step covers, set per layer on its **Depth** line:

- *build*, 60 to 120 minutes a concept: 15 to 30 in the maths drip, 15 to 30 in its quiz with the banked problems, and 30 to 60 typing it in the build;
- *own*, 30 to 60 minutes a concept: the drip or the read, then the quiz and its problems;
- *gloss*, 5 minutes: two plain lines and a link;
- *link only*, 0 minutes: the Skip list names where it is taught.

Each quiz on the college maths beneath a layer adds 20 to 40 minutes. A `--request` rebuild of an idea-walk finding counts as one build concept. The upper end of every range is the box. None of these walks has run yet, so every range is an estimate, and layer 1's retrospective records the time actually used: that ratio rescales the rest. Layers 6 to 9 have no step file, so their lines say "rough".

A walk ends when its bar is checked. A box running out is the warning to switch to `--depth objective` or `faster`, and a missed bar gets one round of diagnose, fix and recheck at half the box.

## The ladder

```text
 setup ─▶ 1 one expert ─▶ 2 composing ─▶ 3 ideal experts ─▶ 4 parts world ─▶ 4b on pictures ─▶ 5 overlap dial
                                                                                                     │
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────┘
 ▼
 reading stop ─▶ 6 guidance ─▶ 7 A without B ─▶ 8 interaction term ─▶ prior art ─▶ 9 correction
                                                                                         │
 ┌───────────────────────────────────────────────────────────────────────────────────────┤
 ▼                                                                                       ▼
 12 the correction at scale (the goal)                                         ◇ 10, ◇ 11 (optional)
```

⭐ marks the critical path. ◇ marks what is optional: layers 10 and 11, the CLEVR reproduction and the Bradley rebuild.

| Layer | File | What you finish with | Your time |
|---|---|---|---|
| setup | [00-setup.md](ladder/00-setup.md) | the environment, and quiz 0 passed | 0.8 to 1.6 h |
| ⭐ 1: one diffusion expert | [01-one-diffusion-expert.md](ladder/01-one-diffusion-expert.md) | your own trained 2D expert | 3.3 to 6.7 h |
| ⭐ 2: composing two experts | [02-composing-two-experts.md](ladder/02-composing-two-experts.md) | your product, annealed ULA and per-sample MALA | 5.8 to 11.7 h |
| ⭐◇ 3: ideal experts | [03-ideal-experts-and-the-seven-kinds.md](ladder/03-ideal-experts-and-the-seven-kinds.md) | ideal experts and four kinds, each succeeding and failing | 4.3 to 8.7 h |
| ⭐ 4: the parts world | [04-the-parts-world.md](ladder/04-the-parts-world.md) | the mosaic, the blend and two animals from one product | 3.3 to 6.7 h |
| ⭐ 4b: the same product on pictures | [the toy failure in six checkpoints](../../../plans/10-what-poe-loses-when-it-composes/today-the-toy-failure-in-six-checkpoints.md) | red circles and blue squares on 8 × 16 pixels: the overlay drawn 200 of 200, the caption dial as a switch, error A in closed form, guided-sum DDIM; checkpoint 6's last three samplers still to walk | 2 to 4 h left, rough |
| ⭐ 5: the overlap dial | [05-the-overlap-dial.md](ladder/05-the-overlap-dial.md) | deep A∩B mass per overlap, exact and sampled | 2.4 to 4.8 h |
| ⭐ reading stop | [05a-reading-stop-projective-composition.md](ladder/05a-reading-stop-projective-composition.md) | the factorized-conditionals condition, placed on both toys | 1.7 to 3.3 h |
| ⭐ 6: guidance as a dial | [06-guidance-as-a-dial.md](ladder/06-guidance-as-a-dial.md) | the guided target for w from 1 to 7.5 | 2.3 to 4.7 h, rough |
| ⭐ 7: A without B | [07-a-without-b.md](ladder/07-a-without-b.md) | when negation stays a density, and the positive label | 2.3 to 4.7 h, rough |
| ⭐ 8: the interaction term | [08-the-interaction-term.md](ladder/08-the-interaction-term.md) | r(x) in the pixels and in the labels | 3.3 to 6.7 h, rough |
| ⭐ correction prior art | [08a-correction-prior-art.md](ladder/08a-correction-prior-art.md) | your correction placed against CO3 and Feynman-Kac | 2 to 4 h, rough |
| ⭐ 9: the correction | [09-the-correction-learned-back.md](ladder/09-the-correction-learned-back.md) | the fair test's answer | 2.3 to 4.7 h, rough |
| ◇ 10: latent space | [10-latent-space-and-vaes.md](ladder/10-latent-space-and-vaes.md) | warp invariance in code, and the smallest VAE check | 2.2 to 4.7 h |
| ◇ 11: the testbed | [11-the-text-conditioned-testbed.md](ladder/11-the-text-conditioned-testbed.md) | the testbed's candidate causes and parser | 1 to 2 h before its build |
| ⭐ 12: the correction at scale | [12-the-correction-at-scale.md](ladder/12-the-correction-at-scale.md) | the drain gap, the transfer plan, the LoRA on held-out pairs | 8.3 to 16.2 h, plus the gated experiment |

**Layer 4b** is the parts world's question asked on pictures. It was walked as its own plan before this guide took it in, so it keeps its own file and its own walk folder (`artifacts/drips/toy-failure-six-checkpoints/`). Its code is typed in `~/playground/deep-learning/poe-structural-toy/`.

## Settle before the layer it decides

Layers 4, 4b and 5 were built on two toys, and the two disagree in places. [The two walks reconciled](../../notes/what-poe-loses-when-it-composes/two-walks-reconciled.md) lists every claim with its evidence. Each disagreement below is settled at the layer it decides, before that layer's build is trusted.

| # | The disagreement | Settled by | At |
|---|---|---|---|
| 1 | Which picture the exact product fuses into: the overlay (4b, 200 of 200) or the ghost and two animals (4, by background spread) | a run: checkpoint 3 with the background at σ 1.0, and the parts world with cat and dog encoded so they do not cancel | before 4b's close |
| 2 | Whether reverse diffusion finds the pair once it is in training: 17 of 200 on 4b against 0.969 of the exact mass on 4 | a run: layer 5's samplers on 4b's 8 × 16 toy at a share of 1 in 20 | layer 5, step 5 |
| 6 | MALA's acceptance: tuned to about one half on 4b, to 0.6 on 2 and 5; the "accepts nearly every move" step accepted 0.377 in 1,536 dimensions | the same run as 2, tuned to one stated acceptance | layer 5, step 5 |
| 11 | Counts alone (4b) against counts with total variation (5) | the same run as 2, reporting total variation to the exact draws beside the counts | layer 5, step 5 |
| 3, 4, 5, 7, 8, 9, 10, 12 | sentences in the write-up and the structural route that state as settled what depends on a setting | a rewrite, no run | before layer 6 |

**Ideas that tie the ladder together.** Eight ideas, each with an exact toy check and a cost, ranked by how far each moves the goal per hour: [the ideas memo](../../ideas/ideas-that-tie-the-ladder-together.md). The layers they attach to link the ones they use.

**The prompt shape.** Every prompt carries the fields `~/.claude/PROMPT_FORMAT.md` asks for. The walks and builds carry all nine (task, starts from, settled facts with their source, outputs, target, progress, proof, limits, hand-off). The maths drips, quiz builds and scenes carry at least starts from, settled facts, outputs and hand-off. Under each prompt, `added:` and `dropped:` lines say how it differs from the guide's earlier version of it. Each background prompt writes its quiz or scene URL back into its own step, in place of a `(URL lands here)` line. Layers 6, 7, 10 and 12 number their steps from 1; the other layers put the one-layer-ahead background prompts at step 0.

**What the critical path costs.** About 48 hours of your time to finish, inside a 91-hour box. That counts quiz 0, two quizzes a layer, the scenes and a `/drip-retro` after each build. The AI's share is about 1.5 hours a layer for the quizzes and their preparation, plus about 45 minutes a scene, nearly all of it beside your work (estimates: neither quiz agent has a measured run). Layer 12's drain experiment is not counted, because it runs only if the drain-gap test passes. Layers 10 and 11 add 3 to 7 hours before layer 11's build, which is timed by the plan its walk writes.

**Start here:** [the setup](ladder/00-setup.md).
