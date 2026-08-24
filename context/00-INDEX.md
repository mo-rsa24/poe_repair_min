# 🌍 Context: poe_repair_min

What this project is about in the real world: why PoE composition on SDXL fails, what a chimera,
an animal pair, the interaction term, and a compose rate are, where the pipeline's data comes
from, and what every field in a `summary.json` or a scorer output means. What the system *is*
(the cluster, the two filesystems, the LaTeX build) lives in
[environment/00-INDEX.md](../environment/00-INDEX.md); how to drive it lives in the runbook, not
yet built as of this pass. This folder is only the meaning.

If what you need is not in here, **stop and ask**. Do not infer a meaning from a column or symbol
name and do not proceed on a plausible assumption; a wrong guess about what `d_T` or `arm` means
gets built on by the next figure or claim. Say so and ask instead.

## Table of contents

- [I want to know...](#i-want-to-know)
- [The themes](#the-themes)
- [Still open](#still-open)
- [Pictures still missing](#pictures-still-missing)

## I want to know...

Navigation: 📋 [TOC](#table-of-contents) | [Next](#the-themes) ➡️

| The question | Where it is answered |
|---|---|
| What problem is this project actually solving, and for whom | [The problem](purpose/01-the-problem.md#who-has-the-problem) |
| What PoE composition is, and what Mono is | [PoE composition § What PoE composition is](world/poe-composition.md#what-poe-composition-is) |
| What a chimera looks like | [Chimera § What it looks like](world/chimera.md#what-it-looks-like) |
| What the interaction term (`r_t`) is | [Interaction term § What the interaction term is](world/interaction-term.md#what-the-interaction-term-is) |
| What the trained LoRA corrector does, and why it never sees the joint prompt | [LoRA corrector § What the LoRA corrector is](world/lora-corrector.md#what-the-lora-corrector-is) |
| What an animal pair is, and why the pool is biased toward failure | [Animal pair § What an animal pair is](world/animal-pair.md#what-an-animal-pair-is) |
| What a compose rate is, and what the scorer can and cannot tell you | [Compose rate § What a compose rate is](world/compose-rate.md#what-a-compose-rate-is) |
| How a prompt pair becomes a scored cell | [Where it comes from § The journey](data/01-where-it-comes-from.md#the-journey) |
| What `pair_slug`, `seed`, `lambda`, `d_T`, or `n_instances` mean | [Dictionary](data/02-dictionary.md) |
| Who reads the output, and what they do with it | [What we produce § Who reads it](purpose/02-what-we-produce.md#who-reads-it) |
| How would we know this project is working | [What working means § The test](purpose/03-what-working-means.md#the-test) |

## The themes

Navigation: ⬅️ [I want to know...](#i-want-to-know) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| Folder | What it answers | Files |
|---|---|---|
| [purpose/](purpose/) | Why this exists, what it produces, what working looks like | 3 |
| [world/](world/) | The real things behind the data: PoE composition, chimera, the interaction term, the LoRA corrector, the animal pair, compose rate | 6 |
| [data/](data/) | Where the data comes from, and what every field means | 2 |

Picture prompts are in [diagram-prompts.md](diagram-prompts.md) (8 prompts, 0 rendered); rendered
images sit in `diagrams/`. Existing pipeline images already copied in sit in `images/world/`.
External documents this folder draws on are listed in [sources.md](sources.md) (one entry, not
yet read past its title).

## Still open

Navigation: ⬅️ [The themes](#the-themes) | 📋 [TOC](#table-of-contents) | [Next](#pictures-still-missing) ➡️

The things this pass could not settle by reading. Each names who or what would settle it.

- [ ] **The interaction term's sign convention is written two ways across the repo's own
      documents**: `MASTER_PLAN.md`'s glossary writes `r_t = ε̃_J − ε̃_PoE`; a maintainer memory
      note referenced (but not read) during this build is recorded as using
      `r_t = ε̃_Mono − ε̃_PoE`. Both are read here as the same Mono-minus-PoE quantity, but this
      has not been confirmed against the code that computes it. Would settle it: read
      `poe_repair/composers/teacher_residual.py` or wherever `d_t_poe_vs_mono` is computed, and
      confirm `ε̃_J` and `ε̃_Mono` name the same prediction. See
      [world/interaction-term.md § What people get wrong](world/interaction-term.md#what-people-get-wrong).
- [ ] **The current animal-pair pool's size is not settled.** `artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md`
      lists 17 pairs; `EXPERIMENTS.md` also says 17 "in the current pool"; but
      `plans/retrofit-poe-repair-min.md` (a later document) names "the 20 animal pairs" as the
      scope boundary. Would settle it: list the pair directories actually present under the
      current pool's output folder and recount. See
      [world/animal-pair.md § What people get wrong](world/animal-pair.md#what-people-get-wrong).
- [ ] **Whether `a_butterfly__x__a_flower_meadow` (named as the one pair that composes, and the
      pair the paper opens on) has a recorded fail-rate anywhere.** It does not appear in
      `artifacts/results/does-the-fix-reach-unseen-pairs/fail_rate.md`. Would settle it: find or run its scorer
      result. See [world/animal-pair.md § What an animal pair is](world/animal-pair.md#what-an-animal-pair-is).
- [ ] **The one external paper registered in `sources.md` has not been read past its title**,
      because `pdftoppm` is not installed in this session's environment. Would settle it: install
      `poppler-utils` or open the PDF another way, then log it in
      `plans/standing/literature/plans/01-reading-register.md` (itself still an empty task as of
      this build).
- [ ] **The interaction term's causal role is only backed, inside this repository, by this
      project's own paper draft.** The maintainer's working memory (not filed in the repo) refers
      to a seven-paper external literature reconciliation on this exact question, but
      `plans/standing/literature/plans/01-reading-register.md`'s own task list confirms the
      register is still empty. Would settle it: back-fill that register, then this file's citation
      of the paper's abstract can be joined by external corroboration.

## Pictures still missing

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

- [ ] 📷 [world/lora-corrector.md § What it looks like](world/lora-corrector.md#what-it-looks-like):
      one cell's PoE render, LoRA-corrected render, and Mono render side by side. No search tried
      yet beyond noting a candidate source (`artifacts/results/can-we-trust-the-compose-score/compose-scorer-validation/scorer_validated.json`'s
      passing cells). Save as `images/world/05-lora-corrected-vs-poe-vs-mono.png`.
- [ ] 🖼️ All 8 pieces in [diagram-prompts.md](diagram-prompts.md) are written and unrendered: 4
      subject-lane pieces plus capstone are missing from this count (3 pieces + capstone = 4
      subject, 3 pieces + capstone = 4 process). Paste each into ChatGPT, save under `diagrams/`,
      and check against its faithfulness note before embedding, per that file's own instructions.
