# How this research gets done

**The practice itself is in [`~/.claude/RESEARCH_PRACTICE.md`](file:///home-mscluster/mmolefe/.claude/RESEARCH_PRACTICE.md)
and applies to every project.** Read it: write the results section before running the
experiments, name a run's cost and what it buys before it starts, look at the data before
designing on top of it, which visual skill to pick by what you are pointing at, the seven kinds
of result and what each may do to the paper, the six-step diagnosis procedure, attack the work
before a reviewer does, and kill work that is bleeding rather than only work that failed.

The mechanics are elsewhere and not repeated in either place: `~/.claude/EXPERIMENT_CONVENTIONS.md`
for what a run may change, and `~/.claude/skills/WORKFLOWS.md` for the end-to-end skill chains
with their handoff files marked.

This file holds only what is true of **this** project: where each general rule lands here, and
the specific mistakes this project has already made. Read it with the root
[MASTER_PLAN.md](../MASTER_PLAN.md), whose four lists say what is happening now, what is
happening in the background, and what to do next.

## Where the general rules land in this repo

| The rule | Here, concretely |
|---|---|
| Write the results section first | `plans/closing-the-compositional-gap/plans/writing-the-paper/plans/writing-05-the-results-skeleton.md` |
| Every run names its cost and what it buys | Two lines in the design plan. Worked example: `480 cells at 50s, about 6h. Answers question 1, feeds figure F2.` The constraint here is GPU-days |
| The reason a figure exists goes in its own document | [figures/what-each-figure-argues.md](figures/what-each-figure-argues.md) is the worked example: seven checkboxes in the plan, seven reasons in the document |
| A figure serves its reader | Same document, and it is held to the same one-pass standard as the figures it describes. Here that means: x is the denoising step 0 to 50 with "noise" and "image" labelled at the ends, not log-SNR, because only DDIM is used; four or five named pairs drawn in front with the remaining pairs behind as a band; a strip of decoded frames above the curve, which costs no sampling because every cell already saved `latent_trajectory.pt`; spreads and percentages held back to the appendix table |
| Every read lands in the register | `plans/standing/literature/plans/01-reading-register.md` |
| Suspiciously-good results are contamination until cleared | The canary here is the λ=0 check: injecting nothing must reproduce plain PoE to under 1e-5 |
| Organise on the way out | `python3 scripts/plan_pulse.py`, report-only, four checks, about 7 seconds over the whole tree |
| Keep the handoff on disk | [ENVIRONMENT.md](ENVIRONMENT.md) is why the cluster has never had to be re-explained |

## The mistakes this project has already made

Each of these is why a general rule exists. They are recorded here so the same one does not get
made twice in this repo.

- **The scorer read three instances on an image of two cats.** A threshold chosen from a number
  and never checked against a picture fails silently. It also counted a 162px limb as a third
  animal. This is why `/visualize-data-samples` runs before any scorer, and why step 2 of the
  diagnosis procedure (score five cells by eye, compare with the scorer) exists.
- **A figure plan became unreadable to its own author.** Lines like "the cure dosed" and "the
  fork elbow as a vertical band" named nothing to anyone who had not been in the conversation.
  The fix is in the writing-style section of `~/.claude/CLAUDE.md`: no private labels, a figure
  line names its axes and its data, and every number carries its unit and its meaning.
- **A figure was designed correct and unreadable.** The size-follows-noise figure was specified
  as seventeen thin lines against log-SNR from -5.15 to 6.37, with a spread of 19.7% annotated on
  the plot. Every part of that is defensible and none of it lands: the axis names a quantity the
  reader has to convert, the lines read as texture, and the number cannot be acted on while
  looking at it. The redesign changed the axis to the denoising step, cut to five named pairs
  over a band holding the other twelve, added decoded frames along the top, and moved the number
  to the appendix. The reason the first version happened is that it was designed against the
  data on disk rather than against a reader.
- **The disk guard checked a filesystem the script was not writing to.** A guard has to look at
  the filesystem the output actually lands on. `/home-mscluster` hit 100% once and silently
  killed checkpointing.
- **A full-strength shortcut would have made every control row reproduce the real one.** At λ=1
  the sampler could skip the arithmetic and use the joined-prompt prediction directly, which
  would have ignored the injected fake entirely and made all three curves look equally good.
  Switching it off during control runs is a decision recorded in the design plan, not a finding.
- **The obvious measure gave the opposite answer, for reasons about the instrument.** In the
  mechanism probe, comparing raw norms said attention moved 1.70× more than painted content,
  because the two maps are not on the same footing and the adapter dims attention by 25%
  overall. The argument for the scale-free measure is in
  [artifacts/results/residual-dynamics/content-change-relative-to-attention-change/measure-fairness.md](artifacts/results/residual-dynamics/content-change-relative-to-attention-change/measure-fairness.md).

## The deadline

Put the submission date at the top of the paper list and count backwards from it once a week. The
tree gives an order and cannot tell you whether the remainder fits.
