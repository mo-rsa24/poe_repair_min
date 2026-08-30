# The words this project uses, and the ones it does not

Every term below was replaced across the plan tree because it was a private label, a metaphor, or
a category name: something that told a reader nothing unless they had been in the room. The right
column is what the work is actually called now. This file is the contract, so a sweep can be
re-run with a different word by changing one row.

The rule that generated it: name what happened, not the category it falls into. If a reader would
have to ask what a word means, the line was wrong, not the reader.

## Replaced everywhere, including file and folder names

| Was | Now | What it actually is |
|---|---|---|
| cell | run | one pair-and-seed combination that was generated |
| gate | check, or what must pass first | a condition that has to hold before the next step starts |
| instrument | measuring tool, or the tool | something built to measure with, before any result is read |
| sweep | series, or the run across values | generating the same thing at many settings in one go |
| probe | test | a small run that answers one yes-or-no question |
| arm | condition | one setting of the thing being varied |
| slot | reserved place | a figure the paper has promised but not yet built |
| the bar | the threshold | the number a result has to beat, written before the run |
| the floor | chance level | what the measure reads when nothing real is happening |
| plateau | where the curve stops rising | a training curve that has flattened |
| spine | section order | the order the paper's sections go in |
| harness | the runner | the code that launches and collects a run |
| smoke run | first short run | a one-epoch run that proves the wiring works |
| oracle | the cached true correction, or the endpoint predictor | two different things shared one word: the ground-truth correction computed from the joined prompt, and a model that predicts where a trajectory finishes |
| preflight | the check before launch | what must pass before a job is submitted |
| canary | the check that must pass before anything runs | the same idea, under a bird's name |
| caustic | the fold where nearby states land in different endings | where two nearby starting points finish in different places |
| endpoint map | where each state would finish | the function from a state to the image it becomes |
| decide-then-descend | decides early, then only descends | the outcome is fixed early and the rest of the run just resolves it |
| cache drum, drum | the training cache | the saved predictions the work reads instead of recomputing |
| dose dial | the multiplier on the correction | the number that scales how much correction is added |
| follower | the watcher that reads checkpoints during training | a process that reads saved checkpoints while a run is going |
| rung | step, or stage | a ladder metaphor from a plan tree that no longer exists |
| crossbar | the train-against-evaluate grid | rows are what a model trained on, columns what it was tested on |
| veracity | whether it is true | |
| hedged | not yet measured | in prose. In a diagram prompt it stays, where it means a dashed outline |

## Kept, because the field genuinely calls it that

These are not private labels, and removing them would cost more than it saves: a reader needs them
to follow a paper. Each gets one plain gloss at its first use in a file, never a rename.

speciation, Langevin, Tweedie, AUC, low-rank, held-out, basin, Mono-free.

## Not changed, and why

Code identifiers keep their names. `ax.spines` and `.bar()` are matplotlib's API, so renaming them
breaks plotting, and the internal names in `poe_repair/` are a separate refactor with test
coverage behind it. A word can therefore appear in a script and not in the prose that describes it.
