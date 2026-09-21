# Does the correction the LoRA learns have a text embedding you could plot beside the joint and solo prompts?

Three quantities are conditioned on text in this project's pipeline and two of them already have
embeddings you can read off directly: the joint prompt and the two solo prompts. The wish was for
a third, a text embedding of the correction the LoRA learns, so all three could sit on one plot.
The walk found the load-bearing claim wrong as stated (a text embedding is an input to the UNet
and the correction is an output, so nothing maps backwards in closed form, and the same pair under
two seeds gives a near-zero cross-seed cosine, so there is no single vector to embed) and repaired
it to a fit: is there a token embedding whose guidance direction reproduces the correction on
average, with a reported error. The walk is parked mid-claim, with two runs already showing that
the cheap readback route (captioning a decoded noise prediction) fails on the ceiling case.

## What is in here

One file, `IDEA_MAP.md`, the full `/drip-idea` walk: six claims (one settled, one repaired and
load-bearing, four still open), two dead ends (both closed-form inversion attempts), two runs
(a signature-mismatch failure and a completed readback run whose stated pass criterion held while
the actual captions disagreed with the ceiling image), and the sources on textual inversion and
null-text inversion the repaired claim leans on.

## Where it came from and what judged it

**Held since 2026-09-05.** No plan file or review file references this idea; the only other
mention in the tree is an incidental rename-table entry in `RENAMES.md` recording the folder's
move under `artifacts/ideas/`. Kept because a `/frame-hypothesis` prompt was already emitted from
claim 4.5 (the x̂_0 readback route) covering the four-rung ladder up to the controls, and the
map's own "Next step" leaves claims 2, 3, and 5 unwalked and claim 4's optimization route
unchecked.
