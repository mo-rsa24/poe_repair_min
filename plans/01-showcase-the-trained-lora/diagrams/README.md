# Diagrams for this scope

## the-joined-prompt-is-only-a-target.png

**What it shows.** Two panels. Training: one noisy state goes through "a cat", "a dog" and the
empty prompt into a cross-attention block carrying the rank-8 adapter, the three predictions
combine, and the result is scored by squared error against the stored target from the cache.
Inference: the same three prompts run twice through the same block, once with the adapter off and
once on, the two combined predictions subtract to give the predicted correction, a multiplier dial
scales it, and adding it back yields two separate animals.

**What it is for.** The one picture that shows why the adapter is Mono-free at inference: the
joined prompt appears only in training, as the stored target, and never in the inference panel.

**Where it came from.** A house-style render (vivid circuit) found loose at `output/imagegen/`
with no prompt entry claiming it, filed here because it draws this scope's shipped artifact. No
prompt in this scope's `diagram-prompts.md` names it, so re-rendering needs the prompt written.

**It carries the old vocabulary.** The dial in the inference panel is labelled "dose dial", a name
this project no longer uses (see [the word list](../../../context/plain-words.md)). The words are
painted into the image, so fixing them means regenerating it, and no prompt in this scope's map
names this file yet. Writing that prompt is what has to happen first.

## Superseded renders

`the-plateau-read.png` and `the-null-input-probe.png` were rendered here and then moved to
`temp/codex-drop/superseded-old-vocabulary/`. Both had a banned word painted into the title banner
("plateau", "probe"). They are kept rather than deleted so a regeneration can be compared against
them. Their prompts, renamed to "reading where the curve flattens" and "the null-input test", are
back at not-rendered in the scope's map.

