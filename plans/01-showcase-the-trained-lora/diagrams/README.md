# Diagrams for this scope

## the-joined-prompt-is-only-a-target.png

**What it shows.** Two panels. Training: one noisy state goes through "a cat", "a dog" and the
empty prompt into a cross-attention block carrying the rank-8 adapter, the three predictions
combine, and the result is scored by squared error against the stored target from the cache.
Inference: the same three prompts run twice through the same block, once with the adapter off and
once on, the two combined predictions subtract to give the predicted correction, a dose dial
scales it, and adding it back yields two separate animals.

**What it is for.** The one picture that shows why the adapter is Mono-free at inference: the
joined prompt appears only in training, as the stored target, and never in the inference panel.

**Where it came from.** A house-style render (vivid circuit) found loose at `output/imagegen/`
with no prompt entry claiming it, filed here because it draws this scope's shipped artifact. No
prompt in this scope's `diagram-prompts.md` names it, so re-rendering needs the prompt written.
