# How did two X posts about reward-guided sampling turn into a scope 06 baseline in one day?

A conversation on 2026-09-05 started by asking whether Diffusion Explorer could show our sampler,
read two posts about Contrastive Distribution Matching, saw that the trained LoRA corrector is the
gradient of the twist those posts describe, and built the selection-only sampler that follows: K
particles on the plain product-of-experts score, reweighted by a learned joint-versus-PoE
classifier and resampled. The run finished the same day and came back inconclusive, because the
classifier memorised its 120 training images. This folder is the record of that path, written for
someone who was not in the conversation.

## What is in here

| File | What it is |
|---|---|
| `note.md` | the record: the first question, the posts and the connection, the hypothesis and its bar, what was built, the three runs, the pictures, the numbers, what it means, where everything is |

The figures the note embeds live in
[the results grouping](../../results/is-the-gap-the-samplers-or-the-models/README.md) and are read
one by one in [its figure explainer](../../results/is-the-gap-the-samplers-or-the-models/figure-explainer.md).
The verdict lives in [the finding](../../../report/is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md).
