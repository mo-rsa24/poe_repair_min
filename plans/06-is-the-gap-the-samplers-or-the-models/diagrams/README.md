# Diagrams for this scope

## which-half-does-the-corrector-take.png

**What it shows.** The 50-step denoising run left to right from high noise to low noise, with the
correction split into two stacked amber coils: the sampler's share, drawn tapering to nothing by
the low-noise end, and the model's share, drawn holding its thickness all the way. A Langevin
corrector sits above the track at three points, running k steps per level. A caliper at the
low-noise end marks the read zone where the split is actually measured.

**What is claimed and what is not.** Every "unconfirmed" tag in the image is load-bearing: the
tapering of the sampler share, the persistence of the model share, and the corrector's effect at
the middle step are all predictions this scope exists to test, not results. The dashed gray legend
entry means not yet measured.

**Where it came from.** A house-style render (vivid circuit, shared with the project map) that
arrived without a prompt entry claiming it, found loose at `output/imagegen/` and filed here
because it illustrates this scope's whole question. No `diagram-prompts.md` in this scope names
it, so re-rendering it would need the prompt written first.
