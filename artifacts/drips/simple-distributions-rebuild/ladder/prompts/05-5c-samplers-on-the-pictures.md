# Layer 5, step 5c: this layer's samplers on layer 4b's pictures

⬅️ [Layer 5, step 5c](../05-the-overlap-dial.md#5c-ai-then-you-this-layers-samplers-on-layer-4bs-pictures)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
beside step 5b.

```text
In /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy, write
rebuild/settle_samplers_on_pictures.py and run it: layer 5's samplers on layer 4b's 8 × 16
circle-and-square experts, against the exact product, at a caption share of 1 in 20.

Background
Layer 4b and layer 5 of the composition ladder (artifacts/drips/simple-distributions-rebuild/)
disagree on whether reverse diffusion finds the pair once the pair is in training. 4b's guided-sum
DDIM draws the two shapes apart 17 of 200 times at a caption share of 1 in 20, while the parts
world's reverse diffusion reaches 0.969 of the exact mass at 5% (the two walks reconciled,
artifacts/notes/what-poe-loses-when-it-composes/two-walks-reconciled.md, disagreement 2). The two
toys also tune MALA to different acceptances, about one half against 0.6 (disagreement 6), and 4b
reports counts where layer 5 reports total variation (disagreement 11). Running layer 5's samplers
on 4b's pictures leaves one difference standing, the size of the picture (1,536 numbers against 2),
so the answer says whether the size is what drives the gap.

Objectives
1. Write toy/rebuild/settle_samplers_on_pictures.py from the starting scripts under What to read.
2. Run six samplers on 4b's pictures at a caption share of 1 in 20: exact draws, reverse diffusion
   on summed scores, guided-sum DDIM at w = 1, annealed ULA, annealed MALA tuned to an acceptance
   of 0.57, and annealed UHA.
3. For each sampler, record counts over read_out's six kinds out of 200, the total variation
   between that six-kind histogram and the exact draws' histogram, and, for MALA, its measured
   acceptance per noise level.
4. Save toy/rebuild/out/settle/samplers-on-pictures.json and a figure,
   toy/rebuild/out/settle/samplers-on-pictures.png, with one row per sampler, eight samples each,
   and its counts on the row.
5. Write the six rows and which of the three outcomes below held into step 5c of
   artifacts/drips/simple-distributions-rebuild/ladder/05-the-overlap-dial.md, under "Result".

What to read
- artifacts/notes/what-poe-loses-when-it-composes/fig_ddim_guided.py: 4b's pictures, namely three
  red circles and three blue squares at columns 3, 8 and 13 of row 4, the pair picture, σ = 0.1,
  the 16-picture library and read_out's six kinds, the cosine schedule, and guided-sum DDIM.
- artifacts/notes/what-poe-loses-when-it-composes/toy_sampler_hybrids.py: an earlier 16 × 32
  version with exact draws, reverse diffusion and annealed MALA on the product.
- toy/rebuild/samplers.py and toy/anneal_samplers.py: layer 5's ULA, MALA, tuned MALA and UHA.
- artifacts/notes/what-poe-loses-when-it-composes/two-walks-reconciled.md, disagreements 2, 6 and
  11: why this run exists.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The exact product at share 1 in 20 and σ = 0.1 puts its weight on the pair with itself, because
  that pairing has d² = 0 against 32 and more for every other
  (today-the-toy-failure-in-six-checkpoints.md, checkpoint 5).
- 4b measured with 200 DDIM steps: with no pair, w = 1 gives 122 overlay and 78 pale-apart
  (fig-ddim-guided.json).
- With the pair at 1 in 20, w = 1 gives 115 overlay, 68 pale-apart and 17 apart
  (fig-ddim-guided.json).
- At 50 steps and w = 7.5 the DDIM overshoots into single shapes, so use 200 steps
  (fig-ddim-guided.json).
- Layer 5's tuned MALA lags the exact mass at low overlap, 0.17 against 0.30 at 5% (phase5_tables).
- The outcomes are written before the run (layer 5, step 5c).
- If reverse diffusion and tuned MALA each put at least 0.9 of their draws on "apart", the gap was
  4b's guided-sum DDIM and not the picture's size, so 4b's sentence "the failure is in the sampler"
  narrows to that sampler (layer 5, step 5c).
- If each puts under 0.3 on "apart", the size drives it, and layer 12's reliance on samplers
  recovering most of the mass is in doubt (layer 5, step 5c).
- Between 0.3 and 0.9 is inconclusive, and the next run doubles the steps (layer 5, step 5c).

Rules
- Run on the CPU or one card under 1 GB in use.
- Finish in under 30 minutes.
- Use seed 0.
- Write nothing outside toy/rebuild/, except the Result lines in step 5c of the ladder file.
- Print one line per sampler as it finishes, with its counts.

What to hand back
- toy/rebuild/settle_samplers_on_pictures.py.
- toy/rebuild/out/settle/samplers-on-pictures.json and samplers-on-pictures.png.
- The six rows and the outcome that held, under "Result" in step 5c of the ladder file.
- The proof: python -m rebuild.settle_samplers_on_pictures --check prints the exact draws'
  histogram, which must put at least 195 of 200 on "apart".

What you will see
Along the way: one line per sampler as it finishes, with its counts.
At the end, a closing report: what was asked, in one sentence; what was done, with the script's,
the JSON's and the figure's paths; which of the three outcomes held, with reverse diffusion's and
tuned MALA's shares on "apart" against 0.9 and 0.3, and the --check line showing at least 195 of
200 exact draws on "apart"; what is still open; at most one question.

I judge, from the six rows and the figure: done when the six rows are filled in under Result, the exact draws put at least 195 of 200 on apart, and one of the three outcomes is named.
```

added: the seven sections, from the field-labelled version · the step's "Why this step exists" paragraph as the Background · the first line written as a plain instruction, since this prompt opens no skill
dropped: nothing
