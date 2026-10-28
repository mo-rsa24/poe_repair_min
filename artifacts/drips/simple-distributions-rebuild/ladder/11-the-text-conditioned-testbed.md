# ◇ Layer 11: the 32 × 32 text-conditioned testbed

⬅️ [Layer 10: latent space](10-latent-space-and-vaes.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 12: at scale](12-the-correction-at-scale.md) ➡️

**Why it is optional.** The toys cannot show failures that come from the text encoder or cross-attention, such as leakage or a dropped concept. The goal does not need that explanation. A paper about Stable Diffusion's hybrids would.

**Depth.** Gloss until the idea walk below compiles; its plan then sets the depth and the quizzes.

## The flow

```text
 AI, in the background                         │  You, in order
───────────────────────────────────────────────┼───────────────────────────────────────────────
                                               │  1. /drip-idea, claim 1: causes, parser
                                               │       ▼
                                               │  2. build the plan the walk's compile writes
                                               │       ▼
 xref-cadence links the layer ◀─────────────── │  3. /drip-retro, then layer 12
```

## The steps

### 1. You: the idea walk

```text
/drip-idea resume artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md at claim 1, with
artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md claims 7 and 12 and
artifacts/ideas/poe-overlap-dial/routes/03-what-a-label-teaches-and-the-toy-axes.prompt.md beside
it, until the testbed's candidate causes and its parser are fixed

Task: walk claim 1 of the testbed map, then claims 3 and 5 as far as the parser and the probes need,
until compile is earned, in repo /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.
Starts from: nothing before it in this layer. The two maps and the route prompt above, read end to
end.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- Claim 1 is current, marked wrong as stated: SDXL's cause of the cat-dog hybrid is not known, so
  the testbed cannot inherit it. The repair: switch candidate causes one at a time, and a cause
  becomes a candidate for SDXL's hybrid when the signature it leaves in the testbed also appears in
  SDXL's cache (testbed map, claim 1).
- SDXL's hybrid comes from one model with two prompts at guidance 7.5 over 50 DDIM steps, composed
  as ε_∅ + w(ε_A − ε_∅) + w(ε_B − ε_∅), which targets p(x)·p(A|x)^w·p(B|x)^w; this is inferred
  from the cache's residual formula gs*(eps_j − eps_a − eps_b + eps_uncond), and the 2D walk
  measured construction (1) only at w = 1 (testbed map, claim 1 and Sources).
- The cache of ε_A, ε_B, ε_joint and ε_∅ at all 50 steps is under
  /datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/; the lower bound on the
  interaction computed from it for 25 animal pairs did not track which pairs blend (testbed map,
  claim 1 and Sources).
- SDXL's hybrids are scored by an instance counter that cannot tell two dogs from a cat and a dog
  (testbed map, Sources: poe_repair_min/context/world/chimera.md).
- Claim 2: total variation to the exact product needs the composed density, which pixel space does
  not give; repair: a parser maps each output back to its drawing parameters, where the generator's
  density is known exactly (testbed map, claim 2).
- Claim 3's check: fix the outcome categories (fusion, dominance, leakage among them) before any
  model trains, then run the parser on 1,000 held-out training draws and on 200 hand-built hybrids
  (a red dot cluster, a blue circle, a circle made of dots); above 99% correct on both makes it the
  scorer (testbed map, claim 3 and Checks outstanding).
- Claim 4: a jointly trained tiny encoder has no pressure to put two words close; repair: a frozen
  embedding table with the two words' cosine set by hand (testbed map, claim 4).
- Claim 5: an attention map is an observation; every probe is paired with an intervention (mask one
  word's attention to its region, swap the two maps, drop cross-attention at one resolution)
  (testbed map, claim 5).
- Claim 6's check: train the smallest model on single-concept captions in co3 on one GPU until the
  parser calls 95% of single-concept samples correct; under 30 minutes means dials run as parallel
  jobs, over an hour means shrink the UNet or the vocabulary (testbed map, Checks outstanding).
- The walk's bar: at compile, claim 1 is settled with at least two candidate causes, each with a
  signature computable both in the testbed and from the SDXL cache, and a table places every probe
  as isolating or not isolating each cause, every row naming its check; time box two sittings of
  about 90 minutes (testbed map, "The bar and the time box").
- Claim 7 of the parent map: a better network only gets closer to its target density; region and
  box conditioning change the target; GLIDE composed as in Liu et al. 2022 is construction (1) on a
  large base; what an image model adds that 2D cannot show is inductive bias across image parts
  (parent map, claim 7).
- Claim 12 of the parent map needs a check: train on single-concept captions, ask for two concepts
  as the joint caption, construction (1) and the plain product, score with a parser; the dials are
  embedding similarity, data overlap and fine-tuning set size, with each token's cross-attention
  map read (parent map, claim 12). Okawa et al. 2023 is confident; Park et al. 2024 is "likely,
  verify" (same claim).
- Route 8, which launched this walk, asks for a return at
  artifacts/ideas/when-poe-reaches-a-and-b/routes/08-sdxl-shaped-hybrid-testbed.return.md; no
  return exists yet (parent map, route 8).
- What the testbed must reproduce first: construction (1) with "contains" labels holds 0.989 in A∩B
  down to a data share of 0.001 while separately fine-tuned experts reach at most 0.5 when each
  one's leftover density sits in the two adjacent quadrants (the answer key's case; leftover on the
  top-right alone gives 0.9997, ladder layer 5's corrected answer key); experts over their own part
  reach 0.9999 with no A∩B data; reverse diffusion with summed scores misses by 0.049 total
  variation above the exact-sample floor; chimera mass 0.66 when both experts saw their object in
  the same place (route 8's prompt; parent map claims 2 and 9, runs 1 and 5).
- The label route proposes that what a product produces is decided by what each label meant in
  training: "contains A" puts 1.00 of the composed mass in A∩B and "exactly A" puts 0.08 (the
  overlap-dial map, claim 1.2). It was never returned, and the overlap-dial map is closed with its
  claim 8 already taken by the 2D-toys claim, so its axes and prediction are read here as material
  for claim 1, not written back there (the route prompt; the overlap-dial map's header).
- Python, NumPy and PyTorch only; the co3 and co3_bw environments; every experiment small enough to
  run in minutes to an hour (route 8's prompt).
Outputs: the testbed map with claim 1 marked settled, the candidate causes and their signatures in
it, claim 3's outcome categories fixed, the probe table; at compile, the plan the compile writes and
the route 8 return file named above.
Target: the map's own bar above, judged by me against the claim ledger.
Progress: each round closes one part with a mark: holds, needs a check, wrong as stated with its
repair, or dead.
Proof: grep -n "^| \*\*1\.\|^| 1\." artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md prints
claim 1's row with its settled mark.
Limits: write only in the testbed map's folder and the route 8 return file; nothing in the parent
map or the overlap-dial map; train nothing during the walk (claim 6's timing run belongs to step
2); no commit.
Hand-off: the compiled plan's path and the route 8 return, which step 2 reads.
```

added: the testbed map's six claims with their repairs and checks, its bar and time box, the SDXL composition and cache facts (from `artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md`) · claims 7 and 12 and route 8's open return (from the parent map) · the 2D findings the testbed must reproduce and the environment limits (from route 8's prompt) · the label route's evidence and why it is read here rather than written back (from the route prompt and the overlap-dial map)
dropped: nothing

⏱ you ~1 to 2 h over two sittings (estimate: it resumes with three files beside it, and the median measured `drip-idea` session is 20 minutes) · AI ~30 min · stop: the candidate causes and the parser are fixed

### 2. You: build the testbed

Once that walk compiles, build its design with `/drip-execute-plan` on the plan its compile writes. That plan's pieces set the build's time.

```text
/drip-execute-plan <the plan file step 1's compile names>

Task: walk that plan one step per message in its order. I type every block myself into
~/Playground/PhD/simple-distributions-walk/walk11.py, and into any module the plan names in the same
folder, and run it with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python.
Starts from: the compiled plan and the testbed map step 1 leaves; the route 8 return.
Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The candidate causes, their signatures and the parser's categories are the ones step 1 settled in
  artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md (from step 1).
- The parser is trusted only above 99% correct on 1,000 held-out training draws and on 200
  hand-built hybrids (testbed map, Checks outstanding, claim 3).
- The smallest model on single-concept captions should reach 95% parser-correct single-concept
  samples in under 30 minutes; over an hour means shrink the UNet or the vocabulary first (testbed
  map, Checks outstanding, claim 6).
- Pixel space has no exact density, so success is scored in the drawing parameters (REBUILD_WALK.md,
  brick "The text-conditioned testbed").
- Train on a card with under 1 GB in use, else wait (the GPU guard in
  ~/Playground/PhD/poe_repair_min/environment/hpc/execution-protocol.md).
Outputs: walk11.py and the modules the plan names; the plan's steps ticked, and the walk map that
holds my place between sittings.
Target: the plan's own bar, which step 1's compile writes; the parser bar (99% on both sets) and the
timing bar (95% within 30 minutes) are checked first. I judge, from the printed lines.
Progress: the parser check prints its accuracy on each set; the timed run prints minutes to 95%;
each later step prints the number the plan names.
Proof: the command the plan names for its bar, plus python walk11.py printing both parser
accuracies.
Limits: never edit rebuild/ or the idea maps beyond the ticks the plan names; I type, I do not
paste; no commit.
Hand-off: the retrospective the walk closes on, and the /drip-retro command it prints.
```

added: the whole build prompt around the existing sentence, with the parser and timing bars (from the testbed map) and the scoring space (from `REBUILD_WALK.md`) and the GPU guard · the file it types into, per the guide's one file per layer
dropped: nothing

⏱ set by the plan the walk compiles

**Outcome:** a testbed where each candidate cause of the hybrid is switched on one at a time:

- single-object captions;
- the guidance weight;
- word similarity;
- shared attention;
- what each label meant during training ("exactly" labels predicting fusion, "contains" labels predicting co-occurrence).

### 3. You: close the layer

Run the `/drip-retro` command the build printed at its close (⏱ about 25 minutes). Then paste this:

```text
In the background, dispatch xref-cadence over the files layer 11 of the composition ladder wrote:
artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, the plan its compile wrote, the route 8
return, and the retrospective the layer 11 walk closed on.

Settled facts:
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and python -m rebuild.<script> runs from toy/.
- The reference sets to link against are the "The text-conditioned testbed" row of the Bricks table
  in artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, claims 7 and 12 of
  artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, and the route prompt
  artifacts/ideas/poe-overlap-dial/routes/03-what-a-label-teaches-and-the-toy-axes.prompt.md. This
  layer has no Study first line.
- The overlap-dial map is closed; link toward it, write nothing inside it (its header).
- ~/Playground/PhD/simple-distributions-walk/ is scratch space; link nothing into it (CLAUDE.md, the
  playground is disposable).
Outputs: links written both ways between those files and the reference sets.
Hand-off: the agent's report, listing every link it wrote, read before layer 12 starts.
```

added: the reference sets, by path (from `REBUILD_WALK.md` and the two idea maps)
dropped: nothing

⏱ you 2 min to paste · AI ~10 min (estimate)

---

⬅️ [Layer 10: latent space](10-latent-space-and-vaes.md) · 🏠 [the guide](../PROMPTS.md) · [Layer 12: at scale](12-the-correction-at-scale.md) ➡️
