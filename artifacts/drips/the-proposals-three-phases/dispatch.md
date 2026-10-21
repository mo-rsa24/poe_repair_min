scope: /home/molef/PhD/poe_repair_min
source: /mnt/c/Users/molef/Downloads/PhD/Proposal/Composability Gap - PhD Proposal (v2).pdf (44 pages, dated 8 May 2026)
why: not stated yet, asked in the opening message

target: the proposal's three phases, each as its question, its hypothesis and its experimental design
mode: none, a plain walk of the document in its own terms (nothing is derived or built)
ladder: 5 pieces (the one problem all three share; Phase 1 measures the gap; Phase 2 tests
  whether the representation causes it; Phase 3 tries the fix on chest X-rays; how the three
  hand over to each other)
read first: terms-ledger.md (17 defined, 7 undefined), contradictions.md (11 disagreements, 5 gaps)

asked [yours]: "Simply put, what are the 3 projects, what is the hypothesis, the question, what
  is the experimental design"

---

rung: the one problem all three share
notation: "logical composition" for the two-prompt score-addition rule (Eq. 2.30), "semantic
  composition" for one prompt naming both things, "composability gap" for the distance between
  the two results
examples: the pair "a cat" and "a dog" from one shared starting noise, once as the single prompt
  "a cat and a dog" and once as two prompts combined by Eq. 2.30 (toy chosen by the walk, the
  proposal names no pair)
clicked [my read]: (empty until the reader responds)
asked [yours]: (none yet)
wants to see move: (empty until the reader says)
moved on with: `next`, no question, no pin

---

rung: Phase 1 measures the gap across four kinds of concept pair
notation: G1 to G4 for the four pair groups, `d_t` for the gap at step t, `d_T` at the final
  step, `J_g` for the group-level divergence, "rerun" for the SD-IPC prompt recovery
examples: the grid 4 groups x 6 pairs x 24 seeds = 576 records, four runs each from one noise
  (solo c1, solo c2, single prompt, PoE); the three windows steps 0-17, 17-34, 34-50
raised: disagreement 4 (group ordering), disagreement 2 (which image is right), and a new gap:
  no rule for assigning a pair to a group
clicked [my read]: (empty until the reader responds)
asked [yours]: (none yet)
wants to see move: (empty until the reader says)
moved on with: `next`, no question, no pin

---

rung: Phase 2 tests whether the way concepts are stored causes the gap
notation: setups A, B, C1, C2, D for the proposal's "regimes"; "withheld combination" for the
  orientation-scale pair removed from training; "joint recovery" for both factors read correctly
  in the same generated image
examples: the proposal's own case (§5.2.3): withheld "large and rotated left", a generated sprite
  read as "rotated left, small" scores 1 on orientation, 0 on scale, 0 on joint recovery
raised: disagreement 3 (H2 predicts a smaller gap, Phase 2 computes none), gap: the diffusion
  model that every setup feeds is not described
clicked [my read]: (empty until the reader responds)
asked [yours]: (none yet)
wants to see move: (empty until the reader says)
