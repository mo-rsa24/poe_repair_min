# Layer 2, step 4c: the second idea-walk rebuild, walk02c.py

⬅️ [Layer 2, step 4c](../02-composing-two-experts.md#4c-you-the-second-idea-walk-rebuild-walk02cpy)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan --request rebuild run 1 of
artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb by hand in
~/Playground/PhD/simple-distributions-walk/walk02c.py; bar: MALA adapted to 0.6 acceptance takes
steps 18 to 82 times larger and reaches the floor on four kinds, while ULA at those steps diverges
(it keeps 1 of 2,000 samples on disjunction)

Background
This is the second idea-walk rebuild in layer 2 of the composition ladder
(artifacts/drips/simple-distributions-rebuild/). Run 1 of the kinds notebook showed that the accept
step is what makes large steps safe, and that large steps are what let a sampler mix. I rebuild it
by hand, starting from walk02.py from step 4a, whose per-sample MALA is the move run 1 tunes; the
notebook's printed run 1 table is the answer key.

Objectives
1. Cut run 1 into 4 to 6 pieces, and write the step file at the open.
2. Build, teach and test one piece per round.
3. I type the tuned MALA and the ULA comparison into walk02c.py by hand and run it with
   /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
4. End with walk02c.py printing one row per kind: acceptance, step ratio, tuned TV, floor, ULA's TV
   and the samples ULA kept.

What to read
- artifacts/ideas/poe-overlap-dial/kinds-of-composition-on-the-papers-toys.ipynb, run 1: the code,
  and its cell 15 table as the answer key.
- toy/rebuild/kinds.py (build_kinds) and toy/rebuild/ideal.py (exact_grid, composed_target, tv, V):
  the seven kinds' exact targets, imported here.
- toy/rebuild/samplers.py, tuned_mala: the reference tuning rule.
- ~/Playground/PhD/simple-distributions-walk/walk02.py from step 4a.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Run 1 runs on the seven kinds' exact targets: ideal experts with σ = 0.05, 800 points per set,
  2,000 samples, 100 levels and 10 moves per level from N(0, I) (notebook, cells 0 and 2;
  toy/rebuild/kinds.py).
- Layer 3 builds those targets by hand; here they are imported from rebuild/kinds.py (build_kinds)
  and rebuild/ideal.py (exact_grid, composed_target, tv, V) (layer 2, step 4c).
- In Figure 1, MALA at 0.05·ν_t accepted 0.99 to 1.00 of its proposals, so it equalled ULA and did
  not yet test the accept step (notebook, cells 8 and 10).
- Tuned MALA carries its step across levels scaled by ν_t and multiplies it by e^(acceptance − 0.6)
  after every move (notebook, cell 14; toy/rebuild/samplers.py, tuned_mala).
- The adaptation runs during sampling, so the chain is not exactly reversible while it adapts
  (notebook, cell 14; toy/rebuild/samplers.py, tuned_mala).
- The notebook's run 1 table, as kind: acceptance, final step over Figure 1's step, tuned TV,
  floor, reads intersection 0.60, 18.6x, 0.11, 0.11; disjunction 0.61, 82.4x, 0.22, 0.23;
  co-occurrence 0.61, 68.9x, 0.11, 0.10; contention 0.60, 43.9x, 0.16, 0.10; dominance 0.60,
  17.6x, 0.30, 0.11; interaction 0.61, 37.5x, 0.08, 0.07; negation 0.60, 28.7x, 0.21, 0.15
  (notebook, cell 15).
- ULA at the same steps keeps 721, 1, 157, 1843, 777, 2000 and 446 of 2,000 samples in that kind
  order, with TV 0.33 to 0.50 where it survives (notebook, cells 15 and 16).
- Dominance stays at 0.30 against a floor of 0.11 even with the large steps, because no local
  sampler moves mass between separated modes to match unequal weights (notebook, cell 16;
  artifacts/ideas/poe-overlap-dial/IDEA_MAP.md, run 5).
- rebuild/phase3_kinds.py reproduces run 1 within 0.03 on every kind, its own tolerance for sampler
  numbers (toy/rebuild/out/phase3/phase3.json, run1).

Rules
- Never edit the notebook, toy/rebuild/ or toy/anneal_samplers.py.
- I type; I do not paste.
- No commit.
- Show each piece's printed lines before moving on: piece 1 prints Figure 1's MALA acceptance per
  kind, the tuning piece prints each kind's acceptance and step ratio, and the last piece prints
  the ULA column.

What to hand back
- walk02c.py, printing one row per kind: acceptance, step ratio, tuned TV, floor, ULA's TV and the
  samples ULA kept.
- The step file the walk writes at its open, ticked piece by piece.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk02c.py prints the seven-row table, and step 5's comparison with python -m
  rebuild.phase3_kinds (Run 1) and the notebook's cell 15 shows each tuned TV within 0.03.

What you will see
Along the way: one piece per round, each ending on the lines it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk02c.py's
path and the step file's; PASS or MISS against tuned acceptance of 0.60 ± 0.03 on every kind, tuned
TV within 0.03 of the floor on intersection, disjunction, co-occurrence and interaction, and ULA at
the same steps keeping under 2,000 samples on disjunction (the notebook kept 1); what is still
open; at most one question.

I judge, from the printed table: done when tuned acceptance is 0.60 ± 0.03 on every kind, tuned TV sits within 0.03 of the floor on the four kinds named, and ULA keeps under 2,000 on disjunction.
```

added: the seven sections, from the field-labelled version · what run 1 showed, as the Background (from the step's Outcome)
dropped: nothing (the name ν_t, in place of v_t, stays from the Codex mathematics review, to match the step file)
