# Layer 5a, optional step: the Bradley rebuild

⬅️ [Layer 5a, optional: the Bradley rebuild](../05a-reading-stop-projective-composition.md#-optional-recommended-if-the-read-raised-questions-the-bradley-rebuild)

Optional, recommended if the combined read raised questions. Paste into a session opened in
`/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`, after the combined read of
step 2 has closed.

```text
/unbuild --walk "projective composition from arXiv 2502.04549 and local scores from arXiv
2509.16447, rebuilt on the quadrant toy and the parts world (background choice, factorized
conditionals, a local score), then in a learned invertible feature space as layer 10's VAE step"

Background
The reading stop of the composition ladder (artifacts/drips/simple-distributions-rebuild/) read the
two Bradley papers. This walk rebuilds both papers' constructions by hand on the ladder's own toys,
checked against the answer keys the read-aheads already hold. Its last piece, the learned
invertible feature space, is the first piece of layer 10's latent brick.

Objectives
1. Rebuild both papers' constructions by hand in build order, one brick per round: background
   choice, factorized conditionals, a local score, then a learned invertible feature space.
2. I type every block into a new folder, ~/Playground/PhD/simple-distributions-walk/bradley/, and
   run it with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python and
   PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain/toy.
3. Check every brick against the read-aheads' answer keys, printing its number beside the key's
   line.
4. Compute the local score on both toys, the split pair and the shared pair.
5. Keep one row per brick, with its check, in a REBUILD_WALK.md in the rebuild folder.

What to read
- COMBINE_MAP.md from this stop's combined read (step 2's hand-off), under
  ~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/drips/ in its
  combine-<idea-slug>/ folder.
- Layer 5's walk05.py in ~/Playground/PhD/simple-distributions-walk/.
- The answer keys in the journey folder
  ~/goal-setting/learning/deep-learning/diffusion-models/01-poe-derivation-foundations/, at
  sources/mechanisms-of-projective-composition/checks/ and
  sources/local-mechanisms-of-compositional/checks/.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "Composing in a latent
  space": layer 10's unbuilt brick.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The answer keys run with uv run --no-project --with numpy --with scipy python checks/<piece>.py
  from the paper's folder, and the 2509.16447 keys import the earlier paper's through
  checks/_earlier.py (READ_MAP.md, Position; read-ahead 00-INDEX.md of 2509.16447).
- Factorized conditionals on the split and shared toy pairs print "split ... FC holds: True" and
  "shared ... FC holds: False" (checks/fc_check.out).
- Composing commutes with noising on the split pair and not the shared one at ᾱ = 0.5, where the
  shared pair's noise-then-compose mean is −0.8485 against compose-then-noise −1.0607
  (checks/noise_commute.out).
- The local conditional score matches the exact score to 0.0 on the split pair at ᾱ = 1.0, 0.5 and
  0.0058, and on the shared pair it misses by up to 0.5 at ᾱ = 1.0, falling to 0.0381 at
  ᾱ = 0.0058 (checks/local_score.out).
- Composing in a feature space, identity, rotation and shear all map the composition back to
  (2.0, 2.0) (checks/fc_experts.out), and the operator's two orders agree to max/min 1.0000 while
  the simple product's vary by 17.0000 (checks/reparam.out).
- On the four-quadrant answer key, experts that model only their own coordinate reach 0.9999
  (plain) and 0.989 (background-divided) in the top-right blob with no A∩B data
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/answer_key.json).
- The parts world with a loose half (variance 1.0) puts A∩B weight 1.000 at 0% overlap, and with an
  empty half 0.000 (step file 05, step 6).
- Across an invertible warp of the four-quadrant toy, construction (1) changes by 4e-17 and the
  plain product by 0.035, 0.080 and 0.126 at area stretches 2.6, 9 and 49
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/README.md).
- Layer 10's latent brick, rebuild/latent.py, is not built yet, and the learned invertible feature
  space is its first piece (REBUILD_WALK.md, brick "Composing in a latent space").

Rules
- Never edit rebuild/, the step files, the read-aheads or their checks/ folders.
- I type; I do not paste.
- No commit.
- Each round ends on a picture of what the brick produced and its check line.

What to hand back
- One script per brick in ~/Playground/PhD/simple-distributions-walk/bradley/, each printing its
  check beside the answer key's line.
- The walk's REBUILD_WALK.md in that folder, one row per brick with its check, which layer 10
  reads.
- The walk's retrospective.
- The proof: the rebuild folder's REBUILD_WALK.md table with every row marked matched.

What you will see
Along the way: one brick per round, each ending on its picture, its check line beside the answer
key's, and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with the rebuild
folder's path; PASS or MISS against every brick's number equal to its answer key to the digits
shown, and the local score computed on both toys (the split pair to 0.0, the shared pair to the
miss values above); what is still open; at most one question.

I judge, from the brick table: done when every row is marked matched and the local score is computed on both toys.
```

added: the seven sections, from the field-labelled version · layer 10's latent brick as the reason for the last piece, in Background (from `REBUILD_WALK.md`)
dropped: nothing
