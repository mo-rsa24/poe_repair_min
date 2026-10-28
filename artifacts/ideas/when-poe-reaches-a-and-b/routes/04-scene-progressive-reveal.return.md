```
origin: artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md · claim 2 scenes · route 4
ask: one taught concept per press, with what's already taught dimmed

deliverable:
  Kept and republished: Three Ways a Product Reaches A-and-B,
    https://claude.ai/artifact/9eFnG2rWEciPsceMwdem4u (version 3), source
    artifacts/scenes/three-ways-a-product-reaches-a-and-b/. 48 presses. Each press reveals one region,
    label, number or formula; what was taught earlier stays on screen in grey; nothing not yet taught
    is drawn. A prediction is its own press, before the press that reveals its answer. The ledger
    "What each label and number means" gains one row per press, newest on top in colour. Back
    renders each press from scratch, so it un-reveals in the same order. A bottom-bar box
    "Grey what is already taught" turns the greying off.
  Merged: the two scenes taught the same toy, rescues, questions and key rows, so they are one scene
    now. Scene 1 was kept. From Where the Product Lands (https://claude.ai/artifact/2A7YDnfKe8EPsS7yFDVcNT,
    14 presses) two things moved into scene 1: dragging expert B's blob (presses 15 and 16) and the
    fourth answer to the first prediction, "Nowhere: the product is zero". Its row of factor panels
    (A × B ÷ p(x) = product as separate pictures) was dropped, since scene 1 shows that as formulas.
    Scene 2 stays published unchanged; its README points to scene 1 as the one kept current.
  Final press list, scene 1:
    frame 1, support and the zero rule
       1 four grey blobs, one kind of chest film each     2 A is the right half, x1 > 0
       3 B is the top half, x2 > 0                        4 top-right circle: A and B, the rare film
       5 expert A, "A, not B"                             6 expert B, "B, not A"
       7 support formula                                  8 sum of scores = score of p_A·p_B
       9 predict: where the product goes                 10 reveal: the blend at the origin
      11 zero rule                                        12 0.989 in the origin circle
      13 0.006 in each blob circle (tails toggle)         14 Z = 7.9 × 10⁻¹⁰, B opposite A
      15 B next door: Z = 5.3 × 10⁻⁵                      16 B on A: Z = 3.54
    frame 2, rescue (a), pretraining
      17 the pretrained model                             18 p_A = (1 − 3ε)BR + ε(TR + TL + BL)
      19 ε = 0.25, no fine-tuning                         20 predict: top-right share at ε = 0.25
      21 reveal: 0.250 in every corner                    22 ε/(2 − 4ε) with the ε slider
      23 "neither" ties top-right (0.0278 at ε = 0.05)    24 0.472 to A, not B and B, not A
      25 next-door leftover formulas                      26 predict: next door at ε = 0.1
      27 reveal: 0.500 and 0.500                          28 x → −x, so top-right ≤ ½ (toggle, slider)
    frame 3, rescue (a), one model with contains labels
      29 contains labels, r = 0.25 data                   30 q_r = p(x,A) p(x,B)/p(x)
      31 r = 0.05                                         32 predict: top-right share at r = 0.05
      33 reveal: 0.989                                    34 Z ≈ r
      35 rare is enough: r slider, 0.9886 at 0.001        36 absent: r = 0, 0.970 at the origin
    frame 4, rescue (b), experts that learn parts
      37 parts expert A                                   38 parts expert B
      39 predict: where the parts product goes            40 reveal: top-right
      41 0.9999 in the circle                             42 spread 0.106 against 0.150
      43 divide out the background                        44 0.989, spread 0.150
      45 the catch: a drain changes the lung it treats
    frame 5, rescue (c), samplers
      46 the exact target (one model at r = 0.05)         47 empty sampler slot, not run
    48 Explore
  Checks: check_numbers.mjs 123 of 123 pass (114 slider stops plus 9 for B's three blob positions,
    which scene_key.py also checks against scene 2's key). check_scene.mjs --region walked all 48
    presses at 390 px dark and 1440 px light with no must-fix items. Screenshots of presses 10, 16,
    22 and 44 are in artifacts/scenes/three-ways-a-product-reaches-a-and-b/screens/.

changes for the walk: no claim changed. One number on scene 1 was corrected: the one-model
  top-right share at 1 film in 100,000 reads 0.955 (key 0.95546), where the old page had 0.956 typed
  in. Scene 1 now also shows Z = 5.3 × 10⁻⁵ and 3.54 (B's blob next door and on A), the values scene 2
  already showed. IDEA_MAP.md line 165 still describes scene 1 as "15 presses in five" frames; it is 48.
```
