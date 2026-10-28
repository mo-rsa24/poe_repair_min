```
origin: artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md · claim 2, child 2b · route 5
ask: chimera or counterfactual, and which use comes first

deliverable:
  1. Training data. A treated lung with no drain is close to real films: the check film after a drain
     is pulled, and a small pneumothorax watched without one (BTS guidance, high confidence). It helps
     against the drain shortcut only if the air rim stays visible; a fully re-expanded lung labelled
     positive is label noise. Generated negatives must come from the same generator, or "synthetic"
     becomes the next shortcut. Training does not need composition on NIH at all: 1,906 of the 3,543
     NEATX pneumothorax films have no drain (b-shortcut-through-a-device, pneumothorax-score-per-film.csv),
     so any generated set must beat simply upweighting those real films.
  2. Stress test. The same film with only the drain removed is a counterfactual edit, since a product
     p_A·p_B draws a new patient each time. RadEdit is itself a masked composition: the edit region is
     prompt-guided and the keep region is pinned to the inverted real film, which is 2b with one expert
     replaced by the real film. Unanchored composition is the wrong tool here; anchored composition is
     the edit. Both edits exist: RadEdit removes pneumothorax and keeps the drain (CANDID-PTX classifier
     93.3 on real films, 17.9 on 629 edited ones; Pérez-García et al., arXiv 2312.12865, Table 2, high
     confidence), and FastDiME removes and adds drains on the 3,543 NEATX films (Weng et al., arXiv
     2312.14223, medium confidence, about three quarters read).
  3. Stress-testing comes first, because it is the ruler any training fix is judged by and it needs
     only an existing edit. The stress-test film: drain detector below threshold, no tube by eye, no
     pixel change over the pneumothorax edge, scored against a VAE round-trip of the original (the one
     RadEdit edit on disk, 00000732_005, garbled the "AP" marker outside its mask). A training film:
     visible air rim, size range matching the 1,906 real drain-free films, no drain track, negatives
     through the same generator.
  Gate on all three: whether the shortcut exists on NIH is unshown in this project. Oakden-Rayner et
  al. 2019 report AUC 0.940 with a drain and 0.770 without on the NIH test set (full text read, high
  confidence); Thian et al. 2021 find 0.95 against 0.94 at six outside hospitals (abstract, medium);
  TorchXRayVision's DenseNet moves a median of +0.001 between drain and drain-free films of the same
  patient over 237 pairs. B1 route 1 (task 10.1 of 01-measure-the-drain-gap.md) settles it.

changes for the walk: 2b's open risk is a feature for stress-testing (the treated lung without a drain
  is the counterfactual, made by an edit anchored to the real film), and a risk for training only where
  the air rim vanishes. Separately, claim 2's medical example needs restating, because pneumothorax
  without a drain is 54% of NEATX films, not rare; the job left for unanchored composition is a shortcut
  that cannot be masked (AP against PA view, scanner), as B1's where-we-left-off file proposes.
```
