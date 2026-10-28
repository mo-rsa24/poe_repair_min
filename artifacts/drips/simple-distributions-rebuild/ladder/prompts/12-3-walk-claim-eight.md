# Layer 12, step 3: walk claim 8

⬅️ [Layer 12, step 3](../12-the-correction-at-scale.md#3-you-then-the-open-questions-in-one-walk)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`, after step 2's verdict is recorded. It runs whatever the verdict is.

```text
/drip-idea resume artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md at claim 8, with the
drain-gap result: which shortcut leads (the drain, or one that cannot be masked such as view or
scanner); whether NIH's drainless pneumothoraces are mostly large (measured with the SIIM-ACR masks
at /datasets/mmolefe/siim_acr); whether FastDiME trains classifiers on its edits; and how many of
the 4,312 follow-up films with no pneumothorax label still carry a drain (estimated at about 1,100)

Background
Layer 12 of the composition ladder (artifacts/drips/simple-distributions-rebuild/) asks whether a
correction for composition helps on chest X-rays. Step 2 measured the drain gap in B1's project,
~/Playground/PhD/medical-imaging/compositional_diffusion/b-shortcut-through-a-device/b1-drain-shortcut-in-pneumothorax/.
Claim 8 of the idea map says what composition's medical job is, and four questions about it are
still open. This walk settles them in the repo
/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.

Objectives
1. Walk claim 8 with the four questions in the skill line, one per round.
2. Close each question with an answer or a "needs a check" entry, counted ones with their counts.
3. Name the shortcut composition should target.
4. Mark whether the drain case stands or ends.
5. Integrate route 6's return into the map if the walk takes it in.

What to read
- Instruction 11.1 of B1's plans/01-measure-the-drain-gap.md: step 2's verdict.
- The sidecar CSV beside sensitivity-by-drain__nih-neatx.png in B1's
  artifacts/results/does-the-classifier-use-the-drain/: the gap's numbers.
- artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claims 6 and 8 and its Sources: the claim
  being walked.
- artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md:
  route 6's return.
- B1's pick-up page, plans/00-where-we-left-off.md: the follow-up pool and the unanswered proposals.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Claim 8 is wrong as stated (the map, claim 8).
- Its repair puts composition on the drain shortcut as one arm against upweighting the real
  drain-free films and against masked editing, run only after B1's drain gap shows the shortcut
  exists (the map, claim 8).
- The same repair gives composition its own job on a shortcut that cannot be masked, such as AP
  against PA view or scanner (the map, claim 8).
- 1,906 of the 3,543 NEATX pneumothorax films have no drain, so the drainless cell is common and not
  missing (claims 6 and 8).
- Oakden-Rayner et al. report AUC 0.940 with a drain and 0.770 without on the NIH test set (claim 8).
- Thian et al. 2021 find 0.95 against 0.94 at six outside hospitals (claim 8).
- TorchXRayVision's DenseNet moves a median of +0.001 over 237 same-patient pairs, with the drain
  film higher in 134 (claim 8; B1's results card does-the-classifier-look-at-the-drain/README.md).
- That model puts 180 of 1,637 drain films against 514 of 1,906 drain-free films below its threshold
  (claim 8; the same results card).
- The split holds within each view, PA 131 of 1,030 against 407 of 1,281 and AP 49 of 607 against
  107 of 625, which may come from which patients get a drain rather than from the tube (claim 8; the
  same results card).
- RadEdit removes the pneumothorax and keeps the drain (claim 8; the map's Sources).
- A classifier trained on CANDID-PTX falls from 93.3 to 17.9 accuracy on 629 RadEdit-edited films
  (claim 8; the map's Sources).
- RadEdit tests classifiers and never trains on its edits (claim 8; the map's Sources).
- Route 6's return answers the FastDiME question: FastDiME trains ResNet-18 at 100, 75 and 50% drain
  association and reports AUROC 0.98 with the association and 0.65 without, detection only
  (routes/06-composition-against-masked-editing.return.md).
- FastDiME's discussion names training on counterfactuals as an untested next step (the route 6
  return).
- The route 6 return is not yet integrated: the map's Sources still say "unchecked" and its Routes
  row says "emitted" (the route 6 return).
- Route 6's return names two missing arms: a jointly conditioned model p(x | pneumothorax, no drain),
  and prevalence recalibration (Kina and Petersen: AUROC 0.226 to 0.726 on films where the drain
  points the wrong way) (the route 6 return).
- Route 6's return proposes the bounded positive label "no drain" in place of negation (the route 6
  return).
- Route 6's return gives test sizes of about ±0.14 at 25 films against 25 and ±0.07 at 100 against
  100 (the route 6 return).
- The follow-up pool is 4,312 later NIH films of NEATX drain patients with no pneumothorax label,
  from 385 patients (B1's pick-up page, route 9's answer; claim 6).
- About 1,100 of those probably still carry a drain, range 900 to 1,400, from three counted group
  sizes times estimated shares, none looked at (B1's pick-up page, route 9's answer; claim 6).
- 907 of the follow-up films sit between two drain films, and 397 of those are labelled "No
  Finding" (B1's pick-up page, route 9's answer; claim 6).
- /datasets/mmolefe/siim_acr holds positive_imgs.npy, positive_masks.npy and
  positive_imgs_names.npy (listed 2026-10-18).
- Data_Entry_2017.csv carries View Position, image size and pixel spacing per film and no scanner or
  site column (its header line, in B1's data/).
- B1 never settled which shortcut leads: a session proposed keeping the drain as the lead and adding
  a non-localised shortcut (AP against PA view, or scanner and site) only if the drain result is a
  null or ties with RadEdit (B1's pick-up page, "Proposals never answered").

Rules
- Write only in that idea map and its routes/ and runs/ folders.
- Read B1's files and never edit them.
- No commit.
- Each round closes one question with a mark, shown before the next round opens.

What to hand back
- Claim 8's row and section in the map, with the shortcut to target and the drain case marked
  standing or ended.
- An answer or a "needs a check" entry per question, counted ones with their counts.
- Route 6 integrated, if the walk takes it in.
- Claim 8's row and section are what step 4 reads if the gate passed, and what step 11's cross-links
  read.
- The proof: grep -n "^| 8\." artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md prints claim 8's
  new mark.

What you will see
Along the way: one question per round, each closing on its mark and the walk's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the map's path;
PASS or MISS against the shortcut named, the drain case marked, and both counted questions carrying a
number; what is still open; at most one question for me.

I judge, from claim 8's row: done when the shortcut to target is named, the drain case is marked standing or ended, and both counted questions carry a number.
```

added: the seven sections, from the field-labelled version · the Background paragraph placing this walk after the gate (from this layer's "Why it is critical" line)
dropped: nothing
