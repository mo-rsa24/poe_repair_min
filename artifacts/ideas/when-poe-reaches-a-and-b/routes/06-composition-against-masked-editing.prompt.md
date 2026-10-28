/pressure-test Is composing diffusion models (product and negation of separately trained or conditional experts) still a worthwhile PhD direction for the chest-drain shortcut in pneumothorax classification, now that masked diffusion editing can remove drains from real films?

## The idea, in my words

A pneumothorax classifier trained on NIH ChestX-ray14 leans on chest drains, because most positive films show one. The films it fails on are pneumothorax without a drain, which are rare in the data and mostly untreated in the clinic, with a larger collapse. My plan was to generate those films by composing diffusion experts: pneumothorax AND no drain, or pneumothorax AND NOT drain (negation), with annealed MCMC samplers (Du et al. 2023, Reduce, Reuse, Recycle), and to understand on 2D toy data first when such composition works and when it fails.

## What the idea walk has settled

The walk's map is /home-mscluster/mmolefe/PhD/reduce-reuse-recycle/artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md. Read claims 2, 7 and 8 and the Sources table.
- Claim 2: a product p_A·p_B only puts samples where both experts give probability. There are three ways round it. Borrow it: support kept from pretraining, strongest as one conditional model with "contains" labels, which holds `0.989` of the product in the rare corner of a four-quadrant toy for any rare-data share down to `0.001`; separately fine-tuned experts reach at most `0.5`. Build it: experts that model only their own image part (projective composition, Bradley et al. 2025a) reach `0.9999` with no rare data at all, but on real films a drain changes the lung it treats, so a factorised expert risks a treated-looking lung with no drain. Grade with it: the toy as an exact answer key for samplers.
- Claim 7: model sophistication with the same target density does not change any of this; region or box conditioning and GLIDE-style composition are rescue configurations rather than better models.
- Claim 8, needs a check: masked editing removes a drain from a film that has one and keeps that film's treated lung, so it yields the same treated-lung-without-drain film. The check is one classifier trained three ways (original films; plus drain-removed edits; plus composed films), tested on real pneumothorax films without a drain (NEATX drain labels).

## The specific gap

A scoped search found, at abstract level only: RadEdit (arXiv 2312.12865) uses masked editing to stress-test biomedical vision models under acquisition, manifestation and population shift, without retraining any classifier; a search summary says its masked edit removes the pneumothorax and keeps the chest drain (verify in the paper). Weng et al., ECCV 2024 (Fast Diffusion-Based Counterfactuals for Shortcut Removal and Generation) adds or removes shortcut features to assess how they influence predictions, with a pacemaker as its named example; its page does not mention drains or retraining. Neither, at abstract level, retrains a classifier on edits or tests on untreated pneumothorax.

## The ask

1. Has anyone retrained a chest X-ray classifier on counterfactual edits or generated films to remove the drain shortcut, and measured AUROC on pneumothorax without a drain? Name the papers and what they measured.
2. Has anyone used composition of diffusion models (product, negation, or classifier-free AND/NOT) for medical counterfactuals or shortcut mitigation?
3. Your verdict: is "composition against masked editing for the drain shortcut" real, already done, or wrong, and what would make the composition line worth a thesis chapter even if editing wins the clinical comparison?

Return. Close with a fenced return package, and also write it to /home-mscluster/mmolefe/PhD/reduce-reuse-recycle/artifacts/ideas/when-poe-reaches-a-and-b/routes/06-composition-against-masked-editing.return.md:
  origin: artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md · claim 8 · route 6
  ask: novelty and worth of composition against masked editing for the drain shortcut
  deliverable: the verdict, the papers answering questions 1 and 2 with what each measured, and confidence flags
  changes for the walk: one line on whether claim 8 stands, and what the PhD's composition line should claim
Then I paste it into the idea walk and say `integrate`.
