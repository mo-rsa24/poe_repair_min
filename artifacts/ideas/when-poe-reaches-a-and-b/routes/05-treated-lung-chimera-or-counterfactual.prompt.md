I want to think through one question freely, outside my idea walk: is "a treated-looking lung with no chest drain" a chimera to avoid, or exactly the counterfactual a shortcut study needs?

## Where this comes from

Repo: /home-mscluster/mmolefe/PhD/reduce-reuse-recycle. The walk is at artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md (read claim 2 and its children). It asks when composing two diffusion experts (adding their scores, a product of experts p_A·p_B) can generate chest films that are rare in NIH ChestX-ray14: pneumothorax without a chest drain. A pneumothorax classifier leans on drains. In secondary reports, a DenseNet-121 on NIH scores AUROC `0.940` on pneumothorax films with a drain and `0.770` without one, attributed to Oakden-Rayner et al. 2020. I have not verified those numbers against the paper.

What the walk has established:
- A product only puts samples where both experts give probability (the support rule). There are three ways round it: borrow it (support kept from pretraining; strongest as one conditional model with "contains" labels, which holds `0.989` of the product in the rare corner down to a data share of `0.001` on the four-quadrant toy), build it (experts that model only their own image part, projective composition, `0.9999` with no rare data at all), and grade with it (the toy as a sampler answer key).
- The open risk for "build it": the parts are not independent. A drain treats the lung, so films with a drain usually show a partly re-expanded lung, and pneumothorax without a drain is usually untreated, with a larger collapse. A factorised pneumothorax expert trained mostly on drain films would pair a treated-looking lung with a clean chest wall: a film that may not exist clinically.

## The question

1. For training data that teaches a classifier pneumothorax without a drain, a treated lung without a drain looks like a chimera. Does it hurt, help, or do nothing for the shortcut?
2. For stress-testing the classifier, the ideal image is the same film with only the drain removed, which is also a treated lung without a drain. Is that a counterfactual edit (RadEdit, arXiv 2312.12865; Weng et al. ECCV 2024) rather than a composition, and does that make composition the wrong tool for this half of the job?
3. Which of the two uses (augmenting training data, or stress-testing) does my PhD need first, and what does each one require the generated film to look like?

Useful facts: NEATX gives manual drain labels for 3,709 pneumothorax-positive NIH films only; Jiménez-Sánchez et al. 2023 (Detecting Shortcuts in Medical Images) released drain annotations for a subset; there is a conda environment `radedit` at /datasets/mmolefe/shortcut_companion/envs/radedit, and a `cxr` environment. Check what /datasets/mmolefe/shortcut_companion holds before assuming anything about it.

Talk it through with me in short rounds. Bring in a paper only when a round needs it, and flag how confident each citation is.

## Return

When we are done, close with a fenced return package and also write it to artifacts/ideas/when-poe-reaches-a-and-b/routes/05-treated-lung-chimera-or-counterfactual.return.md:
  origin: artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md · claim 2, child 2b · route 5
  ask: chimera or counterfactual, and which use comes first
  deliverable: the answer to each of the three questions in one or two lines, with sources
  changes for the walk: one line on whether 2b's open risk is a risk, a feature, or both, depending on the use
Then I paste it into the idea walk and say `integrate`.
