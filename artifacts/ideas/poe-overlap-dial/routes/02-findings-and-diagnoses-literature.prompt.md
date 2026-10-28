/paper-scout Find the literature on how chest X-ray labels split into findings and diagnoses, and how those labels compose, so the project's medical composition terms rest on citations.

Context (cold start). Repo: /home-mscluster/mmolefe/PhD/reduce-reuse-recycle. Read artifacts/ideas/poe-overlap-dial/composition-terms.md first, especially "The kinds, with examples" and "Findings, diagnoses and diseases", both marked [proposed]. Route 1's return (routes/01-composition-taxonomy-literature.return.md) covers the general composition literature; this route covers the medical side only.

The terms to check, with the project's current examples:
- finding: a visible sign (cardiomegaly, pleural effusion, pleural line)
- diagnosis built from findings: heart failure read from cardiomegaly, effusion and oedema
- comorbidity: two diagnoses with no effect on each other's appearance
- compound disease: two causes changing the same tissue (silicotuberculosis)
- consequence: a device or change that follows a diagnosis (a drain after a pneumothorax), the source of shortcuts

Search for, 2020 to 2026 weighted:
1. Label schemes that separate findings from diagnoses: how NIH ChestX-ray14, CheXpert, MIMIC-CXR and PadChest define their labels, and any radiology ontology (for example the Fleischner glossary or RadLex) that defines "finding".
2. Label co-occurrence and dependence in chest X-ray datasets: measured co-occurrence rates, label hierarchies, and the drain and pneumothorax confounder (Oakden-Rayner and others), with the reported drain rates.
3. Chest X-ray generative models conditioned on several findings at once, and whether any composes separately trained models or reports composition failures.
4. Clinical descriptions of silicotuberculosis on chest X-ray, and of other compound patterns (tuberculosis with HIV, pneumothorax in emphysema), as examples of interaction.

Re-rank on whether a paper defines the finding/diagnosis split, measures label co-occurrence, or generates images from combined findings. Shortlist 8 to 12. File the PDFs on the paper shelf (paper-cadence takes in unfiled PDFs from ~/goal-setting/learning/papers/_incoming/).

Then /drip-read --combine --lean, one idea per walk, two to four sources each:
  (a) what counts as a finding and what as a diagnosis, per dataset
  (b) how often labels co-occur, and which co-occurrences are consequences (devices, treatment)
  (c) compound diseases on chest X-ray: what changes in appearance
Use `differences` at the end of each walk.

Write-back: add citations under the [proposed] medical entries in composition-terms.md, with a confidence flag per citation. Do not rewrite [yours] or [proposed] text; put changes in "Proposed changes from the literature". Do not edit IDEA_MAP.md.

Return: write artifacts/ideas/poe-overlap-dial/routes/02-findings-and-diagnoses-literature.return.md with (1) the shortlist with ids, (2) per term, whether the literature supports, splits or contradicts it, (3) the measured drain rate among NIH pneumothorax images with its source, (4) whether any chest X-ray generator composes findings from separate models.
