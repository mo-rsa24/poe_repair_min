# Composing medical image generators: merged starting prompt

/drip-execute-plan plans/10-does-roentgen-draw-two-findings/plans/01-the-cheapest-test.md --request

Context for the walk (written for a reader who is learning as they go; define AUROC, detector, prompt and "both findings drawn" rate the first time each appears).

## Direction

Continue the one chest X-ray experiment that three earlier sessions converged on: does a generator (RoentGen-v2) draw two findings in one image when the pair is rare in NIH labels, and does composing two single-finding predictions (Liu et al. 2022, product of experts) do better than writing both findings in one prompt. The first run was uninformative. The user wants to understand and reproduce the authors' pipeline before testing new ideas. A replication ladder (R1 and R2 first) was proposed but not started and not yet approved: begin by reading plan 10 and its walk map, explain the proposed first check in plain words, and resolve that open choice before launching anything.

Treat the reader as starting from scratch, with this history as background. Explain one idea in plain words before showing its numbers, define new terms when they first matter, and end each round with one action the reader can perform and a check they can understand. Keep published results visibly separate from measurements made in this project. Everything below about nodes, packages, git status and script defaults is what the earlier sessions observed, not a current inventory: recheck git status, script contents, the `co3` environment and existing outputs before copying or changing anything.

**The four problem types**

The earlier framing: A is a rare combined class, B is a classifier's shortcut through a companion such as a chest drain, C is a detector weakened by a second condition, D is a disease label that bundles other signs. Plan 10 tests the generation question behind A. B was discussed as a possible dry run and not started; the user judged C and D weak. No decision says the chest X-ray result must gate the liver CT work.

## State of play

**Why this experiment exists**

Silicosis plus TB data (Silicodata: silicosis 1,057, TB 750, silicotuberculosis 752, normal 481, 3,044 frontal images, baseline F1 for the combined class 0.58 to 0.67) needs a signed email the user will not send. The earlier searches found no paper that composes single-disease medical generators as a product of experts and then tests a downstream classifier, and no cited benchmark of composed-disease classification near 75%. That is absence in these searches, not proof of absence, and scarcity as the cause of any gap is unestablished. Prakash et al. reported better downstream F1 with single-disease prompts than multi-disease prompts; that does not show an image-level failure to draw both findings. Cardiomegaly and pleural effusion are radiographic findings, and their co-occurrence alone does not establish two diagnoses, so this project uses them as a finding-level comparison, not as a combined disease class. TB and silicosis are disease diagnoses.

**The chest X-ray test (plan 10)**

The pairs were chosen from NIH ChestX-ray14 labels (112,120 images, 14 findings, 91 pairs). Cardiomegaly with pneumothorax is 49 images against 131 expected if independent (ratio 0.37). The control pair, cardiomegaly with effusion, is 1,063 images (ratio 3.2). Adapter training sets would be 2,727 cardiomegaly-without-pneumothorax and 5,253 pneumothorax-without-cardiomegaly images. Rarity in RoentGen-v2's own MIMIC-CXR training reports is unknown, so do not call either pair rare or common there. The proposed approximation counts each impression once in the released dataset's prompt labels and compares observed co-occurrence with the independence expectation; the relevant MIMIC split is unverified. The control changes the finding (effusion for pneumothorax) as well as the NIH frequency, so a difference between the two pairs does not isolate rarity; read it beside each single-finding result. The pair table is `artifacts/results/does-roentgen-draw-two-findings/pair-table.csv`.

**Vocabulary in this repo**

Mono is one joint prompt naming both things. Inference-time PoE evaluates the model under each single-concept condition at every denoising step and combines each conditional prediction's difference from the shared unconditional prediction; with unit weights this adds the two conditional predictions and subtracts the unconditional one once. It needs no adapter training and no joint-prompt training, and demoting plan 10's adapter-training task was proposed and not implemented. Evidence from one pair on eight seeds: cat x dog, seeds 9 to 16, compose rate 1.0 for Mono and 0.0 for plain PoE and the unweighted control (`report/00-INDEX.md` line 152 at the earlier check). Use Mono as the reference baseline in the medical test; whether PoE improves on it is open. For each method report four detector outcomes: both findings detected, only A, only B, neither. These are threshold-based detector outcomes, not confirmed image content; blending, anatomical plausibility and interactions between findings are separate questions a classifier cannot answer.

**The first run was uninformative**

320 images (5 prompts x 64 seeds), generated on mscluster107 in 1598 s (about 27 min): both-findings rate 0% for the rare pair and for the control. Pneumothorax was drawn 0 of 64 times and effusion 1 of 64. Cardiomegaly was drawn in 37.5% under its own prompt and 45.3% under the effusion prompt. The prompts were a demographic prefix plus short finding text, for example "60 year old White male. Pneumothorax." Pneumothorax passed the detector threshold in 0 of 64 images and effusion in 1 of 64, while cardiomegaly passed 37.5%. The bar code returns "null" for 0 against 0, so read this as no information, not as a null result. The cause (prompt wording, generator, or detector transfer) is untested.

**The detector can see findings**

Detector: TorchXRayVision, weights `densenet121-res224-all`, scored on 4,999 real NIH images (shard images_001). AUROC (area under the curve of true positive rate against false positive rate, 0.5 is chance): cardiomegaly 0.901, pneumothorax 0.771. It saw NIH and MIMIC images in training, so these are optimistic. On 17,126 rows of the authors' released images (shard 16, no normal films) AUROC was 0.675 (cardiomegaly), 0.744 (pneumothorax), 0.836 (effusion). It fired on 35.7% of pneumothorax-positive images against 10.9% of negatives, and 48.2% of images without cardiomegaly still passed the cardiomegaly threshold. The cause of the 0 of 64 (prompt wording, generator, or detector transfer) is untested.

**Published numbers to reproduce**

RoentGen-v2 Supplementary Table 1 (released 10k-step checkpoint, same detector): cardiomegaly 0.77, pneumothorax 0.90, effusion 0.78 on synthetic images; 0.83, 0.92, 0.81 on real; five-disease average 0.81 synthetic, 0.88 real. TorchXRayVision's own benchmark for the all-weights model gives NIH cardiomegaly 0.88 and pneumothorax 0.75; nothing was re-run to compare. Prakash et al. (arXiv 2411.18602) cannot be replicated verbatim: no code, no prompt templates, earlier weights with ControlNet.

**Authors' released resources**

Dataset `stanfordmimi/RoentGen-v2-synthetic-dataset`: 33 parquet shards, 565,154 images, 15.7 GB, gated, sorted by label (shard 0 is all "No Finding"). Prompt form: `<AGE> year old <RACE> <SEX>. <IMPRESSION>.`, 75 steps, guidance 4.0, 512 x 512. Code: github.com/StanfordMIMI/RoentGen-v2 (`inference_code/run_inference.py`, `run_inference_w_quality_check.py`, `configs/`), not yet read. The user accepted the model's access terms on 2026-10-06 and later the synthetic dataset's terms; authenticated downloads worked from mscluster107. Access to the underlying MIMIC-CXR files and the authors' validation subset was not established.

**Where things live**

On mscluster107 (two Quadro RTX 8000, 49 GB each, allocated to the user): `/datasets/mmolefe/poe_repair_min/pair_test/` holds `full/` (the 320 images), `thresholds*.json` for six detectors, `nih_real/`, `rg2_synth/` (shards 0, 16, 32) and `hf_cache` (6.1 GB). Environment `co3` has torch 2.5.1, diffusers 0.29.2, peft 0.19.1, torchxrayvision 1.5.2. NIH labels: `data/nih_cxr14/Data_Entry_2017_v2020.csv`. Scripts, all untracked in git: `scripts/medical_pair_table.py`, `medical_pair_bar.py`, `medical_pair_generate.py`, `medical_pair_score.py`, `medical_pair_check_authors.py`. Walk map: `artifacts/drips/does-roentgen-draw-two-findings/EXECUTE_WALK.md`.

**Side material, not yet used**

Liver CT (Luo 2025, Scientific Data, CC BY 4.0, doi 10.57760/sciencedb.12207): HCC 94, ICC 99, combined cHCC-ICC 85, controls 83, pathology confirmed; the combined class is one tumour with two phenotypes, so it is a weaker fit for "two diseases". The tar is 42,051,328,000 bytes; the download to `/datasets/mmolefe/poe_repair_min/data/liver-cect/` was started by the user via `scripts/download_liver_cect.sh`, using public links with no login (the file-tree listing's login prompt caused the earlier confusion). Completion is unconfirmed: one session saw an `.extracted` marker dated 6 Oct 16:38. The script's expected final counts are 1,444 CT files, 1,112 lesion-mask files and 1,444 liver-mask files; check those and the archive MD5 before restarting anything. `scripts/liver_cect_gallery.py` renders one PNG per patient plus an index.html and was tested only on synthetic volumes. Ten NIOH silicosis and TB teaching films are in `data/silicotb-teaching-films/`. They are teaching examples, not a validated training set: copyright restricts them, labels are experienced readers' interpretations, and bacteriological confirmation is incomplete. The DICOM headers hold real patient identifiers, so use only the PNGs in `png/`, never upload the raw files and never repeat the identifiers. The folder is excluded through `.git/info/exclude`, which does not travel to a new checkout. The NEATX join (3,543 pneumothorax images, 1,637 with a drain, 1,906 without) supports sensitivity comparisons between those groups but cannot reproduce drain-stratified AUROC, because it lacks drain-labelled negatives; any held-out evaluation must split by patient.

## Decisions in force

**Run on mscluster107.** The user allocated it to themselves; confirm that is still current before launching. Code lives in `/home/molef/PhD/poe_repair_min` locally and `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min` on the cluster; large outputs go under `/datasets/mmolefe/poe_repair_min/`. `scripts/cluster.sh run` refuses uncommitted tracked changes, and the plan 10 scripts are untracked, so earlier they were copied through mscluster84 with `scp` and executed on 107. Say so each time.

**Rare pair and control.** Cardiomegaly plus pneumothorax (rare in NIH labels) and cardiomegaly plus effusion (common in NIH labels, the comparison). Chosen by the assistant with the user's delegation ("You can do no.2 for me"), before any image was generated.

**Bar, written before any image.** 64 images per prompt; "drawn" means the detector score is above the 90%-specificity threshold on real NIH images. Null if the rare-pair rate P is at least 0.5 times the control rate C. Support requires P below 0.5 x C, the composed rate K at least P + 0.20, K at least 0.8 x C, and each finding's detected rate in composed images at least 0.8 of its alone-prompt rate. Inconclusive if detector AUROC is under 0.75. A later amendment, the single-finding validity gate, is not yet confirmed in the bar code. Sample size: with 64 images a rate near 50% has a standard error near 6 percentage points. An assistant suggested treating gaps under about 12 points as no difference; that was not adopted, and the real Mono against PoE comparison (shared seeds) needs its own uncertainty check. The 0.75 bar is not lowered after seeing CheXpert-only pneumothorax at 0.693.

**Detector choice.** All-datasets weights as primary, PadChest as secondary. CheXpert-only and MIMIC-trained detectors all fell under 0.75 on pneumothorax.

**Start where the authors stopped.** Replicate the authors' pipeline before testing own ideas. Proposed ladder, not yet run or approved: R1 score all 33 released shards with the detector as a broader alignment check; R2 clone the repo and regenerate with their code; R3 rerun the short prompts through it as a controlled diagnostic. R4 is synthetic pretraining then fine-tuning on a small NIH subset, tested on held-out real NIH patients against a baseline without synthetic pretraining; it differs from the authors' pipeline. R5 names the Prakash target, which the available code, templates and data did not allow. Supplementary Table 1 is a reference, not a pass target: the authors evaluated MIMIC validation prompts (AUROC against paired real-report labels, four seeds per prompt, area interpolation from 512 to 224 pixels), and the earlier session did not have that validation data. Its five synthetic AUROCs at 10k steps are Atelectasis 0.76, Cardiomegaly 0.77, Edema 0.82, Effusion 0.78, Pneumothorax 0.90; confirm the evaluation details against the code first. No published result has yet been replicated. Prakash verbatim is rejected as impossible.

**Two separate questions.** Keep the generative method test (does composition draw both findings) apart from the clinical use (does the synthetic image help a classifier). The CrossMed claim is dropped from the writing until it is checked. Rarity in the generator's training distribution must be measured, not asserted.

## Do not redo

- Do not repeat the first 320-image run unchanged as the main test; the proposed R3 keeps its short prompts only as a controlled diagnostic.
- Do not rely on the default weights in `medical_pair_score.py`. At the last check both subcommands defaulted to `densenet121-res224-chex` (lines 104 and 109), which misses the bar. Pass `--weights densenet121-res224-all` explicitly and check the current defaults first.
- Do not set `HF_HOME` to `/datasets`; it moved token lookup and gave HTTP 401. Set only `HF_HUB_CACHE`.
- Do not read shard 0 of the released dataset for positives; it is all "No Finding".
- Do not offer cardiomegaly plus effusion as a combined class.
- Silicodata, VinDr-CXR (PhysioNet credentialing) and CANDID-PTX (ethics course plus emailed agreement) are behind forms and out of scope. SVBCX has no combined class. NEATX joined to NIH has no drain-without-pneumothorax cell (3,543 images, 1,011 patients, all pneumothorax; 1,637 with a drain, 1,906 without).
- A combined `sleep 90` then ssh command was blocked by the tool; Monitor worked in that session, so use whatever background-wait mechanism the new session has.

## Open, most important first

1. Decide whether to start the replication ladder (R1 and R2). The assistant asked "Do you want me to start R1 and R2?"; the user said "Keep this in mind".
2. Fold in five proposed edits to the user's "What I am trying to find out" paragraph: detector error rates beside every four-way table; PoE needs no adapter training, so plan 10 task 5.1 should be demoted and the cheapest PoE test uses RoentGen-v2 twice; sample-size note (under about 12 points is not a difference); rarity measured in the released dataset's prompt labels (each impression counted once) as a proxy, MIMIC split unverified; and the single-finding gate below. Unanswered.
3. The single-finding gate: the user adopted 25% as a pragmatic pilot gate, not a clinical or published standard, and the transcripts do not show it in `medical_pair_bar.py`. Settle whether equality passes ("above" against "at least") before implementing it. The assistant noted cardiomegaly cleared 25% under every prompt, and suggested requiring a rate of at least 25% under its own prompt plus a margin over other prompts. The margin (the assistant proposed 15 points) is the user's to set.
4. Repair plan 10: the control pair is never composed; there is no baseline fine-tuned on the 49 real joint images; rarity is measured in NIH labels while RoentGen-v2 trained on MIMIC-CXR reports; the hypothesis wording "composition beats the joint prompt" must be marked open; the plan is in none of the root running-order lists, lacks a review file and skeleton sections, and numbers Tasks 1, 3, 5 beside Instructions 2, 3, 5 (run `/verify-plan`).
5. Housekeeping: fix the scorer default weights; commit the untracked scripts; correct `environment/hpc/nodes.md`, which still lists mscluster107 as down; inspect the cadence queue (one session saw 10 entries and a denied drain attempt; sibling audits saw an empty queue or duplicates) and the applicable permissions before acting.
6. Unchecked: whether the paper's first dataset column is MIMIC training (pneumothorax 2.5%, 1,696 studies); the released-image pneumothorax AUROC near 0.90 once normals are included; the liver MD5 and extraction; TorchXRayVision's published NIH numbers against the 4,999-image shard.
7. Unanswered choices: whether the chest X-ray result gates the liver CT work; which CT generator (no public text-conditioned liver CT model; MAISI is 3D and mask-conditioned); whether to create `~/PhD/xray-composition/` as a parent folder for sibling problems A, B, C and D (nothing created); whether to add a cat-and-dog zero-training step to the master-plan prompt; whether a Prakash single-disease baseline on RoentGen-v2 is added as step 0.
8. Counts that disagree: NEATX is described as 3,500 annotations in one place and 3,543 joined rows in another; recheck against the CSV before citing either.

## Requests so far

Early shared part (all three sessions):

1. "/paper-scout Objective: find public datasets and papers that can stand in for silicosis plus TB, which we cannot access. The research question is whether two single-disease image generators, composed with product-of-experts, can synthesize a rare co-occurrence class (disease A and disease B in one patient) well enough to augment a scarce training set and improve a downstream classifier. I need two lists, and they feed each other. [full brief: List 1 proxy datasets with the three required pieces, List 2 how diseases co-occur classed as (a) separate lesions, (b) same region changed appearance, (c) new marker; 5 to 8 per list]". Done: two lists, Silicodata the strongest dataset, several counts "not retrieved".
2. "The importance is the composition of diseases and the ability to have motivation of applying the composition of pretrained diffusion in scarce datasets, and what different options we have. It is vital to ensure that we are solving a problem that actually exists. ... So we just need to know that". Done: no published 75% gap found.
3. "So there is no concept of composition of diseases ? In other words, HIV occuring together with TB manifests in a specific way that pretrained classifiers/benchmarks fail as a result of the composition. ... nothing of that sort ?" Done, partial evidence.
4. "But I won't be able to get that data unless I send that signed email, and I won't be able to. Are there any other replacements ?" Done.
5. "How do you imagine that composition of pretrained diffusion models (of Liu 2022) could be applied in clinical settings ?" Done.
6. "I like 1 and 3 but I need to see the need fall out of literature where we can either cite it, and if we cannot then it is motivated by a limitation in literature. Moreover, I would need examples to see what a rare combined class in real life could be. Do these exist ? and which instances is not having negation bad, and how can negation help ?" Done.
7. "I want to see if I would be able to have readily available datasets that I can download and look into the images for both instances of Real rare combined classes and Where lacking negation hurts," Done.
8. "No I want to understand how rare combined classes actually manifest on an x-ray. ... Do they see that a combination of signs make up a disease, can we have different signs be signs of different diseases ? for example Tb is a disease, silicosis is a disease ?" Done.
9. "So let's revisit the different problem statements where one of them was "Rare combined classes", Where lacking negation hurts,", maybe there's more, and can you tabulate these again". Done: rows A to G.
10. "We need the table again but for open data to look at, and literature, we need more web fetch and search because we have to know the dataset, what classes it has, which classes are relevant from the actual problem. So that we have a better idea. We need some examples". Done.
11. "No for each of the problems, take a use case scenario of different patients leading up to the problem, and motivation for solving, what is the problem in literature. Give me a very high-level overview of the deep learning solution. ... Then where does Liu et al play a role ?" Done.

Branch that set up liver CT and the folder plan (sessions 0ac4f7d7 and 05f748c6):

12. "Let's start with A. Because we might need a few /drip-read for related papers, so we can start downloading x-rays, organizing them, doing the necessary cleaning and validation, opening them up, looking at the signs myself, /video-scout for specialists to explain then see combination and interpret how it manifests". Partly done: ten NIOH films downloaded and checked; drip-read and video-scout deferred until images exist.
13. "Are these the only images we can find ? Can we find a similar or same problem with other diseases ? Does Tb and Pneumonia have the same issues as silicosis and TB?" Done.
14. "What about the following pairs and modalites - do they still fall under A: [eight pairs] Or is it better to focus on B ? Because our own problem readily available data. The silicoTB have long turnaround time, writing and sending emails, waiting, etc". Done: only liver CT fits A with open data.
15. "I downloaded the csv. This is where we can leverage the mscluster. ... Then you can help provide instructions of the quickest and most efficient manner to load from the images from that folder." Partly done: script written, tar download completion unconfirmed.
16. "We need to think of a convenient way to view them. Isn't it dangerous to open a vs code remote ssh on the /datasets/molefe mount, won't it affect other students because it's a shared cluster. What's quick and simple to be able to scroll in a folder, double click and see some x-rays". Done: gallery script, untested on real data.
17. "Let's carefully walk B. Reiterate it again and what does it entail". Done: B restated; nothing started.
18. "I think these steps need to enable us to create a new empty repo with an init-master-plan, and simplified diagrams ... then write the prompts that I should write in order". Partly done: folder tree and eleven prompts written, nothing created.

Branch that ran plan 10 (session f5a87047):

19. "C and D seem really weak. A seems to only have HIV & TB, or silicosis and TB as the only use cases. How useful then can composition be in medical x-rays ... Whereas in the wild, it has applications. Because SD 3.5 fails at certain compositional tasks that SDXL can outperform simply by using PoE ... And the equivalent of "a cat and a dog" vs "a cat" and "a dog"". Done: led to plan 10.
20. "/drip-execute-plan --request The cheapest test: Build a table of finding pairs from the report labels, with the count of reports containing both. Prompt RoentGen-v2 for each pair and for each finding alone. Run a detector for each finding, and score how often both appear. Compose the two single-finding adapters, and score again." Partly done: table, bar, generation and scoring done; composition not started.
21. "a. I allocated mscluster107 to myself so we should use that" Done. "defaults" Done. "You can do no.2 for me" Done. "Go but what is the generation script for mscluster107 ?" Done. "I just accepted the stuff for https://huggingface.co/stanfordmimi/RoentGen-v2" Done. "It looks like image 1 and 3 have cardiomegaly. Hard to determine pneuothorax when I am not a radiologist" Recorded.
22. "1. Please provide a concise primer on which (between A, B, C, D) this experiment is. Please remind us on what the problem/gap in literature is, what do we hypothesize is the solution, what is the dataset, what is the model, what is the control and why - what is the control accomplishing in this case, what are we trying to show with the rare pair, what is the experimental design 2. Go on with detector script then fix plan's numbering and missing sections afterward". Partly done: primer and detector script done; plan numbering and sections not fixed.
23. "go, and you can find the best models on huggingface or benchmarks" Done. "Nice keep going" Done.
24. "Is there any codebase that you can use as reference in which the agent is forgetting to leverage ? In other words, is the agent failing because there's something in the literature and the codebase of the authors that we are not using appropriately that is not providing expected results. Can we not start from where somebody left off ? For example, having the Prakash paper, try to replicate their results verbatim and having that as a starting point. We want to start where a paper, or a result in literature stops". Done: RoentGen-v2 repo and dataset found; Prakash not replicable.
25. "Done" (accepted the dataset terms). Done: shards 0, 16, 32 downloaded.
26. "very simply and slowly walk me through the results, which ones are from published score cards or literature, which did we run to replicate their results ?" Done: published and run numbers separated; nothing replicated yet.
27. "How can we run something using each of their code, datasets, and basically anything we can to replicate their results before we start training or testing any of our ideas ?" Done as a proposal (R1 to R5); not executed.
28. "1. Keep this in mind. I am going to paste two separate prompts where I want you to summarize this interaction ... then join the 3 prompts into a single prompt that we will use to start a new session. It should be in drip-style ... Be efficient, and remove redundancy." and "The idea is that in the session, I am basically starting from scratch but I am using this session as context and background for the drip-execute-plan --request. Assume that I myself do not really know much, but will gain knowledge as I walk through the pieces of the drip walk. So I just need a single step which is a prompt in a new session". Partly done: a prompt was drafted; this merge replaces it.
29. "There are some inaccuracies in the prompt itself like ' ... a cat and a dog draws two animals not one". Done: Mono and PoE roles corrected from `context/world/poe-composition.md`.
30. "What do you think about " [the user's rewritten "What I am trying to find out" paragraph: Mono against PoE with the Liu formula, a four-way outcome, MIMIC-based rarity, an AUROC definition, a 25% gate]. Done: five changes offered; awaiting a go-ahead.
31. "Can you see f5a87047-c4c9-464a-b11b-972a4056f6ee, and see if we cannot join this session into that one to have a consolidated interaction ?" then "You can just give me a prompt that will summarize what we have done and concluded here, and it will ask that session to summarize what its done and concluded, then have a pass and join the prompts to make one consolidated prompt, to start a brand new session in". Done: this merge is that consolidation.

## Preferences

- Show the problem is real before building: cite it from the literature or state it plainly as a limitation. Give concrete examples and real datasets to download and look at.
- Distinguish signs or findings from diseases.
- Prefer open data over anything behind forms, emails or long waits.
- Start where a published result stops: replicate the authors' code, data and numbers first.
- Walk results slowly and plainly, with published numbers and numbers this work ran kept visibly apart.
- Judgment calls (pair choice, bar thresholds) are delegated to the assistant with veto before images are generated. The 25% single-finding gate is the user's own and is not changed silently.
- The user is not a radiologist and cannot judge pneumothorax by eye; measure with detectors, and show images with labels.
- Use the repo's own definitions (Mono is the joint prompt, PoE is composition) and check `context/` before rewriting prompts.
- Write in plain words with no dashes as clause separators and no private labels; every number carries its unit and meaning.
- Never use the raw DICOM files in `data/silicotb-teaching-films/`.

## Overruled

- **Cardiomegaly plus effusion as a combined class.** The early sessions offered it; the user corrected that these are findings, not two diseases. It stays as the comparison pair only because it is common in NIH labels.
- **RoentGen-v2 gating.** An early claim that the weights are open lost to the later verified 401 and the accepted MIMIC-CXR terms.
- **Mono and PoE roles.** The first starter prompt reversed them; the later check of `context/world/poe-composition.md` and `report/00-INDEX.md` line 152 wins, and only for one pair on eight seeds.
- **Whether the first run is a null.** The bar code prints "null" for 0 against 0; the later session's reading (uninformative) wins because pneumothorax and effusion were almost never detected even alone.
- **Liver CT download status.** One session had about 1 h 57 min left at 5.69 MB/s; another saw an `.extracted` marker dated 6 Oct 16:38. Treated as unconfirmed, not resolved.
- **Where the fork point sits.** The two forks disagree on which items belong to the shared early part (the 05f748c6 digest placed the A to D write-up and B walk before its fork marker). Requests are merged with duplicates kept once, so this does not change the content.

---

## Validation log

Checked: 29 strings (counts, AUROCs, byte size, arXiv ids, weights names, node and env names) by grep across the three transcripts; every one appears in at least one transcript. 8 paths checked on disk: plan 10 file, walk map, results folder (pair-table.csv, smoke sheet), five `medical_pair_*` scripts, `download_liver_cect.sh`, `liver_cect_gallery.py`, `data/silicotb-teaching-films/` (manifest.csv, png, raw), `data/nih_cxr14/`. All exist. Cluster paths and the liver tar status were not checked from this machine and keep their tag. Not changed: no number was removed. Caveats carried as stated, not verified: CAD4TB 24.0% against 60.3% (search snippet), Oakden-Rayner AUROC figures (secondary summary), the Table 1 numbers (read from paper PDF text), the liver counts (paper summaries). The quoted requests are taken from the digests, which quote the transcripts; two long pasted briefs (request 1, request 28) are shortened with bracketed notes, so a new session wanting the exact pasted brief should read the parent transcript.

## Codex review

Done: job 20261007-163355-578e75 returned 24 findings (10 must, 14 should) and 3 accuracy flags. Each was checked against the transcripts (for example the prompt form and Table 1 values appear in the f5a87047 transcript). All were applied as edits, mostly adding qualifications: the opening contradiction about starting the replication ladder, the support rule missing "P below 0.5 x C", the 12-point rule demoted from a decision to a suggestion, the effusion and one-word-prompt overstatements, the PoE definition, the four-way detector outcomes, and stale-state caveats. None was dropped. The Codex job is returned and not yet run through triage_job.py; the edits were made by hand from its response file.
