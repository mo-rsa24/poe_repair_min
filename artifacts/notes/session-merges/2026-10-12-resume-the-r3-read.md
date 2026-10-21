# Resume the Reduce, Reuse, Recycle read, with the composition corrections carried in

## Direction

Resume the `/drip-read` walk of Du et al., *Reduce, Reuse, Recycle* (arXiv 2302.11552), from where its map stands. This time the walk also reads the paper for the conditions under which composing two experts succeeds, because the reader's clinical goal (drain-free pneumothorax x-rays by negation) depends on those conditions.

## State of play

**The walk's map is the authority.** It lives at `/home/molef/goal-setting/learning/deep-learning/diffusion-models/02-sampler-correctors-for-composition/drips/read-reduce-reuse-recycle/READ_MAP.md` (483 lines, last written 2026-10-09 05:54). Read it first, with a listing of its `routes/` folder. Its "Current chunk" row reads "Abstract, supporting batch (chunks 2 and 7), then the skippable list". The claim list marks abstract chunks 5 and 6 as "walked lean". Two parts of the map are stale against that row. The "States" row still counts 165 not reached, and the "Next step" section still describes the opening round. Fix both in the first round's map write.

**The reasons the lean sort uses** were chosen by the reader on 2026-10-09: whether the paper's argument predicts the compose-rate cliff at denoising steps 0 to 10, and why a product of experts makes hybrids rather than co-occurrence. The 165 chunks are sorted into core, supporting and skippable against those two reasons. This session proposes a third reason (below), so chunks may move.

**No object is approved yet.** The proposal is the journey's two-Gaussian toy (experts with means -1 and 1, variances 0.5 and 0.8; product mean -0.23077, variance 0.30769), with the paper's ring-times-bar from Figure 2 as an optional extra stage. At ᾱ 0.5 the two routes differ. Multiply-then-noise gives mean -0.163178 and variance 0.653846, and noise-then-multiply gives mean -0.064282 and variance 0.409091.

**The read-ahead is complete and integrated.** It produced `sources/reduce-reuse-recycle/read-ahead/00-INDEX.md` plus 14 section files covering 165 chunks, and its return is `routes/read-ahead-return.md`. It found 40 slips in the printed paper. These go to the reader one at a time, in the round that walks each chunk, never as a list. The check scripts sit in `sources/reduce-reuse-recycle/checks/` (38 `.py` and 36 `.out` files on disk today). With exact scores, reverse diffusion on the summed score ends at variance 0.2102 against the product's 0.3077. MALA reaches 0.3076.

**Two routes landed after the walk's last round.** `routes/code-vs-paper.md` (2026-10-09) audits the released code against the paper in 60 rows and finds 28 differences that can change results, including the PyTorch MALA proposal variance and the HMC rejection momentum. Read it before trusting any sampler in the authors' code. `routes/inventory.md` (2026-10-11) has not been integrated into the map.

**The playground** is `/home/molef/playground/deep-learning/reduce-reuse-recycle/`. The authors' code is cloned in `reference/` at commit `81177d9`, and `reference/notebooks/simple_distributions.ipynb` covers the 2D densities. The venv built on 2026-10-02 is still at `.venv/`, and numpy, scipy and matplotlib import from it. A plain `ls` hides it, which is why session 69356b18's listing showed only `reference/`. The CLEVR data (`dataset_train_larger.npz`, README line 42) and the checkpoint (`ebm-49x1874.pt`, README line 59) are not downloaded.

**Jobs running** when this merge was written: none. Check for live jobs before launching anything.

### Five corrections to the reader's composition intuition

They come from session 69356b18. The reader drew A (red) and B (blue) shapes and asked whether "shape A expert should see blue". The walk should test each correction against the chunk that carries it.

**Support.** A product is high where the experts overlap, which holds. Within the overlap, mass is weighted by both experts' heights, and the conditional form p(x|A)p(x|B)/p(x) also divides by the base density p(x). The two-Gaussian toy multiplies raw densities with no p(x) term, and the Factorization correction below is about the conditional form. Each expert must put mass where both findings are true, which is necessary but not sufficient. The A expert never needs to know what B is. It needs nonzero mass where B is also true. If B's region falls in a gap of A, the product is empty and a sampler returns whatever the tails give. Learned scores are poor in sparse regions, so the testable question is how much coverage is enough. In the clinical scope, cardiomegaly with pneumothorax appears in 49 NIH images against 131 expected under independence, a ratio of 0.37 (`plans/10-does-roentgen-draw-two-findings/plans/01-the-cheapest-test.md`).

**Factorization.** This one comes from the ChatGPT chat. The product of experts equals the joint only when P(A,B|x) = P(A|x)·P(B|x). At t=0 that holds if the labels are deterministic functions of the image. At noisy t it fails, and the gap is r_t = ∇log[P(A,B|x_t) / (P(A|x_t)P(B|x_t))].

**Sampler.** This one comes from Du. Summed scores at time t are not the score of the noised product, so reverse diffusion misses the product even with perfect experts (the 0.2102 against 0.3077 variance above). With deterministic clean labels, factorization and sampler are the same gap seen from two sides. Du fixes it with MCMC at each noise level, and the adapter fixes the intermediate distributions directly. Neither fixes a product whose peak is itself a hybrid. The open prediction, untested, is that if the clean product favours chimeras, a better sampler (HMC) makes chimeras more frequent.

**A label is learned as a contrast.** The expert does not know which finding is meant. s_C − s₀ = ∇log P(C|x_t) is a classifier learned from which images carried the label, so anything correlated with the label (atelectasis, lines, drains) is folded in at its co-occurrence rate. The product then reweights by both implicit classifiers. This is hidden stratification moved into the generator. An unlabelled finding has no expert at all.

**Drain negation needs three conditions.** Liu's negation ε₀ + w(ε_PTX − ε₀) − w̃(ε_drain − ε₀) works only if (a) the pneumothorax expert's data contains pneumothorax without a drain (support again), (b) a drain label exists, which NIH CXR14's 14 labels do not include, and (c) the drain direction is separable from the pneumothorax direction. If nearly every pneumothorax image has a drain, negation has nowhere to go. R3's appendix E has a 2D negation example to test this on.

### The ChatGPT chat "Diffusion Sampling Basics"

Saved as raw text at `/home/molef/PhD/poe_repair_min/.playwright-mcp/share.txt` (60,283 bytes, untracked). The file runs to the page's closing footer, but session 69356b18 found the last turn cut off mid-section, so the conversation itself may be incomplete. It has four turns. The first covers score against sampler (DDPM, DDIM, reverse SDE, probability-flow ODE, MCMC and HMC). The second says samplers agree in distribution but not image by image, and poses H1 (the composed score field is wrong) against H2 (the sampler fails to explore). The third derives that the product equals the joint only under conditional independence, with a toy where the hybrid wins (0.050 against 0.015), and defines r = s_AB − s_PoE. The fourth covers LoRA, classifier against classifier-free guidance with the multi-condition form s₀ + w(s_A + s_B − 2s₀), the mismatch between a curated LoRA and SDXL's own unconditional, and a staged classifier experiment. No clean markdown copy exists in the journey's `sources/` yet.

## Decisions in force

The paper and maths routing below are session a0661fcd's answers to the reader's questions. The third sort reason, the reproduction targets and the experiment steps come from session 69356b18's prompts, which the reader had not sent when this merge was written.

**The proposed object is the two-Gaussian toy**, awaiting the reader's approval, because the journey's plans 14 to 16 already build on it.

**A third sort reason is proposed**: the conditions under which composition succeeds (support, factorization, sampler), carried forward to drain negation. It comes from prompt 3 of session 69356b18's sequence, so offer it in the first round for the reader to confirm. Under it, sections 4.2 and 5.1 (Table 1, Figure 2), 5.2 (CLEVR, Table 2) and appendices D and E count as core.

**The maths routes to the journey's plans**, not into the read: plans 10 to 13, 21, 23 and 33 to 36 (Markov chains, detailed balance, Metropolis, Langevin, the Hamiltonian corrector, energies). Plans 17 to 19 (Brownian motion, Itô, Fokker-Planck) are skipped unless a chunk needs them, at the reader's request.

**Other papers get their own walks, in this order.** Bradley et al. 2502.04549 (when the product is the right target at all) walks in `01-poe-derivation-foundations` and is read first when the hybrid question is live. CO3 2509.25940 walks in the same journey after Bradley. SuperDiff 2412.17762 is covered by `/drip-execute-plan` on journey plan 31. Product of Experts for Visual Generation 2506.08894 gets `/unpack-paper` only when cited.

**Reproduction uses `/drip-execute-plan --reproduce 2302.11552`**, with `--request` kept for new experiments. Its targets, in order, are Table 1 and Figure 2 (2D product and mixture), Figure A1 and the appendix E negation, then Table 2 (CLEVR, 1 to 5 cubes). Write each bar before running, use the drip-repo walk's clone, and mark anything the 28 code/paper differences make unreproducible as out of reach, with the reason. The clinical drain test waits until the 2D coverage and negation sweeps give a verdict.

**Proposal ideas ride inside walks where the toy carries them.** `d_t` (distance over time, proposal section 5.1.4) and independence with a partitioned latent (Phase 2 regimes A, B and D) fit inside the Bradley walk. β-VAE, FactorVAE, total correlation and DCI on dSprites (regime C) need a separate journey. Riemannian geometry is out. MDS stays a drawing tool, because journey `09-trajectory-manifold-by-hand` already builds trajectory distances in its plan 11, "distances with definitions". The real 2D and 3D shapes datasets need training, so they belong in Phase 2's own plans, which do not exist yet. At the chunk carrying Bradley's definition, test whether a higher DCI score implies the separated conditionals Bradley's condition needs; session a0661fcd said it does not obviously do so.

## Do not redo

**Reading the ChatGPT share page.** WebFetch returned only the title and Claude in Chrome stuck on a new tab. Playwright worked, and the text is already saved.

**A merge of session a0661fcd alone.** It adds nothing, because the map holds that walk's state.

**Correcting the plan numbers.** Plans 14, 15, 16 and 31 were already corrected. Master plan row 3.1 keeps its numbers on purpose, because it describes the 2D toy where they are right.

## Open

1. Resume the walk at the map's current chunk (abstract chunks 2 and 7, then the skippable list), after updating the stale "States" and "Next step" sections and re-sorting under the third reason once the reader confirms it.
2. Propose the object and get it approved.
3. Integrate `routes/inventory.md`, which landed 2026-10-11 and is not yet in the map.
4. The reader has not confirmed the reading of their A/B drawing ("Tell me if I've misread it").
5. "TCOE" was read as total correlation and never confirmed.
6. Drain labels: NIH CXR14's own 14 labels have none. Session 69356b18 named Oakden-Rayner's group's annotations and CANDID-PTX chest-tube labels as leads. The earlier merge in this folder ([composing medical generators](2026-10-07-composing-medical-generators.md)) already records a NEATX join with drain labels: 3,543 pneumothorax images, 1,637 with a drain and 1,906 without, and no drain-labelled negatives. Check that before searching further.
7. Where the permanent R3 clone should live, which decides whether CLEVR runs from poe_repair_min's cluster path or a new checkout. How a second repo launches on mscluster is not written in `environment/`. Ask, do not infer.
8. Whether to delete `.playwright-mcp/share.txt` once a clean copy is in the journey's `sources/`.

The steps from session 69356b18's sequence, none run yet, each with what it waits on. `/drip-repo` on the reference clone runs alongside the read, starting with `notebooks/simple_distributions.ipynb` on CPU and then CLEVR with `inf_sample.py --sampler MALA`. `/drip-execute-plan --from-read <READ_MAP.md> --depth objective` opens as a thread off the read when a chunk needs the maths. `/drip-idea` on the three conditions starts once the read reaches section 5. `/drip-execute-plan --reproduce 2302.11552` waits until the drip-repo walk has the code working. `--request` for 2D experiments (coverage swept at 0, 1, 10 and 50% of B's points inside A∩B, one curve per sampler, plus the HMC chimera prediction) waits for both the drip-idea's compiled experiment list and the reproduction. Last comes `/merge-sessions --retro` on the experiment session, which ends with the `/drip-retro <retro file>` line to paste next.

## Requests so far

1. "/drip-read https://arxiv.org/pdf/2302.11552 where the code is https://colab.research.google.com/github/yilundu/reduce_reuse_recycle/blob/main/notebooks/image_tapestry.ipynb, https://github.com/yilundu/reduce_reuse_recycle": **partly done** (setup, read-ahead, lean sort and abstract chunks 5 and 6 walked; the rest of the walk remains).
2. "integrate": **done**.
3. "1. fix-plans 2. Will this help me understand product of experts more ? If I understood this at a deeper level could I understand why product of experts creates hybrids instead of co-occurence. Does it align with this paper https://arxiv.org/abs/2502.04549 or should I start a new /drip-read for https://arxiv.org/abs/2502.04549 and later link it": **done** (four plans fixed; R3 covers only the sampler half; Bradley gets its own walk).
4. "what about https://arxiv.org/abs/2412.17762, https://arxiv.org/abs/2506.08894, https://arxiv.org/html/2509.25940v1 - should they be separate calls of /drip-read as well. Which should I start with": **done**.
5. "In @"Composability Gap - PhD Proposal (v2).pdf" , to extend or integrate with the drip-read for Bradley, what about also having the idea of trying to see whether the ideas in project/phase 1 and 2 cannot be infused with some of the examples to test the ideas directly and either prove or disprove by naturally following the learning journey to understand the paper ? In other words, in the middle could we try the euclid distance over time, can we try the mds, can we look into the VAE and independence thing, should we use the shapes 2D, 3D dataset ? what about exploring riemannian geometry to see if concepts stay on the manifold ? what if we tried beta-VAE, factorVAE, looked at TCOE and DCI in light of composition ? Is that helpful or a separate journey ?": **done** (split recorded under Decisions).
6. "let's get back to the original request, and its response and controls": **done**.
7. "next": **done** (claim index).
8. "are you done to where we can walk ? there is into more integration, reading ahead ?": **not done** (interrupted; answer is yes, plus the inventory route to integrate).
9. "Can you read what is happening here https://chatgpt.com/share/6acc400b-8ba0-83ea-b40d-7fa4557590f1 ?": **done**.
10. The intuition-and-sequence brief, quoted in full in the session 69356b18 transcript, opening "Let's consider also this paper https://arxiv.org/pdf/2302.11552. The kinds of experiments that they perform and their findings. Their 2D data and CLEVR dataset experiments. I have drawn up, in the last attached image, my interpretation..." and asking (1) to have the clinical composition intuition corrected and (2) for a sequence of skill prompts to reproduce the paper, learn the maths at the right time, and design intuition-led experiments: **partly done** (five corrections and nine prompts written; the paper's 2D and CLEVR findings not summarised in detail; no prompt run).

## Preferences

**One piece per message**, with the position block and the `next`/`also` control lines on every reply. After a side question, steer back to the walk and restate the controls.

**Learn the maths only when a chunk needs it.** ULA, MALA, HMC and MCMC are welcome, and Fokker-Planck stays out unless needed.

**Test the proposal's ideas inside the walks** wherever the toy can carry them, and design experiments from the reader's own intuition.

**Slips and code differences arrive in the round that meets them**, never as an upfront list.

## Overruled

**Walk position.** Session 69356b18 said 0 of 165 chunks walked, which it read from the map's "States" row. The map's claim list and timings say chunks 5 and 6 were walked lean on 2026-10-09 and the current chunk is 2 and 7. The claim list wins, because it is the more specific record and the "States" row was not updated in that round.

**Check count.** Session 69356b18 listed the checks folder as "18 NumPy checks (ULA, MALA, HMC, annealed MCMC)", and session a0661fcd reported 35 scripts run on CPU at the read-ahead's end. The two disagree. The disk holds 38 `.py` and 36 `.out` files, so the disk count is used, and it counts files rather than checks known to have run.

**Reverse-diffusion spread.** Session a0661fcd gave 0.21 against 0.31 and session 69356b18 gave 0.2102 against 0.3077. These are the same variances at different precision, so the four-decimal values are used.

---

## Validation log

38 claims checked by exact-string grep against both transcripts and the chat file, plus disk checks on the map, its routes, the read-ahead files, the checks folder, the playground clone (HEAD `81177d9` confirmed), journey plans 10 to 36 and the clinical plan file (49, 131 and 0.37 confirmed at line 44).

**Replaced:** "18 NumPy checks" (session 69356b18's figure, which disagrees with session a0661fcd's 35) by the disk count (see Overruled).

**Corrected:** walk position (see Overruled); the playground venv, which is present at `.venv/` (birth time 2026-10-02 15:46). The map's "Next step" section and "States" row are flagged as stale rather than trusted.

**Kept as stated, not rerun:** the 0.2102 and 0.3076 variances (from the journey's check outputs); the 28 code/paper differences (from `routes/code-vs-paper.md`, not reread here); Bradley's role as the other half of the hybrid question (from memory notes, never reread).

**Not checked:** cluster paths.

## Codex review

Codex review: done (job `20261012-050605-db69af`, returned 2026-10-12 05:07 SAST). Its response file went when the stage folder in `/tmp` was cleaned, so the findings were recovered from Codex's own session log (`~/.codex/sessions/2026/10/12/rollout-2026-10-12T05-06-06-01a12e13-1a32-7541-a687-a772d5dc65fe.jsonl`). 11 findings: 9 applied, two of them in part, and 2 dropped. Each was checked against both transcripts and the disk.

**Applied:** the "18 NumPy checks" string is in session 69356b18 (Overruled and Validation log); the chat's last turn is cut off (the chat section); the venv is present and the job state is dated (State of play); the object, third sort reason and prompt steps are marked as proposals (Decisions in force); the steps' dependencies and the `/drip-retro` handoff (Open); the reproduction targets and bars (Decisions in force); the conditional form's p(x) term (Support); MDS, the shapes datasets and the DCI question (Decisions in force).

**Applied in part:** the chat file's 60,283-byte size stays, since the disk confirms it. The validation-log finding held only for the "18 NumPy checks" line; the merge-time disk checks it could not see were rerun and hold.

**Dropped:** the map position, lean-sort reasons and route dates, because the map committed to the learning tree at 2026-10-12 08:27 (`b68a4b6`) carries the current-chunk row and chunks 5 and 6 walked lean, and the routes' file dates match. The walk controls (`split`, `pin`, `restyle A`, `xref`, `map`, `spell-it-out`, `claims`), because the drip-read skill defines them and the resumed walk loads it.
