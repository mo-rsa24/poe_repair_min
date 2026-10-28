/paper-scout Find the current literature for a taxonomy of compositional diffusion models, seeded by five papers, then read it idea by idea with /drip-read --combine --lean and write what it finds into the project's composition terms file.

Context (cold start). Repo: /home-mscluster/mmolefe/PhD/reduce-reuse-recycle. The idea walk is artifacts/ideas/poe-overlap-dial/IDEA_MAP.md; read it and composition-terms.md beside it first. The project asks when product-of-experts (PoE) composition of diffusion models produces samples that are both A and B, on 2D toys first and chest X-rays later (pneumothorax and chest drain). The user proposes that "composition" names several different things and wants to define them:

- co-occurrence: two concepts fully and distinctly present in separate regions (cat beside dog, drain beside pneumothorax); a product over disjoint regions plus a layout
- intersection: one thing satisfying both on the same pixels (square inside circle); a product over the same variables, which is what PoE computes
- interaction: the concepts change each other (rain + dog = wet dog; tuberculosis + silicosis)
- contention: the failure where both experts claim the same pixels and produce a chimera

Settled so far: a density expert learns its training set's density and labels matter only where something reads them (claim 1). Proposed, not settled (claim 7): interaction is not a third operator, because the interaction term log p(A,B|x) - log p(A|x) - log p(B|x) is zero when labels are fixed functions of the image; the kinds reduce to two operators (product over a partition, product over the same variables) plus two data conditions (did each expert see its object in more than one place; did any expert see A as it looks under B). Worked 2D evidence is in what-an-expert-learns-and-kinds-of-composition.ipynb.

The gap: there is no accepted taxonomy of compositional generation, and the project has not checked how recent work names or splits these cases.

Step 1, /paper-scout. Seed papers (fetch each abstract; do not assume their titles):
  https://arxiv.org/abs/2509.25940
  https://arxiv.org/abs/2307.14012
  https://arxiv.org/abs/2607.11758
  https://arxiv.org/abs/2605.07078
  https://arxiv.org/abs/2506.08894
Search outward from them for the latest work (2024 to 2026 weighted) on: compositional diffusion (PoE, energy-based composition, Composable Diffusion, Reduce Reuse Recycle and its successors), compositional text-to-image benchmarks and their category schemes (T2I-CompBench and others), attribute binding and attribute leakage, multi-object and region-restricted composition (MultiDiffusion and similar), superposition of diffusion models, and any paper that explicitly proposes a taxonomy of composition. Re-rank on whether a paper defines or separates kinds of composition, or explains when a product fails (sampler, support, contention). Shortlist about 10 to 15.

Step 2, file the chosen papers on the paper shelf so /drip-read --combine can find them (paper-cadence takes in unfiled PDFs).

Step 3, /drip-read --combine --lean, one idea per walk with two to four sources each, in this order:
  (a) what each source means by "composition" and how it splits it into kinds
  (b) co-occurrence and contention: region-restricted products, attribute leakage, chimeras
  (c) intersection: PoE, logical AND, why summed scores under reverse diffusion miss the product
  (d) interaction and attribute binding: whether any source treats it as needing more than a product
Use `differences` at the end of each walk.

Write-back. Add what each walk establishes to artifacts/ideas/poe-overlap-dial/composition-terms.md: for each kind, the field's names with citations and a confidence flag, where sources agree or disagree with the project's definition, and any kind the project has not named. Put literature-backed additions under each kind's entry, and any new kind as its own entry marked [field]. Do not rewrite the entries marked [yours]; propose changes to them in a "Proposed changes from the literature" section at the end. Do not edit IDEA_MAP.md.

Finish line: walks (a) to (d) are done and composition-terms.md carries the citations.

Return: write artifacts/ideas/poe-overlap-dial/routes/01-composition-taxonomy-literature.return.md with (1) the shortlist with arXiv ids, (2) per kind, whether the literature supports, splits or contradicts the project's definition, (3) whether any source treats interaction as needing a term beyond the product (this decides claim 7), and (4) the proposed changes to the [yours] entries. Then say `integrate` in the idea walk.
