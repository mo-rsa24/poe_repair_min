/demonstrate Show me, as pictures I can look at, what a "hidden B" is and what the two conditioning rules "contains A" and "exactly A" each train on.

Context. Repo: /home-mscluster/mmolefe/PhD/reduce-reuse-recycle. I am walking an idea about product-of-experts composition of diffusion models, in artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md. Concept A is "a red, clean circle". Concept B is "blue, scattered dots". Expert A trains on a set labelled "A" that quietly contains images that also show B, unlabelled; set B likewise hides some A. The walk has marked this claim wrong as stated: "an expert trained on A-labelled data learns the concept A". A density expert learns its training set's density, and the label only matters where a conditioning scheme or a classifier reads it.

Words this uses:
- Hidden B: an image that shows B but whose label does not say so.
- "Contains A": every image whose label set includes A trains condition A.
- "Exactly A": only images labelled A alone train condition A; an image labelled {A, B} trains a separate condition.

What to build (Python, NumPy, matplotlib only; throwaway code under scripts/ or beside the figures):
1. Procedural 64×64 RGB images, so the ground truth is exact. Draw A as a filled red circle and B as 6 to 12 small blue dots, at random positions and sizes, on a plain background. Record for every image which concepts are drawn and each concept's bounding box from the drawing parameters (never a detector).
2. Three label cases, all present: clean (label matches the drawing), hidden (image shows A and B, label says only "A", or only "B"), labelled-both (label is {A, B}). Overlap dial f = the share of each set that also shows the other concept, at f in {0, 0.05, 0.2, 0.5, 1.0}, plus a share of the overlap that is labelled-both rather than hidden.
3. Figure 1, "what the label says against what the image holds": a grid of about 8 images from set B and 8 from set A. Each panel's title is the stored label, exactly as the dataset holds it ("label: B"). A concept drawn but missing from the label gets a dashed bounding box captioned "A, not in the label". Concepts in the label get a solid thin box.
4. Figure 2, "the piles each rule trains on": one multi-label pool (clean, hidden and labelled-both images mixed), sorted into the images that train condition A under "contains A" and under "exactly A", and into the unconditional pile. Show a few thumbnails per pile with their stored labels.
5. The number beside the pictures: a table and a terminal report, per f, of the share of condition A's training images that actually show B, under each rule, split into hidden and labelled-both. The finding I expect to see and want checked: "exactly A" removes only the labelled-both images, never the hidden ones, because a hidden-B image is labelled "A" alone. Also report, among images that show B, the share labelled B (what a B classifier would be taught).
6. Unit tests showing the checks can fail: a deliberately mislabelled box fails the box-from-drawing check; a pile computed with the wrong rule fails the count check.

Where it lands: artifacts/results/what-the-label-hides-from-the-expert/ (figures, the count table as JSON with the dial value f in every row, a README.md card saying how each figure was made).

Return. Close with a fenced return package, and also write it to artifacts/ideas/when-poe-reaches-a-and-b/routes/01-hidden-labels-pictured.return.md:
  origin: artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md · claim 1 (and claim 3's conditioning rules) · route 1
  ask: pictures of hidden labels and of what each conditioning rule trains on
  deliverable: the figure paths and the count table path
  changes for the walk: one line on whether "exactly A removes only labelled overlap" held, with the share at f = 0.2
Then I paste it into the walk and say `integrate`.
