I want to fix and improve two interactive scenes that teach the same material, working with you one change at a time while my idea walk carries on in another session. Start with progressive reveal and dimming. Then we keep improving them together.

## What the scenes teach

Repo: /home-mscluster/mmolefe/PhD/reduce-reuse-recycle. Both scenes serve claim 2 of the idea map at artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md: a product of two diffusion experts (composition by adding their scores, p_A·p_B) can only put samples where both experts already give probability, and there are three ways round that, remembered as "borrow it, build it, or grade with it":
- borrow it (2a): support kept from pretraining. Two experts fine-tuned separately from one pretrained model start at a top-right share of `0.25` and fall as ε/(2 − 4ε), with `0.5` as an exact ceiling. The strong form is one conditional model with "contains" labels, which keeps `0.989` in the top-right blob for any share r of A-and-B data down to `0.001`, and collapses to a blend at the origin at r = 0.
- build it (2b): experts that model only their own part put `0.9999` (plain product) or `0.989` (background-divided) in the top-right blob with no A-and-B data at all.
- grade with it (2c): the 2D toy becomes an exact target for testing samplers.

The toy is four Gaussian blobs at (±0.5, ±0.5), std `0.15`. A is x1 > 0 and means pneumothorax. B is x2 > 0 and means no chest drain. Top-right is A and B, pneumothorax without a drain, the rare film. Bottom-right is "A, not B", top-left is "B, not A", and bottom-left is "neither". Keep those words. Every number comes from artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/ (README.md, answer_key.json) or the scenes' own key scripts, which reproduce it. Never invent one.

## The two scenes

1. Three Ways a Product Reaches A-and-B, https://claude.ai/artifact/9eFnG2rWEciPsceMwdem4u. 15 presses in five frames. Source: artifacts/scenes/three-ways-a-product-reaches-a-and-b/ (page.src.html, build.py, scene_key.py, check_numbers.mjs, README.md).
2. Where the Product Lands, https://claude.ai/artifact/2A7YDnfKe8EPsS7yFDVcNT. 14 presses. Source: artifacts/scenes/when-a-product-reaches-a-and-b/ (where-the-product-lands.src.html, checks/build.py, checks/scene_key.py, checks/check_page.mjs, README.md).

Both follow ~/.claude/SCENE_LOOK.md and ~/.claude/skills/picture-speak/references/frame-walk-engine.md. Read those, both READMEs and both source pages before proposing anything.

## The first change: one taught concept at a time

Right now a press can put several new things on screen at once. I want this instead:
- Each press reveals exactly one taught concept: one region, one label, one number, or one formula. Nothing new appears until I press next.
- Everything already taught stays visible but dims (lower contrast, grey), so it is context and not competition. Only the concept just revealed is at full colour, with its label and number on it.
- Anything not yet taught is hidden entirely: not dimmed, not outlined, not in a legend.
- The "what each number means" ledger grows by one row per press, and earlier rows dim the same way.
- A prediction still comes before its reveal, and the reveal is its own press.
- Pressing back un-reveals in the same order.

Work on scene 1 first. Before editing, show me the new press list as a table: press number, the one concept it reveals, and what dims. When I say yes, edit page.src.html, rebuild, rerun check_numbers.mjs (every number must still pass), take phone and desktop screenshots of three presses, and republish to the same URL. Then do the same for scene 2, or tell me if the two should become one scene, and which presses from each to keep.

After that, keep going with whatever I ask, one change at a time, showing the diff and a screenshot before each republish.

## Return

When we are done, close with a fenced return package and also write it to artifacts/ideas/when-poe-reaches-a-and-b/routes/04-scene-progressive-reveal.return.md:
  origin: artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md · claim 2 scenes · route 4
  ask: one taught concept per press, with what's already taught dimmed
  deliverable: the republished URLs, the final press list for each, and whether the scenes were merged and which one was kept
  changes for the walk: one line on whether any number or claim on the scenes changed
Then I paste it into the idea walk and say `integrate`.
