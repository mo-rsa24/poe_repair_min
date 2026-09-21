# Run 5: what instrument can find a training target that shows the wrong thing (tasks 7.2, 7.1)

Claim 7 says cells whose joint-prompt render does not itself show both concepts are teaching the
adapter the wrong correction. The claim needs an instrument that can find those cells across 790
cached targets. This run validated three candidate reads against ground truth, and only the third
survives.

Script `scripts/showcase/target_quality_scan.py`, in-session on this node's free GPU, 2026-09-07.
Output `/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/`.

## The ground truth, by eye

The eight `mono.png` targets of held-out `a_cat__x__a_dog`, read directly rather than taken from
run 2, whose render notes describe the adapter's corrected renders and not these targets.

| seed | what the target shows | verdict |
|---|---|---|
| 9 | tabby cat and white labrador | good |
| 10 | two dogs | bad |
| 11 | white cat and grey dog, illustration | good |
| 12 | white cat and brown dog | good |
| 13 | french bulldog and white poodle | bad |
| 14 | two cats | bad |
| 15 | border collie and kitten | good |
| 16 | jack russell and ginger cat | good |

Three of eight targets on this pair show one concept twice. The premise of claim 7 holds, and the
three cells are seeds 10, 13 and 14.

## The three reads

| Read | What it does | Score on the three bad targets |
|---|---|---|
| generic instance count, the shipped compose scorer | counts distinct boxes for the query "animal" | 0 of 3. Returns 2 on all eight, including two cats and two dogs |
| per-name detection | GroundingDINO once per animal word, a box required for each | 0 of 3, and inverted. The two-cat target of seed 14 returns "a dog" at 0.787, above the 0.615 the real dog in seed 9 gets. Good seed 16 has box counts (2, 3), identical to bad seeds 10 and 13 |
| forced choice | detect boxes generically, crop each, make CLIP choose between the pair's two names; good when both names win a box | 7 of 8 overall, 3 of 3 on the bad targets |

The per-name failure has one cause worth keeping: an open-vocabulary detector will put a confident
box for any animal word on any animal. Detection was never the missing piece. Discriminating
between two named candidates is, and a forced choice on a crop is that question asked directly.

The forced choice's one error is seed 15, where a border collie is called "a cat" at 0.53, a coin
flip at the decision boundary. The error direction is a false negative: the read calls a good
target bad, so it can only drop usable cells and never keep broken ones. The bar was set here,
against images labelled by eye, before the sweep over 790 targets ran.

## The sweep, and why its numbers are an upper bound

The forced-choice read ran over all 790 cached targets. Restricted to the pool's 11 train and 8
held-out pairs it flags 74 of 93 train targets (80%) and 43 of 73 held-out targets (59%) as showing
one concept twice. Those figures are not a count. Two things inflate them, and both were found by
looking at the pictures.

**False negatives rise on pairs whose two animals are similar.** On cat x dog the read missed one
good target in eight. On lion x tiger it flags six of eight, and the eye says seeds 1 and 3 plainly
show a lion beside a tiger. The read was validated on the easiest discrimination in the pool and
applied to pairs chosen precisely because the two concepts are confusable.

**Some pairs are undecidable, not bad.** On rabbit x hare neither the read nor the eye can say
which animal is which, so a 100% flag rate there means "cannot tell", not "the target is wrong".
The same holds for crow x raven, dolphin x porpoise and turtle x tortoise.

**What the eye does confirm.** On the pairs whose two animals are visually separable, the targets
really are frequently wrong, and the rate is nowhere near a handful.

| Pair | What the targets show | Read against eye |
|---|---|---|
| cheetah x cougar | every seed sampled shows two spotted cats; a cougar has no spots | flags 8 of 8, and the eye agrees on the four sampled |
| horse x zebra | three of the four sampled show two striped animals and no plain horse | flags 8 of 8, eye says 3 of 4 |
| donkey x pony | the four sampled show two donkeys | flags 8 of 8, eye agrees |
| cat x dog, held out | seeds 10 and 13 show two dogs, seed 14 two cats | flags those three, plus one good target |
| lion x tiger | seeds 1, 3, 4 and 6 show a lion beside a tiger; seed 5 is a single blended face | flags 6 of 8, eye says at most 4 |

**Why this was likely.** The train pool was chosen for blend-prone pairs, which means near-synonyms.
Those are exactly the pairs where the joint prompt also fails, so the pool was selected in a way
that makes its own targets unreliable. Run 3 already put these same pairs at the bottom of the
interaction bound; they are now also the pairs whose targets are worst.

**What settles the count.** The pool is 152 cells, which is nineteen contact sheets of eight. A
direct eye pass over those gives ground truth rather than a proxy, at no GPU cost, and it is the
only way to get a number worth acting on for pairs where the forced choice cannot separate.
