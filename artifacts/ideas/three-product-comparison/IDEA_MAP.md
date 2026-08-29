# 💡 The probe battery: cheap products whose outcomes the theory predicts in advance

Routed from the held claim 2 of [the merge walk's map](../merge-supervisor-structure/IDEA_MAP.md). Its verdict returns there via `integrate`. The walk opened on the supervisors' three-product comparison alone; the idea reshaped in round 2 into a battery of probe products, with the three products as one family among several. Claims 1 to 5 are the original cut and are not renumbered; claims 6 to 9 are appended. All claims are settled; the walk is compile-ready.

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. The symbolic argument survives in epsilon-space | absorbed into 6 | the algebra is the prediction machinery for every probe |
| 2. A wording for "and thing" exists that does not itself paint a second object | absorbed into 8 | wording decided beside its family's measurement |
| 3. The validated scorer can score products (b) and (c) as planned | absorbed into 8 | the class-blind rule's coverage judged per family |
| 4. Product (c) still fails to compose | absorbed into the battery | one family's run, prediction written first |
| 5. The literature has already named or run this | settled: no name found, minting licensed | broad scan plus one scoped search; nearest neighbours are prompt rewriting and in-context image generation, neither is this object; a later /pressure-test could still surface one |
| 6. Every probe's outcome is predictable in advance from the epsilon-space rule | holds | the prediction table; a probe with no writable prediction is cut |
| 7. The probe list cuts into families that each vary one thing | holds as recut | six families, each internally one axis; the supervisors' hybrid marked mixed, kept as illustration only; families organized by the equation's consequences |
| 8. Each family has a stated measurement | holds as decided | the per-family measurement table below; the presence-context wording carries its own instrument check |
| 9. The existing pipeline runs k-expert products cheaply | holds for two experts, one small code task for three | run_cfg_poe (poe_repair/methods/_sampling.py:133) takes arbitrary prompt embeddings, so every two-expert probe runs today; noise.chunk(3) at line 168 hard-codes two experts, so k=3 needs a generalization to chunk(k+1) with a summed push loop |

Load-bearing: claim 6, and it holds. Compile lands at destination A.

## The idea, as it stands

Generate many cheap probe products at λ=0, each chosen so the epsilon-space composition rule (ε_PoE = Σᵢ wᵢ(ε(cᵢ) − ε(∅)) + ε(∅), iclr2027_conference.tex:99) yields a written prediction before the run. Six families, each varying one axis against the plain product "a cat" × "a dog", each rendering its joint-prompt counterpart as the ceiling. Predictions are recorded before scoring, in the project's pre-registration pattern. The battery would enter the paper as section 1's worked examples, subject to the parent walk's placement decision.

## The prediction table (claim 6's demonstration)

| Probe | In epsilon-space | Predicted outcome | What it tests |
|---|---|---|---|
| "a cat" × "a cat" vs joint "a cat and a cat" | identical pushes = double-weight CFG on one prompt | product: one cat, saturated; joint: two only if the model counts | plurality isolated, no class conflict |
| "a cat and a bird" × "a dog" | one factor carries its own internal binding | cat+bird survives to the extent the joint pair composes alone; dog uncoupled | binding lives inside a factor, never across |
| "a cat on the left" × "a dog in the scene" | pushes localized to different regions, still no cross term | partial rescue plausible (layout is early; the early window composes); full rescue is a finding against sufficiency of r_t | contention vs binding as the failure's content |
| "a cat" × "a dog" × "a bird" | three pushes, contention grows | compose-3 near 0, worse than pairs | failure scales with expert count |
| "a cat" × "a dog" × "beside one another" | relation push tied to neither animal token | little rescue; any rescue means a text-only expert approximates r_t | can a relation expert supply the coupling |
| "a cat" × "a dog" × "a dog" | dog pushed twice | dog-dominant, cat lost; the axis is literally w_dog | overlap = double counting |
| p(cat and thing)·p(dog and thing) | augmented factors, still a sum | fails if binding is the missing piece; composes = training-free baseline | the supervisors' comparison, one family |

## The family and measurement table (claims 7 and 8, settled)

| Family | Products | The one axis | Measurement | Wording note |
|---|---|---|---|---|
| duplication | "a cat" × "a cat", joint "a cat and a cat" | the pair's concepts | scorer as-is; predicted counts 1 vs 2 | none needed |
| presence context | plain, one factor augmented, both augmented | placeholder count 0, 1, 2 | scorer as-is plus the audit protocol on compose-scored cells | placeholder names no animal: "a cat and something else in the scene"; instrument check: each augmented prompt run solo through run_cfg first, and the wording is disqualified if it alone paints a second animal on more than a small fraction of seeds |
| spatial context | plain, one location, two locations | location count 0, 1, 2 | scorer as-is | locations license no extra animal |
| expert count | cat × dog vs cat × dog × bird, joints as ceilings | k = 2 → 3 | compose-3 = n_instances ≥ 3, a threshold read on the recorded field, no new detector work | joint three-animal prompt rendered as the ceiling |
| relation expert | cat × dog × "beside one another", control cat × dog × bird | third push content: object vs relation | scorer as-is | relation phrase names no animal |
| overlap | cat × dog vs cat × dog × dog | w_dog = 1 → 2 | scorer as-is plus audit: two dogs would pass the class-blind counter, so compose-scored cells in this family are eyeballed | none needed |

The supervisors' hybrid "a cat and dog" × "a dog on the left" varies three axes at once and is marked mixed: kept as an illustration, excluded from clean comparisons. Gaps between the presence and spatial ladders are read per family, never across.

## What the words are

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| product of experts on prompts | composable diffusion / conjunction operator | summing conditional noise predictions minus the unconditional (Liu et al. 2022) | confident |
| "a cat" × "a cat" collapsing to one cat | classifier-free guidance reweighting | identical pushes sum to one push at double weight | confident, follows from the rule |
| the missing coupling | the interaction term / r_t | joint-prompt prediction minus the PoE prediction | confident, project term |
| context-augmented experts | no established name; minting licensed | each expert conditioned on its concept plus a placeholder for the other's presence; define at first use | settled by claim 5's search |

## Held claims

| Claim | What is unresolved | What would settle it |
|---|---|---|

## Dead ends

| Claim | The workaround | Why it failed |
|---|---|---|

## Checks outstanding

| Claim | The check | What each outcome means |
|---|---|---|
| 8, presence family | the solo instrument check on the chosen wording | passes: the ladder runs as designed; fails: try the next non-animal wording, and record the failed one here |
| 9 | the k=3 generalization of run_cfg_poe | small task; until done, only two-expert families run |

## Runs

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|

## Sources

| Source | What it gives the idea | Confidence |
|---|---|---|
| paper/iclr/iclr2027_conference.tex lines 96 to 214 | the epsilon-space rule, r_t, the λ-correction | read this session |
| context/world/compose-rate.md | the validated scorer: ≥2 animal instances, class-blind; the 87 to 94% bound; class-specific queries considered and dropped | read this session |
| poe_repair/methods/_sampling.py lines 133 to 186 | run_cfg_poe: arbitrary prompt embeddings, exactly two experts hard-coded via noise.chunk(3) | read this session |
| artifacts/ideas/merge-supervisor-structure/IDEA_MAP.md | the parent walk; the comparison parked as its held claim 2 | read this session |
| Liu et al. 2022, Composable Diffusion (arXiv 2206.01714) | the conjunction rule | confident |
| Feynman-Kac Correctors (arXiv 2503.02819) | samplers for products of diffusions, no joint term | likely, verify |
| CO3 / mode-collision steering (arXiv 2509.25940) | training-free corrective sampling for conjunction failures | likely, verify |
| claim 5's scoped search (2026-08-25) | no established name for the augmented-expert object; nearest neighbours are prompt rewriting, RAG-augmented prompting, in-context image generation | one search; a /pressure-test could still surface a name |

## Next step

`compile`. The load-bearing claim holds, so compile lands at destination A: the plain statement, the claim ledger, the ordered build (two-expert families first, the k=3 generalization, then the three-expert families), and the route out via /frame-hypothesis for the falsification bars, then the parent walk's `integrate`.
