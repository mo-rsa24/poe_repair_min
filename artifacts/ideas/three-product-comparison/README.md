# Do cheap product-of-experts probes let the theory predict their outcome in advance?

The idea started as one comparison the supervisors proposed (`p(cat)·p(dog)` against
`p(cat and thing)·p(dog)` against `p(cat and thing)·p(dog and thing)`) and grew into a battery of
six probe families, each varying one axis against the plain "a cat" times "a dog" product, with
the outcome written down before any run using the paper's epsilon-space composition rule. All nine
claims are settled and the walk is compile-ready: no probe has been run yet, but the prediction
table, the family list, and the per-family measurement are all decided.

## What is in here

One file, `IDEA_MAP.md`, the full `/drip-idea` walk: the claim ledger, the prediction table for
six probe families, the per-family measurement table, and the sources it read (the paper's
epsilon-space rule, the compose-rate scorer, and the sampling code that runs two-expert products
today).

## Where it came from and what judged it

Routed from held claim 2 of [the merge-supervisor-structure walk](../merge-supervisor-structure/IDEA_MAP.md),
which parked the supervisors' three-product comparison here for its own `/drip-idea` treatment.
Its verdict returns to that walk through `integrate` once the battery runs. No plan file or
review file references it yet, because nothing in the battery has executed: the next step named
in the map is `compile`, which would order the two-expert families first, then the k=3
generalization needed for the three-expert families.
