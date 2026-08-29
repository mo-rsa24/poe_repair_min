# 💡 What other composition samplers do with the dropped factor

State: parked, not yet walked. This map exists so the idea survives; a `/drip-idea` walk cuts it
into claims when it is picked up.

## The idea, as it stands

Section 3.6 shows that the plain product-of-experts rule is the exact joint with one factor, the
co-occurrence factor R_t, set to one. Other compositional sampling techniques exist that do not
simply add the two scores and reverse-diffuse: SuperDiff, Hamiltonian Monte Carlo samplers over
the product, and Feynman-Kac-style corrections. Each is, in effect, a different answer to the
same dropped factor. Running or at least formally comparing them on this paper's pairs would
illustrate what each technique does to the product, and would locate this paper's correction
(inject the measured r_t) inside a family rather than leaving it as the only alternative to
naive addition.

## Constraints carried from the paper

The interaction term is deliberately not the paper's object, because it is too broad to cover
all instances of compositional failure. The animal-pair setting was chosen to narrow it to one
aspect, plurality, and see what can be uncovered there. Any sampler comparison inherits that
narrowing: the question is what each technique does to plurality on these pairs, not whether it
fixes composition in general.

## What picking this up would need

A reading pass on the three technique families with their exact objects stated (what each one
samples from, and where the co-occurrence factor appears in its formulation), then a decision on
whether the comparison is run (cost: sampler implementations) or written (a related-work
paragraph in section 2 with the formal comparison only). The reading register currently has no
rows for any of them.
