# What do other compositional samplers (SuperDiff, Hamiltonian Monte Carlo, Feynman-Kac) do with the same dropped factor?

Section 3.6 of the paper shows plain product-of-experts is the exact joint distribution with one
factor, the co-occurrence term `R_t`, set to one. Other compositional samplers answer that same
dropped factor differently, and comparing them, formally or by running them on this paper's
animal pairs, would locate this paper's fix (injecting the measured `r_t`) inside a family rather
than leaving it as the only alternative to naive score addition. The idea has not been picked up:
it carries no claims yet, only the question and what picking it up would need.

## What is in here

One file, `IDEA_MAP.md`, a pre-walk note: the idea as stated, the constraint carried from the
paper (the comparison has to stay narrowed to plurality on these pairs, not composition in
general), and what a first pass would need (a reading pass on the three technique families with
their exact objects stated, then a decision between running the comparison or writing it as a
related-work paragraph).

## Where it came from and what judged it

**Held since 2026-09-05.** No plan, review file, or draft section references this idea yet,
and the reading register carries no rows for SuperDiff, Hamiltonian Monte Carlo correctors, or
Feynman-Kac corrections. Kept because it is a real candidate to strengthen section 2's
related-work framing once the paper's other open walks (the sampler-versus-model split in scope
06, the merge-supervisor-structure section list) settle.
