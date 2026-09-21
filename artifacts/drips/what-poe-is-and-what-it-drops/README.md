# Which unstated rules turn Eq (9) into Eq (10), and can the denominator really be thrown away?

A screenshot from the product-of-experts composition literature shows
$p(x \mid c_1,c_2) = p(x\mid c_1)\,p(x\mid c_2) / p(x)$ derived from Bayes and an independence
assumption, with several factors silently deleted along the way. This `/drip --math` walk works
out exactly which rules license each deletion, why $p(x)$ survives while $p(c_1)$ and
$p(c_1,c_2)$ do not, and ends by connecting the same derivation to what this project's correction
term costs when the independence assumption is false. The ladder has five planned pieces; the
first, "what proportional-to actually licenses," is taught through its three children, and the
walk is paused mid-piece at rung 1.2.

## What is in here

Four taught pieces of a five-piece ladder (`01` through `04`), one routing record to a follow-on
scene (`05-route-math-scene.md`), the ladder's own index (`00-INDEX.md`), and the thread's
compressed record (`dispatch.md`). No images.

## The pieces

**Rung 1.1, the free variable** (`01-the-free-variable.md`). A conditional density is a function
of its free variable only: rows of a joint table sum to 1 when conditioning on a column, columns
sum to 1 when conditioning on a row, and a 16-cell toy universe of cat, dog, chimera, and car
images shows that conditioning on both labels at once picks the chimera, at $p(x\mid c_1{=}1,
c_2{=}1) = (0.077, 0.077, 0.846, 0.000)$.

**Rung 1.1.1, density or likelihood** (`02-density-or-likelihood.md`). The same joint cell,
divided by two different totals, gives two different numbers 3.6 times apart depending on which
variable is held free: $p(x_3\mid c_1{=}1) = 0.24$ against $p(c_1{=}1\mid x_3) = 0.857$ from the
same 0.12 joint value.

**Rung 1.1.2, x continuous** (`03-x-continuous.md`). Densities can exceed 1 (a Gaussian with
$\sigma=0.1$ peaks at 3.989) while still integrating to 1, and in 786,432 dimensions the true
normalizing constant is astronomically large ($\log Z \approx 722{,}684$ nats), which is why the
derivation is forced to work with proportionality rather than exact densities. A Riemann-sum
check across three bell widths confirms the area stays 1 in every case.

**Rung 1.1.3, the other direction** (`04-the-other-direction.md`). The same joint table read as a
generator ($p(x\mid c)$, normalize down a row) against as a classifier ($p(c\mid x)$, normalize
down a column) ranks images differently: a whisker close-up with $p(x_5)=0.001$ reads as the most
certain cat to the classifier (1.000) but as almost never what a generator produces when asked
for a cat (0.002). The thing separating the two readings is exactly $p(x)$, the term that
survives in the boxed equation.

**Rung 1.2, what proportional-to lets you delete** (recorded in `dispatch.md`, not yet filed to
its own numbered piece). Dividing a distribution's row by $p(x)$ is shown, on the toy table, not
to be a valid rescaling: it changes the chimera's share from 0.24 to 0.434 after renormalizing,
so $p(x)$ cannot be deleted the way $p(c_1)$ can, and a named trap is that $p(c_1\mid x)$ is not
deletable either, despite $c_1$ sitting on the equation's left.

**What the walk has not reached.** Pieces 2 through 5, conditional probability and Bayes,
conditional independence as an assumption, the line-by-line derivation of Eq (9) to Eq (10), and
what the assumption costs in density and score form, are unwritten. The last of these is the
piece that would connect directly to this project's correction term; a separate note,
`artifacts/notes/interaction-term-as-pmi-gradient/`, has already worked out that connection
independently, ahead of this walk reaching it.

**Held since 2026-09-05.** No plan, review file, or context entry in this repository points back
at this walk by path. It is kept because it is a working derivation of the exact independence
assumption the paper's argument rests on, reusable when section 5 or its background is drafted,
even though nothing has claimed it yet.
