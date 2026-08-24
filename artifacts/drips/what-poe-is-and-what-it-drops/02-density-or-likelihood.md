> Verbatim capture of the chat. Position and control lines are kept as delivered.

```
Eq (9) -> Eq (10) > what ∝ actually licenses > the free variable > density or likelihood
depth 2 · child 1 of 3 · 0 visited, 2 left here
```

Three children under "the free variable", traced from what the screenshot's lines actually do:

1. **Density or likelihood: the same symbols read two ways.** Eq (9) has $p(c_1 \mid x)$ and Eq (10) has $p(x \mid c_1)$, so the derivation swaps which slot is free halfway through and never says so.
2. **What changes when $x$ is a continuous image, not one of four.** Sums become integrals, and the thing $\propto$ hides stops being a number you could compute.
3. **Reading the joint table in the other direction.** Where $p(c \mid x)$ comes from at all. This one is the doorway into piece 2, so it hands straight over rather than teaching Bayes twice.

Opening the first.

---

## Density or likelihood: the free variable decides which object you are holding

Back to rung 1's table, all eight cells:

| $p(x, c_1)$ | $x_1$: cat photo | $x_2$: dog photo | $x_3$: chimera | $x_4$: car | row total |
|---|---|---|---|---|---|
| $c_1 = 1$ | 0.35 | 0.01 | 0.12 | 0.02 | **0.50** |
| $c_1 = 0$ | 0.01 | 0.32 | 0.02 | 0.15 | **0.50** |

Rung 1 said the sum over $x$ is 1 and the sum over $c_1$ is not. That was a fact about arithmetic. Here is what it means.

**When $x$ is free you are holding a probability distribution; when $c_1$ is free you are holding a likelihood, and the notation $p(x \mid c_1)$ is identical in both cases.**

A distribution answers "how do the possibilities share out". A likelihood answers "how well does each setting explain the thing I saw". They are different questions and only the first one has to sum to 1.

### Worked example 1: read the same numbers along both directions

**Along a row, $x$ free, $c_1$ fixed at 1.** Divide by the row total $0.50$:

$$p(x \mid c_1{=}1) = (0.70,\ 0.02,\ 0.24,\ 0.04), \qquad \text{sum} = 1.00. \ ✓$$

The question this answers: given a cat is present, which image is it? A cat photo, most likely.

**Along a column, $c_1$ free, $x$ fixed at the chimera $x_3$.** Take the same conditional numbers, but now the two that share $x_3$:

$$p(x_3 \mid c_1{=}1) = 0.24, \qquad p(x_3 \mid c_1{=}0) = 0.04, \qquad \text{sum} = 0.28.$$

The question this answers: which label setting better explains having seen a chimera? Their sum, $0.28$, means nothing at all. Their ratio does:

$$\frac{p(x_3 \mid c_1{=}1)}{p(x_3 \mid c_1{=}0)} = \frac{0.24}{0.04} = 6.$$

"A cat is present" explains the chimera six times better than "no cat is present". That ratio is the entire content of a likelihood, and it survives any rescaling of the column, which is the first hint of why $\propto$ is going to be allowed to throw things away.

### Worked example 2: one cell, two divisors, two different answers

This is the one that bites. Take a single number out of the table: the joint $p(x_3, c_1{=}1) = 0.12$, the probability that the image is the chimera **and** a cat is present.

Divide it two ways.

**By the row total**, $p(c_1{=}1) = 0.50$:

$$p(x_3 \mid c_1{=}1) = \frac{0.12}{0.50} = 0.24.$$

**By the column total**, $p(x_3) = 0.14$ (from the four-cell column sum $0.12 + 0.02$):

$$p(c_1{=}1 \mid x_3) = \frac{0.12}{0.14} = 0.857.$$

$0.24$ and $0.857$. Same cell, same $0.12$ on top. The only difference is which total went underneath, and the notation records that difference by which symbol sits on which side of the bar. Nothing else in the expression changes.

$$\boxed{p(x_3 \mid c_1{=}1) = 0.24 \neq 0.857 = p(c_1{=}1 \mid x_3)}$$

Two things follow, and both are load-bearing for the screenshot.

**The bar is not symmetric and reading it as "and" destroys the derivation.** Eq (9) is built from $p(c_i \mid x)$ and Eq (10) from $p(x \mid c_i)$, so the whole second half of that screenshot is the work of turning $0.857$-shaped quantities into $0.24$-shaped ones.

**The divisor is the entire difference between them.** Your question was whether the denominator can be thrown away. Here are two numbers that differ by nothing except their denominator, and they differ by a factor of 3.6. The rule that lets you delete a denominator is therefore not "denominators do not matter", it is something much narrower, and that narrower thing is rung 2.

---

```
next: what changes when x is a continuous image rather than one of four choices
expand: nothing below this, it bottoms out at "a function has named arguments", which is
        the school line. `example` or `parent` instead.
example: the same two-divisor split done on x1 (the cat photo) where the two numbers come
        out 0.70 and 0.97, so the gap is small and the point is easier to miss
also: parent, top, map
```
