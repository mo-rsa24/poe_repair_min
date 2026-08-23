> Verbatim capture of the chat. Position and control lines are kept as delivered.

```
Eq (9) -> Eq (10) > what ∝ actually licenses > the free variable
depth 1 · rung 1 of 3 · 0 visited, 2 left here
```

## Rung 1: the bar splits the symbols into one free variable and some fixed labels

Read $p(x \mid c_1, c_2)$ out loud as a machine: it takes an image $x$ and returns a number. The $c_1, c_2$ are not inputs to that machine. They are labels saying *which* machine you picked up.

**The consequence: the numbers sum to 1 when you sweep the free variable $x$, and they do not sum to anything in particular when you sweep a conditioning variable.**

$$\sum_x p(x \mid c_1, c_2) = 1 \qquad \text{but} \qquad \sum_{c_1} p(x \mid c_1) \neq 1 \text{ in general.}$$

That asymmetry is the entire licence for everything the screenshot does later, so it is worth seeing on numbers before anything else.

### Worked example 1: read a conditional off a joint table

Universe of four images:

| | $x_1$: cat photo | $x_2$: dog photo | $x_3$: chimera | $x_4$: car |
|---|---|---|---|---|

One binary label, $c_1 = 1$ meaning "a cat is in the image". The joint probability $p(x, c_1)$, meaning the probability that both things are true at once:

| $p(x, c_1)$ | $x_1$ | $x_2$ | $x_3$ | $x_4$ | row total |
|---|---|---|---|---|---|
| $c_1 = 1$ | 0.35 | 0.01 | 0.12 | 0.02 | **0.50** |
| $c_1 = 0$ | 0.01 | 0.32 | 0.02 | 0.15 | **0.50** |

All eight cells sum to $1.00$.

Now condition on $c_1 = 1$. That means: take the top row and nothing else, then rescale it so it sums to 1.

$$p(x \mid c_1 = 1) = \frac{p(x, c_1=1)}{0.50}$$

Term by term:

$$p(x_1 \mid c_1{=}1) = \frac{0.35}{0.50} = 0.70, \quad p(x_2 \mid c_1{=}1) = \frac{0.01}{0.50} = 0.02,$$

$$p(x_3 \mid c_1{=}1) = \frac{0.12}{0.50} = 0.24, \quad p(x_4 \mid c_1{=}1) = \frac{0.02}{0.50} = 0.04.$$

Check: $0.70 + 0.02 + 0.24 + 0.04 = 1.00$. ✓

The conditioning value picked the row. The division by the row total is what made it a probability distribution again.

### Worked example 2: the sum that does *not* come out to 1

Same table, other row: $p(x \mid c_1 = 0) = (0.02,\ 0.64,\ 0.04,\ 0.30)$, which also sums to $1.00$.

Now sweep the conditioning slot instead of the free slot. Fix $x = x_1$ (the cat photo) and add across both machines:

$$p(x_1 \mid c_1{=}1) + p(x_1 \mid c_1{=}0) = 0.70 + 0.02 = 0.72.$$

Not 1, and there is no rule saying it should be. Try $x_2$: $0.02 + 0.64 = 0.66$. Try $x_3$: $0.24 + 0.04 = 0.28$. Three different totals, none of them meaningful.

$$\boxed{\text{Only the free variable is normalized. The conditioning slot is a label, not a probability slot.}}$$

This is why, later on, a factor like $p(c_1)$ can be deleted and a factor like $p(x)$ cannot. One of them changes as you sweep the normalized variable and one of them does not.

---

```
next: what ∝ lets you delete, and why the deleted factor is recoverable for free
expand: 3 inside (why "free variable" has to be stated out loud, what changes when x is
        continuous instead of four images, reading one joint table in both directions)
example: this same table with a second label c2 ("a dog is in the image") added, so you
        can see p(x | c1, c2) pick a cell instead of a row
also: skip, map
```

---

```
Eq (9) -> Eq (10) > what ∝ actually licenses > the free variable > example
depth 1 · rung 1 of 3 · 0 visited, 2 left here
```

## Example: add the second label, and the conditional picks a cell instead of a row

Second binary label: $c_2 = 1$ meaning "a dog is in the image". Now the joint has three slots, $p(x, c_1, c_2)$, so 16 cells. Same four images, same universe.

| $p(x, c_1, c_2)$ | $x_1$: cat photo | $x_2$: dog photo | $x_3$: chimera | $x_4$: car | row total |
|---|---|---|---|---|---|
| $c_1{=}1, c_2{=}1$ | 0.01 | 0.01 | 0.11 | 0.00 | **0.13** |
| $c_1{=}1, c_2{=}0$ | 0.34 | 0.00 | 0.01 | 0.02 | **0.37** |
| $c_1{=}0, c_2{=}1$ | 0.00 | 0.31 | 0.02 | 0.01 | **0.34** |
| $c_1{=}0, c_2{=}0$ | 0.01 | 0.01 | 0.00 | 0.14 | **0.16** |

All sixteen cells sum to $0.13 + 0.37 + 0.34 + 0.16 = 1.00$. ✓

**It is the same table as before, split finer.** Add the two $c_2$ rows that share a $c_1$ value and you get rung 1's table back. For $c_1 = 1$:

$$0.01 + 0.34 = 0.35,\quad 0.01 + 0.00 = 0.01,\quad 0.11 + 0.01 = 0.12,\quad 0.00 + 0.02 = 0.02,$$

which is $(0.35, 0.01, 0.12, 0.02)$, the top row of rung 1's table exactly. Adding away a slot you no longer care about is called marginalizing, and it is the only operation that has happened here.

### Now condition on both

$p(x \mid c_1 = 1, c_2 = 1)$ means: take the one row where both labels hold, and rescale it to sum to 1. That row totals $0.13$.

$$p(x_1 \mid c_1{=}1, c_2{=}1) = \frac{0.01}{0.13} = 0.077, \qquad p(x_2 \mid c_1{=}1, c_2{=}1) = \frac{0.01}{0.13} = 0.077,$$

$$p(x_3 \mid c_1{=}1, c_2{=}1) = \frac{0.11}{0.13} = 0.846, \qquad p(x_4 \mid c_1{=}1, c_2{=}1) = \frac{0.00}{0.13} = 0.000.$$

Check: $0.077 + 0.077 + 0.846 + 0.000 = 1.000$. ✓

Two labels, one row. Three labels would be one row of a bigger table. **The number of conditioning slots never changes what gets normalized: it only changes how narrow the selected row is.** The free variable is still $x$, and the sum over $x$ is still exactly 1.

Read the answer: asked for an image containing both a cat and a dog, this distribution puts $84.6\%$ of its mass on the chimera. That is the joint doing what your research says the product of experts fails to do, and it is here because the table was built with the interaction between the two labels intact. Piece 5 is where that gets taken away.

### The rung-1 claim, re-checked on the bigger table

Fix the chimera, $x = x_3$, and sweep the conditioning slots across all four machines:

$$\underbrace{0.846}_{(1,1)} + \underbrace{\tfrac{0.01}{0.37} = 0.027}_{(1,0)} + \underbrace{\tfrac{0.02}{0.34} = 0.059}_{(0,1)} + \underbrace{\tfrac{0.00}{0.16} = 0.000}_{(0,0)} = 0.932.$$

Still not 1, still no reason it should be. Four conditioning slots instead of one changed nothing about that.

### Two knobs

**Change the chimera's entry in the $(1,1)$ row.** Drop $0.11$ to $0.02$ and that row totals $0.04$; then $p(x_3 \mid c_1{=}1, c_2{=}1) = 0.02/0.04 = 0.500$ and $p(x_1 \mid \cdot) = 0.01/0.04 = 0.250$. The argmax survives but the confidence halves. This single cell is the "does both-labels-present actually mean one blended object" knob.

**Condition on $(c_1{=}1, c_2{=}0)$ instead**, meaning a cat and no dog. Row total $0.37$:

$$p(x \mid c_1{=}1, c_2{=}0) = \left(\tfrac{0.34}{0.37},\ \tfrac{0.00}{0.37},\ \tfrac{0.01}{0.37},\ \tfrac{0.02}{0.37}\right) = (0.919,\ 0.000,\ 0.027,\ 0.054),$$

summing to $1.000$. ✓ Same table, different row, mass moves to the cat photo. Nothing about the machinery changed.

One thing worth keeping: this table has every quantity the boxed equation needs. Its column sums are $p(x) = (0.36, 0.33, 0.14, 0.17)$, and $p(x \mid c_1)$ and $p(x \mid c_2)$ both read straight off it. So at piece 4 we can compute both sides of Eq (10) numerically and see exactly how far apart they are.

---

```
next: what ∝ lets you delete, and why the deleted factor is recoverable for free
expand: 3 inside (why "free variable" has to be stated out loud, what changes when x is
        continuous instead of four images, reading one joint table in both directions)
also: skip, map
```
