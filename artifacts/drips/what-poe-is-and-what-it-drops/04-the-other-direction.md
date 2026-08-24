> Verbatim capture of the chat. Position and control lines are kept as delivered.

```
Eq (9) -> Eq (10) > what ∝ actually licenses > the free variable > the other direction
depth 2 · child 3 of 3 · 2 visited, 0 left here
```

## Reading the joint the other way, which is where $p(c \mid x)$ comes from

Short one, as promised. Child 1 divided one cell by two different totals. This says what those two answers are called and why the screenshot needs both.

The table again, with both totals shown:

| $p(x, c_1)$ | $x_1$: cat photo | $x_2$: dog photo | $x_3$: chimera | $x_4$: car | row total |
|---|---|---|---|---|---|
| $c_1 = 1$ | 0.35 | 0.01 | 0.12 | 0.02 | 0.50 |
| $c_1 = 0$ | 0.01 | 0.32 | 0.02 | 0.15 | 0.50 |
| **column total** | **0.36** | **0.33** | **0.14** | **0.17** | 1.00 |

**There is only one object here, the joint table, and both conditionals are it normalized along different axes.** Divide by a row total and you get $p(x \mid c_1)$. Divide by a column total and you get $p(c_1 \mid x)$. Nothing else is involved.

Names worth having, because they are the two models you actually build:

- **$p(x \mid c_1)$ is a generator.** Give it a label, it hands back a distribution over images. Normalize along a row.
- **$p(c_1 \mid x)$ is a classifier.** Give it an image, it hands back a distribution over labels. Normalize along a column.

Eq (9) in the screenshot is written entirely in classifiers and Eq (10) entirely in generators. **So that derivation is a recipe for converting classifiers into a generator**, which is why classifier guidance and this equation are the same piece of algebra wearing different clothes.

### Worked example 1: normalize down the columns

Divide each column by its own total:

$$p(c_1{=}1 \mid x_1) = \frac{0.35}{0.36} = 0.972, \qquad p(c_1{=}1 \mid x_2) = \frac{0.01}{0.33} = 0.030,$$

$$p(c_1{=}1 \mid x_3) = \frac{0.12}{0.14} = 0.857, \qquad p(c_1{=}1 \mid x_4) = \frac{0.02}{0.17} = 0.118.$$

And the complements, $p(c_1{=}0 \mid x) = (0.028,\ 0.970,\ 0.143,\ 0.882)$.

Check the free slot: for the chimera, $0.857 + 0.143 = 1.000$. ✓ Rung 1 said sweeping $c_1$ does not sum to 1, and here it does, because $c_1$ has moved to the free side of the bar. Same rule, and which side of the bar a symbol sits on is the only thing that decides it.

### Worked example 2: the two readings disagree about which image is "most cat"

Add a fifth image $x_5$, an extreme close-up of a single cat whisker. It is very rare, $p(x_5) = 0.001$, and every instance of it contains a cat, so the whole $0.001$ sits in the $c_1 = 1$ row.

**Classifier reading**, divide by the column total $0.001$:

$$p(c_1{=}1 \mid x_5) = \frac{0.001}{0.001} = 1.000.$$

The most certain cat in the universe, beating the cat photo's $0.972$.

**Generator reading**, divide by the row total $0.50$:

$$p(x_5 \mid c_1{=}1) = \frac{0.001}{0.50} = 0.002.$$

Almost never what you get when you ask for a cat, against the cat photo's $0.70$, a factor of 350 behind.

$$\boxed{\text{classifier ranks } x_5 \text{ first, generator ranks it fourth. Same table.}}$$

That flip is the thing that goes wrong in practice when a classifier is used to steer a generator, and the quantity standing between the two readings is the column total, $p(x)$. Which is precisely the $p(x)$ left sitting in the denominator of the screenshot's boxed result. That equation is not decoration: it is the correction that turns two classifier-shaped opinions back into a generator.

---

**"The free variable" is done, all 3 children visited.** Back up to piece 1, "what $\propto$ actually licenses", where 2 rungs are still untouched: what $\propto$ lets you delete, and which factors in the screenshot qualify.

```
next: rung 2 of piece 1, what ∝ lets you delete and why the deleted factor costs nothing
      (this is the direct answer to your denominator question, with the narrow rule stated)
map: the tree so far, 3 children closed under rung 1, 4 pieces never opened
also: parent, top
```
