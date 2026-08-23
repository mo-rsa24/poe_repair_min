> Verbatim capture of the chat. Position and control lines are kept as delivered.

```
Eq (9) -> Eq (10) > what ∝ actually licenses > the free variable > x continuous
depth 2 · child 2 of 3 · 1 visited, 1 left here
```

## What changes when $x$ is a real image instead of one of four

Everything so far ran on four images and a table you could add up by hand. Your actual $x$ is a $512 \times 512 \times 3$ tensor, which is $786{,}432$ real numbers. Three things change and one very important thing does not.

### Change 1: probability moves from the point to the region

With four images, $p(x_3) = 0.14$ was a probability: pick an image at random, $14\%$ of the time it is the chimera.

With continuous $x$, the probability of landing on any *exact* tensor is zero. There are uncountably many of them. So $p(x)$ stops reporting a probability and starts reporting a **density**: probability per unit volume. You only get a probability back by integrating over a region.

$$\text{discrete: } \sum_x p(x) = 1 \qquad \longrightarrow \qquad \text{continuous: } \int p(x)\, dx = 1.$$

Every $\sum$ in everything taught so far becomes $\int$. That is the whole translation.

### Change 2: a density can be bigger than 1

This is the part that catches people, because "probability" and "at most 1" are welded together in most people's heads. **A density is not a probability, so it has no ceiling.**

Worked example: a one-dimensional Gaussian, the standard bell curve,

$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).$$

Read term by term: the exponential is the bell shape, peaking at 1 when $x = \mu$ and falling off as you move away; $\sigma$ sets how wide the bell is; and the fraction out front is the normalizer, whatever number makes the area under the whole thing equal 1.

Evaluate at the peak, $x = \mu$, so the exponential is $e^0 = 1$ and only the front factor is left. With $\sigma = 1$:

$$p(\mu) = \frac{1}{1 \times 2.5066} = 0.399.$$

Now squeeze the bell to $\sigma = 0.1$:

$$p(\mu) = \frac{1}{0.1 \times 2.5066} = \frac{1}{0.25066} = 3.989.$$

A value of $3.989$ from a legitimate probability density. Nothing is broken: the bell is now ten times narrower, so it has to be ten times taller to keep its area at 1. The area is still exactly $1.00$. ✓

### Change 3: the normalizer stops being computable

This is the one that matters for your question about denominators.

Any density can be written as some unnormalized shape divided by whatever makes it integrate to 1:

$$p(x) = \frac{f(x)}{Z}, \qquad Z = \int f(x)\, dx.$$

With four images, $Z$ was the row total, $0.50$. You computed it by adding four numbers.

Now take the easy continuous case: an isotropic Gaussian bump in $D$ dimensions, $f(x) = \exp(-\|x - \mu\|^2 / 2\sigma^2)$. Its normalizer has a closed form:

$$Z = (2\pi\sigma^2)^{D/2}.$$

Put your image dimension in. $D = 512 \times 512 \times 3 = 786{,}432$, and $\sigma = 1$:

$$\log Z = \frac{D}{2}\log(2\pi) = 393{,}216 \times 1.8379 = 722{,}684.$$

So $Z = e^{722{,}684}$, a number with roughly $313{,}900$ decimal digits. It overflows a float64 (which dies above about $10^{308}$) by three hundred thousand orders of magnitude.

**And that is the easy case, where a formula exists.** For the distribution your model actually learned over natural images there is no closed form for $Z$ at all, and the integral is over $786{,}432$ dimensions, so no numerical method reaches it either.

$$\boxed{\text{Writing } \propto \text{ instead of } = \text{ is not a convenience in this setting. The } = \text{ version cannot be written down.}}$$

That reframes your original question. It is not "may we throw the denominator away". It is "the derivation was never able to hold the denominator in the first place, so what can still be proved without it?"

### What does not change: any of the algebra in that screenshot

Every step from Eq (9) to Eq (10) is a multiplication or division of functions of $x$. None of them integrates anything. So the entire derivation is identical whether $x$ ranges over four images or over $\mathbb{R}^{786432}$, and you can check it on the four-image table and trust the result.

One thread to leave hanging, because it is where your own work lives: diffusion models never touch $p(x)$, they work with $\nabla_x \log p(x)$. The reason is sitting right here. That intractable $Z$ is a constant with respect to $x$, so taking the log turns it into $-\log Z$, an additive constant, and taking the gradient in $x$ deletes it exactly. Piece 5 does the screenshot's whole result in that form.

---

```
next: reading the joint table in the other direction, where p(c|x) comes from at all
      (this one hands straight into piece 2, so it is short)
expand: 2 inside (why a single image has probability exactly zero yet your sampler still
        produces one, what Z looks like for the diffusion model you actually train)
example: the Gaussian at sigma = 1, 0.1, and 0.01, peak densities 0.399, 3.989, 39.894,
        with the area check done each time so you see it stay at 1.00
also: parent, top, map
```

---

```
Eq (9) -> Eq (10) > what ∝ actually licenses > the free variable > x continuous > example
depth 2 · child 2 of 3 · 1 visited, 1 left here
```

## Example: three bell widths, and the area checked by hand each time

The claim to check is that peak density $3.989$ is not a broken probability. So we compute the area, numerically, on a grid, and watch it come out to 1 all three times.

Setup: Gaussian centred at $\mu = 0$,

$$p(x) = \underbrace{\frac{1}{\sigma\sqrt{2\pi}}}_{\text{front factor}} \cdot \underbrace{\exp\!\left(-\frac{x^2}{2\sigma^2}\right)}_{\text{bell}}.$$

Area on a grid is just heights times spacing added up: $\text{area} \approx \Delta x \sum_i p(x_i)$. Use 17 grid points spanning $\pm 4\sigma$, which is where the bell is effectively over.

### Case $\sigma = 0.1$: peak $3.989$, spacing $\Delta x = 0.05$

Front factor: $1 / (0.1 \times 2.5066) = 3.9894$. Bell: $\exp(-x^2/0.02) = \exp(-50x^2)$.

| $x$ | $\exp(-50x^2)$ | $p(x)$ |
|---|---|---|
| 0.00 | 1.0000 | **3.9894** |
| ±0.05 | 0.8825 | 3.5206 |
| ±0.10 | 0.6065 | 2.4197 |
| ±0.15 | 0.3247 | 1.2953 |
| ±0.20 | 0.1353 | 0.5399 |
| ±0.25 | 0.0439 | 0.1753 |
| ±0.30 | 0.0111 | 0.0443 |
| ±0.35 | 0.0022 | 0.0087 |
| ±0.40 | 0.0003 | 0.0013 |

One side adds to $8.00517$. So all 17 heights:

$$3.9894 + 2 \times 8.00517 = 19.99975.$$

Multiply by the spacing:

$$\text{area} = 0.05 \times 19.99975 = \boxed{0.99999}$$

A density peaking at nearly 4, with area 1. ✓

### Case $\sigma = 1$: peak $0.399$, spacing $\Delta x = 0.5$

Front factor $1/2.5066 = 0.39894$, bell $\exp(-x^2/2)$, grid now $\pm 4$ in steps of $0.5$.

**Every single exponential value is identical to the table above**, because the grid widened by the same factor of 10 that the bell did. Only the front factor changed, by $1/10$:

$$\text{heights sum} = 0.39894 + 2 \times 0.80052 = 1.99997, \qquad \text{area} = 0.5 \times 1.99997 = 0.99999. \ ✓$$

### Case $\sigma = 0.01$: peak $39.894$, spacing $\Delta x = 0.005$

Same arithmetic again, scaled the other way: heights are $10\times$ the $\sigma = 0.1$ table, spacing is $1/10$ of it.

$$\text{area} = 0.005 \times 199.9975 = 0.99999. \ ✓$$

| $\sigma$ | peak density | grid spacing | heights sum | area |
|---|---|---|---|---|
| 1 | 0.399 | 0.5 | 2.000 | 1.000 |
| 0.1 | 3.989 | 0.05 | 20.00 | 1.000 |
| 0.01 | 39.894 | 0.005 | 200.0 | 1.000 |

**Height and width trade off exactly, which is the whole reason a density has no ceiling.** Squeeze the distribution into a tenth of the space and the density must go up tenfold or the area would drop to a tenth.

### Where the front factor comes from, checked

That $1/(\sigma\sqrt{2\pi})$ is not magic, it is the $1/Z$ from the previous rung. Compute $Z$ directly: add up the bare bell with no front factor, using the $\sigma = 0.1$ column of exponentials:

$$Z \approx 0.05 \times \left[1 + 2(0.8825 + 0.6065 + 0.3247 + 0.1353 + 0.0439 + 0.0111 + 0.0022 + 0.0003)\right]$$

$$= 0.05 \times 5.01316 = 0.250658.$$

And the closed form says $Z = \sigma\sqrt{2\pi} = 0.1 \times 2.5066 = 0.250660$. Agreement to five figures. ✓ The front factor is $1/Z$, computed rather than assumed.

### Two knobs

**Truncate the grid instead of widening it.** Keep $\sigma = 1$ but run only $x \in [-1, 1]$: heights $0.39894 + 2(0.35206 + 0.24197) = 1.587$, area $= 0.5 \times 1.587 = 0.79$. The missing $0.21$ is the tails you cut off. Worth seeing, because it is the honest failure mode of every numerical normalizer: the answer looks fine and is short by whatever you did not integrate over.

**Multiply the whole shape by 7 and renormalize.** Take $f(x) = 7\exp(-50x^2)$. Its area is $7 \times 0.25066 = 1.7546$. Divide by that area:

$$\frac{7 \exp(-50x^2)}{1.7546} = \frac{\exp(-50x^2)}{0.25066},$$

the 7 cancels and you get back the identical density, peak $3.9894$, area $1.0$. **A constant factor multiplied into the shape is completely erased by normalizing.** That is exactly what $\propto$ is exploiting, now demonstrated on continuous numbers rather than asserted, and it is the whole content of the rung you are one step away from.

---

```
next: reading the joint table in the other direction, where p(c|x) comes from at all
      (short, and it hands straight into piece 2)
parent: back up to "the free variable" with 1 of 3 children left
also: top, map
```
