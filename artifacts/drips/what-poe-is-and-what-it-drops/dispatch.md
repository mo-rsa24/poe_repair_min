# dispatch: poe-two-concept-factorization

```
ran under:  none, a derivation rather than an experiment
built from: a screenshot of Eq (9) to Eq (10) in the PoE composition literature
why:        to find exactly which independence assumption the product-of-experts formula
            smuggles in, and what it costs when that assumption is false
depends on: none, the derivation follows from the stated axioms and no run can falsify it
```

target: the screenshot's Eq (9) -> Eq (10), p(x|c1,c2) = p(x|c1)p(x|c2)/p(x)
mode: --math
ladder: 5 pieces (what proportional-to licenses; conditional probability and Bayes;
  conditional independence as an assumption; the derivation line by line; what the
  assumption costs)

asked [yours]: which rules are being used silently here; conditional independence,
  Bayes, proportionality; "we can just throw away the denominator?"

---

rung: a conditional density is a function of the free variable only
notation: x free, c1 and c2 fixed labels; sum over x is 1, sum over c1 is not
examples: 4-image toy universe {cat photo, dog photo, chimera, car} with joint table
  p(x, c1); row c1=1 totals 0.50, normalizes to (0.70, 0.02, 0.24, 0.04);
  column check p(x1|c1=1)+p(x1|c1=0) = 0.70+0.02 = 0.72, not 1
examples (on `example`): full 16-cell joint p(x, c1, c2), cell totals
  (1,1)=0.13 (1,0)=0.37 (0,1)=0.34 (0,0)=0.16, margins reproduce p(x,c1) and
  p(x) = (0.36, 0.33, 0.14, 0.17); p(x | c1=1, c2=1) = (0.077, 0.077, 0.846, 0.000),
  argmax is the chimera; conditioning-slot sweep on x3 gives 0.932, not 1.
  This table is reusable to check Eq (10) numerically at piece 4.
clicked [my read]: (pending)
asked [yours]: (pending)
wants to see move: (empty until the reader says)

---

rung: density or likelihood, the same symbols read two ways (child 1 of "the free variable")
notation: "which slot is free" as the question to ask of any p(...); the letter p is
  overloaded across four different functions
examples: rows of the 8-cell table (x free) sum to 1; column x3 swept over c1 gives
  (0.24, 0.04), sum 0.28, but the ratio 6:1 is the meaningful part;
  same joint cell 0.12 gives p(x3|c1=1) = 0.12/0.50 = 0.24 and p(c1=1|x3) = 0.12/0.14
  = 0.857, two numbers from one cell, differing only by the divisor
clicked [my read]: (pending)
asked [yours]: (pending)

---

rung: what changes when x is continuous (child 2 of "the free variable")
notation: sum -> integral; Z as the normalizer; density is not probability
examples: Gaussian with sigma=0.1 has peak density 3.989 > 1 and still integrates to 1;
  Z for an isotropic Gaussian bump in D = 786432 dims (512x512x3) is (2*pi*sigma^2)^(D/2),
  log Z = 722,684 nats, about 313,900 decimal digits, so proportional-to is forced rather
  than chosen; none of the screenshot's algebra changes, only sum -> integral
examples (on `example`): Riemann sum on a grid, 17 points at +-4 sigma.
  sigma=0.1, dx=0.05: heights sum 19.99975, x dx = 0.99999.
  sigma=1, dx=0.5: identical arithmetic scaled, 1.99997 x 0.5 = 0.99999.
  sigma=0.01, dx=0.005: same again. Heights x10, spacing /10, area invariant.
  Unnormalized area check: sum exp(-50x^2) x 0.05 = 0.25066 = sigma*sqrt(2pi) exactly.
  Knob that bites: multiply f by 7, area becomes 7, renormalizing gives back the same
  density. That is proportional-to on continuous numbers, and it sets up rung 2.
clicked [my read]: (pending)
asked [yours]: (pending)

---

rung: the joint read the other way (child 3 of "the free variable", closes that subtree)
notation: generator reading p(x|c) = normalize down a row; classifier reading p(c|x) =
  normalize down a column; one joint table, two readings
examples: p(c1=1|x) = (0.972, 0.030, 0.857, 0.118) from column totals
  (0.36, 0.33, 0.14, 0.17); each column pair sums to 1 now because c1 is the free slot.
  Fifth image x5 = whisker close-up, p(x5) = 0.001, all of it cat: classifier says 1.000
  (the most certain cat in the universe), generator says 0.001/0.50 = 0.002 (almost never
  what you get when you ask for a cat). The two readings rank images differently and the
  thing standing between them is exactly p(x), the divisor in the boxed equation.
clicked [my read]: (pending)
asked [yours]: (pending)

---

route emitted: /math-scene, after rung 1 and its 3 children, pieces 2-5 untaught.
asked [yours]: "a math-scene for everything discussed in this chat so far"
wants to see move [my read]: the normalization axis on the one joint table, since every
  rung so far was the same 8 cells divided by a different total

filed: 00-INDEX.md plus 01 to 05, chat copied word for word including position and
  control lines. Resume point is rung 1.2, "what proportional-to lets you delete".

asked [yours]: how to repeat the verbatim filing with one word.
answer given: /polish, run in-session (it is built for drip ladders, appends to
  drips/<slug>/, and its rule is "reformat freely, rewrite nothing"). /capture-chat is
  the wider net for a multi-skill session. Off-tree, no child minted.

---

rung: what proportional-to lets you delete (rung 1.2)
notation: f(x) ∝ g(x) means f = k g for one k with no x in it; k = 1/Z recoverable
examples: (35, 1, 12, 2) -> (0.35, 0.01, 0.12, 0.02) -> (0.70, 0.02, 0.24, 0.04), three
  ways of writing one distribution, k chained 1/100 then 2 = 1/50;
  the bite: dividing the same row by p(x) = (0.36, 0.33, 0.14, 0.17) gives
  (0.972, 0.030, 0.857, 0.118), which normalizes to (0.492, 0.015, 0.434, 0.060),
  nothing like (0.70, 0.02, 0.24, 0.04). Chimera 0.24 -> 0.434. So p(x) is not a
  rescaling and cannot be deleted, while p(c1) can.
  Trap named: p(c1|x) contains x and is NOT deletable, despite c1 being on the left.
clicked [my read]: (pending)
asked [yours]: (pending)
