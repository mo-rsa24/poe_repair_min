Source: transcript 9484913d-a247-4bdb-bb44-0f11895d96cc.jsonl, where p_k comes from: message 4695b76c-6650-4b3a-88ce-ea57ea736016 at 2026-10-21T07:24:11.815Z

<!-- where p_k comes from: message 4695b76c-6650-4b3a-88ce-ea57ea736016 -->
```
forward process → score > the cosine schedule > the boundary values C_k by hand > where p_k comes from
depth 2 · child 1 of 1 · 0 visited, 0 left here
```

**The handbook is now one click away.** Step 4 has a quick-reference line under the cosine schedule paragraph. It links [the vault's trigonometry handbook](/home-mscluster/mmolefe/Obsidian/Mathematics/references/textbooks/trigonometry/trigonometry-handbook.pdf) at radians (p. 7), the unit circle (p. 10) and the squared cosine (p. 16).

**What k, T and δ are**

**k** is the post number, 0 to 100. In `rebuild/model.py` it is `x = np.linspace(0, timesteps, steps)`.

**T** is the number of steps, 100. It is `timesteps`.

**δ** is the offset, 0.008. It is the code's `s`.

**Where it comes from**

Nichol and Dhariwal (2021, equation 17) write the schedule as

$$f(k)=\cos^2\!\Big(\underbrace{\frac{k/T+s}{1+s}}_{\text{the position}}\cdot\frac{\pi}{2}\Big).$$

**p_k is the walk's own name for the part inside the bracket: how far round the quarter turn post k sits.** The paper gives it as a design choice, with no derivation. Even so, the straight-line form follows from three requirements, so you can derive it.

**Deriving it**

Let u = k/T be the fraction of the schedule done, from 0 to 1. Then apply three requirements.

1. **Posts are evenly spaced.** The cos² curve already makes the noise uneven, so the posts themselves step evenly. That makes p a straight line in u:

$$p=b+a\,u.$$

2. **The last post lands exactly on the quarter turn**, so that C_100 = 0 and no signal is left:

$$p(1)=b+a=1.$$

3. **The first post sits a little past 0**, by the offset. The paper does this by shifting u forward by δ and then rescaling so requirement 2 still holds:

$$p=\frac{u+\delta}{1+\delta}\quad\Rightarrow\quad a=\frac{1}{1+\delta},\qquad b=\frac{\delta}{1+\delta},\qquad b+a=\frac{\delta+1}{1+\delta}=1\ \checkmark$$

$$\boxed{p_k=\frac{k/T+\delta}{1+\delta},\qquad \theta_k=p_k\cdot 90^\circ}$$

Multiply the top and bottom by 100 and you get the (k + 0.8)/100.8 form. In your screenshot δ = 0, which is why it reads (k + 0.0)/100.0.

**Worked: the four posts from the straight line**

With δ = 0.008: a = 1/1.008 = 0.99206 and b = 0.008/1.008 = 0.0079365.

$$
\begin{aligned}
p_0&=0.0079365+0\times0.0099206=0.00794 &&\to\ 0.714^\circ\\
p_1&=0.0079365+1\times0.0099206=0.01786 &&\to\ 1.607^\circ\\
p_{51}&=0.0079365+51\times0.0099206=0.51389 &&\to\ 46.25^\circ\\
p_{100}&=0.0079365+100\times0.0099206=1 &&\to\ 90^\circ
\end{aligned}
$$

Each step moves a/T = 0.0099206 of the way round, which is 0.8929°. Without the offset it would be exactly 0.9°.

**Broken: drop the ÷ (1 + δ)**

Then post 100 sits at 1.008 of the way round, which is 90.72°, past the quarter turn. There cos² = 0.000158, so C_100 = 0.000158 instead of 0, and the last step never removes all of the clean signal. Requirement 2 is the one that broke.

**The same line in the code**

```python
alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
#                         └ u = k/T ┘  └+δ┘  └÷(1+δ)┘   └ ×90° ┘   └ cos² ┘
```

```
parent: back to the boundary values C_k by hand
next: β_t from consecutive posts, the share each step throws away, at t = 0 and t = 50
also: keep: this derivation into Step 4's On paper 4 (adds the three requirements and the broken end at 90.72°), scene: a derive frame for p_k's three requirements, squaring yes, map, spell-it-out, controls
```

**Read more:** [On paper 4 in Step 4](artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md#step4-the-quarter-circle-by-hand) · [the 101 posts on the scene](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#timetable-posts)

No journey covers: unit circle, the noise schedule's curve shape.
