Source: transcript 9484913d-a247-4bdb-bb44-0f11895d96cc.jsonl, the scene return, why and the rung: message 7a43103b-7e70-4eb3-8014-6a0a211c9a7b at 2026-10-21T06:38:18.797Z; the squaring correction: message ffaeba2b-2258-42cb-bd16-7fc83a3d63eb at 2026-10-21T06:39:30.614Z

<!-- the scene return, why and the rung: message 7a43103b-7e70-4eb3-8014-6a0a211c9a7b -->
The first scene build is back, and the circle frames you asked for are now being built in the background. Here is the `why`, then the rung again with the by-hand build-up interleaved.

```
forward process → score > the cosine schedule > the boundary values C_k by hand
depth 1 · piece 1 of 4 · 0 visited, 3 left here
```

**The scene now covers all of piece 1.** Rung 4 is the new [Derive 7](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#derive-7): the toy point, its density bell, a prediction slot, and a switch that breaks the centre. Rungs 1 to 3 open at [Derive 2](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#derive-2), [Derive 3](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#derive-3) and [Derive 5](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#derive-5) to [6](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#derive-6). It was walked in a headless browser at desktop and phone widths. Two things were not done: no Codex mathematics review, and the paper link on $q(x_t\mid x_0)$ wasn't opened to confirm its page.

**why: the offset δ**

The obvious alternative is to start post 0 at exactly $0^\circ$. Near $0^\circ$, $\cos^2$ is flat: its slope is $-2\cos\theta\sin\theta$, which is $0$ at $0^\circ$. So the first step would barely add noise, $0.000247$ instead of $0.000631$. Nichol and Dhariwal (2021, *Improved DDPM*, section 3.2) give this as their reason: very small noise at the start made it hard for the network to predict ε. That citation is from memory, so check the section before quoting it. The value 0.008 itself is a tuned choice, with no derivation behind it.

---

**Why this piece is next**

Piece 1 used numbers like $\bar\alpha_{50}=0.4783$ without saying where they come from. They come from a timetable laid along a squared cosine. This piece computes that timetable by hand. Why a cosine was chosen at all is on the skip list, so the shape is taken as given.

**Fence posts, not gaps**

100 steps have 101 boundaries, the way 100 gaps need 101 fence posts. **$C_k$ is the share of the clean signal's [variance](/home-mscluster/mmolefe/goal-setting/learning/deep-learning/diffusion-models/08-spectral-structure-of-the-correction/plans/03-mean-variance-centering.md) still kept at boundary $k$**, for $k=0,\dots,100$. Steps live between the posts, and step $t$ runs from post $t$ to post $t+1$.

**✏️ On paper 1: the plane and the quarter circle**

Use a scale of 10 cm for 1 unit.

1. Draw the x-axis to the right and the y-axis up, meeting at the origin O.
2. Mark 1 on each axis, 10 cm from O.
3. Draw the quarter circle of radius 10 cm from (1, 0) round to (0, 1).

*Consequence.* Every point on that arc is exactly 1 from O. That exact 1 is what will keep the total variance at 1.
*Scene.* Timetable 2 already draws this arc.

**✏️ On paper 2: polar coordinates, one line from the origin**

1. Pick a point on the arc and draw the line from O to it.
2. Describe the point by its angle $\theta$ from the x-axis and its length 1, instead of by $(x,y)$.
3. Drop a vertical line from the point to the x-axis. You now have a right triangle.
4. The horizontal leg is $\cos\theta$ and the vertical leg is $\sin\theta$.

*Consequence.* $(x,y)=(\cos\theta,\sin\theta)$. Pythagoras on that triangle gives $\cos^2\theta+\sin^2\theta=1$: two shares that always add to one.
*Scene.* Being added: a new frame that draws both legs and their two share bars.

**✏️ On paper 3: measure the angle twice**

1. Protractor: read the angle in degrees.
2. String: lay it along the arc from (1, 0) to the point and measure it. On a 10 cm radius, centimetres ÷ 10 gives radians.
3. Try $46.25^\circ$. The string measures 8.07 cm, which is $0.8072$ rad. At $90^\circ$ it measures 15.71 cm, which is $\pi/2=1.5708$.
4. The rule is radians = degrees × $\pi/180$.

*Consequence.* The $\pi/2$ in the formula is the length of the whole quarter arc, which is $90^\circ$.
*Scene.* Timetable 2 prints both units. The new frame highlights the arc and labels its length.

**✏️ On paper 4: where the position goes**

1. Turn the arc into a ruler: 0 at the x-axis, 1 at the y-axis. Position $p$ sits at angle $p\times90^\circ$.
2. Without the offset, post $k$ sits at $p=k/100$.
3. With the offset, $p_k=(k+0.8)/100.8$. Mark post 0 at $0.0079$ ($0.714^\circ$), post 51 at $0.5139$ ($46.25^\circ$) and post 100 at $1$ ($90^\circ$).

*Consequence.* δ slides the first post just off the axis. Dividing by $1+\delta=1.008$ pulls the last post back onto $90^\circ$.
*Scene.* Being added: a position ruler along the arc with the posts and a δ handle.

**✏️ On paper 5: play with the quarter turn**

1. Put the point at $0^\circ, 15^\circ, 30^\circ, 45^\circ, 60^\circ, 75^\circ, 90^\circ$ and read $x^2$ each time: $1,\ 0.933,\ 0.75,\ 0.5,\ 0.25,\ 0.067,\ 0$.
2. Subtract neighbours: $0.067,\ 0.183,\ 0.25,\ 0.25,\ 0.183,\ 0.067$.
3. Put the point at $46.25^\circ$ and measure both legs: $(0.6915,\ 0.7224)$.

*Consequence.* $x^2$ barely moves near $0^\circ$ and $90^\circ$ and falls fastest in the middle, so the kept share drops slowly at both ends of the schedule. **The point at $46.25^\circ$ is $(0.6915, 0.7224)$, within $0.0001$ of step 50's dog multiplier $0.6916$ and noise multiplier $0.7223$.** Walking the point round the quarter turn is the forward process.
*Scene.* Being added: presets at every $15^\circ$ in the new frame.

**The formula, term by term**

With $T=100$ and offset $\delta=0.008$:

$$C_k=\frac{\cos^2\theta_k}{\cos^2\theta_0},\qquad \theta_k=\underbrace{\frac{k/T+\delta}{1+\delta}}_{\text{position, 0 to 1}}\cdot\underbrace{90^\circ}_{\text{a quarter turn}}=\frac{k+0.8}{100.8}\cdot 90^\circ$$

$k/T$ turns the post number into a position from 0 to 1. Adding $\delta$ nudges every position slightly forward. Dividing by $1+\delta$ pulls the last post back to exactly 1. Multiplying by $90^\circ$ lays the positions along a quarter turn, where the [cosine falls from 1 to 0](/home-mscluster/mmolefe/goal-setting/learning/mathematics/textbooks/trigonometry/trigonometry-handbook.pdf#page=16). Squaring makes the result a share between 0 and 1. Dividing by $\cos^2\theta_0$ makes the first post exactly 1.

**✏️ On paper 6: the fraction, numerator against denominator**

1. Draw two bars side by side. The numerator bar is $x^2$ at post $k$. The denominator bar is $x^2$ at post 0, which is $0.99984$.
2. $C_k$ asks how many denominators fit in the numerator. At post 51 that is $0.47819/0.99984=0.47826$. At post 0 it is $0.99984/0.99984=1$.
3. Now write $\beta_t=1-C_{t+1}/C_t$ with both $C$'s as fractions. The $\cos^2\theta_0$ sits on top and on the bottom.

*Consequence.* **The denominator is one fixed number for every post.** It stretches the whole curve by the same $1.00016$ so that post 0 reads exactly 1. It cancels in every $\beta$, so it never changes how much noise a step adds.
*Scene.* Timetable 2 states the cancellation in words. Being added: the numerator, denominator and ratio as three bars, with a switch that drops the denominator. Switched off, $C_0$ reads $0.99984$ and every β stays the same.

**Worked: post 0**

$$\theta_0=\frac{0.8}{100.8}\cdot90^\circ=0.714^\circ,\quad \cos\theta_0=0.99992,\quad \cos^2\theta_0=0.99984,\quad C_0=\frac{0.99984}{0.99984}=1$$

**Worked: post 1**

$$\theta_1=\frac{1.8}{100.8}\cdot90^\circ=1.607^\circ,\quad \cos^2\theta_1=0.99921,\quad C_1=\frac{0.99921}{0.99984}=0.99937$$

That is the same number as $\bar\alpha_0=0.9994$ in `walk-numbers.json`. The third rung of this piece shows why the cumulative share at code step $t$ lands on post $t+1$.

**Worked: post 51**

$$\theta_{51}=\frac{51.8}{100.8}\cdot90^\circ=46.25^\circ,\quad \cos\theta_{51}=0.6915,\quad \cos^2\theta_{51}=0.47819,\quad C_{51}=\frac{0.47819}{0.99984}=0.47826$$

That is $\bar\alpha_{50}=0.4783$, and its square root $0.6915$ is the dog's multiplier at step 50.

**Worked: post 100**

$\theta_{100}=90^\circ$ and $\cos 90^\circ=0$, so $C_{100}=0$. Nothing of the clean point is left.

**Broken: drop the offset**

With $\delta=0$, post 1 sits at $0.9^\circ$ and $C_1=\cos^2 0.9^\circ=0.99975$. The first step then adds a noise share of only $0.000247$, a typical miss of $0.0157$ against $0.0251$ with the offset. The offset makes sure the first step adds enough noise to matter.

```
next: β_t from consecutive posts, the share each step throws away, at t = 0 and t = 50
example: C_k at k = 25 and 75 by hand, checked against ᾱ_24 and ᾱ_74
also: scene (running: the circle frame and the fraction bars), links: the boundary values (textbook, paper, video, ours, scene frame), map, spell-it-out, controls
```

<!-- the squaring correction: message ffaeba2b-2258-42cb-bd16-7fc83a3d63eb -->
```
forward process → score > the cosine schedule > the boundary values C_k by hand
depth 1 · piece 1 of 4 · 0 visited, 3 left here
```

**A correction from the vault writer.** I wrote that squaring the cosine "makes the result a share between 0 and 1". That's loose: on a quarter turn $\cos\theta$ is already between 0 and 1. On your circle the real reading is this. $\cos\theta$ is the signal's multiplier, and $\cos^2\theta$ is its variance share, because share = multiplier² (rung 1). The paper calls the $\cos^2$ shape itself an arbitrary choice.

The boundary values are filed in a new note, [The Cosine Noise Schedule](/home-mscluster/mmolefe/Obsidian/Artificial%20Intelligence/Generative%20Modeling/Diffusion%20Models/The%20Cosine%20Noise%20Schedule.md). Your circle build-up is being filed beside it, and the circle frames for the scene are still being built.

```
next: β_t from consecutive posts, the share each step throws away, at t = 0 and t = 50
also: scene (running: the circle frame and the fraction bars), links: the boundary values (textbook, paper, video, ours, scene frame), map, spell-it-out, controls
```
