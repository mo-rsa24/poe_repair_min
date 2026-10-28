# The mathematics of sampling an image

This chapter writes out every piece of mathematics the three scenes of [From 173 cm to a Dog](../../scenes/what-sampling-an-image-means/README.md) use, in the order the scenes build it: one number, then many numbers, then going back from noise. Each section states the idea, derives it from what came before, and checks it on the same toys the scenes use: the ten heights with yours at 173 cm, a 2 × 2 grey image with two real images, and Step 4's cosine schedule. Every number here is recomputed by `check_numbers.py` beside this file, and each one carries its label from `check_numbers.out` in square brackets, so any claim can be traced to the line that verifies it.

**Colour key.** A sample or a noisy value is $\textcolor{#00838f}{x}$, a clean training example is $\textcolor{#1565c0}{x_0}$, a density is $\textcolor{#ad1457}{p}$, the kept share is $\textcolor{#2e7d32}{\bar\alpha_t}$, a noise scale is $\textcolor{#ef6c00}{\sqrt{1-\bar\alpha_t}}$, and a random number fed in is $\textcolor{#6a1b9a}{u}$, $\textcolor{#6a1b9a}{z}$ or $\textcolor{#6a1b9a}{\varepsilon}$. The scenes use the same colours.

## Contents

- [Part 1: one number](#part-1-one-number)
- [Part 2: many numbers](#part-2-many-numbers)
- [Part 3: going back](#part-3-going-back)
- [The assumptions, counted](#the-assumptions-counted)

---

## Part 1: one number

### 1.1 A sample and a dataset

A sample is one value drawn from a distribution. The ten heights are ten samples, written $x_1,\dots,x_{10}$, and your height is $x_7 = 173$ cm. The heights are example values chosen for the scenes. The two summary numbers are the mean and the variance, both dividing by 10:

$$
\underbrace{\bar x}_{\text{mean}}=\frac{1}{10}\underbrace{\sum_{i=1}^{10}x_i}_{1708}=170.8\ \text{cm},
\qquad
\underbrace{s^2}_{\text{variance}}=\frac{1}{10}\underbrace{\sum_{i=1}^{10}(x_i-\bar x)^2}_{541.6}=54.16\ \text{cm}^2,
\qquad s=7.36\ \text{cm}.
$$

The sum adds every height once, and dividing by 10 turns a total into an average. The square in the variance makes a height 5 cm below the mean count the same as one 5 cm above, and it makes far heights count more than near ones. [1.1]

The scenes treat the ten heights as independent draws from one distribution. That is the first assumption: nothing about one person's height changes the chance of another's.

### 1.2 A density, and why its area is 1

A density $\textcolor{#ad1457}{p(x)}$ says how thickly samples pile up near $x$. It is measured per unit of $x$ (per cm here), and it is never a chance on its own. Its one rule is that its total area is 1:

$$
\int_{-\infty}^{\infty}\textcolor{#ad1457}{p(x)}\,dx=1 .
$$

The Gaussian density with centre $\mu$ and width $\sigma$ is

$$
\textcolor{#ad1457}{p(x;\mu,\sigma)}=\underbrace{\frac{1}{\sigma\sqrt{2\pi}}}_{\text{normaliser}}\ \underbrace{\exp\!\Big(-\underbrace{\frac{(x-\mu)^2}{2\sigma^2}}_{\text{squared distance in widths}}\Big)}_{\text{bell shape, peak 1}} .
$$

Each piece has one job:

- The square $(x-\mu)^2$ makes the bell symmetric around $\mu$.
- Dividing by $2\sigma^2$ measures that distance in units of the width, so a wider $\sigma$ gives a wider bell.
- The exponential with a minus sign turns distance into a height between 0 and 1: 1 at the centre, falling fast far away.
- The normaliser $1/(\sigma\sqrt{2\pi})$ is what makes the area exactly 1.

**Where $\sqrt{2\pi}$ comes from.** Substitute $u=(x-\mu)/\sigma$, so $dx=\sigma\,du$, and the area of the bell shape is $\sigma\int e^{-u^2/2}du$. Call $I=\int e^{-u^2/2}du$. Then

$$
I^2=\int\!\!\int e^{-(u^2+v^2)/2}\,du\,dv=\int_0^{2\pi}\!\!\int_0^\infty e^{-r^2/2}\,r\,dr\,d\theta=2\pi\cdot 1 ,
$$

using polar coordinates, where $u^2+v^2=r^2$ and the area element is $r\,dr\,d\theta$. So $I=\sqrt{2\pi}$ [1.2 Gaussian integral], the shape's area is $\sigma\sqrt{2\pi}$, and dividing by it leaves area 1.

**On the toy.** The best bell has $\mu=170.8$ and $\sigma=7.36$. Its normaliser is 0.0542 per cm, and its height at your 173 cm is $\textcolor{#ad1457}{p(173)}=0.0518$ per cm [1.2]. A density can exceed 1: a bell of width 0.1 cm peaks at 3.99 per cm, and its area is still 1 [1.2]. Shown in [scene 1, frame 4](../../scenes/what-sampling-an-image-means/index.html#f4).

### 1.3 A chance is an area

The chance of landing between $a$ and $b$ is the area under the density over that range:

$$
P(a\le X\le b)=\int_a^b\textcolor{#ad1457}{p(x)}\,dx\ \approx\ \textcolor{#ad1457}{p(x)}\cdot\underbrace{(b-a)}_{\text{width }w}\quad\text{for a narrow strip.}
$$

For the strip from 172 to 174 cm the area is 0.1034, and height times width gives 0.1037 [1.3]. As the width goes to 0 the area goes to 0, so any single exact height has chance zero even where the density is high. Shown in [scene 1, frame 5](../../scenes/what-sampling-an-image-means/index.html#f5).

### 1.4 Likelihood: the same formula, read the other way

Hold $\mu$ and $\sigma$ fixed and let $x$ move, and $p(x;\mu,\sigma)$ is a density over heights. Hold the data fixed and let $\mu$ or $\sigma$ move, and the same formula scores each possible bell. That second reading is the likelihood. For ten independent heights the chances multiply, so

$$
L(\mu,\sigma)=\prod_{i=1}^{10}\textcolor{#ad1457}{p(x_i;\mu,\sigma)},
\qquad
\log L(\mu,\sigma)=\sum_{i=1}^{10}\log\textcolor{#ad1457}{p(x_i;\mu,\sigma)} .
$$

The product is there because the heights are independent, so the chance of all ten together is the product of their chances. The log turns that product into a sum, which keeps the numbers readable: $L=1.477\times10^{-15}$ at the best bell, while $\log L=-34.15$ [1.4].

**A likelihood is not a density over the parameters.** For one height, $p(173;\mu,\sigma)$ read over $\mu$ happens to have area 1, because the formula is symmetric in $x$ and $\mu$ [1.4]. For ten heights the area of $L(\mu,\sigma)$ over $\mu$ is $8.61\times10^{-15}$, nowhere near 1 [1.4]. A likelihood ranks parameter values; it does not spread a chance over them. Shown in [scene 1, frame 6](../../scenes/what-sampling-an-image-means/index.html#f6).

### 1.5 Fitting by likelihood gives the mean and the variance

Write out the log-likelihood:

$$
\log L(\mu,\sigma)=-10\log\sigma-5\log(2\pi)-\frac{1}{2\sigma^2}\sum_{i=1}^{10}(x_i-\mu)^2 .
$$

The best bell sits where both slopes are zero. In $\mu$:

$$
\frac{\partial\log L}{\partial\mu}=\frac{1}{\sigma^2}\sum_{i=1}^{10}(x_i-\mu)=0
\ \Longrightarrow\ \mu=\frac{1}{10}\sum_{i=1}^{10}x_i=\bar x .
$$

The slope is the sum of the ten signed distances, so it is zero exactly where the distances cancel, at the average. At $\mu=177$ the distances add to −62 cm and the slope is −1.145 [1.5]. In $\sigma$:

$$
\frac{\partial\log L}{\partial\sigma}=-\frac{10}{\sigma}+\frac{1}{\sigma^3}\sum_{i=1}^{10}(x_i-\bar x)^2=0
\ \Longrightarrow\ \sigma^2=\frac{1}{10}\sum_{i=1}^{10}(x_i-\bar x)^2=s^2 .
$$

The first term rewards a narrow bell, because the normaliser grows as $\sigma$ shrinks. The second penalises a bell too narrow to cover the data. They balance at the variance. A numerical search finds the same top, $\mu=170.8$ and $\sigma=7.3593$ [1.5]. Dividing by 9 instead gives $\sigma=7.76$ and a lower score, −34.18 [1.5]. Shown in [scene 1, frames 8 and 9](../../scenes/what-sampling-an-image-means/index.html#f8).

### 1.6 One height alone sends the width to zero

With only $x=173$, put $\mu=173$:

$$
\log L(\sigma)=-\log\sigma-\tfrac12\log(2\pi),
$$

which grows without limit as $\sigma\to0$: −0.92 at $\sigma=1$, 1.38 at $\sigma=0.1$ [1.6]. The best fit is a spike of zero width, so a fit needs more than one sample. Journey 03 records the same result by simulation: with one sample the fitted variance was 0 in every trial. Shown in [scene 1, frame 10](../../scenes/what-sampling-an-image-means/index.html#f10).

### 1.7 The running area is the CDF

The cumulative distribution function is the running total of area from the far left:

$$
F(x)=\int_{-\infty}^{x}\textcolor{#ad1457}{p(s)}\,ds,
\qquad
\frac{dF}{dx}=\textcolor{#ad1457}{p(x)} .
$$

The second equation is the fundamental theorem of calculus: the rate at which the running area grows is the height of the curve being added. $F$ climbs from 0 to 1 and never falls, because $p$ is never negative. On the toy, $F(173)=0.6175$, and its numerical slope at 173 is 0.0518, equal to $p(173)$ [1.7]. Shown in [scene 1, frame 11](../../scenes/what-sampling-an-image-means/index.html#f11).

### 1.8 Sampling by inverse CDF

Because $F$ only rises, each share $u$ between 0 and 1 has exactly one height $x$ with $F(x)=u$. Call that height $F^{-1}(u)$. The claim is that if $\textcolor{#6a1b9a}{U}$ is uniform on (0, 1), then $X=F^{-1}(\textcolor{#6a1b9a}{U})$ has the density $p$. The proof is one line:

$$
P\big(F^{-1}(\textcolor{#6a1b9a}{U})\le x\big)=P\big(\textcolor{#6a1b9a}{U}\le F(x)\big)=F(x).
$$

The first step holds because $F$ is increasing, so $F^{-1}(u)\le x$ exactly when $u\le F(x)$. The second holds because a uniform number falls below any $c$ in (0, 1) with chance $c$. Since $X$ has the same running area $F$, it has the same density.

For a Gaussian, $F^{-1}(u)=\mu+\sigma\,\Phi^{-1}(u)$, where $\Phi^{-1}$ is the inverse CDF of the standard bell. With $u=0.975$, $\Phi^{-1}(0.975)=1.96$ and the height is $170.8+1.96\times7.36=185.2$ cm [1.8]. Of 200,000 draws, a share 0.6175 lands at or below 173 cm, matching $F(173)$ [1.8]. All the randomness sits in $\textcolor{#6a1b9a}{u}$; $F^{-1}$ is a fixed rule that moves it. Shown in [scene 1, frames 12 and 13](../../scenes/what-sampling-an-image-means/index.html#f12).

---

## Part 2: many numbers

### 2.1 An image is a point

A grey 2 × 2 image is four numbers. List them in a fixed order, row by row, and the image is one point $\textcolor{#00838f}{x}=(x_1,x_2,x_3,x_4)\in\mathbb R^4$. A 128 × 128 colour photo is $128\times128\times3=49{,}152$ numbers, so it is one point in $\mathbb R^{49152}$ [2.1]. Values run from −1 (black) to 1 (white), the scale Step 4 uses. Shown in [scene 2, frame 1](../../scenes/what-sampling-an-image-means/many-numbers.html#f1).

### 2.2 A density over whole images

A density over images assigns a value to every point of $\mathbb R^4$, and its volume is 1: $\int_{\mathbb R^4}p(x)\,dx=1$. The toy has two real images, top bright $a=(1,1,-1,-1)$ and bottom bright $b=(-1,-1,1,1)$, each with spread $s=0.1$:

$$
\textcolor{#ad1457}{p(x)}=\tfrac12\,\mathcal N(x;a,s^2I)+\tfrac12\,\mathcal N(x;b,s^2I),
\qquad
\mathcal N(x;m,s^2I)=\underbrace{(2\pi s^2)^{-d/2}}_{\text{normaliser}}\exp\!\Big(-\frac{\lVert x-m\rVert^2}{2s^2}\Big).
$$

The weights ½ say each real image is equally common, and they add to 1, so the mix keeps volume 1 (each bell has volume 1, and ½ + ½ = 1). The normaliser is the 1-D one raised to the power $d$, because a bell with independent coordinates is a product of $d$ one-dimensional bells. It is 3.989 for one number, 15.92 for two and 253.3 for four [2.2]: the same total of 1 is packed into a region that shrinks as the dimensions multiply. At the top-bright image the density is $p(a)=126.65$, and at the checkerboard (1, −1, 1, −1) it is $4.85\times10^{-172}$ ($\log_{10}=-171.31$) [2.2]. Multiplying four 1-D integrals per bump gives a volume of 1 [2.2]. Shown in [scene 2, frame 2](../../scenes/what-sampling-an-image-means/many-numbers.html#f2).

### 2.3 One pixel alone is a marginal

The density of pixel 1 alone integrates the other three away:

$$
\textcolor{#ad1457}{p(x_1)}=\int\!\!\int\!\!\int \textcolor{#ad1457}{p(x)}\,dx_2\,dx_3\,dx_4=\tfrac12\,\mathcal N(x_1;1,s^2)+\tfrac12\,\mathcal N(x_1;-1,s^2).
$$

Each bell's other coordinates integrate to 1 on their own, leaving its first coordinate. Pixel 1 alone therefore has two bumps, at +1 and −1, each $\tfrac12\cdot\frac{1}{0.1\sqrt{2\pi}}=1.995$ high [2.3]. Shown in [scene 2, frame 3](../../scenes/what-sampling-an-image-means/many-numbers.html#f3).

### 2.4 Four numbers, drawn in order

One uniform $u$ makes one number, so an image needs four. The chain rule writes any joint density as a product of conditionals:

$$
p(x_1,x_2,x_3,x_4)=p(x_1)\;p(x_2\mid x_1)\;p(x_3\mid x_1,x_2)\;p(x_4\mid x_1,x_2,x_3).
$$

It follows from the definition $p(y\mid x)=p(x,y)/p(x)$ applied three times. Drawing in that order, one inverse CDF per pixel, samples the whole image:

$$
x_i=F_i^{-1}\big(\textcolor{#6a1b9a}{u_i}\mid x_1,\dots,x_{i-1}\big),\qquad i=1,2,3,4.
$$

Each step is section 1.8 applied to a conditional density, so each pixel has the right distribution given the ones before it, and by the chain rule the four together have the joint density. This construction is the Rosenblatt transform.

For the toy, each conditional is a mix of the two bumps, with weights updated by Bayes from the pixels already drawn:

$$
w_a(x_{<i})=\frac{\tfrac12\prod_{j<i}\mathcal N(x_j;a_j,s^2)}{\tfrac12\prod_{j<i}\mathcal N(x_j;a_j,s^2)+\tfrac12\prod_{j<i}\mathcal N(x_j;b_j,s^2)},
\qquad
F_i(v\mid x_{<i})=w_a\,\Phi\!\Big(\frac{v-a_i}{s}\Big)+w_b\,\Phi\!\Big(\frac{v-b_i}{s}\Big).
$$

With $\textcolor{#6a1b9a}{u}=(0.7,0.4,0.6,0.5)$: pixel 1 falls in the top-bright bump at 0.975, since $1+0.1\,\Phi^{-1}(0.4)=1-0.0253$ and $\Phi^{-1}(0.4)=-0.2533$. After that the weight on top bright is 1.000, so the other three follow it, and the image is (0.975, 0.975, −0.975, −1.000) [2.4]. Any pixel order works when the conditionals are known; PixelCNN-style networks learn them. Shown in [scene 2, frame 4](../../scenes/what-sampling-an-image-means/many-numbers.html#f4).

### 2.5 Skipping the conditioning breaks the image

Drawing each pixel from its own marginal assumes $p(x)=p(x_1)\,p(x_2)\,p(x_3)\,p(x_4)$, which says the pixels ignore each other. Each pixel is right on its own, but each of the 16 bright and dark patterns then has chance 1/16, and only 2 of them are real images: a share of 0.125 [2.5], against 1 when drawing in order. Journey 03's four-pixel stripes toy gives the same 0.125.

Writing the conditionals as a table fails for size. A table of chances over every image has one entry per possible image, minus one because they must add to 1: $2^{64}-1=1.845\times10^{19}$ free numbers for 8 × 8 black and white, and a count 118,370 digits long for 128 × 128 colour at 256 levels [2.5]. That is why the conditionals, or something equivalent, have to be learned by a network. Shown in [scene 2, frame 5](../../scenes/what-sampling-an-image-means/many-numbers.html#f5).

### 2.6 A network outputs numbers; a formula makes them a density

A network is a fixed function: weights in, numbers out. Its output vector, the same shape as the image, is never itself a density. It becomes one in one of three ways.

**Parameters of a fixed family.** The network outputs a mean $\mu_\theta$ for each pixel, and the density is

$$
p(x\mid\mu_\theta)=\prod_{i}\frac{1}{\sigma\sqrt{2\pi}}\exp\!\Big(-\frac{(x_i-\mu_{\theta,i})^2}{2\sigma^2}\Big).
$$

Each factor has area 1 for any $\mu$ (0.6 or anything else [2.6]), and integrating a product of independent factors coordinate by coordinate gives 1 × 1 × ⋯ = 1. The normaliser comes from the formula, and the network never has to produce it. A softmax over 256 levels per pixel works the same way: dividing by the sum makes the chances add to 1 whatever numbers the network gives. Diffusion's reverse step is this case, with the network supplying the mean.

**Transport.** Section 2.7.

**Score.** If $p(x)=\tilde p(x)/Z$ with an unknown constant $Z$, then $\nabla_x\log p(x)=\nabla_x\log\tilde p(x)$, because $\log Z$ does not depend on $x$. A network that learns the score $\nabla_x\log p$ never needs $Z$. This is Step 5 of the composition ladder, and it is why a product of experts can be sampled without ever computing its normaliser. Shown in [scene 2, frame 6](../../scenes/what-sampling-an-image-means/many-numbers.html#f6).

### 2.7 Transport: pushing a fixed distribution through a map

Draw $\textcolor{#6a1b9a}{z}$ from a fixed, easy distribution and output $x=G(\textcolor{#6a1b9a}{z})$. The distribution of $x$ is the pushforward: for any region $A$,

$$
P(x\in A)=P\big(\textcolor{#6a1b9a}{z}\in G^{-1}(A)\big).
$$

Each $z$ carries its share of chance to one $x$, so chance is moved and never created. The total stays 1 without anyone computing a normaliser. When $G$ is invertible and smooth, the density follows from the change of variables:

$$
p_X(x)=p_Z\big(G^{-1}(x)\big)\,\Big|\det\frac{\partial G^{-1}}{\partial x}\Big| .
$$

The determinant corrects for how much $G$ stretches or squeezes space: where $G$ stretches, the same chance spreads over more room and the density falls. In one dimension, $x=170.8+7.36\,z$ with $z\sim\mathcal N(0,1)$ gives $p_X(173)=\varphi\big((173-170.8)/7.36\big)/7.36=0.0518$, the best bell again [2.7]. Scene 1's $F^{-1}$ is a transport from Uniform(0, 1) to the bell.

The toy's transport takes five Gaussian numbers:

$$
G(\textcolor{#6a1b9a}{z})=\operatorname{sign}(\textcolor{#6a1b9a}{z_0})\,a+0.1\,(\textcolor{#6a1b9a}{z_1},\textcolor{#6a1b9a}{z_2},\textcolor{#6a1b9a}{z_3},\textcolor{#6a1b9a}{z_4}).
$$

The sign of $z_0$ is positive half the time, which picks a or b with chance ½ each. The other four, scaled by 0.1, add the spread. In 200,000 draws, a share 0.500 lands on top bright, with pixel 1's spread 0.100 inside a bump [2.7], which is the toy's two-bump density exactly. $G$ is written by hand and not trained. A GAN generator or a VAE decoder is a learned $G$; a normalizing flow is an invertible $G$ whose density follows from the change-of-variables formula. Shown in [scene 2, frame 7](../../scenes/what-sampling-an-image-means/many-numbers.html#f7).

### 2.8 The noise is fixed; the map is learned

$\mathcal N(0,I)$ is chosen in advance and never learned. What is learned is $G$. In the scene, $G$ has one parameter, a scale $c$, and under it the images follow $\tfrac12\mathcal N(x;ca,s^2I)+\tfrac12\mathcal N(x;-ca,s^2I)$. Fitting $c$ by likelihood on eight training images is section 1.5 again:

$$
\log L(c)=\sum_{n=1}^{8}\log\Big[\tfrac12\mathcal N(x^{(n)};ca,s^2I)+\tfrac12\mathcal N(x^{(n)};-ca,s^2I)\Big].
$$

On eight fresh training images the score peaks at $c=1.00$ [2.8], the scale the toy was built with. Flows train $G$ on likelihood exactly and VAEs on a bound on it. A GAN uses a discriminator's score instead, and diffusion trains the best guess of section 3.2. Shown in [scene 2, frame 8](../../scenes/what-sampling-an-image-means/many-numbers.html#f8).

### 2.9 Adding noise to one photo

Step 4's forward process adds Gaussian noise to one training example $\textcolor{#1565c0}{x_0}$ over $T=100$ steps, and the steps combine into one jump:

$$
q(\textcolor{#00838f}{x_t}\mid\textcolor{#1565c0}{x_0})=\mathcal N\big(\underbrace{\textcolor{#2e7d32}{\sqrt{\bar\alpha_t}}\,\textcolor{#1565c0}{x_0}}_{\text{mean, for this photo}},\ \underbrace{(1-\textcolor{#2e7d32}{\bar\alpha_t})}_{\text{variance}}\,I\big),
\qquad
\textcolor{#00838f}{x_t}=\textcolor{#2e7d32}{\sqrt{\bar\alpha_t}}\,\textcolor{#1565c0}{x_0}+\textcolor{#ef6c00}{\sqrt{1-\bar\alpha_t}}\,\textcolor{#6a1b9a}{\varepsilon}.
$$

The mean $\sqrt{\bar\alpha_t}\,x_0$ belongs to one photo; it is not the mean of all dogs. The schedule recomputed here gives $\beta_0=0.000631$, $\bar\alpha_{50}=0.4783$ and $\bar\alpha_{99}=2.43\times10^{-7}$ [3.0]. The derivation of the one-jump form is Step 4's, in [the one-expert walk](../../drips/simple-distributions-rebuild/01-one-expert-from-scratch.md). Shown in [scene 2, frame 9](../../scenes/what-sampling-an-image-means/many-numbers.html#f9).

---

## Part 3: going back

### 3.1 Bayes turns the noise step around

Sampling starts from noise and never has $x_0$. With the ten heights as the whole training set, each rescaled to mean 0 and spread 1 (written $z_i$), the prior is $p(x_0=z_i)=1/10$, and Bayes' theorem gives the chance of each height behind one noisy value:

$$
p(x_0=z_i\mid\textcolor{#00838f}{x_t})=\frac{\overbrace{q(\textcolor{#00838f}{x_t}\mid z_i)}^{\text{likelihood}}\ \overbrace{p(z_i)}^{\text{prior}}}{\underbrace{\sum_{j=1}^{10}q(\textcolor{#00838f}{x_t}\mid z_j)\,p(z_j)}_{p(x_t),\ \text{the total}}},
\qquad
q(\textcolor{#00838f}{x_t}\mid z_i)=\mathcal N\big(\textcolor{#00838f}{x_t};\ \textcolor{#2e7d32}{\sqrt{\bar\alpha_t}}\,z_i,\ 1-\textcolor{#2e7d32}{\bar\alpha_t}\big).
$$

The numerator is one cell of the joint table $p(x_0,x_t)$, and the denominator is the sum of its column, so dividing makes the weights add to 1. Read along a row, the table is the forward process; read down a column and divided by its sum, it is the posterior. At $t=50$ and $x_t=0.3$ the weights add to 1 and the largest, 0.147, is on 173 cm. With an uneven prior that counts 173 cm five times, it rises to 0.463 [3.1]. Shown in [scene 3, frames 1 and 2](../../scenes/what-sampling-an-image-means/going-back.html#f1).

### 3.2 The best guess is the posterior mean

The best single guess of $x_0$ under squared error is the weighted average:

$$
\textcolor{#1565c0}{\hat x_0}(\textcolor{#00838f}{x_t})=\mathbb E[x_0\mid x_t]=\sum_{i=1}^{10}w_i\,z_i,\qquad w_i=p(x_0=z_i\mid x_t).
$$

Why the mean: the expected squared error of a guess $c$ is $\sum_iw_i(z_i-c)^2$. Its slope in $c$ is $-2\sum_iw_i(z_i-c)$, which is zero exactly at $c=\sum_iw_iz_i$, and a grid search agrees [3.2]. At $t=50$, $x_t=0.3$ the guess is 172.4 cm. At $t=99$, where $\bar\alpha_{99}$ is $2.43\times10^{-7}$, every height explains the noise almost equally, so the guess is the plain mean 170.8 cm [3.2], the $\bar x$ of section 1.1. This is why $x_0$ can look like a mean: the guess of $x_0$ starts at the dataset's mean and sharpens as $t$ falls.

The guess is tied to the score by Tweedie's formula, $\mathbb E[x_0\mid x_t]=\big(x_t+(1-\bar\alpha_t)\nabla_{x_t}\log p(x_t)\big)/\sqrt{\bar\alpha_t}$, which is how a network that predicts noise, a network that predicts $x_0$, and a network that predicts the score are the same network in different units (Step 5). Shown in [scene 3, frame 3](../../scenes/what-sampling-an-image-means/going-back.html#f3).

### 3.3 One step back, with $x_0$ known

Knowing both $x_t$ and $x_0$, Bayes gives the earlier value:

$$
q(\textcolor{#00838f}{x_{t-1}}\mid\textcolor{#00838f}{x_t},\textcolor{#1565c0}{x_0})\ \propto\ \underbrace{q(\textcolor{#00838f}{x_t}\mid\textcolor{#00838f}{x_{t-1}})}_{\text{one forward step}}\ \underbrace{q(\textcolor{#00838f}{x_{t-1}}\mid\textcolor{#1565c0}{x_0})}_{\text{the jump to }t-1}.
$$

As functions of $x_{t-1}$, both factors are bells. The first, $\mathcal N(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t)$, is a bell in $x_{t-1}$ centred at $x_t/\sqrt{1-\beta_t}$ with variance $\beta_t/(1-\beta_t)$. The second is centred at $\sqrt{\bar\alpha_{t-1}}\,x_0$ with variance $1-\bar\alpha_{t-1}$. A product of two bells is a bell whose precision (one over the variance) is the sum of the two precisions, and whose mean is the precision-weighted mean of the two centres:

$$
\frac{1}{\sigma_t^2}=\frac{1-\beta_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}=\frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})},
$$

using $1-\bar\alpha_t=1-(1-\beta_t)\bar\alpha_{t-1}$. Collecting the terms gives

$$
q(\textcolor{#00838f}{x_{t-1}}\mid\textcolor{#00838f}{x_t},\textcolor{#1565c0}{x_0})=\mathcal N\big(a_t\,\textcolor{#1565c0}{x_0}+b_t\,\textcolor{#00838f}{x_t},\ \sigma_t^2\big),
\quad
a_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t},\quad
b_t=\frac{\sqrt{1-\beta_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t},\quad
\sigma_t^2=\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t},
$$

which are equations 6 and 7 of Ho, Jain and Abbeel (2020). At $t=50$: $a_{50}=0.0425$, $b_{50}=0.9547$ and $\sigma_{50}^2=0.0306$. The precision sum and the precision-weighted mean reproduce them exactly. For $x_0=173$ cm and $x_t=0.3$ the mean is 0.2991 [3.3]. At the last step, $t=0$, $\bar\alpha_{-1}=1$, so $a_0=1$, $b_0=0$ and $\sigma_0=0$: the final step returns the guess itself [3.5]. Shown in [scene 3, frame 4](../../scenes/what-sampling-an-image-means/going-back.html#f4).

### 3.4 In practice, the best guess replaces $x_0$

Without $x_0$, the exact step back mixes the ten bells of section 3.3, one per training height, with the posterior weights:

$$
p(\textcolor{#00838f}{x_{t-1}}\mid\textcolor{#00838f}{x_t})=\sum_{i=1}^{10}w_i\,q(\textcolor{#00838f}{x_{t-1}}\mid\textcolor{#00838f}{x_t},z_i)\ \approx\ \mathcal N\big(a_t\,\textcolor{#1565c0}{\hat x_0}+b_t\,\textcolor{#00838f}{x_t},\ \sigma_t^2\big).
$$

The mix's mean is $a_t\sum_iw_iz_i+b_tx_t=a_t\hat x_0+b_tx_t$, the same as the single bell, because the mean is linear in $x_0$. Its variance is larger by the spread of the parts' centres: $\sigma_t^2+a_t^2\,\mathrm{Var}_w(z_i)$. At $t=50$, $x_t=0.3$ the spreads are 0.1779 for the mix and 0.1749 for the single bell [3.4]. The gap scales with $a_t^2$, which is small when $\beta_t$ is small, so the single-bell step is accurate when steps are small. Shown in [scene 3, frame 5](../../scenes/what-sampling-an-image-means/going-back.html#f5).

### 3.5 The chain, and what fixes it

Start from $x_{99}\sim\mathcal N(0,1)$ and repeat for $t=99,\dots,0$:

$$
\textcolor{#00838f}{x_{t-1}}=a_t\,\textcolor{#1565c0}{\hat x_0}(\textcolor{#00838f}{x_t})+b_t\,\textcolor{#00838f}{x_t}+\sigma_t\,\textcolor{#6a1b9a}{\varepsilon_t},\qquad\textcolor{#6a1b9a}{\varepsilon_t}\sim\mathcal N(0,1).
$$

With the exact best guess over the ten heights, all 200 chains end within 0.5 cm of a training height [3.6]. On the four dog photos the guess at step 99 is the average of the four dogs, and it sharpens into one dog as $t$ falls. With the fresh noise $\sigma_t\varepsilon_t$ removed, using the update $x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\hat x_0+\sqrt{1-\bar\alpha_{t-1}}\,\hat\varepsilon$ with $\hat\varepsilon=(x_t-\sqrt{\bar\alpha_t}\,\hat x_0)/\sqrt{1-\bar\alpha_t}$ (Song, Meng and Ermon 2021), the 100 steps become a fixed function of the starting noise: the transport $G$ of section 2.7. Shown in [scene 3, frames 6 and 7](../../scenes/what-sampling-an-image-means/going-back.html#f6).

### 3.6 What the training data decides

A network is trained to predict $x_0$ (or the noise) from $x_t$ by minimising squared error, averaged over training images and noise. By section 3.2 the minimiser is the posterior mean $\mathbb E[x_0\mid x_t]$ under the training set, so the training data alone defines the target. A perfect fit to that target is the exact best guess, and section 3.5 showed where it leads: every sample is a copy of a training example. New samples come from the network falling short of that target in a smooth way; the scene contrasts the exact guess with a fitted bell, which returns heights the data never had. That trained networks generalise for this reason is a published result outside these pages (Kadkhodaie et al. 2024). Squared error is itself a Gaussian likelihood, since $\log\mathcal N(y;\hat y,\sigma^2)=-\frac{(y-\hat y)^2}{2\sigma^2}+\text{const}$. Shown in [scene 3, frame 9](../../scenes/what-sampling-an-image-means/going-back.html#f9).

### 3.7 A label enters by Bayes

Conditioning on a label multiplies the density by the label's chance at each $x$ and renormalises:

$$
p(x\mid\text{tall})=\frac{p(\text{tall}\mid x)\ \textcolor{#ad1457}{p(x)}}{P(\text{tall})},\qquad P(\text{tall})=\int p(\text{tall}\mid x)\,\textcolor{#ad1457}{p(x)}\,dx .
$$

With the illustrative step $p(\text{tall}\mid x)=1/(1+e^{-(x-175)/2})$ on the best bell, $P(\text{tall})=0.304$, the result has area 1, and its mean moves from 170.8 to 178.4 cm [3.7]. For images, "dog" plays the role of "tall". Taking the gradient of the log, $\nabla\log p(x\mid y)=\nabla\log p(x)+\nabla\log p(y\mid x)$, and the normaliser $P(y)$ drops out; classifier guidance (Dhariwal and Nichol 2021) steers a sampler with exactly this sum. Shown in [scene 3, frame 8](../../scenes/what-sampling-an-image-means/going-back.html#f8).

---

## The assumptions, counted

Five assumptions carry the chain from ten heights to a sampled dog. One is a free choice, and four are bets about the data and the network.

**The training examples are independent draws from one distribution.** It lets the likelihood be a product (section 1.4) and lets the dataset stand in for $p(x\mid\text{dog})$ (section 2.2).

**The forward noise is Gaussian.** This one is a choice. It makes the one-jump form and the product of section 3.3 closed-form.

**Steps are small.** It keeps the exact step back close to one bell (section 3.4).

**Squared error is the right loss.** It is Gaussian likelihood in disguise (section 3.6), and its minimiser is the posterior mean (section 3.2).

**The network can represent the best guess.** Architecture enters only here: a convnet's local, shared filters and a transformer's all-pairs attention are preferences about which functions are easy to learn, and neither layer computes a probability. The probability comes from the Gaussian noise, the loss and the sampler. Shown in [scene 3, frames 10 and 11](../../scenes/what-sampling-an-image-means/going-back.html#f10).
