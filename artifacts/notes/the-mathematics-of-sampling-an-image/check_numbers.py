"""Every number in the-mathematics-of-sampling-an-image.md, recomputed independently of the scenes.

Run:  python3 check_numbers.py > check_numbers.out
Each line prints a label the chapter cites, its value, and PASS or FAIL against the value the
chapter states. Nothing here imports the scenes' JavaScript; it is NumPy and SciPy from first
principles, so a PASS means two independent computations agree.
"""
import itertools
import numpy as np
from scipy import integrate, optimize, stats

rows, fails = [], 0


def show(label, value, stated, tol):
    global fails
    ok = abs(value - stated) <= tol
    fails += not ok
    rows.append(f"{'PASS' if ok else 'FAIL'}  {label:<58} {value:>14.6g}   chapter says {stated}")


# ---------- part 1: one number ----------
H = np.array([158, 163, 165, 168, 170, 172, 173, 176, 179, 184], float)
mu, ss = H.mean(), ((H - H.mean()) ** 2).sum()
var, sd = ss / 10, np.sqrt(ss / 10)
show("1.1 sum of the ten heights", H.sum(), 1708, 0)
show("1.1 mean x-bar (cm)", mu, 170.8, 1e-9)
show("1.1 sum of squared distances", ss, 541.6, 1e-6)
show("1.1 variance s^2, dividing by 10 (cm^2)", var, 54.16, 1e-6)
show("1.1 spread s (cm)", sd, 7.3593, 5e-5)
show("1.1 variance dividing by 9 (cm^2)", ss / 9, 60.18, 5e-3)
show("1.2 Gaussian integral of exp(-u^2/2)", integrate.quad(lambda u: np.exp(-u * u / 2), -np.inf, np.inf)[0], np.sqrt(2 * np.pi), 1e-9)
show("1.2 front factor 1/(s sqrt(2 pi))", 1 / (sd * np.sqrt(2 * np.pi)), 0.0542, 5e-5)
show("1.2 p(173) under the best bell (per cm)", stats.norm.pdf(173, mu, sd), 0.0518, 5e-5)
show("1.2 area under the best bell", integrate.quad(lambda x: stats.norm.pdf(x, mu, sd), 100, 250, points=[mu])[0], 1.0, 1e-9)
show("1.2 peak of a bell with sigma 0.1 (per cm)", stats.norm.pdf(0, 0, 0.1), 3.99, 5e-3)
show("1.3 P(172 <= X <= 174)", stats.norm.cdf(174, mu, sd) - stats.norm.cdf(172, mu, sd), 0.1034, 5e-5)
show("1.3 p(173) x 2 cm", 2 * stats.norm.pdf(173, mu, sd), 0.1037, 5e-5)
L = lambda m, s: stats.norm.logpdf(H, m, s).sum()
show("1.4 log L at the best bell", L(mu, sd), -34.15, 5e-3)
show("1.4 L at the best bell", np.exp(L(mu, sd)), 1.477e-15, 5e-18)
show("1.4 integral over mu of p(173; mu, s)", integrate.quad(lambda m: stats.norm.pdf(173, m, sd), 100, 250, points=[173])[0], 1.0, 1e-9)
show("1.4 integral over mu of L(mu, s) for ten heights", integrate.quad(lambda m: np.exp(L(m, sd)), 100, 250, points=[mu])[0], 8.611e-15, 5e-18)
show("1.5 slope in mu at mu = 177", ((H - 177) / sd ** 2).sum(), -1.145, 5e-4)
opt = optimize.minimize(lambda p: -L(p[0], p[1]), [175, 10], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12})
show("1.5 numerical maximiser mu (cm)", opt.x[0], 170.8, 1e-4)
show("1.5 numerical maximiser sigma (cm)", opt.x[1], 7.3593, 1e-4)
show("1.5 log L with sigma dividing by 9", L(mu, np.sqrt(ss / 9)), -34.18, 5e-3)
show("1.6 one height: log L at sigma = 1", stats.norm.logpdf(173, 173, 1), -0.92, 5e-3)
show("1.6 one height: log L at sigma = 0.1", stats.norm.logpdf(173, 173, 0.1), 1.38, 5e-3)
show("1.7 F(173)", stats.norm.cdf(173, mu, sd), 0.6175, 5e-5)
show("1.7 numerical slope of F at 173", (stats.norm.cdf(173.001, mu, sd) - stats.norm.cdf(172.999, mu, sd)) / 0.002, 0.0518, 5e-5)
show("1.8 Phi^-1(0.975)", stats.norm.ppf(0.975), 1.96, 5e-3)
show("1.8 F^-1(0.975) (cm)", stats.norm.ppf(0.975, mu, sd), 185.2, 5e-2)
u = np.random.default_rng(0).random(200_000)
show("1.8 share of F^-1(U) below 173 in 200,000 draws", (stats.norm.ppf(u, mu, sd) <= 173).mean(), 0.6175, 3e-3)

# ---------- part 2: many numbers ----------
s, a = 0.1, np.array([1, 1, -1, -1.0])
b = -a
norm = lambda d: (2 * np.pi * s * s) ** (-d / 2)
N4 = lambda x, m: norm(4) * np.exp(-((x - m) ** 2).sum() / (2 * s * s))
p = lambda x: 0.5 * N4(x, a) + 0.5 * N4(x, b)
show("2.1 numbers in a 128 x 128 colour photo", 128 * 128 * 3, 49152, 0)
show("2.2 normaliser, d = 1", norm(1), 3.989, 5e-4)
show("2.2 normaliser, d = 2", norm(2), 15.92, 5e-3)
show("2.2 normaliser, d = 4", norm(4), 253.3, 5e-2)
show("2.2 p(a), the top-bright image", p(a), 126.65, 5e-3)
show("2.2 log10 p(1, -1, 1, -1), equally far from a and b", np.log10(norm(4)) - 8 / (2 * s * s) / np.log(10), -171.31, 5e-3)
one = lambda m: integrate.quad(lambda v: stats.norm.pdf(v, m, s), -3, 3, points=[m])[0]
show("2.2 volume of p over R^4 (product of four 1-D integrals per bump)", 0.5 * np.prod([one(m) for m in a]) + 0.5 * np.prod([one(m) for m in b]), 1.0, 1e-9)
m1 = lambda v: 0.5 * stats.norm.pdf(v, 1, s) + 0.5 * stats.norm.pdf(v, -1, s)
show("2.3 marginal p(x1) at x1 = 1", m1(1), 1.995, 5e-4)
show("2.3 integral over x2 of p(x1=1, x2)", integrate.quad(lambda x2: 0.5 * (stats.norm.pdf(1, 1, s) * stats.norm.pdf(x2, 1, s) + stats.norm.pdf(1, -1, s) * stats.norm.pdf(x2, -1, s)), -3, 3, points=[-1, 1])[0], 1.995, 5e-4)


def in_order(us):
    x = []
    for i, ui in enumerate(us):
        lw = [np.log(0.5) + sum(stats.norm.logpdf(v, m[j], s) for j, v in enumerate(x)) for m in (a, b)]
        w = np.exp(np.array(lw) - max(lw)); w /= w.sum()
        F = lambda v: w[0] * stats.norm.cdf(v, a[i], s) + w[1] * stats.norm.cdf(v, b[i], s)
        x.append(optimize.brentq(lambda v: F(v) - ui, -3, 3, xtol=1e-12))
    return np.array(x), w


xo, w_after = in_order([0.7, 0.4, 0.6, 0.5])
for i, stated in enumerate([0.975, 0.975, -0.975, -1.0]):
    show(f"2.4 in-order draw, pixel {i + 1}", xo[i], stated, 5e-4)
show("2.4 weight on top bright after pixel 1", in_order([0.7, 0.4])[1][0], 1.0, 5e-4)
show("2.4 Phi^-1(0.4)", stats.norm.ppf(0.4), -0.2533, 5e-5)
real = lambda x: tuple((x > 0).astype(int)) in {(1, 1, 0, 0), (0, 0, 1, 1)}
show("2.5 exact share of real images, pixels drawn alone", sum(real(np.array(sg) * 2 - 1) for sg in itertools.product([0, 1], repeat=4)) / 16, 0.125, 0)
rng = np.random.default_rng(2)
alone = rng.choice([-1, 1], size=(100_000, 4)) + s * rng.normal(size=(100_000, 4))
show("2.5 share real in 100,000 draws, pixels alone", np.mean([real(x) for x in alone]), 0.125, 3e-3)
show("2.5 free numbers in an 8 x 8 black-and-white table", 2.0 ** 64 - 1, 1.845e19, 5e15)
show("2.5 digits in the 128 x 128 colour count", np.floor(49152 * np.log10(256)) + 1, 118370, 0)
show("2.6 area of N(mu1 = 0.6, sigma = 0.3)", integrate.quad(lambda v: stats.norm.pdf(v, 0.6, 0.3), -np.inf, np.inf)[0], 1.0, 1e-9)
z = np.random.default_rng(3).normal(size=(200_000, 5))
G = np.sign(z[:, :1]) * a + s * z[:, 1:]
show("2.7 share of G(z) near top bright", (G[:, 0] > 0).mean(), 0.5, 3e-3)
show("2.7 spread of pixel 1 within a bump", G[G[:, 0] > 0, 0].std(), 0.1, 2e-3)
show("2.7 change of variables: x = 170.8 + 7.36 z, density at 173", stats.norm.pdf((173 - mu) / sd) / sd, 0.0518, 5e-5)
# frame 8's eight training images use the page's seeded generator; here a fresh seeded set shows the same fit
tr = np.where(np.random.default_rng(4).random(8)[:, None] < 0.5, a, b) + s * np.random.default_rng(5).normal(size=(8, 4))
lc = lambda c: sum(np.log(0.5 * N4(x, c * a) + 0.5 * N4(x, -c * a)) for x in tr)
cs = np.linspace(0, 1.6, 1601)
show("2.8 best scale c on eight fresh training images", cs[np.argmax([lc(c) for c in cs])], 1.0, 0.05)

# ---------- part 3: going back ----------
T, dl = 100, 0.008
C = np.cos(((np.arange(T + 1) / T) + dl) / (1 + dl) * np.pi / 2) ** 2
C /= C[0]
beta = np.clip(1 - C[1:] / C[:-1], 0, 0.999)
abar = np.cumprod(1 - beta)
ab = lambda t: 1.0 if t < 0 else abar[t]
show("3.0 beta_0", beta[0], 0.000631, 5e-7)
show("3.0 abar_50", abar[50], 0.4783, 5e-5)
show("3.0 abar_99", abar[99], 2.43e-7, 5e-9)
Z = (H - mu) / sd


def post(xt, t, prior=None):
    pw = np.ones(10) if prior is None else np.asarray(prior, float)
    lw = np.log(pw) - (xt - np.sqrt(abar[t]) * Z) ** 2 / (2 * (1 - abar[t]))
    w = np.exp(lw - lw.max()); return w / w.sum()


w = post(0.3, 50)
show("3.1 posterior weights add to", w.sum(), 1.0, 1e-12)
show("3.1 largest weight at t = 50, x_t = 0.3", w.max(), 0.147, 5e-4)
show("3.1 uneven prior: weight on 173 cm", post(0.3, 50, [5 if h == 173 else 1 for h in H]).max(), 0.463, 5e-4)
show("3.2 best guess at t = 50, x_t = 0.3 (cm)", mu + sd * (w @ Z), 172.4, 5e-2)
show("3.2 best guess at t = 99 (cm)", mu + sd * (post(0.3, 99) @ Z), 170.8, 5e-2)
cands = np.linspace(-2, 2, 40001)
show("3.2 c minimising expected squared error equals the posterior mean", cands[np.argmin([(w * (Z - c) ** 2).sum() for c in cands])], w @ Z, 1e-4)
t = 50
A_ = np.sqrt(ab(t - 1)) * beta[t] / (1 - abar[t])
B_ = np.sqrt(1 - beta[t]) * (1 - ab(t - 1)) / (1 - abar[t])
V_ = beta[t] * (1 - ab(t - 1)) / (1 - abar[t])
show("3.3 a_50", A_, 0.0425, 5e-5)
show("3.3 b_50", B_, 0.9547, 5e-5)
show("3.3 sigma_50^2", V_, 0.0306, 5e-5)
x0, xt = Z[6], 0.3
prec = (1 - beta[t]) / beta[t] + 1 / (1 - ab(t - 1))
mean = ((np.sqrt(1 - beta[t]) / beta[t]) * xt + np.sqrt(ab(t - 1)) / (1 - ab(t - 1)) * x0) / prec
show("3.3 product of two bells: variance = 1/precision", 1 / prec, V_, 1e-12)
show("3.3 product of two bells: mean", mean, A_ * x0 + B_ * xt, 1e-12)
show("3.3 mean for x0 = 173 cm, x_t = 0.3", mean, 0.2991, 5e-5)
mix_var = V_ + A_ ** 2 * (w * (Z - w @ Z) ** 2).sum()
show("3.4 spread of the exact ten-bell mix", np.sqrt(mix_var), 0.1779, 5e-5)
show("3.4 spread of the single bell", np.sqrt(V_), 0.1749, 5e-5)
A0 = np.sqrt(ab(-1)) * beta[0] / (1 - abar[0])
show("3.5 a_0 (the last step returns the best guess)", A0, 1.0, 1e-12)
# DDPM chains on the ten heights with the exact best guess
rng = np.random.default_rng(6)
ends = []
for c in range(200):
    x = rng.normal()
    for t in range(99, -1, -1):
        g = post(x, t) @ Z
        a_t = np.sqrt(ab(t - 1)) * beta[t] / (1 - abar[t]); b_t = np.sqrt(1 - beta[t]) * (1 - ab(t - 1)) / (1 - abar[t]); v_t = beta[t] * (1 - ab(t - 1)) / (1 - abar[t])
        x = a_t * g + b_t * x + (np.sqrt(v_t) * rng.normal() if v_t > 0 else 0)
    ends.append(mu + sd * x)
ends = np.array(ends)
show("3.6 share of 200 chains ending within 0.5 cm of a height", np.mean(np.min(np.abs(ends[:, None] - H[None]), 1) < 0.5), 1.0, 0)
pt = integrate.quad(lambda x: stats.norm.pdf(x, mu, sd) / (1 + np.exp(-(x - 175) / 2)), 100, 250)[0]
show("3.7 P(tall) with threshold 175 cm", pt, 0.304, 5e-4)
show("3.7 area of p(x | tall)", integrate.quad(lambda x: stats.norm.pdf(x, mu, sd) / (1 + np.exp(-(x - 175) / 2)) / pt, 100, 250)[0], 1.0, 1e-6)
show("3.7 mean of p(x | tall) (cm)", integrate.quad(lambda x: x * stats.norm.pdf(x, mu, sd) / (1 + np.exp(-(x - 175) / 2)) / pt, 100, 250)[0], 178.4, 5e-2)

print("\n".join(rows))
print(f"\n{len(rows)} numbers, {fails} FAIL")
