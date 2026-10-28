"""Print every number in quiz-1-1-solutions.md, one block per solution.

Run from anywhere with numpy installed:
    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python quiz-1-1-solutions-check.py

Inputs taken from the step file and walk_numbers (rebuild/out/walk-numbers.json):
abar_50 = 0.4783 and sqrt(1 - abar_50) = 0.7223 (p1.abar_50, p1.sqrt_1m_abar_50), sqrt(abar_50) = 0.6916
(step 4), the clean toy point (0, 0.5), and step 1's Gaussian N(0, 0.5^2) with densities 0.7979 at 0
and 0.4839 at 0.5 (p1.normal_pdf_0_mu0_sd0.5, p1.normal_pdf_0.5_mu0_sd0.5). Book inputs are cited
in each block.
"""
import numpy as np

ABAR = 0.4783          # p1.abar_50
SIG = 0.7223           # p1.sqrt_1m_abar_50
ROOT = 0.6916          # sqrt(0.4783), step 4 of the step file
NOISE_VAR = 0.5217     # 1 - 0.4783, step 5 of the step file
X0 = np.array([0.0, 0.5])
DATA_VAR = 0.25        # step 1's Gaussian has sd 0.5
rng = np.random.default_rng(0)
N = 1_000_000


def r4(v):
    return np.round(v, 4)


print("== S1: the mean and variance of a sum of independent variables (B1.2)")
var_signal = ABAR * DATA_VAR
print("Var[0.6916 x], x ~ N(0, 0.25):", r4(var_signal))
print("Var[0.7223 eps]:", r4(SIG ** 2), " written as", NOISE_VAR)
print("sum of variances:", r4(var_signal + NOISE_VAR))
sd_sum = ROOT * 0.5 + SIG
print("wrong: sds added", r4(ROOT * 0.5), "+", SIG, "=", r4(sd_sum), " squared", r4(sd_sum ** 2))
x = 0.5 * rng.standard_normal(N)
eps = rng.standard_normal(N)
print("simulated Var[0.6916 x + 0.7223 eps], 1e6 draws, seed 0:", r4((ROOT * x + SIG * eps).var()))
print("wrong: z = x itself, Var[2 x] =", 4 * DATA_VAR, "against Var[x] + Var[x] =", 2 * DATA_VAR)

print("\n== S2: y = A x + b + w (B1.6, MML Exercise 6.12)")
A = ROOT * np.eye(2)
Q = NOISE_VAR * np.eye(2)
print("(a) mean of p(y | x0):", r4(A @ X0), " covariance diag:", np.diag(Q))
Sx = DATA_VAR * np.eye(2)
mu_y = A @ X0
S_y = A @ Sx @ A.T + Q
print("(b) x ~ N((0, 0.5), 0.25 I): mu_y", r4(mu_y), " Sigma_y diag", r4(np.diag(S_y)))
wrong = A @ Sx + Q
print("wrong: A Sigma_x + Q (no A^T), diag", r4(np.diag(wrong)))
xs = X0 + 0.5 * rng.standard_normal((N, 2))
ys = xs @ A.T + SIG * rng.standard_normal((N, 2))
print("simulated mean", r4(ys.mean(0)), " simulated variances", r4(ys.var(0)))

print("\n== S3: the sum of two Gaussians with unequal variances (B1.7, Chan Theorem 5.12)")
mu1, v1 = ROOT * 0.5, ABAR * DATA_VAR
mu2, v2 = 0.0, NOISE_VAR
print("X1 ~ N(", r4(mu1), ",", r4(v1), ")   X2 ~ N(", mu2, ",", v2, ")")
print("sum ~ N(", r4(mu1 + mu2), ",", r4(v1 + v2), ")")
lam = 1.0
mgf1 = np.exp(lam * mu1 + lam ** 2 * v1 / 2)
mgf2 = np.exp(lam * mu2 + lam ** 2 * v2 / 2)
mgf_sum = np.exp(lam * (mu1 + mu2) + lam ** 2 * (v1 + v2) / 2)
print("MGF at lambda = 1: product", r4(mgf1 * mgf2), " closed form", r4(mgf_sum))
print("wrong: Chan's 2 sigma^2 with sigma^2 = Var X1:", r4(2 * v1), " with sigma^2 = Var X2:", r4(2 * v2))
s = mu1 + np.sqrt(v1) * rng.standard_normal(N) + np.sqrt(v2) * rng.standard_normal(N)
print("simulated mean", r4(s.mean()), " simulated variance", r4(s.var()))

print("\n== S4: the mode of a Gaussian is its mean (B2.3, Bishop Exercise 1.9)")
mu, sd = 0.0, 0.5


def logp(v):
    return -(v - mu) ** 2 / (2 * sd ** 2) - 0.5 * np.log(2 * np.pi * sd ** 2)


print("density at 0:", r4(np.exp(logp(0.0))), " at 0.5:", r4(np.exp(logp(0.5))))
print("score at 0.5: -(0.5 - 0) / 0.25 =", -(0.5 - mu) / sd ** 2, "  second derivative:", -1 / sd ** 2)
h = 1e-6
print("finite difference of log p at 0.5:", r4((logp(0.5 + h) - logp(0.5 - h)) / (2 * h)))
print("finite difference of log p at 0:", r4((logp(0.0 + h) - logp(0.0 - h)) / (2 * h)))
print("wrong sign: (x - mu)/sigma^2 at 0.5 =", (0.5 - mu) / sd ** 2, " its second derivative", 1 / sd ** 2)
print("forward-draw Gaussian q(x_50 | x0): mode 0.6916 x0 =", r4(ROOT * X0))

print("\n== S5: the chain rule through exp(-z/2) (B2.6, MML Exercise 5.8a)")
xv = np.array([3.0, 4.0])
Sinv = np.diag([1.0, 0.25])
y = xv - 0.0
z = y @ Sinv @ y
f = np.exp(-z / 2)
df = -np.exp(-z / 2) * (y @ Sinv)
print("z:", z, " f = exp(-6.5):", f"{f:.7f}")
print("df/dx:", np.round(df, 7))
print("score = (df/dx) / f:", r4(df / f))
half = -0.5 * np.exp(-z / 2) * (y @ Sinv)
print("wrong: dropped the 2 from dz/dy, score", r4(half / f))


def F(v):
    return np.exp(-0.5 * v @ Sinv @ v)


fd = np.array([(F(xv + h * e) - F(xv - h * e)) / (2 * h) for e in np.eye(2)])
print("finite difference of f:", np.round(fd, 7))

print("\n== S6: Example 6.4 without its sixth person (B3.4, Introduction to Statistics Exercise 6.5)")
S_all = np.array([20000, 24500, 23000, 25000, 20000, 22500], float)
E_all = np.array([2.8, 3.4, 3.2, 3.8, 3.2, 3.4])
for label, S, E in [("all six", S_all, E_all), ("first five", S_all[:5], E_all[:5])]:
    n = len(S)
    sE, sS, sES, sEE = E.sum(), S.sum(), (E * S).sum(), (E * E).sum()
    Sxy = sES - sE * sS / n
    Sxx = sEE - sE ** 2 / n
    b1 = Sxy / Sxx
    b0 = S.mean() - b1 * E.mean()
    print(f"{label}: n={n} sum Ed={sE:.1f} sum S={sS:.0f} sum S*Ed={sES:.0f} sum Ed^2={sEE:.2f}")
    print(f"   S_xy={Sxy:.1f} S_xx={Sxx:.3f} slope={b1:.2f} intercept={b0:.2f} predict at 2.8: {b0 + b1 * 2.8:.2f}")
    print("   polyfit check:", np.round(np.polyfit(E, S, 1), 2))
S5, E5 = S_all[:5], E_all[:5]
b_origin = (E5 * S5).sum() / (E5 * E5).sum()
print(f"wrong: no intercept, slope = sum S*Ed / sum Ed^2 = {b_origin:.2f}, predicts {b_origin * 2.8:.2f} at 2.8")
print("book's rounding on all six: 22500 - 5370 * 3.3 =", 22500 - 5370 * 3.3)

print("\n== S7: fit_line by lstsq on Anton's four points (B3.6, Chan Theorem 7.1 and listing)")


def fit_line(xd, yd):
    X = np.column_stack((xd, np.ones(len(xd))))
    return np.linalg.lstsq(X, yd, rcond=None)[0]


xd = np.array([0.0, 1, 2, 3])
yd = np.array([1.0, 3, 4, 4])
theta = fit_line(xd, yd)
X = np.column_stack((xd, np.ones(4)))
print("theta (slope, intercept):", r4(theta))
print("X^T X:", (X.T @ X).tolist(), " X^T y:", (X.T @ yd).tolist(), " X^T X theta:", r4(X.T @ X @ theta))
res = yd - X @ theta
print("residuals:", r4(res), " RSS:", r4(res @ res), " X^T r:", r4(X.T @ res))
b_o = (xd * yd).sum() / (xd * xd).sum()
res_o = yd - b_o * xd
print("wrong: no column of ones, slope", r4(b_o), " RSS", r4(res_o @ res_o))
print("wrong: theta read as (intercept, slope): predicts", theta[0] + theta[1] * 3, "at x = 3 against", theta[1] + theta[0] * 3)
