"""The answer key for the five walk documents: every worked-example number they quote, computed here.

    python -m rebuild.walk_numbers            # prints every number and writes rebuild/out/walk-numbers.json

Results of the runs themselves (TVs, A∩B masses) are read from rebuild/out/phase*/ by the documents directly.
The p2.step9_ keys are the one exception read here: they come from rebuild/out/phase2/step9.json, which
python -m rebuild.phase2_step9 writes (it reruns step 9 of the phase 2 walk, about a minute).
"""
import json
import math
import os

import numpy as np
import torch
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

from rebuild.model import cosine_beta_schedule, make_model

K = {}


def put(key, value):
    K[key] = value
    print(f"{key:>48}: {value}")


def phase1():
    put("p1.normal_pdf_0.5_mu0_sd0.5", round(float(norm.pdf(0.5, 0, 0.5)), 4))
    put("p1.normal_pdf_0_mu0_sd0.5", round(float(norm.pdf(0.0, 0, 0.5)), 4))
    means = 0.5 * np.c_[np.cos(2 * np.pi * np.arange(8) / 8), np.sin(2 * np.pi * np.arange(8) / 8)]
    def ring(x, s=0.03):
        d2 = ((x[None] - means) ** 2).sum(1)
        return float(np.mean(np.exp(-d2 / (2 * s * s)) / (2 * np.pi * s * s)))
    put("p1.ring_density_at_(0.5,0)", round(ring(np.array([0.5, 0.0])), 2))
    put("p1.ring_density_at_(0.53,0)", round(ring(np.array([0.53, 0.0])), 2))
    put("p1.ring_density_at_(0,0)", float(f"{ring(np.array([0.0, 0.0])):.3g}"))
    put("p1.box0.7_density", round(1 / 1.4 ** 2, 4))
    put("p1.bar0.2_density", round(1 / (0.4 * 2), 4))
    b = cosine_beta_schedule(100); ab = np.cumprod(1 - b)
    put("p1.beta_0", float(f"{b[0]:.3g}"))
    put("p1.beta_99", round(float(b[99]), 4))
    for t in (0, 25, 50, 75, 99):
        put(f"p1.abar_{t}", round(float(ab[t]), 4))
        put(f"p1.sqrt_1m_abar_{t}", round(float(np.sqrt(1 - ab[t])), 4))
    # one forward-noising draw and its score
    x0, eps, t = np.array([0.0, 0.5]), np.array([0.3, -1.2]), 50
    xt = np.sqrt(ab[t]) * x0 + np.sqrt(1 - ab[t]) * eps
    score = -(xt - np.sqrt(ab[t]) * x0) / (1 - ab[t])
    put("p1.example_x_t", np.round(xt, 4).tolist())
    put("p1.example_score", np.round(score, 4).tolist())
    put("p1.example_minus_eps_over_sigma", np.round(-eps / np.sqrt(1 - ab[t]), 4).tolist())
    # parameter count, block by block
    lin = lambda i, o: i * o + o
    block = 2 * 128 + lin(128, 256) + lin(32, 256) + lin(256, 256) + lin(256, 128)
    put("p1.params_embedding", 100 * 32)
    put("p1.params_input_linear", lin(2, 128))
    put("p1.params_one_block", block)
    put("p1.params_output_linear", lin(128, 2))
    put("p1.params_total_by_hand", 100 * 32 + lin(2, 128) + 4 * block + lin(128, 2))
    put("p1.params_total_torch", sum(p.numel() for p in make_model().parameters()))
    # energy at init
    m = make_model(); x = torch.tensor([[0.3, -0.1]]); tt = torch.tensor([50])
    put("p1.init_energy_at_(0.3,-0.1)", round(float(m.net.neg_logp_unnorm(x, tt)), 4))
    put("p1.init_eps_at_(0.3,-0.1)", np.round(m.net(x, tt, create_graph=False).numpy()[0], 4).tolist())
    # loss of the init network on bar data: E|2 x_t - eps|^2 / 2
    rng = np.random.default_rng(0); n = 200000
    xb = np.c_[rng.uniform(-0.2, 0.2, n), rng.uniform(-1, 1, n)]
    ts = rng.integers(0, 100, n); e = rng.standard_normal((n, 2))
    xt_ = np.sqrt(ab[ts])[:, None] * xb + np.sqrt(1 - ab[ts])[:, None] * e
    put("p1.init_loss_on_bar", round(float(((2 * xt_ - e) ** 2).mean()), 3))
    put("p1.ema_half_life_steps", round(math.log(0.5) / math.log(0.999), 1))
    abp = np.append(1.0, ab[:-1])
    for t in (50, 0):
        put(f"p1.post_c1_{t}", round(float(b[t] * np.sqrt(abp[t]) / (1 - ab[t])), 5))
        put(f"p1.post_c2_{t}", round(float((1 - abp[t]) * np.sqrt(1 - b[t]) / (1 - ab[t])), 5))
        put(f"p1.post_var_{t}", float(f"{b[t] * (1 - abp[t]) / (1 - ab[t]):.4g}"))
        put(f"p1.beta_{t}_sd", round(float(np.sqrt(b[t])), 4))
    put("p1.ring_mode_sd_with_t0_noise", round(float(np.sqrt(0.03 ** 2 + b[0])), 4))


def phase2():
    # 1D product of N(0,1) and N(2,1)
    put("p2.prod_mean", 1.0); put("p2.prod_var", 0.5)
    put("p2.prod_logZ", round(float(norm.logpdf(0, 2, np.sqrt(2))), 4))
    # noised product against product of noised, A = N(1, 0.1^2), B = N(-1, 1), abar = 0.5
    a = 0.5
    pa, pb = 1 / 0.01, 1.0
    clean_mean = (pa * 1 - pb * 1) / (pa + pb)
    va, vb = a * 0.01 + 1 - a, a * 1 + 1 - a
    noised_prod_of_noised = (np.sqrt(a) / va - np.sqrt(a) / vb) / (1 / va + 1 / vb)
    put("p2.clean_product_mean", round(clean_mean, 4))
    put("p2.noised_product_mean", round(np.sqrt(a) * clean_mean, 4))
    put("p2.product_of_noised_mean", round(float(noised_prod_of_noised), 4))
    # the notebook's mixture of energies with temperature 3.5
    E = np.array([1.0, 3.0])
    put("p2.mixture_energy_E1=1_E2=3", round(float(-np.log(np.exp(-3.5 * E).sum())), 4))
    put("p2.negation_value_p1=0.5_p2=2", round(0.5 ** 1.3 * 2 ** -0.3, 4))
    # ULA on N(0, 1): x' = (1 - ss) x + sqrt(2 ss) z
    ss = 0.9
    put("p2.ula_stationary_var_ss0.9", round(2 / (2 - ss), 4))
    rng = np.random.default_rng(0); x = rng.standard_normal(200000) * 3
    for _ in range(300):
        x = (1 - ss) * x + np.sqrt(2 * ss) * rng.standard_normal(len(x))
    put("p2.ula_simulated_var_ss0.9", round(float(x.var()), 3))
    # one MALA move on N(0, 1), both divisors
    x, ss, z = 1.0, 0.5, 0.3
    g = -x; xp = x + ss * g + np.sqrt(2 * ss) * z; gp = -xp
    dlogp = -0.5 * xp ** 2 + 0.5 * x ** 2
    for name, div in [("fixed", 4 * ss), ("old", 2 * ss ** 2)]:
        fwd = -(xp - x - ss * g) ** 2 / div; bwd = -(x - xp - ss * gp) ** 2 / div
        la = dlogp + bwd - fwd
        put(f"p2.mala_{name}_divisor", div)
        put(f"p2.mala_{name}_log_fwd", round(fwd, 4)); put(f"p2.mala_{name}_log_bwd", round(bwd, 4))
        put(f"p2.mala_{name}_log_alpha", round(la, 4)); put(f"p2.mala_{name}_accept_prob", round(min(1.0, math.exp(la)), 4))
    put("p2.mala_example_x_prime", round(xp, 4)); put("p2.mala_example_dlogp", round(dlogp, 4))
    # one leapfrog step on N(0, 1)
    x, v, h = 1.0, 0.5, 0.3
    H0 = 0.5 * x ** 2 + 0.5 * v ** 2
    v1 = v + 0.5 * h * (-x); x1 = x + h * v1; v2 = v1 + 0.5 * h * (-x1)
    put("p2.leapfrog_v_half", round(v1, 4)); put("p2.leapfrog_x", round(x1, 4)); put("p2.leapfrog_v", round(v2, 5))
    put("p2.leapfrog_H_before", round(H0, 4)); put("p2.leapfrog_H_after", round(0.5 * x1 ** 2 + 0.5 * v2 ** 2, 5))
    # the CHA aliasing: the in-place leapfrog returns the same tensor it was given
    from anneal_samplers import leapfrog_step_c
    vp = torch.tensor([[0.5]]); xx = torch.tensor([[1.0]])
    _, vn = leapfrog_step_c(xx, vp, lambda q: (None, -q), 0.3, 1.0, 1, -xx.clone())
    put("p2.cha_alias_same_tensor", bool(vn is vp))
    # step 2: noising first blurs A from variance 0.01 to 0.505, so its precision falls from 100 to 1.98
    put("p2.noised_A_var_abar0.5", round(va, 4)); put("p2.noised_A_precision_abar0.5", round(1 / va, 2))
    # step 3: the soft minimum written exactly, 3.5 m - log(1 + exp(-3.5 Delta)), at E_A = 1, E_B = 3
    put("p2.mixture_softmin_formula_E1=1_E2=3", round(3.5 * 1.0 - math.log1p(math.exp(-3.5 * 2.0)), 5))
    # step 4: N(0,1)^(1+w) / N(0,s^2)^w has a finite normaliser only when s^2 > w / (1 + w)
    put("p2.negation_finite_Z_needs_s2_above_w0.3", round(0.3 / 1.3, 4))
    # step 7: one Metropolis-Hastings decision for the whole batch (scalar energy), started at the target,
    # 200 moves at ss = 1.5; the fraction of moves accepted collapses as the batch grows
    from anneal_samplers import AnnealedMALASampler
    lp = lambda q: -0.5 * (q ** 2).sum(1)
    for n in (1, 10, 100, 4000):
        torch.manual_seed(1)
        smp = AnnealedMALASampler(1, 200, [1.5], lambda q, t: (lp(q).sum(), -q))
        smp.sample_step(torch.randn(n, 1), 0, None, {})
        put(f"p2.batch_mala_joint_accept_n{n}", round(float(smp.last_accept[0]), 3))
    # step 8: flip v', take the same leapfrog step, and it lands on the start with its momentum flipped
    def lf(x, v, h=0.3):
        v = v + 0.5 * h * (-x); x = x + h * v; return x, v + 0.5 * h * (-x)
    x1, v1 = lf(1.0, 0.5); xr, vr = lf(x1, -v1)
    put("p2.leapfrog_step_from_flipped_end", [round(xr, 6), round(vr, 6)])
    # step 9: the schedule's two ends, and step 9's run with and without a clean final level
    b = cosine_beta_schedule(100); ab = np.cumprod(1 - b)
    put("p2.abar_99", float(f"{ab[99]:.2g}")); put("p2.sqrt_abar_0", round(float(np.sqrt(ab[0])), 4))
    # step 9's run takes about a minute, so its numbers are read from the file python -m rebuild.phase2_step9 writes
    from rebuild.phase2_step9 import OUT as STEP9
    if not os.path.exists(STEP9):
        raise SystemExit(f"{STEP9} is missing: run `python -m rebuild.phase2_step9` from the repo root first "
                         "(it reruns step 9 of 02-composing-and-sampling.md and writes that file)")
    R = json.load(open(STEP9))
    put("p2.step9_exact_share_clean", round(R["exact_share_clean"], 3))
    put("p2.step9_exact_share_pi0", round(R["exact_share_pi0"], 3))
    put("p2.step9_clean_mass_within_0.2_of_in_bar_centres", round(R["clean_mass_within"]["in_bar_0.2"], 3))
    put("p2.step9_clean_mass_within_0.15_of_in_bar_centres", round(R["clean_mass_within"]["in_bar_0.15"], 3))
    put("p2.step9_clean_mass_within_0.2_of_other_six", round(R["clean_mass_within"]["other_six_0.2"], 3))
    put("p2.step9_clean_mass_beyond_0.2_of_every_centre", round(R["clean_mass_within"]["beyond_0.2_of_every_centre"], 3))
    put("p2.step9_share_perfect_1000", round(R["share_perfect_1000"], 3))
    put("p2.step9_share_reverse_diffusion", round(R["share_reverse_diffusion"], 3))
    for k in ("ULA", "MALA", "UHA"):
        put(f"p2.step9_share_{k}_annealed_to_pi0", round(R["share_annealed"][k], 3))
        put(f"p2.step9_share_{k}_plus_clean_level", round(R["share_annealed_then_clean_level"][k], 3))


def phase3():
    A = np.array([[-0.5, -0.5], [-0.6, -0.4], [-0.5, 0.5]]); s = 0.15; q = np.array([[-0.5, 0.0]])
    d2 = ((q[:, None] - A[None]) ** 2).sum(-1)
    ours = float(np.log(np.mean(np.exp(-d2 / (2 * s * s)))) - np.log(2 * np.pi * s * s))
    kde = KernelDensity(kernel="gaussian", bandwidth=s).fit(A)
    put("p3.ideal_logdensity_formula", round(ours, 4))
    put("p3.ideal_logdensity_sklearn", round(float(kde.score_samples(q)[0]), 4))
    b = cosine_beta_schedule(100); ab = np.cumprod(1 - b)
    for t in (99, 50, 0):
        put(f"p3.noised_ideal_var_t{t}_sig0.05", float(f"{1 - ab[t] + ab[t] * 0.05 ** 2:.4g}"))
    put("p3.grid_cell_area", float(f"{(2.2 / 240) ** 2:.4g}"))
    put("p3.grid_points", 241 * 241)
    from rebuild import ideal, kinds as KK
    # step 2: the mixture score Sum_i r_i (sqrt(abar) x_i - x) / v, against autograd through logq, at t = 50
    X = np.random.default_rng(3).normal(size=(5, 2)); t = 50
    x = torch.tensor([[0.2, -0.1]], requires_grad=True)
    ideal.logq(x, torch.tensor(X, dtype=torch.float32), t).backward()
    a = ideal.ABAR[t]; v = 1 - a + a * 0.05 ** 2; mu = np.sqrt(a) * X; xx = x.detach().numpy()[0]
    lphi = -((xx - mu) ** 2).sum(1) / (2 * v); r = np.exp(lphi - lphi.max()); r /= r.sum()
    put("p3.mixture_score_autograd", np.round(x.grad.numpy()[0].astype(float), 4).tolist())
    put("p3.mixture_score_formula", np.round((r[:, None] * (mu - xx) / v).sum(0), 4).tolist())
    # step 8: the gradient of logaddexp(a, b) is r_A s_A + r_B s_B, not s_A + s_B
    rng0 = np.random.default_rng(3); rng0.normal(size=(5, 2))
    y = torch.tensor([[0.1, 0.3]], requires_grad=True)
    A = torch.tensor(rng0.normal(size=(4, 2)), dtype=torch.float32); B = torch.tensor(rng0.normal(size=(4, 2)) + 1, dtype=torch.float32)
    la, lb = ideal.logq(y, A, t), ideal.logq(y, B, t)
    g = torch.autograd.grad(torch.logaddexp(la, lb).sum(), y)[0]
    ga = torch.autograd.grad(ideal.logq(y, A, t).sum(), y)[0]; gb = torch.autograd.grad(ideal.logq(y, B, t).sum(), y)[0]
    rA = torch.sigmoid(la - lb).item()
    put("p3.logaddexp_grad", np.round(g.numpy()[0].astype(float), 4).tolist())
    put("p3.logaddexp_responsibility_weighted", np.round((rA * ga + (1 - rA) * gb).numpy()[0].astype(float), 4).tolist())
    put("p3.logaddexp_plain_sum", np.round((ga + gb).numpy()[0].astype(float), 4).tolist())
    # the seven kinds, built as phase3_kinds builds them, with exact grids on the CPU
    rng = np.random.default_rng(0); torch.manual_seed(0); cpu = torch.device("cpu")
    KINDS = KK.build_kinds(rng)
    for k in KINDS:
        k["p"] = ideal.exact_grid(*(torch.tensor(d, dtype=torch.float32) for d in k["data"]), k["compose"], cpu)
    # step 4: the perfect-sample baseline is random; 200 redraws of 2000 perfect samples per kind
    r2 = np.random.default_rng(1); spread = {}
    for k in KINDS:
        tvs = np.array([ideal.tv(ideal.perfect_samples(k["p"], 2000, r2), k["p"]) for _ in range(200)])
        spread[k["name"]] = dict(mean=round(float(tvs.mean()), 4), sd=round(float(tvs.std()), 4),
                                 min=round(float(tvs.min()), 4), max=round(float(tvs.max()), 4))
    put("p3.perfect_tv_200_redraws", spread)
    sds = [s["sd"] for s in spread.values()]
    put("p3.perfect_tv_200_redraws_sd_range", [min(sds), max(sds)])
    # step 5: intersection's mass near the two in-bar modes (0, +-0.5), by ball radius, and the upper mode's spread
    pi = KINDS[0]["p"] * ideal.CELL
    for rad in (0.08, 0.12, 0.25):
        put(f"p3.intersection_mass_within_{rad}", round(float((pi * KK.near(ideal.X1, ideal.X2, KK.RING[[2, 6]], rad)).sum()), 4))
    pu = pi * (ideal.X2 > 0); pu = pu / pu.sum(); m2 = (pu * ideal.X2).sum()
    put("p3.intersection_upper_mode_sd_along_bar", round(float(np.sqrt((pu * (ideal.X2 - m2) ** 2).sum())), 4))
    # step 12: dominance, ring x blob; the inward pull and the unequal mode weights
    ring, blob = KINDS[4]["data"]; pd = KINDS[4]["p"] * ideal.CELL
    counts = np.bincount((np.round(np.arctan2(ring[:, 1], ring[:, 0]) / (2 * np.pi / 8)) % 8).astype(int), minlength=8)
    put("p3.dominance_ring_points_per_mode", counts.tolist())
    lblob = ideal.logq(torch.tensor(KK.RING, dtype=torch.float32), torch.tensor(blob, dtype=torch.float32), -1).numpy()
    put("p3.dominance_blob_logdensity_at_modes", np.round(lblob.astype(float), 2).tolist())
    sR2 = 0.03 ** 2 + 0.05 ** 2
    wpred = np.array([counts[j] * np.mean(np.exp(-((blob - KK.RING[j]) ** 2).sum(1) / (2 * (sR2 + 0.05 ** 2)))) for j in range(8)])
    put("p3.dominance_predicted_mode_weights", np.round(wpred / wpred.sum(), 3).tolist())
    put("p3.dominance_exact_mode_mass_within_0.25",
        [round(float((pd * KK.near(ideal.X1, ideal.X2, KK.RING[[j]], 0.25)).sum()), 3) for j in range(8)])
    shrink = 0.0125 / (0.0034 + 0.0125)
    put("p3.dominance_shrink_factor", round(shrink, 3)); put("p3.dominance_shrunk_radius", round(0.5 * shrink, 3))
    put("p3.dominance_exact_mean_radius", round(float((pd * np.hypot(ideal.X1, ideal.X2)).sum()), 4))


def phase4():
    # the parts world's 16 pixels: cat tight (variance 0.01) on the head pixels 0-7, dog tight on the body 8-15
    D = 16; HEAD, BODY = np.arange(8), np.arange(8, 16)
    vcat = np.full(D, 1.0); vcat[HEAD] = 0.01
    lp = lambda x, m, v: float(-0.5 * (((x - m) ** 2) / v + np.log(2 * np.pi * v)).sum())
    one = np.ones(D)
    # step 2: the cat expert's log-density of the all-cat image against the all-dog image, head and body parts
    put("p4.cat_expert_cat_minus_dog_image_nats", lp(one, one, vcat) - lp(-one, one, vcat))
    put("p4.cat_expert_gap_head_part", 8 * 2 ** 2 / (2 * 0.01)); put("p4.cat_expert_gap_body_part", 8 * 2 ** 2 / 2.0)
    # step 5: example 9(c), the side-by-side and same-half pair weights, and what float64 does to them
    lside = math.log(0.25) + 16 * (-0.5 * (1 / 1.01 + math.log(2 * math.pi * 1.01)))
    lsame = math.log(0.25) + 8 * (-0.5 * (4 / 0.02 + math.log(2 * math.pi * 0.02))) + 8 * (-0.5 * math.log(4 * math.pi))
    delta = lside - lsame
    put("p4.logw_side_by_side", round(lside, 4)); put("p4.logw_same_half", round(lsame, 4))
    put("p4.delta_side_minus_same", round(delta, 4))
    def sci(log10v):   # a number below float64's range, written from its base-10 logarithm
        e = math.floor(log10v); return f"{10 ** (log10v - e):.2f}e{e}"
    put("p4.exp_minus_delta", sci(-delta / math.log(10)))
    put("p4.exp_minus_delta_float64", math.exp(-delta))
    # w_same = e^-Delta / (2 (1 + e^-Delta)); e^-Delta is so small that 1 + e^-Delta is 1
    put("p4.w_same_half_exact", sci(-delta / math.log(10) - math.log10(2)))
    put("p4.w_side_by_side_float64", 1 / (2 * (1 + math.exp(-delta))))
    put("p4.float64_smallest_positive", float(np.nextafter(0, 1)))


def phase5():
    # the 0% product near the corner: Phi(-x/s) Phi(x/s) is symmetric in x, so each quadrant gets the same share
    s = 0.05
    xs = np.linspace(-0.3, 0.3, 2001)
    f = norm.cdf(-xs / s) * norm.cdf(xs / s)
    put("p5.corner_profile_peak_at_0", round(float(f[1000]), 4))
    put("p5.corner_profile_half_mass_within", round(float(np.interp(0.5, np.cumsum(f[1000:]) / f[1000:].sum(), xs[1000:])), 4))
    put("p5.overlap_points_of_800_at_1pct", int(0.01 * 800)); put("p5.overlap_points_of_800_at_5pct", int(0.05 * 800))
    # the parts-world flip: matching pair against the ghost pair, empty background, f = 0.001
    f_ = 0.001
    mismatch = 16 * 1.0 ** 2 / (2 * 0.02)
    weights = 2 * math.log((f_ / 2) / ((1 - f_) / 2))
    put("p5.ghost_mismatch_nats", mismatch); put("p5.overlap_weight_cost_nats", round(weights, 2))
    put("p5.matching_pair_advantage_nats", round(mismatch + weights, 1))
    put("p5.interior_area_share_margin0.15", round(0.85 ** 2, 4))
    # step 4: the deep mass averaged over the data draw, M_D(f) = f^2 C_D / (f^2 C + f(1-f) C_1 + (1-f)^2 C_0),
    # with the unit squares smoothed by sigma = 0.05 and integrated on a 1601-point grid over [-1.6, 1.6] per axis
    g = np.linspace(-1.6, 1.6, 1601); d = g[1] - g[0]
    sq = lambda lo, hi: norm.cdf((g - lo) / s) - norm.cdf((g - hi) / s)
    lo_, hi_ = sq(-1, 0), sq(0, 1)                 # the same two profiles serve x1 (left, right) and x2 (bottom, top)
    I = lambda h, m=None: float((h if m is None else h * m).sum() * d)
    x1in, x2in = g < -0.15, g > 0.15
    CD = I(lo_ * lo_, x1in) * I(hi_ * hi_, x2in); C = I(lo_ * lo_) * I(hi_ * hi_)
    C1 = I(lo_ * lo_) * I(lo_ * hi_) + I(lo_ * hi_) * I(hi_ * hi_); C0 = I(lo_ * hi_) * I(lo_ * hi_)
    put("p5.deep_mass_C0_corner", float(f"{C0:.3g}")); put("p5.deep_mass_C1_edges", round(C1, 4))
    put("p5.deep_mass_C_square", round(C, 4)); put("p5.deep_mass_CD_deep_square", round(CD, 4))
    MD = lambda f: f * f * CD / (f * f * C + f * (1 - f) * C1 + (1 - f) ** 2 * C0)
    put("p5.deep_mass_formula_f", [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    put("p5.deep_mass_formula_MD", [round(MD(f), 3) for f in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]])
    put("p5.deep_mass_formula_numerator_f0.05", round(0.05 ** 2 * CD, 4))
    put("p5.deep_mass_formula_denominator_f0.05", round(0.05 ** 2 * C + 0.05 * 0.95 * C1 + 0.95 ** 2 * C0, 4))
    put("p5.phi_minus_3", round(float(norm.cdf(-3)), 4))
    # the answer key's 0.5 is a property of adjacent leftover: leftover on the top-right alone gives almost all
    # of the product to A and B's shared quadrant (blobs of std 0.15 at (+-0.5, +-0.5), 441 x 441 grid)
    S_ = 0.15; gg = np.linspace(-1.1, 1.1, 441); X1, X2 = np.meshgrid(gg, gg)
    blob = {q: np.exp(-((X1 - m[0]) ** 2 + (X2 - m[1]) ** 2) / (2 * S_ * S_)) / (2 * np.pi * S_ * S_)
            for q, m in {"TR": (0.5, 0.5), "BR": (0.5, -0.5), "TL": (-0.5, 0.5)}.items()}
    near_tr = (X1 - 0.5) ** 2 + (X2 - 0.5) ** 2 < (3 * S_) ** 2
    for eps in (0.1, 0.01):
        p = ((1 - eps) * blob["BR"] + eps * blob["TR"]) * ((1 - eps) * blob["TL"] + eps * blob["TR"])
        put(f"p5.leftover_top_right_only_eps{eps}_share", round(float((p * near_tr).sum() / p.sum()), 4))
    # the three constructions' closed forms, and the by-hand Z of two blobs of std 0.15
    put("p5.mass_within_3sd_2d", round(1 - math.exp(-4.5), 4)); put("p5.mass_within_3sd_2d_half_variance", round(1 - math.exp(-9), 4))
    pre = 1 / (4 * math.pi * S_ ** 2)
    put("p5.Z_prefactor_sd0.15", round(pre, 5))
    put("p5.Z_next_door_by_hand", float(f"{pre * math.exp(-1 / (4 * S_ ** 2)):.3g}"))
    put("p5.Z_opposite_by_hand", float(f"{pre * math.exp(-2 / (4 * S_ ** 2)):.3g}"))


if __name__ == "__main__":
    for f in (phase1, phase2, phase3, phase4, phase5):
        print(f"--- {f.__name__}")
        f()
    json.dump(K, open("rebuild/out/walk-numbers.json", "w"), indent=1)
