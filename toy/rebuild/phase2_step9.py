"""Phase 2, step 9 of the walk: what the annealed samplers end on, and what a clean final level adds.

    python -m rebuild.phase2_step9      # prints the shares and writes rebuild/out/phase2/step9.json

Step 9 of 02-composing-and-sampling.md anneals ULA, MALA and UHA through pi_99, ..., pi_0, where pi_t is the
product of the two ideal experts noised at code step t. The schedule stops at abar_0 = 0.999369, so pi_0 is
the clean composition smoothed a little further (component variance nu_0 = 0.00313 against the clean 0.0025).
This module reproduces the step's run exactly (it executes the step file's python blocks 0 to 10 first, so the
random streams are the ones walk02.py has when it reaches step 9) and adds:

  - the exact in-bar share under pi_0 and under the clean composition, and how the clean composition's mass
    splits between the two in-bar centres, the other six ring centres and everything farther away;
  - each annealed sampler with one more level after the anneal: the same 10 moves on the clean composition
    ("the target itself", logq0 in the step), with the step size the step's rule gives at abar = 1, that is
    nu = SIG ** 2: 0.05 * SIG ** 2 for ULA and MALA, h = 0.3 * SIG for UHA. The clean level continues each
    chain from the torch random state its anneal left, and the state is then put back, so every later
    sampler in the step draws exactly what it draws on the page.

phase2.json is a list of rows, one per trained composition, so these numbers go in a file of their own.
"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP_FILE = os.path.join(REPO, "artifacts/drips/simple-distributions-rebuild/02-composing-and-sampling.md")
OUT = os.path.join(REPO, "rebuild/out/phase2/step9.json")


def walk_state():
    """Execute the step file's python blocks before step 9's, quietly, and return their namespace."""
    from rebuild.run_walk_doc import blocks
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import anneal_samplers  # noqa: F401  (imported before the chdir so the blocks find it)
    bs = blocks(STEP_FILE)
    i9 = [i for i, b in enumerate(bs) if b.startswith("SIG = 0.05; V = ")]
    assert len(i9) == 1, "step 9's block not found"
    ns = {"__name__": "__walk__"}
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        os.chdir(tmp)
        try:
            exec(compile("\n".join(bs[:i9[0]]), "walk02-blocks-0-to-10", "exec"), ns)
        finally:
            os.chdir(here)
    return ns


def compute(samplers=True):
    from anneal_samplers import AnnealedMALASampler, AnnealedULASampler, AnnealedUHASampler
    ns = walk_state()
    rng, T, betas, abar = ns["rng"], ns["T"], ns["betas"], ns["abar"]
    # step 9's block, line for line, up to the data and the composed score
    SIG = 0.05; V = np.array([1 - abar[t] + abar[t] * SIG ** 2 for t in range(T - 1, -1, -1)])
    RING = 0.5 * np.c_[np.cos(2 * np.pi * np.arange(8) / 8), np.sin(2 * np.pi * np.arange(8) / 8)]
    ring = torch.tensor(RING[rng.integers(0, 8, 400)] + 0.03 * rng.standard_normal((400, 2)), dtype=torch.float32)
    barpts = torch.tensor(np.c_[rng.uniform(-0.2, 0.2, 400), rng.uniform(-1, 1, 400)], dtype=torch.float32)

    def logq(x, data, t):
        a = abar[t]; v = 1 - a + a * SIG ** 2
        return torch.logsumexp(-torch.cdist(x, math.sqrt(a) * data) ** 2 / (2 * v), 1)

    def both(x, t):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True); lp = logq(x, ring, t) + logq(x, barpts, t)
            return lp.detach(), torch.autograd.grad(lp.sum(), x)[0]

    def logq0(x, data):   # the experts with no diffusion noise: the target itself
        return torch.logsumexp(-torch.cdist(x, data) ** 2 / (2 * SIG ** 2), 1)

    def clean(x, t=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True); lp = logq0(x, ring) + logq0(x, barpts)
            return lp.detach(), torch.autograd.grad(lp.sum(), x)[0]

    d_in = lambda s: np.minimum(np.hypot(s[:, 0], s[:, 1] - 0.5), np.hypot(s[:, 0], s[:, 1] + 0.5))
    in_bar = lambda s: float((d_in(s) < 0.1).mean())
    R = {}

    # the exact shares on step 9's grid: the clean composition (what the step prints as "exact") and pi_0
    G = 241; gx = np.linspace(-1.1, 1.1, G); X1, X2 = np.meshgrid(gx, gx)
    g = torch.tensor(np.c_[X1.ravel(), X2.ravel()], dtype=torch.float32)
    pts = np.c_[X1.ravel(), X2.ravel()]
    norm_w = lambda lp: (lambda w: w / w.sum())(np.exp(lp - lp.max()))
    w = norm_w((logq0(g, ring) + logq0(g, barpts)).numpy())
    w0 = norm_w((logq(g, ring, 0) + logq(g, barpts, 0)).numpy())
    near = d_in(pts) < 0.1
    others = RING[np.abs(RING[:, 0]) > 1e-9]          # the six ring centres outside the bar
    d_out = np.min(np.hypot(pts[:, None, 0] - others[None, :, 0], pts[:, None, 1] - others[None, :, 1]), 1)
    R["exact_share_clean"] = float((w * near).sum())
    R["exact_share_pi0"] = float((w0 * near).sum())
    R["abar_0"] = float(abar[0]); R["nu_0"] = float(V[-1]); R["nu_clean"] = SIG ** 2
    R["clean_mass_within"] = {
        "in_bar_0.1": float((w * (d_in(pts) < 0.1)).sum()),
        "in_bar_0.15": float((w * (d_in(pts) < 0.15)).sum()),
        "in_bar_0.2": float((w * (d_in(pts) < 0.2)).sum()),
        "other_six_0.2": float((w * (d_out < 0.2)).sum()),
        "beyond_0.2_of_every_centre": float((w * ((d_in(pts) >= 0.2) & (d_out >= 0.2))).sum()),
    }
    if not samplers:
        return R

    def run(sampler, final, n=1000):
        """Step 9's run, then one clean level of 10 moves on the same chain; the torch state is restored."""
        x = torch.randn(n, 2)
        for i, t in enumerate(range(T - 1, -1, -1)): x = sampler.sample_step(x, i, t, {})
        state = torch.get_rng_state()
        xc = final.sample_step(x.clone(), 0, None, {})
        torch.set_rng_state(state)
        return x.numpy(), xc.numpy()

    ss_clean, h_clean = 0.05 * SIG ** 2, 0.3 * SIG
    pairs = {
        "ULA": run(AnnealedULASampler(T, 10, list(0.05 * V), lambda x, t: both(x, t)[1]),
                   AnnealedULASampler(1, 10, [ss_clean], lambda x, t: clean(x)[1])),
        "MALA": run(AnnealedMALASampler(T, 10, list(0.05 * V), both),
                    AnnealedMALASampler(1, 10, [ss_clean], clean)),
        "UHA": run(AnnealedUHASampler(T, 10, list(0.3 * np.sqrt(V)), 0.5, [torch.ones(1)] * T, 3, lambda x, t: both(x, t)[1]),
                   AnnealedUHASampler(1, 10, [h_clean], 0.5, [torch.ones(1)], 3, lambda x, t: clean(x)[1])),
    }
    x = torch.randn(1000, 2)
    for t in range(T - 1, -1, -1):
        s_ = both(x, t)[1]; var = betas[t] * (1 - (abar[t - 1] if t > 0 else 1.0)) / (1 - abar[t])
        x = (x + betas[t] * s_) / math.sqrt(1 - betas[t]) + (math.sqrt(var) * torch.randn_like(x) if t > 0 else 0)
    perfect = pts[rng.choice(len(w), 1000, p=w)]
    R["clean_level"] = dict(moves=10, ula_mala_step=ss_clean, uha_leapfrog_step=h_clean, uha_damping=0.5, uha_leapfrog_steps=3)
    R["share_perfect_1000"] = in_bar(perfect)
    R["share_reverse_diffusion"] = in_bar(x.numpy())
    R["share_annealed"] = {k: in_bar(a) for k, (a, _) in pairs.items()}
    R["share_annealed_then_clean_level"] = {k: in_bar(c) for k, (_, c) in pairs.items()}
    return R


if __name__ == "__main__":
    R = compute()
    print(f"exact share within 0.1 of the two in-bar centres: clean composition {R['exact_share_clean']:.3f}, "
          f"pi_0 {R['exact_share_pi0']:.3f} (abar_0 {R['abar_0']:.6f}, nu_0 {R['nu_0']:.6f}, clean {R['nu_clean']})")
    print("clean composition's mass:", {k: round(v, 4) for k, v in R["clean_mass_within"].items()})
    print(f"perfect, 1000 draws {R['share_perfect_1000']:.3f}; reverse diffusion {R['share_reverse_diffusion']:.3f}")
    for k in R["share_annealed"]:
        print(f"{k:>5}: annealed to pi_0 {R['share_annealed'][k]:.3f}  (read against {R['exact_share_pi0']:.3f}); "
              f"plus a clean level {R['share_annealed_then_clean_level'][k]:.3f}  (read against {R['exact_share_clean']:.3f})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(R, open(OUT, "w"), indent=1)
