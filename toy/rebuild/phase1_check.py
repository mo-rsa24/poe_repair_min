"""Phase 1 check: each trained model's reverse-diffusion samples against its dataset.

    python -m rebuild.phase1_check --ckpt rebuild/out/phase1/ckpt --out rebuild/out/phase1

Draws a figure laid out like datasets-1000-samples-each.png, with the model's 1000 samples under each
dataset panel, and prints TV between 5000 model samples and 5000 fresh data samples on a 40 x 40 binning of
[-1.5, 1.5]^2. The floor is TV between two independent 5000-sample draws of the data. A panel matches when
its TV is within 0.10 of the floor.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rebuild.datasets import PANELS
from rebuild.device import describe, pick_device
from rebuild.model import make_model
from rebuild.train import slug

EDGES = np.linspace(-1.5, 1.5, 41)
TOL = 0.10


def tv_samples(a, b):
    ha = np.histogram2d(a[:, 0], a[:, 1], [EDGES, EDGES])[0] / len(a)
    hb = np.histogram2d(b[:, 0], b[:, 1], [EDGES, EDGES])[0] / len(b)
    return 0.5 * np.abs(ha - hb).sum()


def sample_last_mean(m, n, dev):
    """The notebook's reverse diffusion, but the last step returns its mean. The notebook adds noise at t = 0 too,
    with sd sqrt(beta_0) = 0.025 on the cosine schedule, which widens modes as narrow as the ring's 0.03."""
    x = torch.randn(n, m.dim, device=dev)
    for j in range(m.n_steps - 1, 0, -1):
        x = m.p_sample(x, torch.full((n,), j, dtype=torch.long, device=dev))
    t0 = torch.zeros(n, dtype=torch.long, device=dev)
    x0 = m._ex(m.sqrt_recip_ac, t0) * x - m._ex(m.sqrt_recipm1_ac, t0) * m.net(x, t0, create_graph=False)
    return m._ex(m.post_c1, t0) * x0 + m._ex(m.post_c2, t0) * x


def load(path, dev):
    m = make_model().to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    return m.eval()


def main(ckpt, out):
    dev = pick_device(); print(describe(dev))
    torch.manual_seed(1); rng = np.random.default_rng(1)
    rows, fig = [], plt.figure(figsize=(16.5, 13))
    fig.suptitle("1000 samples per dataset (rows 1, 3) and from each trained PyTorch model by reverse diffusion (rows 2, 4)")
    for j, (name, fn) in enumerate(PANELS.items()):
        m = load(os.path.join(ckpt, slug(name) + ".pt"), dev)
        with torch.no_grad():
            ms = m.sample(5000, dev).cpu().numpy()
            mm = sample_last_mean(m, 5000, dev).cpu().numpy()
        d1, d2 = fn(5000, rng=rng), fn(5000, rng=rng)
        tv, tv_mean, floor = tv_samples(ms, d1), tv_samples(mm, d1), tv_samples(d2, d1)
        ok = tv - floor <= TOL
        ok_mean = tv_mean - floor <= TOL
        mk = "matches" if ok else ("matches without the last-step noise" if ok_mean else "differs")
        rows.append(dict(panel=name, tv=tv, tv_last_step_mean=tv_mean, floor=floor, mark=mk))
        print(f"{name:>18}: TV model {tv:.3f} (last-step mean {tv_mean:.3f})  floor {floor:.3f}  gap {tv - floor:+.3f}  {mk}")
        r, c = 2 * (j // 5), j % 5
        for rr, pts, title, col in [(r, d1[:1000], name, "tab:blue"), (r + 1, ms[:1000], f"model, TV {tv:.2f} (floor {floor:.2f})", "tab:orange")]:
            a = fig.add_subplot(4, 5, rr * 5 + c + 1)
            a.scatter(pts[:, 0], pts[:, 1], s=2, c=col)
            a.set_xlim(-1.1, 1.1); a.set_ylim(-1.1, 1.1); a.set_title(title, fontsize=9); a.set_aspect("equal")
    plt.tight_layout(); plt.savefig(os.path.join(out, "datasets-against-trained-models.png"), dpi=90)
    json.dump(dict(tolerance=TOL, rows=rows), open(os.path.join(out, "phase1.json"), "w"), indent=1)
    print("all match" if all(r["mark"].startswith("matches") for r in rows) else "some differ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="rebuild/out/phase1/ckpt")
    ap.add_argument("--out", default="rebuild/out/phase1")
    a = ap.parse_args()
    main(a.ckpt, a.out)
