"""Phase 5, fine-tune variant on the quadrant toy: does pretraining on all placements leave A∩B support behind?

    python -m rebuild.phase5_finetune --out rebuild/out/phase5

One energy MLP (the notebook's architecture) is pretrained on all four quadrants, uniform on [-1, 1]^2, for
10,000 steps (Adam 1e-3, EMA 0.999). Two copies are fine-tuned at 0% overlap: A on the bottom-left square,
B on the top-right square (Adam 1e-4, batch 1000, raw weights). At each checkpoint the product of the two
fine-tuned experts is read three ways: its exact composition on the grid (the summed energies at t = 0,
normalised), tuned MALA and UHA through all 100 levels. The number is the A∩B mass, at least 0.15 inside the
top-left quadrant and in the whole quadrant. Baseline: two experts trained from scratch on the same 0% data
for 10,000 steps.
"""
import argparse
import copy
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rebuild import ideal, samplers as S
from rebuild.compose import ProductEBMDiffusionModel
from rebuild.device import describe, pick_device
from rebuild.model import PortableDiffusionModel, make_model
from rebuild.phase2_compose import trained_target
from rebuild.phase5_dial import MARGIN, NS
from rebuild.train import train

CHECKPOINTS = [0, 10, 30, 100, 300, 1000, 3000, 10000]
all_quadrants = lambda n, rng=None: rng.uniform(-1, 1, (n, 2))
a_only = lambda n, rng=None: rng.uniform(-1, 0, (n, 2))          # A at 0% overlap: bottom-left
b_only = lambda n, rng=None: np.c_[rng.uniform(0, 1, n), rng.uniform(0, 1, n)]   # B at 0% overlap: top-right


def ab_mass(s):
    return dict(interior=float(((s[:, 0] < -MARGIN) & (s[:, 1] > MARGIN)).mean()),
                quadrant=float(((s[:, 0] < 0) & (s[:, 1] > 0)).mean()))


def evaluate(mA, mB, dev, tag):
    ddpm = PortableDiffusionModel(2, 100, ProductEBMDiffusionModel(mA.net, mB.net)).to(dev)
    tgt = trained_target(ddpm, dev)
    p = ideal.grid_density(lambda g: tgt.logp(g, 0), dev)
    w = p * ideal.CELL
    row = dict(exact=dict(interior=float((w * ((ideal.X1 < -MARGIN) & (ideal.X2 > MARGIN))).sum()),
                          quadrant=float((w * ((ideal.X1 < 0) & (ideal.X2 > 0))).sum())))
    smp = {"MALA, tuned": S.tuned_mala(tgt, NS, 2, ideal.V, device=dev)[0].cpu().numpy(),
           "UHA": S.uha(tgt, NS, 2, 0.3 * np.sqrt(ideal.V), device=dev).cpu().numpy()}
    for c, s in smp.items():
        s = s[np.isfinite(s).all(1)]
        row[c] = dict(ab_mass(s), tv=float(ideal.tv(s, p)), kept=len(s))
    print(f"{tag}: exact interior {row['exact']['interior']:.3f} quadrant {row['exact']['quadrant']:.3f} | "
          + " | ".join(f"{c} interior {row[c]['interior']:.3f} quadrant {row[c]['quadrant']:.3f} TV {row[c]['tv']:.2f}" for c in smp), flush=True)
    return row, p, smp


def finetune(pre, dev, steps_list, lr=1e-4, batch=1000, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    # train() returns its EMA copy with gradients switched off, so the copies are switched back on here
    mA, mB = (copy.deepcopy(pre).requires_grad_(True).train() for _ in range(2))
    start = [p.detach().clone() for p in mA.parameters()]
    oA, oB = torch.optim.Adam(mA.parameters(), lr=lr), torch.optim.Adam(mB.parameters(), lr=lr)
    done = 0
    for target in steps_list:
        while done < target:
            for m, o, fn in [(mA, oA, a_only), (mB, oB, b_only)]:
                x = torch.tensor(fn(batch, rng=rng), dtype=torch.float32, device=dev)
                loss = m.loss(x).mean(); o.zero_grad(set_to_none=True); loss.backward(); o.step()
            done += 1
        moved = sum(float((p.detach() - s).norm() ** 2) for p, s in zip(mA.parameters(), start)) ** 0.5
        print(f"fine-tune step {target}: expert A's weights moved {moved:.4f} (L2) from the pretrained ones", flush=True)
        yield target, mA, mB


def main(out):
    os.makedirs(out, exist_ok=True)
    dev = pick_device(); print(describe(dev), flush=True)
    path = os.path.join(out, "pretrained-all-quadrants.pt")
    if os.path.exists(path):
        pre = make_model().to(dev); pre.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
        print("loaded", path, flush=True)
    else:
        pre, _ = train(all_quadrants, steps=10000, seed=0, device=dev, log_every=2000)
        torch.save(pre.state_dict(), path)
    rows, pics = [], {}
    for k, mA, mB in finetune(pre, dev, CHECKPOINTS):
        row, p, smp = evaluate(mA, mB, dev, f"fine-tune step {k}")
        row["steps"] = k; rows.append(row)
        if k in (0, 300, 10000):
            pics[k] = (p, smp)
    sA, _ = train(a_only, steps=10000, seed=1, device=dev, log_every=0)
    sB, _ = train(b_only, steps=10000, seed=2, device=dev, log_every=0)
    scratch, p, smp = evaluate(sA, sB, dev, "from scratch, 10000 steps")
    pics["scratch"] = (p, smp)
    json.dump(dict(checkpoints=rows, from_scratch=scratch, lr=1e-4, pretrain_steps=10000),
              open(os.path.join(out, "finetune.json"), "w"), indent=1, default=float)

    colors = {"exact": "k", "MALA, tuned": "tab:orange", "UHA": "tab:green"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    xs = np.arange(len(CHECKPOINTS))
    for a, key in zip(ax, ["interior", "quadrant"]):
        for c in ["exact", "MALA, tuned", "UHA"]:
            a.plot(xs, [r[c][key] for r in rows], marker="o", c=colors[c], label=c if c != "exact" else "exact composition of the fine-tuned experts")
            a.axhline(scratch[c][key], c=colors[c], ls=":", lw=1)
        a.set_xticks(xs); a.set_xticklabels([str(k) for k in CHECKPOINTS])
        a.set_xlabel("fine-tune steps at 0% overlap (dotted: experts trained from scratch on the same data)")
        a.set_ylabel("share of the product's mass")
        a.set_title("A∩B mass " + ("at least 0.15 inside the top-left quadrant" if key == "interior" else "in the whole top-left quadrant"), fontsize=9)
        a.set_ylim(-0.02, 0.5); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(out, "finetune-curves.png"), dpi=100); plt.close()

    keys = [0, 300, 10000, "scratch"]
    fig, ax = plt.subplots(len(keys), 3, figsize=(7.5, 2.5 * len(keys)))
    for r, k in enumerate(keys):
        p, smp = pics[k]
        ax[r, 0].imshow(p, origin="lower", extent=[-ideal.L, ideal.L, -ideal.L, ideal.L], cmap="magma")
        ax[r, 0].set_ylabel(f"fine-tune {k}" if k != "scratch" else "from scratch")
        ax[r, 0].set_title("exact, trained experts", fontsize=8)
        for j, c in enumerate(smp, start=1):
            ax[r, j].scatter(*smp[c].T, s=1, c="k", alpha=.4); ax[r, j].set_title(c, fontsize=8)
        for a in ax[r]:
            a.set_xlim(-ideal.L, ideal.L); a.set_ylim(-ideal.L, ideal.L); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            a.axhline(0, c="grey", lw=.4); a.axvline(0, c="grey", lw=.4)
    plt.tight_layout(); plt.savefig(os.path.join(out, "finetune-pictures.png"), dpi=90); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rebuild/out/phase5")
    main(ap.parse_args().out)
