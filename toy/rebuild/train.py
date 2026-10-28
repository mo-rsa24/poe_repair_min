"""Training loop of the notebook's cells 32, 33, 37, 38, 42 and 43: Adam 1e-3, batch 1000, EMA 0.999, fresh data each step.

    python -m rebuild.train --panel "bar(scale=0.2)" --out rebuild/out/phase1/ckpt
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from rebuild.datasets import PANELS
from rebuild.device import describe, pick_device
from rebuild.model import make_model


def train(sample_fn, steps=15001, batch=1000, lr=1e-3, ema=0.999, seed=0, device="cpu", model=None,
          log_every=1000, use_ema=True):
    """sample_fn(n, rng=rng) -> (n, D) array. Returns (ema or raw model, list of (step, loss))."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = (model or make_model(data_dim=sample_fn(2, rng=rng).shape[1])).to(device)
    ema_model = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    params, ema_params = list(model.parameters()), list(ema_model.parameters())
    log = []
    for it in range(steps):
        x = torch.tensor(sample_fn(batch, rng=rng), dtype=torch.float32, device=device)
        loss = model.loss(x).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            torch._foreach_mul_(ema_params, ema)
            torch._foreach_add_(ema_params, params, alpha=1 - ema)
        if it % 100 == 0:   # reading the loss syncs the GPU, so only every 100 steps
            log.append((it, float(loss)))
            if log_every and it % log_every == 0:
                print(it, f"{log[-1][1]:.4f}", flush=True)
    return (ema_model if use_ema else model), log


def slug(panel):
    return "".join(c if c.isalnum() else "_" for c in panel).strip("_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=list(PANELS))
    ap.add_argument("--steps", type=int, default=15001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rebuild/out/phase1/ckpt")
    a = ap.parse_args()
    dev = pick_device()
    print(describe(dev), "| panel", a.panel, flush=True)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    model, log = train(PANELS[a.panel], steps=a.steps, seed=a.seed, device=dev)
    torch.save(model.state_dict(), os.path.join(a.out, slug(a.panel) + ".pt"))
    json.dump(dict(panel=a.panel, steps=a.steps, seed=a.seed, seconds=time.time() - t0, loss=log),
              open(os.path.join(a.out, slug(a.panel) + ".json"), "w"))
    print(f"done in {time.time() - t0:.0f} s", flush=True)
