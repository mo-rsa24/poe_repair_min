"""Do per-concept functions transfer to a pair they were never fitted on?

The in-sample floor said the best per-concept additive fit leaves 3.1% of the correction
at step 0. That was fitted on every pair at once. This asks the question that actually
licenses the approach: fit the per-concept functions on 80% of the pairs, then predict the
held-out 20% from their two concepts alone.

Reported per seed, five folds:

    in_sample   residual fraction of the correction, fitted on the training folds
    held_out    same fraction on the held-out pairs, predicted not fitted
    poe_base    1.0 by definition, the plain composition's own error

A held-out value near the in-sample one means the functions are per-concept and transfer.
A held-out value near 1.0 means the in-sample fit was memorising pairs.
"""
from __future__ import annotations
import collections, json, os, time
import numpy as np
import torch

ROOT = "/datasets/mmolefe/poe_repair_min/outputs/training_cache/train"
OUT = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/additive_transfer_step0.json"
FOLDS = 5


def load(seed):
    slugs = [p for p in sorted(os.listdir(ROOT))
             if os.path.exists(os.path.join(ROOT, p, seed, "residuals", "step_000.pt"))]
    concepts, rows, H, HP = {}, [], [], []
    t0 = time.time()
    for n, s in enumerate(slugs):
        a, b = s.split("__x__")
        for c in (a, b):
            concepts.setdefault(c, len(concepts))
        d = torch.load(os.path.join(ROOT, s, seed, "residuals", "step_000.pt"),
                       map_location="cpu", weights_only=False)
        eu = d["eps_uncond"].float().flatten()
        H.append((d["eps_j_raw"].float().flatten() - eu).numpy())
        HP.append((d["eps_a_raw"].float().flatten()
                   + d["eps_b_raw"].float().flatten() - 2.0 * eu).numpy())
        rows.append((concepts[a], concepts[b]))
        if n % 250 == 0:
            print(f"  [{seed}] {n}/{len(slugs)} {time.time()-t0:.0f}s", flush=True)
    return slugs, concepts, rows, np.stack(H).astype(np.float32), np.stack(HP).astype(np.float32)


def design(rows, N):
    A = np.zeros((len(rows), N), dtype=np.float64)
    for k, (i, j) in enumerate(rows):
        A[k, i] += 1.0
        A[k, j] += 1.0
    return A


def run(seed):
    slugs, concepts, rows, H, HP = load(seed)
    P, N = len(rows), len(concepts)
    A = design(rows, N)
    rng = np.random.default_rng(0)
    fold = rng.permutation(P) % FOLDS
    out = {"seed": seed, "pairs": P, "concepts": N, "folds": []}
    for f in range(FOLDS):
        te = fold == f
        tr = ~te
        Atr = A[tr]
        seen = Atr.sum(axis=0) > 0
        # a held-out pair is only predictable if both its concepts appear in the training folds
        ok = np.array([seen[i] and seen[j] for i, j in rows]) & te
        AtA = Atr.T @ Atr
        AtA += 1e-6 * np.trace(AtA) / N * np.eye(N)
        F = np.linalg.solve(AtA, Atr.T @ H[tr].astype(np.float64))
        pred_tr = Atr @ F
        pred_te = A[ok] @ F
        ins = float(((H[tr] - pred_tr) ** 2).sum()) / float(((H[tr] - HP[tr]) ** 2).sum())
        oos = float(((H[ok] - pred_te) ** 2).sum()) / float(((H[ok] - HP[ok]) ** 2).sum())
        out["folds"].append({"fold": f, "n_test": int(ok.sum()),
                             "n_test_dropped": int(te.sum() - ok.sum()),
                             "in_sample": ins, "held_out": oos})
        print(f"[{seed}] fold {f}: test {int(ok.sum())} "
              f"(dropped {int(te.sum()-ok.sum())}), in-sample {ins:.4f}, held-out {oos:.4f}",
              flush=True)
    out["in_sample_mean"] = float(np.mean([x["in_sample"] for x in out["folds"]]))
    out["held_out_mean"] = float(np.mean([x["held_out"] for x in out["folds"]]))
    print(f"[{seed}] MEAN in-sample {out['in_sample_mean']:.4f}  "
          f"held-out {out['held_out_mean']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = [run(s) for s in ("seed_1", "seed_2")]
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps([{k: v for k, v in r.items() if k != "folds"} for r in res], indent=2))
    print(f"written {OUT}", flush=True)
