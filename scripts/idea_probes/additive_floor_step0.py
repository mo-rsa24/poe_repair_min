"""The additive floor at step 0.

At the first denoising step the cached latent is bit-identical across every pair sharing a
seed, so every branch prediction is one vector per prompt. Any adapter that conditions each
branch on a single prompt therefore produces a composition of the form f_i + f_j, whatever
its capacity. The least-squares residual of that family against the joint-prompt prediction
is a hard floor on what such an adapter can reach at step 0.

Reported per seed:

    r_sq     total squared error of the plain composition against the joint prediction,
             which is the correction the adapter exists to remove
    res_sq   squared error of the best per-concept additive fit
    R_rel    res_sq / r_sq, the fraction of the correction NO per-concept adapter can
             remove at step 0. 1 - R_rel is the reachable fraction.

Controls: the plain composition as target must return R_rel = 0 by algebra (fit validator),
and permuting the pair labels gives the no-structure ceiling. The fit is reported twice,
once over all pairs and once restricted to pairs whose two concepts each appear in at least
two pairs, because a concept appearing in exactly one pair can absorb that pair's residual
on its own.
"""
from __future__ import annotations
import collections, json, os, sys, time
import numpy as np
import torch

ROOT = "/datasets/mmolefe/poe_repair_min/outputs/training_cache/train"
OUT = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/additive_floor_step0.json"


def cells(seed: str):
    out = []
    for p in sorted(os.listdir(ROOT)):
        f = os.path.join(ROOT, p, seed, "residuals", "step_000.pt")
        if os.path.exists(f):
            out.append((p, f))
    return out


def solve(A: np.ndarray, H: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Least squares for H ~= A F, returning the fitted A F."""
    AtA = A.T @ A
    AtA = AtA + ridge * np.trace(AtA) / AtA.shape[0] * np.eye(AtA.shape[0])
    F = np.linalg.solve(AtA, A.T @ H)
    return A @ F


def run(seed: str) -> dict:
    cs = cells(seed)
    print(f"[{seed}] {len(cs)} cells", flush=True)
    concepts: dict[str, int] = {}
    rows, J, POE = [], [], []
    t0 = time.time()
    for n, (slug, f) in enumerate(cs):
        a, b = slug.split("__x__")
        for c in (a, b):
            concepts.setdefault(c, len(concepts))
        d = torch.load(f, map_location="cpu", weights_only=False)
        eu = d["eps_uncond"].float().flatten()
        J.append((d["eps_j_raw"].float().flatten() - eu).numpy())
        POE.append((d["eps_a_raw"].float().flatten()
                    + d["eps_b_raw"].float().flatten() - 2.0 * eu).numpy())
        rows.append((concepts[a], concepts[b]))
        if n % 200 == 0:
            print(f"  {n}/{len(cs)}  {time.time()-t0:.0f}s", flush=True)
    P, N = len(rows), len(concepts)
    H = np.stack(J).astype(np.float32)
    Hp = np.stack(POE).astype(np.float32)
    A = np.zeros((P, N), dtype=np.float64)
    for k, (i, j) in enumerate(rows):
        A[k, i] += 1.0
        A[k, j] += 1.0
    deg = collections.Counter()
    for i, j in rows:
        deg[i] += 1
        deg[j] += 1
    keep = np.array([deg[i] >= 2 and deg[j] >= 2 for i, j in rows])
    print(f"[{seed}] P={P} N={N} rank={np.linalg.matrix_rank(A)} "
          f"pairs with both concepts degree>=2: {int(keep.sum())}", flush=True)

    res = {"seed": seed, "pairs": P, "concepts": N,
           "rank": int(np.linalg.matrix_rank(A)),
           "pairs_deg2": int(keep.sum())}

    r_sq = float(((H - Hp) ** 2).sum())
    fit = solve(A, H.astype(np.float64))
    res_sq = float(((H - fit) ** 2).sum())
    res["r_sq"] = r_sq
    res["res_sq"] = res_sq
    res["R_rel"] = res_sq / r_sq
    res["R_vs_target"] = res_sq / float((H ** 2).sum())

    # fit validator: the plain composition is inside the family by algebra
    fit_p = solve(A, Hp.astype(np.float64))
    res["control_poe_R_rel"] = float(((Hp - fit_p) ** 2).sum()) / float((Hp ** 2).sum())

    # no-structure ceiling
    rng = np.random.default_rng(0)
    perm = rng.permutation(P)
    fit_s = solve(A, H[perm].astype(np.float64))
    res["control_shuffled_R_rel"] = float(((H[perm] - fit_s) ** 2).sum()) / float(
        ((H[perm] - Hp[perm]) ** 2).sum())

    # restricted to pairs whose concepts each recur
    if keep.sum() > 0:
        fit_k = solve(A[keep], H[keep].astype(np.float64))
        res["R_rel_deg2"] = float(((H[keep] - fit_k) ** 2).sum()) / float(
            ((H[keep] - Hp[keep]) ** 2).sum())

    # worst and best pairs by per-pair reachable fraction
    per_res = ((H - fit) ** 2).sum(axis=1)
    per_r = ((H - Hp) ** 2).sum(axis=1)
    frac = per_res / np.maximum(per_r, 1e-12)
    order = np.argsort(frac)
    res["best_pairs"] = [(cs[i][0], float(frac[i])) for i in order[:5]]
    res["worst_pairs"] = [(cs[i][0], float(frac[i])) for i in order[-5:]]
    return res


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    all_res = [run(s) for s in ("seed_1", "seed_2")]
    with open(OUT, "w") as fh:
        json.dump(all_res, fh, indent=2)
    for r in all_res:
        print(json.dumps(r, indent=2), flush=True)
    print(f"written {OUT}", flush=True)
