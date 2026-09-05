#!/usr/bin/env python
"""How much of the correction does a Langevin corrector remove, and how much is left?

Steps 25 and 26 of ``plans/06-is-the-gap-the-samplers-or-the-models``. One file holds the
leak checks, the step-size search, the k grid, the three-way verdict and the figure, so
every threshold sits beside the code that applies it and moving one shows up in a diff.

Modes, in the order the plans run them:

    --check-identity --pair a_cat__x__a_dog --seed 9
        k=0 must reproduce plain product-of-experts byte for byte. The reference is a k=0
        run of the new composer with no window; it is compared against a k=0 run with a
        window placed past the last step (only the window logic can differ) and, as a
        second line, against ``run_cfg_poe`` itself (same ops, so this too should match).
    --check-identity --window off --k 200 --pair a_cat__x__a_dog --seed 9
        k=200 with the corrector window past the last step must ALSO match the k=0
        reference. This is the only check that catches a corrector ignoring its window.
    --step-size-search --k 20 --pair a_cat__x__a_dog --seed 9
        c in SEARCH_C at k=20: per c, the median relative chain displacement, the max
        latent norm as a multiple of the chain's start, and whether the residual ratio
        rises with the inner count. Picks the largest c that neither stalls nor diverges.
    --smoke --k 0,1 --pair a_cat__x__a_dog --seed 9
        the first short run: wall time per noise level against the cost table.
    --grid [--k 0,1,5,20,100,200] [--pairs ...] [--seed 9] [--c <picked>]
        the measurement. Resumable per (pair, seed, k); aggregates into residual_curves.json.
    --verdict
        applies the three-way threshold to residual_curves.json and prints which branch fired.
    --plot
        the figure with its .json sidecar under paper/iclr/figures/when-the-correction-arrives/mcmc/.

Every number is float32, upcast from the fp16 the models run in. Output lands under
/datasets only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poe_repair.composers import poe_langevin as cmp_lg  # noqa: E402
from poe_repair.composers._helpers import (  # noqa: E402
    encode_pair, get_joint_embeds, init_latents_for_cell,
)
from poe_repair.experiments.interaction_term.cell import cell_from_slug  # noqa: E402
from poe_repair.methods._poe_langevin import run_poe_langevin  # noqa: E402
from poe_repair.methods._sampling import run_cfg_poe  # noqa: E402
from poe_repair.run import make_ctx  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered thresholds. Written before any corrector existed (plan 26, task 1.1 and
# plan 25, task 2.3). Moving one after the answer is visible shows up in a diff.
# ---------------------------------------------------------------------------

MIN_DROP_FOR_SPLIT = 0.20       # fraction of the k=0 ratio that must go away (read zone)
MIN_REMAINDER_FOR_SPLIT = 0.20  # fraction of the k=0 ratio that must still be there
MAX_DRIFT_FOR_NULL = 0.05       # |ratio_k - ratio_0| / ratio_0 under this at EVERY step,
                                # with the chain provably moved, is a null
MIN_CHAIN_DISPLACEMENT = 0.05   # median ‖x_t^(k) - x_t^(0)‖/‖x_t^(0)‖ over the 50 levels
                                # below this is a failed instrument, never a null
MAX_K_INSTABILITY = 0.05        # k=100 against k=200 differing by more than this fraction
                                # of the k=100 read-zone ratio is not equilibrated
MAX_LATENT_NORM_REL = 1.5       # a chain whose latent norm passes 1.5x its start diverged
READ_ZONE_STEPS = 5             # the last five denoising steps, where the sampler's share
                                # has vanished by construction
UNATTRIBUTABLE_ZONE = (0, 10)   # steps where the two errors cannot be told apart

# The search (plan 25). The vendored sampler's 0.035 at k=20 is the centre of the range.
SEARCH_C = (0.01, 0.035, 0.1, 0.3, 1.0)
SEARCH_K = 20
SEARCH_PROBE_INNER = (0, 5, 10)  # plus the end of the chain (k itself), for the rise-with-k read
SEARCH_PAIR = "a_cat__x__a_dog"
SEARCH_SEED = 9

# The grid (plan 26).
GRID_K = (0, 1, 5, 20, 100, 200)
GRID_PAIRS = ("a_cat__x__a_dog", "a_butterfly__x__a_flower_meadow")
GRID_SEED = 9
COMPOSING_PAIR = "a_butterfly__x__a_flower_meadow"
FAILING_PAIR = "a_cat__x__a_dog"

OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector")
SEARCH_JSON = OUT / "step_size_search.json"
CURVES_JSON = OUT / "residual_curves.json"
CURVE_CELLS = OUT / "curves"
FIG_DIR = Path(__file__).resolve().parent.parent / "paper/iclr/figures/when-the-correction-arrives/mcmc"
FIG_NAME = "how-much-of-the-correction-a-corrector-removes"

# Cost model from the plan: one noise level at corrector count k costs 3k + 5 UNet
# evaluations (3 per Langevin step, 3 for the settled PoE call, 2 for the joint observer).
def unet_evals_per_level(k: int, measure: bool = True) -> int:
    return 3 * int(k) + 3 + (2 if measure else 0)


def _host() -> dict:
    return {"node": socket.gethostname(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "python": platform.python_version(), "torch": torch.__version__}


def _disk_guard(root: Path) -> None:
    st = os.statvfs(str(root if root.exists() else root.parent))
    used = 1.0 - st.f_bavail / max(st.f_blocks, 1)
    if used >= 0.90:
        raise SystemExit(f"disk guard: {root} filesystem at {used:.0%}, refusing to write")


def parse_window(spec: str | None, n: int) -> tuple[int, int] | None:
    if spec is None or spec.strip().lower() == "all":
        return None
    s = spec.strip().lower()
    if s == "off":
        return (n + 10, n + 20)     # past the last step: the corrector never fires
    a, b = (int(x) for x in s.split(","))
    if a >= b:
        raise SystemExit(f"bad --window {spec!r}: start must be < end")
    return (a, b)


# ---------------------------------------------------------------------------
# Leak checks
# ---------------------------------------------------------------------------


def check_identity(pair: str, seed: int, *, window: str | None, k: int, c: float,
                   steps: int | None) -> int:
    cell = cell_from_slug(pair, seed)
    ctx = make_ctx(num_inference_steps=steps) if steps else make_ctx()
    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    n = ctx.num_inference_steps
    common = dict(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale, num_inference_steps=n,
        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
    )

    def clear():
        gc.collect(); torch.cuda.empty_cache()

    t0 = time.time()
    ref = run_poe_langevin(**common, k=0, c=c, corrector_window=None, noise_seed=seed, decode=False)
    t_ref = time.time() - t0
    clear()
    print(f"reference: k=0, no window, {n} steps, {t_ref:.1f}s  ({t_ref / n:.2f}s per level, "
          f"{unet_evals_per_level(0, False)} UNet evals per level)")

    if window is None:
        # Check 1: window parsing at k=0 changes nothing (and the run is deterministic).
        past = run_poe_langevin(**common, k=0, c=c, corrector_window=(n + 10, n + 20),
                                noise_seed=seed, decode=False)
        clear()
        label = "k=0 with a window past the last step"
        # Check 0, informational: the reference sampler itself, same ops, batch of three.
        poe = run_cfg_poe(**common)
        clear()
        d0 = (ref.latents.float() - poe.latents.float()).abs().max().item()
        print(f"against run_cfg_poe (the three-branch reference sampler): "
              f"{'byte-identical' if torch.equal(ref.latents, poe.latents) else f'max |diff| = {d0:.3e}'}")
    else:
        w = parse_window(window, n)
        past = run_poe_langevin(**common, k=k, c=c, corrector_window=w, noise_seed=seed, decode=False)
        clear()
        fired = sum(1 for r in past.extras["per_step"] if r["k_applied"] > 0)
        label = f"k={k} with window {w}"
        if fired:
            print(f"IDENTITY FAILED: the corrector fired at {fired} of {n} levels with window {w}.",
                  file=sys.stderr)
            return 1

    if torch.equal(ref.latents, past.latents):
        print(f"corrector off is byte-identical to plain PoE   pair={pair} seed={seed}  [{label}]")
        return 0
    delta = (ref.latents.float() - past.latents.float()).abs().max().item()
    print(f"IDENTITY FAILED: max |diff| = {delta:.3e}  [{label}]. Nothing downstream may run.",
          file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# One measured run, shared by the search, the smoke and the grid
# ---------------------------------------------------------------------------


def measured_run(ctx, cell, *, k: int, c: float, probes=(), decode=False):
    """The composer with the joint observer on. Returns (per_step rows, seconds)."""
    t0 = time.time()
    _, out = cmp_lg.run(
        cell, ctx, k=k, c=c, corrector_window=None, measure_residual=True,
        probe_inner_counts=tuple(probes), decode=decode, exp_name="interaction_term/corrector",
        overwrite=True, return_outputs=True,
    )
    secs = time.time() - t0
    gc.collect(); torch.cuda.empty_cache()
    return out.extras["per_step"], secs


def step_size_search(pair: str, seed: int, *, k: int, cs, out_path: Path) -> int:
    _disk_guard(OUT)
    cell = cell_from_slug(pair, seed)
    ctx = make_ctx()
    probes = tuple(p for p in SEARCH_PROBE_INNER if p < k) + (k,)
    rows = []
    if out_path.exists():
        rows = json.loads(out_path.read_text()).get("rows", [])
        rows = [r for r in rows if r["c"] in cs]
    done = {r["c"] for r in rows}
    for c in cs:
        if c in done:
            print(f"c={c}: already measured, skipping", flush=True)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] c={c} k={k} ...", flush=True)
        per_step, secs = measured_run(ctx, cell, k=k, c=c, probes=probes)
        disp = np.array([r["chain_disp_rel"] for r in per_step])
        max_norm = max(r["max_latent_norm_rel"] for r in per_step)
        # Rise-with-k read: at each level, does the ratio go up from inner count 0 to k?
        # Recorded as the fraction of levels where the probed sequence is increasing and
        # the median change from start to end of the chain, both against the same k=20 run.
        seq_keys = [str(p) for p in probes]
        increasing = []
        end_minus_start = []
        for r in per_step:
            pr = r.get("probes", {})
            if not all(kk in pr for kk in seq_keys):
                continue
            vals = [pr[kk]["ratio"] for kk in seq_keys]
            increasing.append(all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)))
            end_minus_start.append(vals[-1] - vals[0])
        frac_increasing = float(np.mean(increasing)) if increasing else float("nan")
        med_change = float(np.median(end_minus_start)) if end_minus_start else float("nan")
        rises = bool(frac_increasing > 0.5 and med_change > 0.0)
        stalled = bool(float(np.median(disp)) < MIN_CHAIN_DISPLACEMENT)
        diverged = bool(max_norm > MAX_LATENT_NORM_REL or rises)
        row = {
            "c": c, "k": k, "pair": pair, "seed": seed,
            "median_chain_disp_rel": float(np.median(disp)),
            "chain_disp_rel_min": float(disp.min()), "chain_disp_rel_max": float(disp.max()),
            "max_latent_norm_rel": float(max_norm),
            "ratio_rises_with_k": rises,
            "frac_levels_ratio_increasing": frac_increasing,
            "median_ratio_change_start_to_end": med_change,
            "read_zone_ratio_at_k": float(np.mean([r["ratio"] for r in per_step[-READ_ZONE_STEPS:]])),
            "read_zone_ratio_at_inner0": float(np.mean(
                [r["probes"]["0"]["ratio"] for r in per_step[-READ_ZONE_STEPS:] if "probes" in r])),
            "stalled": stalled, "diverged": diverged,
            "verdict": "stalled" if stalled else ("diverged" if diverged else "usable"),
            "seconds": secs, "per_step": per_step,
        }
        rows.append(row)
        rows.sort(key=lambda r: r["c"])
        _write_search(out_path, rows, k, pair, seed)
        print(f"  median disp {row['median_chain_disp_rel']:.4f}  max norm x{max_norm:.3f}  "
              f"rises={rises} (frac {frac_increasing:.2f}, med change {med_change:+.4f})  "
              f"-> {row['verdict']}  ({secs / 60:.1f} min)", flush=True)
    return 0


def _write_search(out_path: Path, rows: list[dict], k: int, pair: str, seed: int) -> None:
    usable = [r["c"] for r in rows if r["verdict"] == "usable"]
    picked = max(usable) if usable else None
    cs_sorted = sorted(r["c"] for r in rows)
    at_edge = picked is not None and (picked == cs_sorted[0] or picked == cs_sorted[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "pair": pair, "seed": seed, "k": k,
        "thresholds": {"MIN_CHAIN_DISPLACEMENT": MIN_CHAIN_DISPLACEMENT,
                       "MAX_LATENT_NORM_REL": MAX_LATENT_NORM_REL},
        "rule": "pick the largest c that neither stalls (median chain displacement below "
                "MIN_CHAIN_DISPLACEMENT) nor diverges (max latent norm past MAX_LATENT_NORM_REL, "
                "or the residual ratio rising with the inner count at more than half the levels "
                "with a positive median change)",
        "picked_c": picked, "picked_c_at_edge_of_range": at_edge,
        "c_values_tested": cs_sorted, "host": _host(),
        "rows": rows,
    }, indent=1))


def picked_c() -> float:
    if not SEARCH_JSON.exists():
        raise SystemExit(f"no step-size search at {SEARCH_JSON}; run --step-size-search first (plan 25)")
    d = json.loads(SEARCH_JSON.read_text())
    if d.get("picked_c") is None:
        raise SystemExit("the step-size search picked no c: every value stalled or diverged (plan 25 stops the scope here)")
    return float(d["picked_c"])


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def cell_json(pair: str, seed: int, k: int) -> Path:
    return CURVE_CELLS / f"{pair}__seed{seed}__k{k:03d}.json"


def run_grid(pairs, seed: int, ks, c: float, *, smoke: bool = False) -> int:
    _disk_guard(OUT)
    ctx = make_ctx()
    print(f"grid: pairs={list(pairs)} seed={seed} k={list(ks)} c={c} host={_host()}", flush=True)
    for pair in pairs:
        cell = cell_from_slug(pair, seed)
        for k in ks:
            p = cell_json(pair, seed, k)
            if p.exists() and not smoke:
                print(f"[skip] {p.name} exists", flush=True)
                continue
            expect = unet_evals_per_level(k)
            print(f"[{time.strftime('%H:%M:%S')}] {pair} seed {seed} k={k}: "
                  f"{expect} UNet evals per level, {expect * ctx.num_inference_steps} total", flush=True)
            per_step, secs = measured_run(ctx, cell, k=k, c=c)
            rows = [{"pair": pair, "seed": seed, "k": k, "c": c, **r} for r in per_step]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"pair": pair, "seed": seed, "k": k, "c": c,
                                     "seconds": secs, "seconds_per_level": secs / ctx.num_inference_steps,
                                     "unet_evals_per_level": expect, "host": _host(),
                                     "rows": rows}, indent=1))
            rz = float(np.mean([r["ratio"] for r in per_step[-READ_ZONE_STEPS:]]))
            md = float(np.median([r["chain_disp_rel"] for r in per_step]))
            print(f"  done in {secs / 60:.1f} min ({secs / ctx.num_inference_steps:.1f}s per level); "
                  f"read-zone ratio {rz:.4f}, median chain displacement {md:.4f}", flush=True)
            if not smoke:
                aggregate_curves()
    if smoke:
        return 0
    aggregate_curves()
    return 0


def aggregate_curves() -> dict:
    rows, cells = [], []
    for p in sorted(CURVE_CELLS.glob("*.json")):
        d = json.loads(p.read_text())
        if d["k"] not in GRID_K or d["pair"] not in GRID_PAIRS:
            continue
        rows.extend(d["rows"])
        cells.append({k: d[k] for k in ("pair", "seed", "k", "c", "seconds", "seconds_per_level",
                                        "unet_evals_per_level", "host")})
    out = {
        "measure": "per (pair, seed, k, step): numerator ‖eps_J - eps_PoE‖, denominator ‖eps_PoE‖, "
                   "their ratio, and the chain's relative displacement, all float32 at the settled point "
                   "x_t^(k); two corrector counts are two trajectories, not one point wiggled twice",
        "grid": {"k": list(GRID_K), "pairs": list(GRID_PAIRS), "seed": GRID_SEED},
        "expected_rows": len(GRID_K) * len(GRID_PAIRS) * 50,
        "thresholds": {n: globals()[n] for n in (
            "MIN_DROP_FOR_SPLIT", "MIN_REMAINDER_FOR_SPLIT", "MAX_DRIFT_FOR_NULL",
            "MIN_CHAIN_DISPLACEMENT", "MAX_K_INSTABILITY", "MAX_LATENT_NORM_REL", "READ_ZONE_STEPS")},
        "cells": cells, "rows": rows,
    }
    CURVES_JSON.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def _curve(rows, pair, k, key="ratio") -> np.ndarray:
    sel = sorted((r for r in rows if r["pair"] == pair and r["k"] == k), key=lambda r: r["step_index"])
    return np.array([r[key] for r in sel], dtype=float)


def verdict(curves_path: Path = CURVES_JSON, *, quiet: bool = False) -> dict:
    d = json.loads(curves_path.read_text())
    rows = d["rows"]
    n_rows = len(rows)
    lines = [f"rows: {n_rows} of {d['expected_rows']} expected"]
    out = {"n_rows": n_rows, "expected_rows": d["expected_rows"]}
    have = {(r["pair"], r["k"]) for r in rows}
    missing = [(p, k) for p in GRID_PAIRS for k in GRID_K if (p, k) not in have]
    if missing:
        out["branch"] = "not ready"
        out["missing"] = missing
        lines.append(f"missing (pair, k): {missing}")
        if not quiet:
            print("\n".join(lines))
        return out

    rz = slice(-READ_ZONE_STEPS, None)
    K_SETTLED, K_PREV = GRID_K[-1], GRID_K[-2]

    def read_zone(pair, k):
        return float(_curve(rows, pair, k)[rz].mean())

    stats = {}
    for pair in GRID_PAIRS:
        r0, rp, rs = read_zone(pair, 0), read_zone(pair, K_PREV), read_zone(pair, K_SETTLED)
        disp = float(np.median(_curve(rows, pair, K_SETTLED, "chain_disp_rel")))
        max_norm = float(_curve(rows, pair, K_SETTLED, "max_latent_norm_rel").max())
        c0, cs = _curve(rows, pair, 0), _curve(rows, pair, K_SETTLED)
        per_step_rel = np.abs(cs - c0) / np.maximum(c0, 1e-12)
        num0, nums = _curve(rows, pair, 0, "numerator")[rz].mean(), _curve(rows, pair, K_SETTLED, "numerator")[rz].mean()
        den0, dens = _curve(rows, pair, 0, "denominator")[rz].mean(), _curve(rows, pair, K_SETTLED, "denominator")[rz].mean()
        stats[pair] = {
            "read_zone_ratio_k0": r0, f"read_zone_ratio_k{K_PREV}": rp, f"read_zone_ratio_k{K_SETTLED}": rs,
            "remainder_fraction": rs / max(r0, 1e-12), "drop_fraction": 1.0 - rs / max(r0, 1e-12),
            "k_instability": abs(rs - rp) / max(rp, 1e-12),
            "median_chain_disp_rel_settled": disp, "max_latent_norm_rel_settled": max_norm,
            "max_per_step_relative_change": float(per_step_rel.max()),
            "read_zone_numerator_k0": float(num0), f"read_zone_numerator_k{K_SETTLED}": float(nums),
            "read_zone_denominator_k0": float(den0), f"read_zone_denominator_k{K_SETTLED}": float(dens),
            "read_zone_ratio_by_k": {str(k): read_zone(pair, k) for k in GRID_K},
        }
    f, g = stats[FAILING_PAIR], stats[COMPOSING_PAIR]

    # Inconclusive first, because a broken instrument must not be read as a result.
    reasons = []
    for pair, s in stats.items():
        if s["k_instability"] > MAX_K_INSTABILITY:
            reasons.append(f"{pair}: k={K_PREV} against k={K_SETTLED} differ by "
                           f"{s['k_instability']:.3f} of the k={K_PREV} read-zone ratio (bar {MAX_K_INSTABILITY})")
        if s["median_chain_disp_rel_settled"] < MIN_CHAIN_DISPLACEMENT:
            reasons.append(f"{pair}: median chain displacement {s['median_chain_disp_rel_settled']:.4f} "
                           f"under {MIN_CHAIN_DISPLACEMENT}: the chain did not move")
        if s["max_latent_norm_rel_settled"] > MAX_LATENT_NORM_REL:
            reasons.append(f"{pair}: latent norm reached {s['max_latent_norm_rel_settled']:.2f}x its start "
                           f"(bar {MAX_LATENT_NORM_REL}): the chain diverged")
    if g["remainder_fraction"] > 1.0 + MAX_DRIFT_FOR_NULL:
        reasons.append(f"the composing pair's read-zone ratio ROSE with k, from {g['read_zone_ratio_k0']:.4f} "
                       f"to {g[f'read_zone_ratio_k{K_SETTLED}']:.4f}: the joint branch is degrading "
                       f"off-distribution and the measurement is reading itself")
    if reasons:
        branch = "inconclusive"
    elif f["drop_fraction"] >= MIN_DROP_FOR_SPLIT and f["remainder_fraction"] >= MIN_REMAINDER_FOR_SPLIT:
        branch = "support"
    elif f["max_per_step_relative_change"] < MAX_DRIFT_FOR_NULL:
        branch = "null"
    else:
        branch = "no branch fired"
        reasons.append(f"the failing pair's read-zone ratio changed by {f['drop_fraction']:+.3f} of its k=0 "
                       f"value (support needs a drop of at least {MIN_DROP_FOR_SPLIT} with at least "
                       f"{MIN_REMAINDER_FOR_SPLIT} left; null needs under {MAX_DRIFT_FOR_NULL} at every step, "
                       f"and the largest per-step change is {f['max_per_step_relative_change']:.3f})")

    out.update({"branch": branch, "reasons": reasons, "k_settled": K_SETTLED, "k_prev": K_PREV,
                "stats": stats, "thresholds": d["thresholds"]})
    lines.append(f"BRANCH: {branch}")
    for pair, s in stats.items():
        lines.append(f"  {pair}: read-zone ratio k=0 {s['read_zone_ratio_k0']:.4f} -> k={K_SETTLED} "
                     f"{s[f'read_zone_ratio_k{K_SETTLED}']:.4f} (remainder {s['remainder_fraction']:.3f}, "
                     f"drop {s['drop_fraction']:+.3f}); k={K_PREV} vs {K_SETTLED} instability "
                     f"{s['k_instability']:.3f}; median chain displacement {s['median_chain_disp_rel_settled']:.4f}; "
                     f"max latent norm x{s['max_latent_norm_rel_settled']:.3f}; "
                     f"largest per-step relative change {s['max_per_step_relative_change']:.3f}")
        lines.append(f"    numerator {s['read_zone_numerator_k0']:.2f} -> {s[f'read_zone_numerator_k{K_SETTLED}']:.2f}, "
                     f"denominator {s['read_zone_denominator_k0']:.2f} -> {s[f'read_zone_denominator_k{K_SETTLED}']:.2f} "
                     f"(read zone means)")
        lines.append("    read-zone ratio by k: " + ", ".join(
            f"k={k} {v:.4f}" for k, v in s["read_zone_ratio_by_k"].items()))
    for r in reasons:
        lines.append(f"  - {r}")
    text = "\n".join(lines)
    if not quiet:
        print(text)
    (OUT / "verdict.json").write_text(json.dumps(out, indent=1))
    (OUT / "verdict.txt").write_text(text + "\n")
    return out


def flat_k(curves_path: Path = CURVES_JSON) -> int:
    """The smallest k on the flat part of the failing pair's curve: its read-zone ratio is
    within MAX_K_INSTABILITY of the largest k's. Plan 27 runs the corrector at this k."""
    rows = json.loads(curves_path.read_text())["rows"]
    rz = slice(-READ_ZONE_STEPS, None)
    top = float(_curve(rows, FAILING_PAIR, GRID_K[-1])[rz].mean())
    for k in GRID_K[1:]:
        v = float(_curve(rows, FAILING_PAIR, k)[rz].mean())
        if abs(v - top) / max(top, 1e-12) <= MAX_K_INSTABILITY:
            return k
    return GRID_K[-1]


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def plot(curves_path: Path = CURVES_JSON, fig_dir: Path = FIG_DIR) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(curves_path.read_text())
    rows = d["rows"]
    v = verdict(curves_path, quiet=True)
    ks = [k for k in GRID_K if any(r["k"] == k for r in rows)]
    cmap = plt.get_cmap("viridis")
    colours = {k: cmap(i / max(len(ks) - 1, 1)) for i, k in enumerate(ks)}
    titles = {FAILING_PAIR: "a cat × a dog (blends under plain PoE)",
              COMPOSING_PAIR: "a butterfly × a flower meadow (composes under plain PoE)"}

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.12, "wspace": 0.16})
    for ci, pair in enumerate(GRID_PAIRS):
        ax, ax2 = axes[0][ci], axes[1][ci]
        for zone_ax in (ax, ax2):
            zone_ax.axvspan(UNATTRIBUTABLE_ZONE[0], UNATTRIBUTABLE_ZONE[1], color="0.85", lw=0, zorder=0)
            zone_ax.axvspan(49 - READ_ZONE_STEPS + 1, 49.5, color="#d9ead3", lw=0, zorder=0)
        for k in ks:
            c = _curve(rows, pair, k)
            if c.size == 0:
                continue
            ax.plot(np.arange(c.size), c, color=colours[k], lw=1.6 if k in (0, ks[-1]) else 1.1)
            ax.annotate(f"k={k}", (c.size - 1, c[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=colours[k], va="center")
            num, den = _curve(rows, pair, k, "numerator"), _curve(rows, pair, k, "denominator")
            ax2.plot(np.arange(num.size), num, color=colours[k], lw=1.0)
            ax2.plot(np.arange(den.size), den, color=colours[k], lw=1.0, ls="--")
        ax.set_title(titles.get(pair, pair), fontsize=9)
        ax.text(UNATTRIBUTABLE_ZONE[1] / 2, 0.97, "sampler and model\nerrors cannot be\ntold apart here",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color="0.35")
        ax.text(49 - READ_ZONE_STEPS / 2 + 0.5, 0.97, "read\nzone", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, color="#38761d")
        if ci == 0:
            ax.set_ylabel("‖eps_J − eps_PoE‖ / ‖eps_PoE‖ at the settled point", fontsize=8)
            ax2.set_ylabel("norms: solid ‖eps_J − eps_PoE‖, dashed ‖eps_PoE‖", fontsize=8)
        ax2.set_xlabel("denoising step along the k-corrected path (0 = noise, 49 = image)", fontsize=8)
        ax.set_xlim(0, 52)
        for a in (ax, ax2):
            a.tick_params(labelsize=7)
            a.grid(alpha=0.25)
    branch = v.get("branch", "not ready")
    fig.suptitle(f"How much of the correction a Langevin corrector removes, seed {GRID_SEED}, "
                 f"c = {rows[0]['c'] if rows else '?'}: printed branch \"{branch}\"", fontsize=10)
    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.08, right=0.97)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{FIG_NAME}.{ext}", dpi=200)
    plt.close(fig)
    side = {
        "drawn_from": str(curves_path), "pairs": list(GRID_PAIRS), "seed": GRID_SEED, "k": ks,
        "c": rows[0]["c"] if rows else None, "sampler": "SDXL base, DDIM 50 steps, guidance 7.5, 1024 square, fp16 with float32 norms",
        "y_top": "‖eps_J(x_t^(k)) − eps_PoE(x_t^(k))‖ / ‖eps_PoE(x_t^(k))‖, unitless, at the point the k-step Langevin chain settled to at that level",
        "y_bottom": "the two norms separately, so a falling ratio can be read as numerator falling or denominator rising",
        "x": "denoising step index along the k-corrected path; two corrector counts are two trajectories, not one point wiggled twice",
        "shaded_grey": f"steps {UNATTRIBUTABLE_ZONE[0]} to {UNATTRIBUTABLE_ZONE[1]}: the compose-decisive window, where the non-commutation gap and the model gap add and no k separates them",
        "shaded_green": f"the last {READ_ZONE_STEPS} steps: the read zone, where the sampler's share has vanished by construction and what is left is the model's",
        "caption_owes": ["the axis is the correction along the k-corrected path rather than along one path",
                         "eps_J is evaluated off-distribution at the settled point, and that is a deliberate choice",
                         "the residual norm is a proxy for the distributional gap rather than the gap itself: the corrector changes where eps_J − eps_PoE is evaluated, not the function"],
        "verdict": v,
    }
    (fig_dir / f"{FIG_NAME}.json").write_text(json.dumps(side, indent=1))
    return fig_dir / f"{FIG_NAME}.png"


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-identity", action="store_true")
    ap.add_argument("--step-size-search", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--flat-k", action="store_true", help="print the k plan 27 runs at")
    ap.add_argument("--pair", default=None)
    ap.add_argument("--pairs", default=None, help="comma-separated slugs for --grid")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--k", default=None, help="one int, or a comma list for --smoke/--grid")
    ap.add_argument("--c", type=float, default=None, help="step-size multiplier; --grid reads the picked one by default")
    ap.add_argument("--window", default=None, help="'off', 'all', or 'start,end' (identity check only)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--search-c", default=None, help="override SEARCH_C as a comma list (a wider range)")
    args = ap.parse_args()

    if args.check_identity:
        k = int(args.k) if args.k else 0
        c = args.c if args.c is not None else 0.035
        return check_identity(args.pair or SEARCH_PAIR, args.seed if args.seed is not None else SEARCH_SEED,
                              window=args.window, k=k, c=c, steps=args.steps)
    if args.step_size_search:
        cs = tuple(float(x) for x in args.search_c.split(",")) if args.search_c else SEARCH_C
        return step_size_search(args.pair or SEARCH_PAIR, args.seed if args.seed is not None else SEARCH_SEED,
                                k=int(args.k) if args.k else SEARCH_K, cs=cs, out_path=SEARCH_JSON)
    if args.smoke or args.grid:
        ks = tuple(int(x) for x in args.k.split(",")) if args.k else GRID_K
        pairs = tuple(args.pairs.split(",")) if args.pairs else ((args.pair,) if args.pair else GRID_PAIRS)
        c = args.c if args.c is not None else picked_c()
        return run_grid(pairs, args.seed if args.seed is not None else GRID_SEED, ks, c, smoke=args.smoke)
    if args.verdict:
        aggregate_curves()
        verdict()
        return 0
    if args.flat_k:
        print(flat_k())
        return 0
    if args.plot:
        aggregate_curves()
        print(plot())
        return 0
    ap.error("pass one of --check-identity / --step-size-search / --smoke / --grid / --verdict / --plot / --flat-k")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
