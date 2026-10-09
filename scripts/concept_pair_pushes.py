#!/usr/bin/env python
"""What each concept pushes toward, step by step, on plain PoE's own path.

Reruns plain product-of-experts (PoE) for one pair and seed, from the same cached starting noise
as scripts/concept_pair_sheet.py, and at every one of the 50 steps asks the UNet for four
predictions at the current latent: empty prompt, concept A, concept B and the joint prompt
"{a} and {b}". The step itself is PoE's, so the final picture is the sheet's PoE tile.

The push of a prompt is its prediction minus the empty prompt's. Two pushes (A and B) always lie
in one flat plane, so drawing them in 2D is exact: A along x at its true length, B at its true
angle and length. PoE's push is their sum and lies in the plane too. The joint push generally does
not: its shadow in the plane is drawn, and the length left over, out of the plane, is recorded.
`alpha` and `beta` are the weights on A and B that reproduce the shadow; no reweighting of A and
B can reach the out-of-plane part.

    python scripts/concept_pair_pushes.py --pair "a dog|rain" --seed 10              # on a GPU
    python scripts/concept_pair_pushes.py --plot <pushes_dir> <out_dir> --pair "a dog|rain" --seed 10 \
        [--superdiff-sidecar <superdiff_200.json>]

Writes <OUT>/pushes/<pair>/seedNN/pushes.json and the predicted clean image of the PoE path at a
few steps (x0_stepNN.png).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

OUT = Path(os.environ.get("CONCEPT_PAIR_OUT",
                          "/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet"))
NOISE_PAIR = "a_cat__x__a_dog"
SHOW_STEPS = (0, 5, 10, 20, 35, 49)


def slug_of(a: str, b: str) -> str:
    from concept_pair_sheet import concept_slug
    return f"{concept_slug(a)}__x__{concept_slug(b)}"


def record(a: str, b: str, seed: int) -> Path:
    import torch

    from concept_pair_sheet import parse_pair
    from poe_repair.composers._helpers import encode_pair, get_joint_embeds, init_latents_for_cell
    from poe_repair.experiments._eval_common import cell_for
    from poe_repair.experiments.interaction_term.cell import cell_from_slug
    from poe_repair.methods._sampling import add_time_ids, write_decoded_image
    from poe_repair.run import make_ctx
    from poe_repair.runtime import (decode_latents, ddim_prev_from_x0_eps, guided_eps, poe_eps,
                                    tweedie_mean)

    parse_pair(f"{a}|{b}")
    ctx = make_ctx()
    dev, dt, w = ctx.device, ctx.dtype, ctx.guidance_scale
    cell = cell_for(a, b, seed)
    out_dir = OUT / "pushes" / cell.pair_slug / f"seed{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    init, sigma = init_latents_for_cell(cell_from_slug(NOISE_PAIR, seed), ctx)
    emb = encode_pair(cell, ctx)
    seq_j, pool_j = get_joint_embeds(cell, ctx)
    # PoE's own three predictions go through the UNet as the same batch of three run_cfg_poe uses,
    # and the joint prediction as a batch of its own. In fp16 the batch shape changes the arithmetic
    # slightly, and on a seed near the switch between outcomes that alone changes the final picture.
    pe = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_e"]], dim=0)
    pool = torch.cat([emb["pool_a"], emb["pool_b"], emb["pool_e"]], dim=0)
    cond = {"text_embeds": pool,
            "time_ids": add_time_ids(height=cell.height, width=cell.width, batch_size=3, device=dev, dtype=dt)}
    cond_j = {"text_embeds": pool_j,
              "time_ids": add_time_ids(height=cell.height, width=cell.width, batch_size=1, device=dev, dtype=dt)}
    sched, unet = ctx.scheduler, ctx.models["unet"]
    sched.set_timesteps(ctx.num_inference_steps)
    latents = (init / sigma).to(device=dev, dtype=dt)

    rows = []
    with torch.no_grad():
        for i, t in enumerate(sched.timesteps):
            x_in = sched.scale_model_input(latents.repeat(3, 1, 1, 1), t)
            e_a, e_b, e_u = unet(x_in, t, encoder_hidden_states=pe, added_cond_kwargs=cond,
                                 timestep_cond=None).sample.chunk(3)
            e_j = unet(sched.scale_model_input(latents, t), t, encoder_hidden_states=seq_j,
                       added_cond_kwargs=cond_j, timestep_cond=None).sample
            # The geometry is read in float32; the step itself stays in the sampler's precision.
            da, db, dj = ((e - e_u).float().flatten() for e in (e_a, e_b, e_j))
            na, nb, nj = da.norm().item(), db.norm().item(), dj.norm().item()
            e1 = da / na
            b_par = torch.dot(db, e1).item()
            b_perp = db - b_par * e1
            nbp = b_perp.norm().item()
            e2 = b_perp / nbp
            jx, jy = torch.dot(dj, e1).item(), torch.dot(dj, e2).item()
            j_out = (dj - jx * e1 - jy * e2).norm().item()
            # Weights on A and B that rebuild the joint push's shadow: jx e1 + jy e2 = alpha da + beta db.
            beta = jy / nbp
            alpha = (jx - beta * b_par) / na
            rows.append({"step": i, "timestep": int(t.item()),
                         "norm_a": na, "norm_b": nb, "norm_joint": nj,
                         "angle_ab_deg": math.degrees(math.acos(max(-1.0, min(1.0, b_par / nb)))),
                         "b_xy": [b_par, nbp], "joint_xy": [jx, jy], "joint_out_of_plane": j_out,
                         "joint_out_share": j_out / nj, "alpha": alpha, "beta": beta})

            eps_a, eps_b = guided_eps(e_a, e_u, w), guided_eps(e_b, e_u, w)
            eps_p = poe_eps(eps_a, eps_b, e_u)
            abar = sched.alphas_cumprod[int(t.item())].to(device=dev, dtype=dt)
            x0 = tweedie_mean(latents, abar, eps_p)
            if i in SHOW_STEPS:
                write_decoded_image(decode_latents(ctx.models, x0).cpu(), out_dir / f"x0_step{i:02d}.png")
            latents = ddim_prev_from_x0_eps(scheduler=sched, timestep=t, step_index=i, x0=x0, eps=eps_p)
            print(f"step {i:2d}  |A| {na:7.2f}  |B| {nb:7.2f}  angle {rows[-1]['angle_ab_deg']:5.1f}  "
                  f"joint out {100 * j_out / nj:5.1f}%", flush=True)

        write_decoded_image(decode_latents(ctx.models, latents).cpu(), out_dir / "final.png")

    out = out_dir / "pushes.json"
    out.write_text(json.dumps({"pair": cell.pair_slug, "prompt_a": a, "prompt_b": b, "seed": seed,
                               "path": "plain PoE", "noise_from": NOISE_PAIR, "guidance": w,
                               "steps": ctx.num_inference_steps,
                               "push": "unguided prediction minus empty-prompt prediction",
                               "rows": rows}, indent=1))
    print(f"wrote {out}")
    return out


def plot(pushes_dir: Path, out_dir: Path, a: str, b: str, seed: int, sd_sidecar: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = pushes_dir / slug_of(a, b) / f"seed{seed:02d}" / "pushes.json"
    d = json.loads(src.read_text())
    rows = d["rows"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{slug_of(a, b).replace('__x__', '-and-').replace('_', '-')}-seed{seed:02d}"
    col_a, col_b, col_p, col_j = "#2f6fb0", "#c0504d", "#7a7a7a", "#2e8b57"

    # Figure 1: the plane at a few steps, each panel at its own scale.
    fig, axes = plt.subplots(1, len(SHOW_STEPS), figsize=(3.1 * len(SHOW_STEPS), 3.6))
    for ax, s in zip(axes, SHOW_STEPS):
        r = rows[s]
        A, B, J = (r["norm_a"], 0.0), tuple(r["b_xy"]), tuple(r["joint_xy"])
        P = (A[0] + B[0], A[1] + B[1])
        lim = 1.15 * max(abs(v) for v in (*A, *B, *P, *J))
        for vec, c, ls, name in ((A, col_a, "-", a), (B, col_b, "-", b),
                                 (P, col_p, "--", "PoE"), (J, col_j, "-", "joint")):
            ax.annotate("", xy=vec, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2, ls=ls))
            ax.text(vec[0], vec[1], f" {name}", color=c, fontsize=9, va="bottom")
        ax.set_xlim(-lim * 0.35, lim)
        ax.set_ylim(-lim * 0.35, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="#ddd", lw=0.8)
        ax.axvline(0, color="#ddd", lw=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"step {s}", fontsize=10)
        ax.text(0.02, 0.02, f"{a}–{b} angle {r['angle_ab_deg']:.0f}°\n"
                            f"joint out of plane {100 * r['joint_out_share']:.0f}%",
                transform=ax.transAxes, fontsize=8, color="#333")
    fig.suptitle("Where each prompt pushes, seed %d" % seed, fontsize=12)
    fig.tight_layout()
    f1 = out_dir / f"{stem}-pushes-in-the-plane.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # Figure 2: the same quantities over the whole run, one panel per quantity.
    steps = [r["step"] for r in rows]
    panels = [("angle between the two pushes (degrees)", [r["angle_ab_deg"] for r in rows], "#555"),
              ("share of the joint push outside the plane (%)", [100 * r["joint_out_share"] for r in rows], col_j)]
    sd = None
    if sd_sidecar and sd_sidecar.exists():
        k = json.loads(sd_sidecar.read_text()).get("kappa_used") or []
        if k:
            # SuperDiff runs 200 steps; place each of its steps at the same fraction of the run.
            sd = ([50 * i / len(k) for i in range(len(k))], k)
    n = len(panels) + (1 if sd else 0)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.2 * n), sharex=True)
    for ax, (label, ys, c) in zip(axes, panels):
        ax.plot(steps, ys, color=c, lw=2)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
    if sd:
        ax = axes[-1]
        ax.plot(sd[0], sd[1], color="#8064a2", lw=1.5)
        ax.axhline(0.5, color="#999", lw=0.8, ls=":")
        ax.text(1, 0.52, "equal weight", fontsize=8, color="#666")
        ax.set_ylabel(f"SuperDiff weight on \"{a}\"\n(1 − weight on \"{b}\")", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("denoising step (0 = pure noise, 49 = final)")
    fig.suptitle("The two pushes over the run, seed %d" % seed, fontsize=12)
    fig.tight_layout()
    f2 = out_dir / f"{stem}-pushes-over-the-run.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    print(f"wrote {f1}\nwrote {f2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help='as "a dog|rain"')
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--plot", nargs=2, type=Path, metavar=("PUSHES_DIR", "OUT_DIR"))
    ap.add_argument("--superdiff-sidecar", type=Path)
    args = ap.parse_args()
    a, b = (p.strip() for p in args.pair.split("|"))
    if args.plot:
        return plot(*args.plot, a, b, args.seed, args.superdiff_sidecar)
    record(a, b, args.seed)


if __name__ == "__main__":
    main()
