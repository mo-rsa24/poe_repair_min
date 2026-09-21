#!/usr/bin/env python
"""Rung 3 of "what the correction is made of": the rank-32 adapter's output at the cached
states, projected onto the same three-vector span as r_t.

At every cached (x_t, t) of cat x dog seeds 9 to 16 the UNet is run on the three prompts
"a cat", "a dog", "" twice in the same process: adapter enabled and adapter disabled. The PoE
prediction is formed from each (guided a + guided b - uncond, g = 7.5) and their difference is
the adapter's output delta_hat at lambda 1. delta_hat is projected onto span{eps_a - eps_u,
eps_b - eps_u, eps_u} built from the CACHED raw predictions, exactly as rung 2 projects r_t.

Per (seed, step): in-span and orthogonal share of ||delta_hat||^2; cosine of delta_hat's
orthogonal part against r_t's orthogonal part; cosine of the in-span parts; overall cosine and
norm ratio against r_t; and the sanity cosine between the live adapter-off PoE prediction and
the cached one (must be > 0.99, else the cache and this build disagree).

Writes into artifacts/results/what-the-correction-is-made-of/:
    adapter-span-share-over-steps.png     (a) orthogonal share of delta_hat and of r_t against
                                          step; (b) cos(delta_hat_orth, r_orth) and
                                          cos(delta_hat, r_t) against step; mean and min-max band
    adapter-span-share.json               one row per (seed, step)
    seed-15-adapter-vs-correction-norm-maps.png   two rows of per-position norm maps at the
                                          strip steps: ||delta_hat|| and ||r_t||, one colour
                                          scale per row
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import correction_span_common as C  # noqa: E402

CHECKPOINT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt")
RANK, ALPHA = 32, 32
SANITY_MIN_COS = 0.99


def main() -> int:
    from poe_repair.experiments.one_pair_one_seed import trainer as T
    from poe_repair.experiments.one_pair_one_seed.config import RunConfig
    from poe_repair.experiments.one_pair_one_seed.main import encode_all_prompts
    from poe_repair.methods._sampling import add_time_ids
    from poe_repair.runtime import infer_device, infer_dtype, load_sdxl_models

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available on the pinned device (poe-launch-002); refusing to run on CPU")
    print(f"[rung3] host={socket.gethostname()} pid={os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"device={torch.cuda.get_device_name(0)}", flush=True)

    cfg = RunConfig()
    cfg.lora.rank, cfg.lora.alpha = RANK, ALPHA
    cfg.cell.prompt_a, cfg.cell.prompt_b, cfg.cell.joint_prompt = "a cat", "a dog", "a cat and a dog"
    device = infer_device("cuda"); dtype = infer_dtype("float16", device)
    models = load_sdxl_models(model_id=cfg.model_id, device=device, dtype=dtype)
    unet = models["unet"]
    T.attach_lora(unet, cfg)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    T.load_lora_state(unet, ckpt["lora_state"])
    unet.eval()
    print(f"[rung3] checkpoint step={ckpt.get('step')} rank={RANK} alpha={ALPHA}", flush=True)
    gs = C.GUIDANCE; H, W = cfg.sampler.height, cfg.sampler.width
    emb = encode_all_prompts(cfg, models, device, dtype)
    seq3 = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_e"]], 0).to(device=device, dtype=dtype)
    pool3 = torch.cat([emb["pool_a"], emb["pool_b"], emb["pool_e"]], 0).to(device=device, dtype=dtype)
    cond = {"text_embeds": pool3, "time_ids": add_time_ids(height=H, width=W, batch_size=3, device=device, dtype=dtype)}
    # free what the UNet pass does not need
    for k in ("text_encoder", "text_encoder_2", "vae"):
        models[k].to("cpu")
    torch.cuda.empty_cache()

    def poe_from(noise: torch.Tensor) -> torch.Tensor:
        ea, eb, eu = noise[0:1].float(), noise[1:2].float(), noise[2:3].float()
        return C.guided(ea, eu, gs) + C.guided(eb, eu, gs) - eu

    rows = []
    maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    t0 = time.time()
    for seed in C.SEEDS:
        for k in range(C.num_steps(seed)):
            s = C.load_step(seed, k)
            with torch.no_grad():
                x = s.x_t.to(device=device, dtype=dtype).repeat(3, 1, 1, 1)
                tb = torch.full((3,), int(s.timestep), device=device, dtype=torch.long)
                unet.enable_adapters()
                on = unet(x, tb, encoder_hidden_states=seq3, added_cond_kwargs=cond).sample
                unet.disable_adapters()
                off = unet(x, tb, encoder_hidden_states=seq3, added_cond_kwargs=cond).sample
                unet.enable_adapters()
                d_hat = (poe_from(on) - poe_from(off)).cpu()
                poe_live = poe_from(off).cpu()
            B = C.span_basis(s)
            pr = C.project(B, s.r_t)
            pd = C.project(B, d_hat)
            row = {"seed": seed, "step": k, "timestep": s.timestep,
                   "ortho_share_adapter": pd["ortho_share"], "in_share_adapter": pd["in_share"],
                   "ortho_share_r": pr["ortho_share"],
                   "cos_orth_adapter_vs_r": C.cosine(pd["out"], pr["out"]),
                   "cos_inspan_adapter_vs_r": C.cosine(pd["in"], pr["in"]),
                   "cos_adapter_vs_r": C.cosine(d_hat, s.r_t),
                   "norm_adapter": pd["norm"], "norm_r": pr["norm"], "norm_ratio_adapter_over_r": pd["norm"] / pr["norm"],
                   "coef_adapter_on_a_minus_u": float(pd["coef"][0]), "coef_adapter_on_b_minus_u": float(pd["coef"][1]),
                   "coef_adapter_on_u": float(pd["coef"][2]),
                   "sanity_cos_live_vs_cached_poe": C.cosine(poe_live, s.eps_poe)}
            rows.append(row)
            if seed == C.STRIP_SEED and k in C.STRIP_STEPS:
                maps[k] = (C.per_position_norm(d_hat), C.per_position_norm(s.r_t))
        print(f"[rung3] seed {seed} done ({len(rows)} rows, {time.time() - t0:.0f}s)", flush=True)

    steps = sorted({r["step"] for r in rows})
    def mat(key):
        return np.array([[next(r[key] for r in rows if r["seed"] == sd and r["step"] == k) for k in steps] for sd in C.SEEDS])
    early = [k for k in steps if k in C.EARLY_STEPS]
    sanity_min = min(r["sanity_cos_live_vs_cached_poe"] for r in rows)
    summary = {
        "sanity_cos_live_vs_cached_poe_min": sanity_min, "sanity_ok": sanity_min > SANITY_MIN_COS,
        "ortho_share_adapter_mean_per_step": mat("ortho_share_adapter").mean(0).round(4).tolist(),
        "ortho_share_r_mean_per_step": mat("ortho_share_r").mean(0).round(4).tolist(),
        "cos_orth_adapter_vs_r_mean_per_step": mat("cos_orth_adapter_vs_r").mean(0).round(4).tolist(),
        "cos_inspan_adapter_vs_r_mean_per_step": mat("cos_inspan_adapter_vs_r").mean(0).round(4).tolist(),
        "cos_adapter_vs_r_mean_per_step": mat("cos_adapter_vs_r").mean(0).round(4).tolist(),
        "norm_ratio_adapter_over_r_mean_per_step": mat("norm_ratio_adapter_over_r").mean(0).round(4).tolist(),
        "early_window": {
            "ortho_share_adapter_mean": float(mat("ortho_share_adapter")[:, early].mean()),
            "ortho_share_r_mean": float(mat("ortho_share_r")[:, early].mean()),
            "cos_orth_adapter_vs_r_mean": float(mat("cos_orth_adapter_vs_r")[:, early].mean()),
            "cos_adapter_vs_r_mean": float(mat("cos_adapter_vs_r")[:, early].mean()),
        },
        "all_steps": {"cos_adapter_vs_r_mean": float(mat("cos_adapter_vs_r").mean()),
                      "cos_orth_adapter_vs_r_mean": float(mat("cos_orth_adapter_vs_r").mean())},
    }
    C.write_json(C.RESULTS / "adapter-span-share.json", {
        "pair": C.PAIR, "seeds": list(C.SEEDS), "checkpoint": str(CHECKPOINT), "checkpoint_step": ckpt.get("step"),
        "rank": RANK, "alpha": ALPHA, "lambda_read_at": 1.0, "guidance": gs, "steps": steps,
        "host": socket.gethostname(), "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "SANITY_MIN_COS": SANITY_MIN_COS, "EARLY_STEPS": list(C.EARLY_STEPS),
        "definitions": {
            "delta_hat": "PoE prediction with the adapter enabled minus PoE prediction with it disabled, same process, same cached x_t and t; the adapter's output at lambda 1",
            "span": "alpha eps_u + beta (eps_a - eps_u) + gamma (eps_b - eps_u) from the CACHED raw predictions, the same basis rung 2 uses",
            "ortho_share_adapter": "1 - ||proj_span delta_hat||^2 / ||delta_hat||^2",
            "cos_orth_adapter_vs_r": "cosine between the orthogonal parts of delta_hat and r_t",
            "sanity_cos_live_vs_cached_poe": "cosine between the adapter-off PoE prediction computed now and the one rebuilt from the cache",
        },
        "summary": summary, "rows": rows,
    })
    print("[rung3] sanity min cos live vs cached:", round(sanity_min, 5), "ok" if sanity_min > SANITY_MIN_COS else "FAILED")
    print("[rung3] early window:", {k: round(v, 4) for k, v in summary["early_window"].items()})

    # ---- figure: shares and cosines against step ------------------------------------------------
    x = np.array(steps)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    for key, col, lab in (("ortho_share_adapter", "#1f78b4", "the adapter's output delta_hat"), ("ortho_share_r", "#e7298a", "the correction r_t")):
        m = mat(key); ax.plot(x, m.mean(0), color=col, lw=2.6, label=lab); ax.fill_between(x, m.min(0), m.max(0), color=col, alpha=0.15)
    ax.axvspan(min(C.EARLY_STEPS), max(C.EARLY_STEPS), color="#ffd92f", alpha=0.25, lw=0)
    ax.set_ylim(0, 1); ax.set_ylabel("orthogonal share of the squared norm\n(fraction outside the experts' span)")
    ax.set_title("cat x dog, seeds 9 to 16: the rank-32 adapter (step 30050, held out on this pair) at the cached PoE states,\n"
                 "projected onto span{eps_u, eps_a - eps_u, eps_b - eps_u} the same way as the correction; mean and min-to-max band", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[1]
    for key, col, lab in (("cos_orth_adapter_vs_r", "#a50f15", "cos(orthogonal part of delta_hat, orthogonal part of r_t)"),
                          ("cos_inspan_adapter_vs_r", "#6a3d9a", "cos(in-span part of delta_hat, in-span part of r_t)"),
                          ("cos_adapter_vs_r", "#111111", "cos(delta_hat, r_t), the fit cosine")):
        m = mat(key); ax.plot(x, m.mean(0), color=col, lw=2.2, label=lab); ax.fill_between(x, m.min(0), m.max(0), color=col, alpha=0.12)
    ax.axvspan(min(C.EARLY_STEPS), max(C.EARLY_STEPS), color="#ffd92f", alpha=0.25, lw=0)
    ax.axhline(0, color="#999", lw=0.8); ax.set_ylim(-0.2, 1)
    ax.set_ylabel("cosine"); ax.set_xlabel("denoising step (0 = pure noise, 49 = last cached step)")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(); fig.savefig(C.RESULTS / "adapter-span-share-over-steps.png", dpi=160)

    # ---- figure: per-position norm maps, seed 15 -------------------------------------------------
    ks = [k for k in C.STRIP_STEPS if k in maps]
    fig, axes = plt.subplots(2, len(ks), figsize=(2.05 * len(ks), 4.6))
    for row_i, (lab, vmax) in enumerate((("||delta_hat|| (adapter output)", max(maps[k][0].max() for k in ks)),
                                         ("||r_t|| (the correction)", max(maps[k][1].max() for k in ks)))):
        for j, k in enumerate(ks):
            ax = axes[row_i, j]
            im = ax.imshow(maps[k][row_i], cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title(f"step {k}", fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            r = next(r for r in rows if r["seed"] == C.STRIP_SEED and r["step"] == k)
            if row_i == 1:
                ax.set_xlabel(f"cos {r['cos_adapter_vs_r']:+.2f}\n|d|/|r| {r['norm_ratio_adapter_over_r']:.2f}", fontsize=7.5)
        fig.colorbar(im, ax=axes[row_i, :].tolist(), fraction=0.012, pad=0.01)
    fig.suptitle(f"cat x dog, seed {C.STRIP_SEED}: per-latent-position norm (over 4 channels) of the adapter's output and of the correction\n"
                 "at the same cached states; one colour scale per row", fontsize=10)
    fig.savefig(C.RESULTS / f"seed-{C.STRIP_SEED}-adapter-vs-correction-norm-maps.png", dpi=150, bbox_inches="tight")
    print("[rung3] wrote figures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
