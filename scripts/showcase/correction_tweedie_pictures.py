#!/usr/bin/env python
"""Rung 1 of "what the correction is made of": what each expert, the product, and the joint
prompt think the finished picture is, at the same cached state, and where the two experts
disagree.

For seed 15 at steps 0, 2, 5, 10, 20, 30 and 49 (the last cached step), the Tweedie estimate
x0 = (x_t - sqrt(1 - abar_t) eps) / sqrt(abar_t) is formed under five predictions at the same
cached x_t: the guided cat expert, the guided dog expert, the PoE prediction, the guided joint
prediction, and the unconditional prediction. Each is decoded through the SDXL VAE.

Number, all eight seeds and every step (no decode needed): per latent position p (128 x 128),
m_k(p) = ||x0_k(p) - x0_uncond(p)|| over the 4 channels for k in {a, b}. A position is active
for expert k if m_k(p) is above that map's median (so each expert is active on exactly half the
positions). The same-place share is the fraction of all positions active for both; chance is
0.25. Also the mean cosine, over the positions active for both, between the two 4-channel
departures (positive: the experts push the same way there; negative: they fight).

Because x0_k - x0_uncond = -(sqrt(1 - abar)/sqrt(abar)) * g * (eps_k - eps_u), the masks are
the same whether built on x0 or on eps; x0 is used so the number and the pictures share a unit.

Writes:
    /datasets/.../what_the_correction_is_made_of/tweedie/seed_15/<cond>_step_<kk>.png   the decodes
    artifacts/results/what-the-correction-is-made-of/seed-15-experts-tweedie-strip.png  5 rows of
        decodes x 7 steps, plus a 6th row: heat map of ||x0_a - x0_b|| per latent position
    artifacts/results/what-the-correction-is-made-of/same-place-share.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import correction_span_common as C  # noqa: E402

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONDS = (("cat_expert", "cat expert (guided)"), ("dog_expert", "dog expert (guided)"),
         ("poe", "product of experts"), ("joint", '"a cat and a dog" (guided)'), ("uncond", "empty prompt"))
THUMB = 224


def x0_hat(x_t: torch.Tensor, abar: float, eps: torch.Tensor) -> torch.Tensor:
    return (x_t - (1.0 - abar) ** 0.5 * eps) / abar ** 0.5


def predictions(s: C.Step) -> dict[str, torch.Tensor]:
    return {"cat_expert": C.guided(s.eps_a, s.eps_u), "dog_expert": C.guided(s.eps_b, s.eps_u),
            "poe": s.eps_poe, "joint": s.eps_joint, "uncond": s.eps_u}


def same_place(x0a: torch.Tensor, x0b: torch.Tensor, x0u: torch.Tensor) -> dict:
    da = (x0a - x0u).reshape(4, -1); db = (x0b - x0u).reshape(4, -1)
    ma, mb = da.norm(dim=0), db.norm(dim=0)
    act_a, act_b = ma > ma.median(), mb > mb.median()
    both = act_a & act_b
    cos = torch.nn.functional.cosine_similarity(da[:, both], db[:, both], dim=0)
    return {"same_place_share": float(both.float().mean()),
            "jaccard": float(both.float().sum() / (act_a | act_b).float().sum()),
            "cos_on_shared_positions_mean": float(cos.mean()),
            "cos_on_shared_positions_share_negative": float((cos < 0).float().mean()),
            "cos_all_positions_mean": float(torch.nn.functional.cosine_similarity(da, db, dim=0).mean())}


def main() -> int:
    from diffusers import AutoencoderKL, DDIMScheduler
    from poe_repair._sdxl.runtime import decode_latents_to_tensor

    sched = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    abars = sched.alphas_cumprod.double()

    # ---- the number, every seed and step, CPU ----------------------------------------------
    rows = []
    t0 = time.time()
    for seed in C.SEEDS:
        for k in range(C.num_steps(seed)):
            s = C.load_step(seed, k)
            ab = float(abars[s.timestep])
            p = predictions(s)
            x0 = {c: x0_hat(s.x_t, ab, e) for c, e in p.items()}
            r = same_place(x0["cat_expert"], x0["dog_expert"], x0["uncond"])
            r.update({"seed": seed, "step": k, "timestep": s.timestep, "abar": ab,
                      "norm_x0a_minus_x0b": float((x0["cat_expert"] - x0["dog_expert"]).norm())})
            rows.append(r)
        print(f"seed {seed} same-place done ({time.time() - t0:.0f}s)", flush=True)
    steps = sorted({r["step"] for r in rows})
    def m(key):
        return [float(np.mean([r[key] for r in rows if r["step"] == k])) for k in steps]
    C.write_json(C.RESULTS / "same-place-share.json", {
        "pair": C.PAIR, "seeds": list(C.SEEDS), "chance_level_same_place_share": 0.25,
        "definitions": {
            "same_place_share": "fraction of the 16,384 latent positions where both experts' departure from the unconditional Tweedie estimate (4-channel norm) is above that expert's own median; chance 0.25",
            "jaccard": "positions active for both / positions active for either; chance 1/3",
            "cos_on_shared_positions_mean": "mean cosine between the two 4-channel departures on the shared positions; positive = pushing the same way, negative = fighting",
        },
        "summary": {"same_place_share_mean_per_step": m("same_place_share"),
                    "cos_on_shared_positions_mean_per_step": m("cos_on_shared_positions_mean"),
                    "cos_on_shared_positions_share_negative_per_step": m("cos_on_shared_positions_share_negative"),
                    "cos_all_positions_mean_per_step": m("cos_all_positions_mean"),
                    "steps": steps},
        "rows": rows,
    })
    print("same-place share, mean over seeds, steps 0/5/10/20/30/49:",
          [round(m("same_place_share")[k], 3) for k in (0, 5, 10, 20, 30, 49)])
    print("cos on shared positions:", [round(m("cos_on_shared_positions_mean")[k], 3) for k in (0, 5, 10, 20, 30, 49)])

    # ---- the pictures, seed 15, GPU -------------------------------------------------------------
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available on the pinned device (poe-launch-002); refusing to decode on CPU")
    dev = torch.device("cuda")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float16, use_safetensors=True).to(dev)
    if getattr(vae.config, "force_upcast", False):
        vae = vae.to(dtype=torch.float32)          # same rule as poe_repair/_sdxl/runtime.load_sdxl_models
    out_dir = C.OUT_ROOT / "tweedie" / f"seed_{C.STRIP_SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles: dict[tuple[str, int], Path] = {}
    heat: dict[int, np.ndarray] = {}
    for k in C.STRIP_STEPS:
        s = C.load_step(C.STRIP_SEED, k)
        ab = float(abars[s.timestep])
        p = predictions(s)
        x0 = {c: x0_hat(s.x_t, ab, e) for c, e in p.items()}
        heat[k] = C.per_position_norm(x0["cat_expert"] - x0["dog_expert"])
        for c, _ in CONDS:
            q = out_dir / f"{c}_step_{k:03d}.png"
            if not q.exists():
                with torch.no_grad():
                    img = decode_latents_to_tensor(vae, x0[c].to(device=dev, dtype=vae.dtype))
                arr = (img[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
                Image.fromarray(arr).save(q)
            tiles[(c, k)] = q
        print(f"step {k} decoded ({time.time() - t0:.0f}s)", flush=True)

    # ---- the strip ------------------------------------------------------------------------------
    ncol = len(C.STRIP_STEPS); nrow = len(CONDS) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.1 * nrow))
    vmax = max(h.max() for h in heat.values())
    for j, k in enumerate(C.STRIP_STEPS):
        for i, (c, lab) in enumerate(CONDS):
            ax = axes[i, j]
            ax.imshow(Image.open(tiles[(c, k)]).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS))
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            if i == 0:
                ax.set_title(f"step {k}" + (" (last cached)" if k == 49 else ""), fontsize=9)
        ax = axes[nrow - 1, j]
        im = ax.imshow(heat[k], cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("||x0_cat - x0_dog||\nper latent position", fontsize=8)
        r = next(r for r in rows if r["seed"] == C.STRIP_SEED and r["step"] == k)
        ax.set_xlabel(f"same place {r['same_place_share']:.2f}\ncos there {r['cos_on_shared_positions_mean']:+.2f}", fontsize=7.5)
    fig.colorbar(im, ax=axes[nrow - 1, :].tolist(), fraction=0.012, pad=0.01)
    fig.suptitle(f"cat x dog, seed {C.STRIP_SEED}: the running estimate of the finished picture under each prediction at the SAME cached\n"
                 "state of the plain PoE run, decoded; last row is where the two experts' estimates differ most (one colour scale)",
                 fontsize=10)
    fig.savefig(C.RESULTS / f"seed-{C.STRIP_SEED}-experts-tweedie-strip.png", dpi=150, bbox_inches="tight")
    print("wrote", C.RESULTS / f"seed-{C.STRIP_SEED}-experts-tweedie-strip.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
