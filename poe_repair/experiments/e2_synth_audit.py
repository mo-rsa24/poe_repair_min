"""E2 — Synthesizer audit.

Quantifies how well ê_J approximates e_J on held-out pairs. Two metrics:

  (a) Embedding-level: cosine and MSE between ê_J and e_J for ``seq`` and
      ``pooled`` outputs, per held-out pair.
  (b) UNet-output level: ``‖ε(x_t, t, ê_J) − ε(x_t, t, e_J)‖²`` averaged
      over a small set of (timestep, latent) samples per pair.

Output: a JSON table + one bar chart of (a) per pair.

The (b) metric is the load-bearing one: two embeddings can be far in cosine
yet produce nearly identical UNet outputs (or vice versa). UNet-level
divergence is what determines whether ê_J is a usable substitute for e_J.

Outputs:
    outputs/e2_synth_audit/summary.json
    outputs/e2_synth_audit/figures/embed_cosine_by_pair.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from poe_repair.composers._helpers import init_latents_for_cell
from poe_repair.config import RunConfig, joint_prompt
from poe_repair.embeddings.infer import synthesize_joint
from poe_repair.experiments._eval_common import HELD_OUT_PAIRS, cell_for, slugify
from poe_repair.figures._common import bar_plot
from poe_repair.methods._sampling import add_time_ids
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import encode_prompt_sdxl, ensure_dir, write_json


EXP_NAME = "e2_synth_audit"


def _unet_eps_at(
    *, latent: torch.Tensor, timestep, seq, pool, ctx, height, width,
) -> torch.Tensor:
    """Single conditional ε(x_t, t, e) at one timestep."""
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=1,
            device=ctx.device, dtype=ctx.dtype,
        ),
    }
    out = ctx.models["unet"](
        latent, timestep, encoder_hidden_states=seq,
        added_cond_kwargs=cond, timestep_cond=None,
    ).sample
    return out


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--unet-samples-per-pair", type=int, default=8,
                    help="Number of (t, x_t) samples for the UNet-level metric.")
    args = ap.parse_args()

    cfg = RunConfig()
    ctx: MethodCtx = make_ctx()
    fig_dir = ensure_dir(cfg.paths.output_root / EXP_NAME / "figures")
    ctx.scheduler.set_timesteps(ctx.num_inference_steps)
    timesteps = ctx.scheduler.timesteps

    rows: list[dict] = []
    cosine_by_pair: dict[str, list[float]] = {}

    for prompt_a, prompt_b in HELD_OUT_PAIRS:
        slug = slugify(prompt_a, prompt_b)
        joint = joint_prompt(prompt_a, prompt_b, template=ctx.joint_template)

        seq_J_lit, pool_J_lit = encode_prompt_sdxl(
            joint, models=ctx.models, device=ctx.device, dtype=ctx.dtype,
        )
        synth_out = synthesize_joint(
            ctx.get_synth(), prompt_a=prompt_a, prompt_b=prompt_b,
            models=ctx.models, device=ctx.device, dtype=ctx.dtype,
        )
        seq_J_hat = synth_out.seq.to(ctx.device, ctx.dtype)
        pool_J_hat = synth_out.pooled.to(ctx.device, ctx.dtype)

        seq_cos = float(F.cosine_similarity(
            seq_J_hat.flatten(start_dim=1), seq_J_lit.flatten(start_dim=1), dim=-1,
        ).mean().item())
        pool_cos = float(F.cosine_similarity(
            pool_J_hat, pool_J_lit, dim=-1,
        ).mean().item())
        seq_mse = float(F.mse_loss(seq_J_hat, seq_J_lit).item())
        pool_mse = float(F.mse_loss(pool_J_hat, pool_J_lit).item())

        # UNet-level divergence sampled over (seed, t).
        unet_diffs: list[float] = []
        with torch.no_grad():
            for seed in args.seeds:
                cell = cell_for(prompt_a, prompt_b, seed)
                init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
                lat0 = (init_latents / euler_sigma).to(device=ctx.device, dtype=ctx.dtype)
                step_sample = max(
                    1, ctx.num_inference_steps // args.unet_samples_per_pair,
                )
                for step_index, t in enumerate(timesteps[::step_sample]):
                    latent_input = ctx.scheduler.scale_model_input(lat0, t)
                    eps_lit = _unet_eps_at(
                        latent=latent_input, timestep=t,
                        seq=seq_J_lit, pool=pool_J_lit,
                        ctx=ctx, height=cell.height, width=cell.width,
                    )
                    eps_hat = _unet_eps_at(
                        latent=latent_input, timestep=t,
                        seq=seq_J_hat, pool=pool_J_hat,
                        ctx=ctx, height=cell.height, width=cell.width,
                    )
                    diff = float((eps_lit - eps_hat).pow(2).mean().sqrt().item())
                    unet_diffs.append(diff)
        unet_rmse = float(sum(unet_diffs) / max(1, len(unet_diffs)))

        row = {
            "pair_slug": slug,
            "prompt_a": prompt_a, "prompt_b": prompt_b,
            "seq_cosine": seq_cos, "pool_cosine": pool_cos,
            "seq_mse": seq_mse, "pool_mse": pool_mse,
            "unet_rmse": unet_rmse,
            "unet_n_samples": len(unet_diffs),
        }
        print(f"[E2] {slug}  seq_cos={seq_cos:.3f}  pool_cos={pool_cos:.3f}  "
              f"unet_rmse={unet_rmse:.4f}")
        rows.append(row)
        cosine_by_pair[slug] = [seq_cos, pool_cos]

    summary_path = ensure_dir(cfg.paths.output_root / EXP_NAME) / "summary.json"
    write_json(summary_path, {
        "exp": EXP_NAME,
        "seeds": args.seeds,
        "rows": rows,
    })

    bar_plot(
        list(cosine_by_pair.keys()),
        {
            "seq cosine": [v[0] for v in cosine_by_pair.values()],
            "pool cosine": [v[1] for v in cosine_by_pair.values()],
        },
        fig_dir / "embed_cosine_by_pair.png",
        xlabel="held-out pair",
        ylabel="cosine(ê_J, e_J)",
        title="E2 — Synthesizer reconstruction cosine on held-out pairs",
    )
    print(f"[E2] wrote summary {summary_path}")


if __name__ == "__main__":
    main()
