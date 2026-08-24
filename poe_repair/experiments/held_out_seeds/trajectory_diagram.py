"""Trajectory-divergence diagnostic + diagram.

Renders the geometric picture that motivates LoRA over step-0 injection:
at every step the mono and PoE samplers sit at different latents
(x_t^mono vs x_t^poe), and the four guided-eps predictions
(eps_J(x_t^mono), eps_PoE(x_t^mono), eps_J(x_t^poe), eps_PoE(x_t^poe))
are four different tensors. The cached residual r_oracle reuses the
top bracket at the bottom anchor — that is the off-trajectory error
the figure makes visible.

Subcommands
-----------
``collect``
    Runs mono (single-prompt CFG on the joint prompt) and PoE on the
    given seed from the same pinned init, with a 4-branch UNet forward
    at every step. Dumps per-step (x_t, eps_J_guided, eps_PoE_guided)
    for each trajectory.

``plot``
    Loads the dumps, picks one or more step indices, projects the six
    tensors into a 2D plane spanned by the trajectory-gap direction and
    the average-residual direction, and renders the diagram.

Usage
-----
::

    python -m poe_repair.experiments.held_out_seeds.trajectory_diagram \
        collect --seed 42 \
        --out-dir artifacts/results/can-lora-learn-a-residual-that-corrects-poe/held-out-seeds-index/trajectory_diagram

    python -m poe_repair.experiments.held_out_seeds.trajectory_diagram \
        plot \
        --in-dir artifacts/results/can-lora-learn-a-residual-that-corrects-poe/held-out-seeds-index/trajectory_diagram/seed_42 \
        --steps 0,5,25,45
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from poe_repair.experiments.one_pair_one_seed.main import encode_all_prompts
from poe_repair.experiments.one_pair_one_seed.probe import load_pinned_init_latents
from poe_repair.methods._sampling import add_time_ids, write_decoded_image
from poe_repair.runtime import (
    decode_latents,
    ddim_prev_from_x0_eps,
    ensure_dir,
    guided_eps,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
    poe_eps,
    tweedie_mean,
    write_json,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_with_4branch_log(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    emb: dict,
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    euler_init_noise_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    step_with: str,  # "mono" or "poe"
) -> dict:
    """Run a 4-branch (A, B, J, uncond) sampler. At each step record
    ``(x_t, eps_J_guided, eps_PoE_guided)``; step with whichever
    composite eps matches ``step_with``.

    ``step_with == "mono"`` -> step using eps_J_guided (mono trajectory).
    ``step_with == "poe"``  -> step using eps_PoE_guided (PoE trajectory).
    """
    if step_with not in ("mono", "poe"):
        raise ValueError(f"step_with must be 'mono' or 'poe', got {step_with!r}")

    scheduler.set_timesteps(num_inference_steps)
    latents = (init_latents / euler_init_noise_sigma).to(device=device, dtype=dtype)

    pe = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_j"], emb["seq_e"]], dim=0)
    pool = torch.cat(
        [emb["pool_a"], emb["pool_b"], emb["pool_j"], emb["pool_e"]], dim=0
    )
    cond = {
        "text_embeds": pool,
        "time_ids": add_time_ids(
            height=height, width=width, batch_size=4, device=device, dtype=dtype,
        ),
    }
    unet = models["unet"]

    x_log: list[torch.Tensor] = []
    eps_j_log: list[torch.Tensor] = []
    eps_poe_log: list[torch.Tensor] = []
    timesteps_log: list[int] = []
    step_indices_log: list[int] = []

    for step_index, timestep in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(latents.repeat(4, 1, 1, 1), timestep)
        noise = unet(
            latent_input, timestep, encoder_hidden_states=pe,
            added_cond_kwargs=cond, timestep_cond=None,
        ).sample
        eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond = noise.chunk(4)

        eps_a_g = guided_eps(eps_a_raw, eps_uncond, guidance_scale)
        eps_b_g = guided_eps(eps_b_raw, eps_uncond, guidance_scale)
        eps_j_g = guided_eps(eps_j_raw, eps_uncond, guidance_scale)
        eps_poe_g = poe_eps(eps_a_g, eps_b_g, eps_uncond)

        x_log.append(latents.detach().float().cpu())
        eps_j_log.append(eps_j_g.detach().float().cpu())
        eps_poe_log.append(eps_poe_g.detach().float().cpu())
        timesteps_log.append(int(timestep.item()))
        step_indices_log.append(int(step_index))

        eps_step = eps_j_g if step_with == "mono" else eps_poe_g

        alpha_bar_t = scheduler.alphas_cumprod[int(timestep.item())].to(
            device=device, dtype=dtype,
        )
        x0 = tweedie_mean(latents, alpha_bar_t, eps_step)
        latents = ddim_prev_from_x0_eps(
            scheduler=scheduler, timestep=timestep, step_index=step_index,
            x0=x0, eps=eps_step,
        )

    image = decode_latents(models, latents).cpu()

    return {
        "step_with": step_with,
        "step_indices": step_indices_log,
        "timesteps": timesteps_log,
        "x_t_per_step": torch.stack(x_log, dim=0),       # (T, 1, 4, H, W) fp32 cpu
        "eps_j_per_step": torch.stack(eps_j_log, dim=0),
        "eps_poe_per_step": torch.stack(eps_poe_log, dim=0),
        "final_latents": latents.detach().float().cpu(),
        "final_image": image,
        "guidance_scale": float(guidance_scale),
    }


def cmd_collect(args: argparse.Namespace) -> int:
    pair_slug = "a_cat__x__a_dog"  # current beachhead pair
    seed = int(args.seed)

    cache_root = Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT
    cell = CellPath.from_root(pair_slug, seed, cache_root=cache_root)

    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype, device)
    log.info("device=%s dtype=%s", device, dtype)
    models = load_sdxl_models(model_id=args.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(args.model_id)

    class _PromptShim:
        prompt_a = args.prompt_a
        prompt_b = args.prompt_b
        joint_prompt = args.joint_prompt

    class _CfgShim:
        cell = _PromptShim()

    emb = encode_all_prompts(_CfgShim(), models, device, dtype)

    init = load_pinned_init_latents(
        cell, device=device, dtype=dtype,
        euler_init_noise_sigma=float(args.euler_sigma),
    )

    out_root = Path(args.out_dir)
    seed_dir = ensure_dir(out_root / f"seed_{seed:02d}")

    common = dict(
        init_latents=init, models=models, scheduler=scheduler, emb=emb,
        guidance_scale=float(args.guidance_scale),
        num_inference_steps=int(args.num_inference_steps),
        height=int(args.height), width=int(args.width),
        euler_init_noise_sigma=float(args.euler_sigma),
        device=device, dtype=dtype,
    )

    log.info("collecting mono trajectory ...")
    mono = _run_with_4branch_log(**common, step_with="mono")
    log.info("collecting PoE trajectory ...")
    poe = _run_with_4branch_log(**common, step_with="poe")

    torch.save(
        {k: v for k, v in mono.items() if k != "final_image"},
        seed_dir / "mono.pt",
    )
    torch.save(
        {k: v for k, v in poe.items() if k != "final_image"},
        seed_dir / "poe.pt",
    )
    write_decoded_image(mono["final_image"], seed_dir / "mono.png")
    write_decoded_image(poe["final_image"], seed_dir / "poe.png")

    # Sanity: at step 0 both samplers share x_T (pinned init). Confirm.
    diff0 = float(
        (mono["x_t_per_step"][0] - poe["x_t_per_step"][0]).abs().max().item()
    )
    diff_last = float(
        (mono["x_t_per_step"][-1] - poe["x_t_per_step"][-1]).abs().max().item()
    )
    log.info(
        "sanity: max|x_0^mono - x_0^poe|=%.3e (should be ~0); "
        "max|x_{T-1}^mono - x_{T-1}^poe|=%.3e (should be > 0)",
        diff0, diff_last,
    )

    write_json(seed_dir / "meta.json", {
        "seed": seed,
        "pair_slug": pair_slug,
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "joint_prompt": args.joint_prompt,
        "guidance_scale": float(args.guidance_scale),
        "num_inference_steps": int(args.num_inference_steps),
        "height": int(args.height), "width": int(args.width),
        "euler_sigma": float(args.euler_sigma),
        "shared_x_T_check_max_abs_diff": diff0,
        "final_step_max_abs_diff": diff_last,
        "timesteps": mono["timesteps"],
    })
    log.info("collect → %s", seed_dir)
    return 0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _project_step(
    *,
    x_mono: torch.Tensor, x_poe: torch.Tensor,
    eps_j_mono: torch.Tensor, eps_poe_mono: torch.Tensor,
    eps_j_poe: torch.Tensor, eps_poe_poe: torch.Tensor,
    fallback_u2_seed: int = 0,
) -> dict:
    """Project the six tensors at one step into a 2D (u1, u2) plane.

    ``u1`` points from x_poe to x_mono.
    ``u2`` is the average residual direction, orthogonalised against u1.
    If the two anchors coincide (step 0), u1 falls back to the average-Δ
    direction and u2 to a deterministic orthogonal complement.
    """

    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.detach().float().flatten()

    xm = flat(x_mono)
    xp = flat(x_poe)
    ejm = flat(eps_j_mono)
    epm = flat(eps_poe_mono)
    ejp = flat(eps_j_poe)
    epp = flat(eps_poe_poe)

    delta_mono = ejm - epm
    delta_poe = ejp - epp
    delta_bar = 0.5 * (delta_mono + delta_poe)

    gap = xm - xp
    gap_norm = float(gap.norm().item())
    eps_floor = 1e-8
    if gap_norm < eps_floor:
        # Step 0: anchors coincide. Build u1 from average residual.
        u1 = delta_bar / max(float(delta_bar.norm().item()), eps_floor)
        # Pick a fixed orthogonal direction (random unit, project out u1).
        g = torch.Generator().manual_seed(fallback_u2_seed)
        rand = torch.randn(u1.numel(), generator=g)
        u2 = rand - torch.dot(rand, u1) * u1
        u2 = u2 / max(float(u2.norm().item()), eps_floor)
    else:
        u1 = gap / gap_norm
        # u2 from average residual, orthogonalise against u1.
        v = delta_bar - torch.dot(delta_bar, u1) * u1
        v_norm = float(v.norm().item())
        if v_norm < eps_floor:
            # Residuals fully aligned with u1: fall back to a deterministic
            # orthogonal complement.
            g = torch.Generator().manual_seed(fallback_u2_seed)
            rand = torch.randn(u1.numel(), generator=g)
            v = rand - torch.dot(rand, u1) * u1
            v_norm = float(v.norm().item())
        u2 = v / max(v_norm, eps_floor)

    def coords(t: torch.Tensor) -> tuple[float, float]:
        return float(torch.dot(t, u1).item()), float(torch.dot(t, u2).item())

    return {
        "u1_dim": int(u1.numel()),
        "anchors": {"mono": coords(xm), "poe": coords(xp)},
        "eps_at_mono": {"j": coords(ejm), "poe": coords(epm)},
        "eps_at_poe": {"j": coords(ejp), "poe": coords(epp)},
        "delta_mono": coords(delta_mono),
        "delta_poe": coords(delta_poe),
        "gap_norm": gap_norm,
        "delta_mono_norm": float(delta_mono.norm().item()),
        "delta_poe_norm": float(delta_poe.norm().item()),
    }


def _render_panel(ax, proj: dict, step_index: int, timestep: int) -> None:
    """Draw one (u1, u2) panel. Arrows are scaled to ~0.6 × anchor gap;
    every label is anchored with a fixed offset in *display points* and a
    thin connector line to its tip, so labels never pile up regardless of
    how zoomed-in the data is.
    """
    am = proj["anchors"]["mono"]
    ap = proj["anchors"]["poe"]
    ej_m = proj["eps_at_mono"]["j"]
    ep_m = proj["eps_at_mono"]["poe"]
    ej_p = proj["eps_at_poe"]["j"]
    ep_p = proj["eps_at_poe"]["poe"]

    anchor_gap = (
        ((am[0] - ap[0]) ** 2 + (am[1] - ap[1]) ** 2) ** 0.5
    )
    raw_arrow_lengths = [
        (ej_m[0] ** 2 + ej_m[1] ** 2) ** 0.5,
        (ep_m[0] ** 2 + ep_m[1] ** 2) ** 0.5,
        (ej_p[0] ** 2 + ej_p[1] ** 2) ** 0.5,
        (ep_p[0] ** 2 + ep_p[1] ** 2) ** 0.5,
    ]
    max_raw = max(raw_arrow_lengths) if raw_arrow_lengths else 1.0
    target = max(0.6 * anchor_gap, 1e-3) if anchor_gap > 1e-6 else 1.0
    s = target / max(max_raw, 1e-12)

    drawn_pts: list[tuple[float, float]] = [am, ap]

    def _label_at(xy, text, color, offset_xy, *, italic=False, weight="normal"):
        """Anchor a label at fixed display-point offset, with a thin
        connector line and a translucent box so it always reads cleanly
        against arrows or grid lines.
        """
        ax.annotate(
            text, xy=xy, xytext=offset_xy, textcoords="offset points",
            color=color, fontsize=10, ha="center", va="center",
            style="italic" if italic else "normal", fontweight=weight,
            arrowprops=dict(
                arrowstyle="-", color=color, lw=0.6, alpha=0.7,
                shrinkA=0, shrinkB=2,
            ),
            bbox=dict(
                boxstyle="round,pad=0.25", fc="white", ec=color,
                lw=0.6, alpha=0.92,
            ),
            zorder=5,
        )

    def draw_arrow(base, vec, color, ls, label, label_offset_pts):
        tip = (base[0] + s * vec[0], base[1] + s * vec[1])
        ax.annotate(
            "", xy=tip, xytext=base,
            arrowprops=dict(arrowstyle="->", color=color, linestyle=ls, lw=2.0),
        )
        drawn_pts.append(tip)
        _label_at(tip, label, color, label_offset_pts)
        return tip

    # Anchor gap line first (lowest z-order).
    ax.plot([am[0], ap[0]], [am[1], ap[1]], color="0.7", lw=0.9, zorder=1)

    # The two latent anchors.
    ax.scatter([am[0]], [am[1]], s=130, c="tab:red", zorder=3,
               edgecolor="black", linewidth=0.7)
    ax.scatter([ap[0]], [ap[1]], s=130, c="tab:blue", zorder=3,
               edgecolor="black", linewidth=0.7)
    # When anchors coincide (step 0), splay the two anchor labels in
    # opposite vertical directions so they don't overlap.
    if anchor_gap < 1e-6:
        _label_at(am, r"$x_t^{mono}$", "tab:red", (0, +28), weight="bold")
        _label_at(ap, r"$x_t^{poe}$", "tab:blue", (0, -28), weight="bold")
    else:
        # Mono on the right of the panel by construction of u1.
        _label_at(am, r"$x_t^{mono}$", "tab:red", (+22, +14), weight="bold")
        _label_at(ap, r"$x_t^{poe}$", "tab:blue", (-22, -14), weight="bold")

    # Arrows. Fixed display-point offsets per arrow so labels never collide.
    #   mono anchor (right):  J → upper-right,  PoE → lower-right
    #   poe  anchor (left):   J → upper-left,   PoE → lower-left
    tip_j_m = draw_arrow(am, ej_m, "tab:orange", "-",
                         r"$\tilde{\epsilon}_J(x_t^{mono})$", (+55, +35))
    tip_p_m = draw_arrow(am, ep_m, "tab:red", "-",
                         r"$\tilde{\epsilon}_{PoE}(x_t^{mono})$", (+55, -35))
    tip_j_p = draw_arrow(ap, ej_p, "tab:cyan", "--",
                         r"$\tilde{\epsilon}_J(x_t^{poe})$", (-55, +35))
    tip_p_p = draw_arrow(ap, ep_p, "tab:blue", "--",
                         r"$\tilde{\epsilon}_{PoE}(x_t^{poe})$", (-55, -35))

    # Δ bracket at mono (cached) — sits between the two solid tips.
    ax.plot([tip_p_m[0], tip_j_m[0]], [tip_p_m[1], tip_j_m[1]],
            color="0.4", lw=1.0, ls=":", zorder=2)
    mid_m = (0.5 * (tip_p_m[0] + tip_j_m[0]), 0.5 * (tip_p_m[1] + tip_j_m[1]))
    _label_at(mid_m, r"$\Delta_{mono}$ (cached)", "0.2", (+85, 0))
    drawn_pts.append(mid_m)

    # Δ bracket at poe (needed) — sits between the two dashed tips.
    ax.plot([tip_p_p[0], tip_j_p[0]], [tip_p_p[1], tip_j_p[1]],
            color="0.4", lw=1.0, ls=":", zorder=2)
    mid_p = (0.5 * (tip_p_p[0] + tip_j_p[0]), 0.5 * (tip_p_p[1] + tip_j_p[1]))
    _label_at(mid_p, r"$\Delta_{poe}$ (needed)", "0.2", (-85, 0))
    drawn_pts.append(mid_p)

    # r_oracle ghost: copy Δ_mono and re-anchor at poe dot.
    dm = proj["delta_mono"]
    ghost_tip = (ap[0] + s * dm[0], ap[1] + s * dm[1])
    ax.annotate(
        "", xy=ghost_tip, xytext=ap,
        arrowprops=dict(arrowstyle="->", color="0.5", linestyle="--",
                        lw=1.6, alpha=0.8),
    )
    ax.scatter([ghost_tip[0]], [ghost_tip[1]], marker="x", c="black",
               s=110, zorder=4, linewidth=2.0)
    _label_at(ghost_tip, "r_oracle\n(off-trajectory)", "0.2",
              (0, -45), italic=True)
    drawn_pts.append(ghost_tip)

    ax.set_title(
        f"step {step_index}  (timestep t={timestep})\n"
        rf"$\|x_t^{{mono}}-x_t^{{poe}}\|={proj['gap_norm']:.2f}$    "
        f"arrows ×{s:.2g}",
        fontsize=11,
    )
    ax.set_xlabel(r"$u_1 \propto (x_t^{mono}-x_t^{poe})$", fontsize=10)
    ax.set_ylabel(r"$u_2 \perp u_1$  (avg-$\Delta$ direction)", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # Generous viewport: expand to all drawn points + 20% margin on each
    # side, then square the box so equal-aspect does not override our limits.
    xs = [p[0] for p in drawn_pts]
    ys = [p[1] for p in drawn_pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span_x = max(xmax - xmin, 1e-3)
    span_y = max(ymax - ymin, 1e-3)
    pad_x = 0.20 * span_x
    pad_y = 0.20 * span_y
    xmin -= pad_x; xmax += pad_x
    ymin -= pad_y; ymax += pad_y
    # Make data ranges equal so equal-aspect is satisfied without
    # silently dropping the smaller axis.
    span_x = xmax - xmin
    span_y = ymax - ymin
    if span_x > span_y:
        cy = 0.5 * (ymin + ymax)
        ymin = cy - 0.5 * span_x
        ymax = cy + 0.5 * span_x
    else:
        cx = 0.5 * (xmin + xmax)
        xmin = cx - 0.5 * span_y
        xmax = cx + 0.5 * span_y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    in_dir = Path(args.in_dir)
    mono_d = torch.load(in_dir / "mono.pt", map_location="cpu", weights_only=False)
    poe_d = torch.load(in_dir / "poe.pt", map_location="cpu", weights_only=False)

    step_ids = [int(s) for s in args.steps.split(",") if s.strip()]
    T = int(mono_d["x_t_per_step"].shape[0])
    for s in step_ids:
        if s < 0 or s >= T:
            raise ValueError(f"step {s} outside [0,{T})")

    timesteps = list(mono_d["timesteps"])
    projs: list[tuple[int, int, dict]] = []
    for s in step_ids:
        proj = _project_step(
            x_mono=mono_d["x_t_per_step"][s],
            x_poe=poe_d["x_t_per_step"][s],
            eps_j_mono=mono_d["eps_j_per_step"][s],
            eps_poe_mono=mono_d["eps_poe_per_step"][s],
            eps_j_poe=poe_d["eps_j_per_step"][s],
            eps_poe_poe=poe_d["eps_poe_per_step"][s],
        )
        projs.append((s, timesteps[s], proj))

    n = len(projs)
    # Grid layout: 1 row if n <= 2, else 2 rows, ceil(n/2) cols.
    if n <= 2:
        nrows, ncols = 1, n
    else:
        ncols = (n + 1) // 2
        nrows = 2
    panel_w, panel_h = 9.0, 8.0  # inches per panel
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_w * ncols, panel_h * nrows),
        squeeze=False,
    )
    flat_axes = [ax for row in axes for ax in row]
    for ax, (s, ts, proj) in zip(flat_axes, projs):
        _render_panel(ax, proj, s, ts)
    # Hide any unused axes (e.g. 3 panels in a 2×2 grid).
    for ax in flat_axes[n:]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="tab:red", markersize=10, label=r"$x_t^{mono}$"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="tab:blue", markersize=10, label=r"$x_t^{poe}$"),
        plt.Line2D([0], [0], color="tab:orange", lw=2.2,
                   label=r"$\tilde{\epsilon}_J$ @ mono"),
        plt.Line2D([0], [0], color="tab:red", lw=2.2,
                   label=r"$\tilde{\epsilon}_{PoE}$ @ mono"),
        plt.Line2D([0], [0], color="tab:cyan", lw=2.2, ls="--",
                   label=r"$\tilde{\epsilon}_J$ @ poe"),
        plt.Line2D([0], [0], color="tab:blue", lw=2.2, ls="--",
                   label=r"$\tilde{\epsilon}_{PoE}$ @ poe"),
        plt.Line2D([0], [0], color="0.5", lw=1.8, ls="--",
                   label=r"r_oracle ($\Delta_{mono}$ reanchored at poe)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, fontsize=11,
        frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"Latent-space view of the off-trajectory error  (seed in {in_dir.name})",
        fontsize=14, y=0.995,
    )
    # Reserve top + bottom strips for suptitle + legend.
    bottom_pad = 0.08 if nrows == 1 else 0.05
    fig.tight_layout(rect=(0.01, bottom_pad, 0.99, 0.96))

    out_path = Path(args.out) if args.out else (in_dir / "trajectory_diagram.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("plot → %s", out_path)

    # Also dump the projection numbers as JSON for inspection.
    write_json(in_dir / "projection_meta.json", {
        "step_indices": [s for s, _, _ in projs],
        "timesteps": [ts for _, ts, _ in projs],
        "panels": [
            {"step_index": s, "timestep": ts, **p} for s, ts, p in projs
        ],
    })
    return 0


# ---------------------------------------------------------------------------
# Time-vs-projection plot (continuous-time view, à la SDE sample paths)
# ---------------------------------------------------------------------------


def _flatten_steps(stack: torch.Tensor) -> torch.Tensor:
    """(T, 1, C, H, W) -> (T, D) float32 on CPU."""
    return stack.detach().float().reshape(stack.shape[0], -1)


def _top_pca_directions(M: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k right singular vectors of a (N, D) mean-centred matrix.
    Returns (k, D) tensor of unit-norm directions.
    """
    Mc = M - M.mean(dim=0, keepdim=True)
    # Use torch.linalg.svd; D may be huge so prefer the reduced SVD.
    _U, _S, Vh = torch.linalg.svd(Mc, full_matrices=False)
    return Vh[:k]


def cmd_plot_time(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    in_dir = Path(args.in_dir)
    mono_d = torch.load(in_dir / "mono.pt", map_location="cpu", weights_only=False)
    poe_d = torch.load(in_dir / "poe.pt", map_location="cpu", weights_only=False)

    xm = _flatten_steps(mono_d["x_t_per_step"])        # (T, D)
    xp = _flatten_steps(poe_d["x_t_per_step"])         # (T, D)
    T = xm.shape[0]
    step_idx = list(range(T))
    timesteps = list(mono_d["timesteps"])

    # 1D projection direction: the top PCA direction of the gap-vector
    # trajectory {x_t^mono - x_t^poe}. This is the axis along which the
    # two trajectories separate most strongly over time.
    gaps = xm - xp                                     # (T, D)
    direction = _top_pca_directions(gaps, k=1)[0]      # (D,)
    direction = direction / max(float(direction.norm().item()), 1e-12)

    proj_m = (xm @ direction).tolist()
    proj_p = (xp @ direction).tolist()
    gap_norm_per_step = [
        float((xm[i] - xp[i]).norm().item()) for i in range(T)
    ]

    fig, axes = plt.subplots(
        2, 1, figsize=(11.0, 9.0), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.4]},
    )
    ax_top, ax_bot = axes

    # Top panel: projection vs step.
    ax_top.plot(step_idx, proj_m, color="tab:red", lw=2.2, label=r"$x_t^{mono}$")
    ax_top.plot(step_idx, proj_p, color="tab:blue", lw=2.2, label=r"$x_t^{poe}$")
    ax_top.fill_between(
        step_idx, proj_m, proj_p, color="0.7", alpha=0.30,
        label="off-trajectory error (cached vs needed)",
    )
    ax_top.scatter([0], [proj_m[0]], s=120, c="black", zorder=4)
    ax_top.annotate(
        "shared $x_T$\n(pinned init)", xy=(0, proj_m[0]),
        xytext=(20, -25), textcoords="offset points",
        fontsize=10, ha="left", va="top",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.6),
    )
    ax_top.scatter([T - 1], [proj_m[-1]], s=100, c="tab:red", zorder=4,
                   edgecolor="black", linewidth=0.6)
    ax_top.scatter([T - 1], [proj_p[-1]], s=100, c="tab:blue", zorder=4,
                   edgecolor="black", linewidth=0.6)
    ax_top.annotate(
        "mono basin", xy=(T - 1, proj_m[-1]),
        xytext=(-20, 18), textcoords="offset points",
        fontsize=10, color="tab:red", ha="right",
        arrowprops=dict(arrowstyle="-", color="tab:red", lw=0.7),
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec="tab:red", lw=0.6, alpha=0.9),
    )
    ax_top.annotate(
        "poe basin", xy=(T - 1, proj_p[-1]),
        xytext=(-20, -18), textcoords="offset points",
        fontsize=10, color="tab:blue", ha="right",
        arrowprops=dict(arrowstyle="-", color="tab:blue", lw=0.7),
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec="tab:blue", lw=0.6, alpha=0.9),
    )
    ax_top.set_ylabel("projection of $x_t$ onto top-PC\nof the gap trajectory",
                      fontsize=11)
    ax_top.set_title(
        f"Mono and PoE trajectories over sampling steps  (seed in {in_dir.name})",
        fontsize=13,
    )
    ax_top.legend(loc="upper left", fontsize=10, frameon=True)
    ax_top.grid(True, alpha=0.3)

    # Optional secondary x-axis showing the diffusion timestep.
    ax2 = ax_top.twiny()
    ax2.set_xlim(ax_top.get_xlim())
    tick_idx = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([str(timesteps[i]) for i in tick_idx], fontsize=9)
    ax2.set_xlabel("diffusion timestep $t$ (noisy → clean)", fontsize=10)

    # Bottom panel: ‖x_t^mono - x_t^poe‖ vs step (the gap norm).
    ax_bot.plot(step_idx, gap_norm_per_step, color="black", lw=2.0)
    ax_bot.fill_between(step_idx, 0, gap_norm_per_step, color="0.7", alpha=0.35)
    ax_bot.set_ylabel(r"$\|x_t^{mono} - x_t^{poe}\|$", fontsize=11)
    ax_bot.set_xlabel("sampling step index (noisy → clean)", fontsize=11)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_xlim(0, T - 1)

    fig.tight_layout()
    out_path = Path(args.out) if args.out else (in_dir / "trajectory_time1d.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("plot_time → %s", out_path)

    write_json(in_dir / "trajectory_time1d_meta.json", {
        "step_indices": step_idx,
        "timesteps": timesteps,
        "proj_mono": proj_m,
        "proj_poe": proj_p,
        "gap_norm_per_step": gap_norm_per_step,
    })
    return 0


# ---------------------------------------------------------------------------
# 2D phase-plane plot (continuous trajectories through latent space)
# ---------------------------------------------------------------------------


def cmd_plot_phase(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    in_dir = Path(args.in_dir)
    mono_d = torch.load(in_dir / "mono.pt", map_location="cpu", weights_only=False)
    poe_d = torch.load(in_dir / "poe.pt", map_location="cpu", weights_only=False)

    xm = _flatten_steps(mono_d["x_t_per_step"])
    xp = _flatten_steps(poe_d["x_t_per_step"])
    T = xm.shape[0]
    timesteps = list(mono_d["timesteps"])

    # Fit 2D PCA on the union of both trajectories so the plane captures
    # the geometry both share. Mean-centred for honest PCA.
    union = torch.cat([xm, xp], dim=0)                 # (2T, D)
    dirs = _top_pca_directions(union, k=2)             # (2, D)
    mean = union.mean(dim=0, keepdim=True)

    def project(M: torch.Tensor) -> torch.Tensor:
        return (M - mean) @ dirs.T                     # (N, 2)

    pm = project(xm).numpy()
    pp = project(xp).numpy()

    fig, ax = plt.subplots(figsize=(10.5, 9.0))

    def draw_path(pts, base_color: str, label: str, lw: float = 2.0):
        # Build a color gradient from light (early, t = T-ish) to saturated
        # (late, t ~ 0) along the path.
        import matplotlib.colors as mcolors
        from matplotlib.colors import LinearSegmentedColormap

        c0 = mcolors.to_rgb(base_color)
        c_light = tuple(min(1.0, 0.65 + 0.35 * c) for c in c0)
        cmap = LinearSegmentedColormap.from_list(f"{label}_cmap", [c_light, c0])
        segs = [
            [(pts[i, 0], pts[i, 1]), (pts[i + 1, 0], pts[i + 1, 1])]
            for i in range(len(pts) - 1)
        ]
        colors = [cmap(i / max(1, len(segs) - 1)) for i in range(len(segs))]
        lc = LineCollection(segs, colors=colors, linewidths=lw, capstyle="round")
        ax.add_collection(lc)
        # Marker dots at each step on top of the line.
        ax.scatter(
            pts[:, 0], pts[:, 1], s=18,
            c=[cmap(i / max(1, len(pts) - 1)) for i in range(len(pts))],
            zorder=3, edgecolor="white", linewidth=0.4,
        )
        # Endpoint marker.
        ax.scatter([pts[-1, 0]], [pts[-1, 1]], s=180, c=base_color, zorder=5,
                   edgecolor="black", linewidth=0.8, label=label)

    draw_path(pm, "tab:red", r"$x_t^{mono}$ trajectory")
    draw_path(pp, "tab:blue", r"$x_t^{poe}$ trajectory")

    # Shared start point.
    ax.scatter([pm[0, 0]], [pm[0, 1]], s=220, marker="*", c="black",
               zorder=6, edgecolor="white", linewidth=0.8)
    ax.annotate(
        "shared $x_T$\n(pinned init)", xy=(pm[0, 0], pm[0, 1]),
        xytext=(20, 20), textcoords="offset points",
        fontsize=11, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.6),
    )
    ax.annotate(
        "mono basin\n(endpoint $x_0^{mono}$)", xy=(pm[-1, 0], pm[-1, 1]),
        xytext=(20, 20), textcoords="offset points",
        fontsize=11, color="tab:red", ha="left",
        arrowprops=dict(arrowstyle="-", color="tab:red", lw=0.7),
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="tab:red", lw=0.6, alpha=0.92),
    )
    ax.annotate(
        "poe basin\n(endpoint $x_0^{poe}$)", xy=(pp[-1, 0], pp[-1, 1]),
        xytext=(20, -20), textcoords="offset points",
        fontsize=11, color="tab:blue", ha="left",
        arrowprops=dict(arrowstyle="-", color="tab:blue", lw=0.7),
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="tab:blue", lw=0.6, alpha=0.92),
    )

    # Pair up matching timesteps with thin grey segments at chosen markers,
    # to visualise that at the same t the two paths are at different points.
    markers = [int(x) for x in args.marker_steps.split(",") if x.strip()]
    for k in markers:
        if 0 <= k < T:
            ax.plot([pm[k, 0], pp[k, 0]], [pm[k, 1], pp[k, 1]],
                    color="0.5", lw=0.8, ls=":", zorder=2)
            ax.annotate(
                f"t={timesteps[k]}", xy=(0.5 * (pm[k, 0] + pp[k, 0]),
                                          0.5 * (pm[k, 1] + pp[k, 1])),
                xytext=(0, 0), textcoords="offset points",
                fontsize=8, color="0.3", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="0.5", lw=0.4, alpha=0.85),
            )

    ax.set_xlabel("PC 1 of (mono ∪ poe) latents", fontsize=11)
    ax.set_ylabel("PC 2 of (mono ∪ poe) latents", fontsize=11)
    ax.set_title(
        f"Phase-plane view of the two trajectories  (seed in {in_dir.name})\n"
        "color fades from light (noisy, t≈T) to saturated (clean, t≈0)",
        fontsize=12,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, frameon=True)

    fig.tight_layout()
    out_path = Path(args.out) if args.out else (in_dir / "trajectory_phase2d.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("plot_phase → %s", out_path)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="trajectory_diagram")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="run mono + poe with 4-branch logging")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--prompt-a", default="a cat")
    c.add_argument("--prompt-b", default="a dog")
    c.add_argument("--joint-prompt", default="a cat and a dog")
    c.add_argument("--guidance-scale", type=float, default=7.5)
    c.add_argument("--num-inference-steps", type=int, default=50)
    c.add_argument("--height", type=int, default=1024)
    c.add_argument("--width", type=int, default=1024)
    c.add_argument("--euler-sigma", type=float, default=1.0)
    c.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    c.add_argument("--device", default=None)
    c.add_argument(
        "--dtype", default="float16",
        choices=("float16", "fp16", "float32", "fp32", "bfloat16", "bf16"),
    )
    c.add_argument("--cache-root", default=None)

    p = sub.add_parser("plot", help="render the snapshot trajectory diagram (4 arrow panels)")
    p.add_argument(
        "--in-dir", required=True,
        help="directory containing mono.pt / poe.pt (e.g. .../seed_42)",
    )
    p.add_argument(
        "--steps", default="0,5,25,45",
        help="comma-separated step indices to render as panels",
    )
    p.add_argument("--out", default=None, help="output PNG (default: <in-dir>/trajectory_diagram.png)")

    pt = sub.add_parser("plot_time", help="projection of x_t vs sampling step (continuous-time view)")
    pt.add_argument("--in-dir", required=True)
    pt.add_argument("--out", default=None,
                    help="output PNG (default: <in-dir>/trajectory_time1d.png)")

    pp = sub.add_parser("plot_phase", help="2D phase-plane plot of both trajectories")
    pp.add_argument("--in-dir", required=True)
    pp.add_argument(
        "--marker-steps", default="0,5,15,25,35,45",
        help="comma-separated steps to highlight with paired-position markers",
    )
    pp.add_argument("--out", default=None,
                    help="output PNG (default: <in-dir>/trajectory_phase2d.png)")

    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("CROSS_SEED_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_argparser().parse_args(argv)
    if args.cmd == "collect":
        return cmd_collect(args)
    if args.cmd == "plot":
        return cmd_plot(args)
    if args.cmd == "plot_time":
        return cmd_plot_time(args)
    if args.cmd == "plot_phase":
        return cmd_plot_phase(args)
    raise SystemExit(f"unknown cmd: {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
