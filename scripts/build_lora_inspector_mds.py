"""Pre-render MDS trajectory panels for the LoRA Inspector residual tab.

One PNG per (epoch, lambda) cell, on the same grid the existing decoded
images use. Style copied from neurips2026/scripts/render_taxonomy_paper_figure.py
(palette, axis treatment, path styling, boxed letter labels at endpoints).

The four trajectories on every panel:
    A         — solo prompt-A CFG, static
    B         — solo prompt-B CFG, static
    A ∧ B     — mono (joint-prompt CFG), static
    PoE + λ·R — LoRA-corrected PoE at this cell, dynamic

A global MDS is fit once on the union of every trajectory so that the
static endpoints really do sit at fixed coordinates across all panels and
only the PoE+λ·R path moves with the sliders.

Stages (composable via --stages):
    collect-static  : run solo-A, solo-B, mono samplers; cache 50-step traj
    collect-cells   : per (epoch, λ), load checkpoint, run LoRA sampler,
                      cache 50-step trajectory under mds_cache/
    project         : fit one MDS on the global pool, write coords.json
    render          : per cell, render mds.png from coords
    update-manifest : add mds_cells map to inspector_manifest.json

Designed for cat × dog seed 42 (single-cell beachhead). The script
defaults match that setup.

Usage::

    # Smoke set (4 epochs × full λ grid, ~5 minutes on one A100):
    python scripts/build_lora_inspector_mds.py \\
        --epochs 0,200,800,1600 \\
        --stages collect-static,collect-cells,project,render,update-manifest

    # Production sweep (all 48 epochs × 5 λ in the manifest):
    python scripts/build_lora_inspector_mds.py --epochs all \\
        --stages collect-static,collect-cells,project,render,update-manifest
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from poe_repair.experiments.lora import trainer as lora_trainer
from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    add_time_ids,
    initial_latents_for_pair,
    run_cfg,
    run_lora_residual_inject,
    write_decoded_image,
)
from poe_repair.runtime import (
    decode_latents,
    ensure_dir,
    infer_device,
    infer_dtype,
    load_ddim_scheduler,
    load_sdxl_models,
)
from poe_repair.training_cache import DEFAULT_CACHE_ROOT, CellPath


log = logging.getLogger("build_lora_inspector_mds")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local"
DEFAULT_PAIR_SLUG = "a_cat__x__a_dog"

# Style — copied from neurips2026 render_taxonomy_paper_figure.py.
COND_COLORS = {
    "solo_a":   "#C84C5B",
    "solo_b":   "#2B6F97",
    "mono":     "#3B8D5B",
    "poe_lora": "#D9872B",
}
COND_LABELS = {
    "solo_a":   "A",
    "solo_b":   "B",
    "mono":     "A∧B",
    "poe_lora": "PoE",
}
COND_MARKERS = {
    "solo_a":   "o",
    "solo_b":   "s",
    "mono":     "D",
    "poe_lora": "^",
}
LABEL_OFFSETS = {
    "solo_a":   (10, 10),
    "solo_b":   (10, -12),
    "mono":     (-16, 12),
    "poe_lora": (-18, -12),
}
SUBPLOT_TITLE = "Latent trajectories — A, B, A∧B fixed; PoE + λ·R moves"


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


@dataclass
class Layout:
    """Filesystem layout for one MDS builder run.

    The ``mode`` field selects which *embedding* the trajectories were
    computed in. The embedding-dependent dirs (``cache_dir``,
    ``panels_dir``, ``large_panels_dir``, ``coords_path``) differ by
    mode; the embedding-*independent* artefacts (decoded terminal
    thumbnails, the cached flat-latent trajectories used as a fallback
    for non-probed LoRA terminals) always live under the latent-mode
    ``mds_cache/`` so they can be reused across modes.

    - ``mode="latent"``: original behaviour (Plan-04 raw-z_t MDS).
    - ``mode="semantic"``: Plan-13 DINOv2-on-x̂₀(t) MDS, with its own
      cache / panels / coords. Re-uses the latent-mode decoded
      thumbnails (which don't depend on what we embed).
    """

    results_root: Path
    cache_dir: Path           # mds_cache/  (or mds_cache_semantic/)
    panels_dir: Path          # mds_probes/ (or mds_probes_semantic/)
    large_panels_dir: Path    # mds_probes_large/ (or mds_probes_large_semantic/)
    coords_path: Path
    mode: str = "latent"

    @classmethod
    def build(cls, results_root: Path, mode: str = "latent") -> "Layout":
        if mode == "latent":
            return cls(
                results_root=results_root,
                cache_dir=results_root / "mds_cache",
                panels_dir=results_root / "mds_probes",
                large_panels_dir=results_root / "mds_probes_large",
                coords_path=results_root / "mds_cache" / "coords.json",
                mode=mode,
            )
        if mode == "semantic":
            return cls(
                results_root=results_root,
                cache_dir=results_root / "mds_cache_semantic",
                panels_dir=results_root / "mds_probes_semantic",
                large_panels_dir=results_root / "mds_probes_large_semantic",
                coords_path=(
                    results_root / "mds_cache_semantic" / "coords_semantic.json"
                ),
                mode=mode,
            )
        raise ValueError(f"unknown Layout mode: {mode!r}")

    # ---- embedding-dependent (per-mode) ----

    def cell_traj_path(self, epoch: int, lam: float) -> Path:
        """In latent mode this holds the flat (T+1, D) trajectory.
        In semantic mode it holds the (T_sub+1, 384) DINOv2 CLS sequence."""
        return (
            self.cache_dir / "cells" / f"epoch_{epoch:04d}"
            / f"lambda_{lam:.2f}.npy"
        )

    def static_traj_path(self, name: str) -> Path:
        return self.cache_dir / "static" / f"{name}.npy"

    def panel_path(self, epoch: int, lam: float) -> Path:
        return (
            self.panels_dir / f"epoch_{epoch:04d}"
            / f"lambda_{lam:.2f}" / "mds.png"
        )

    def large_panel_path(self, epoch: int, lam: float) -> Path:
        """Large single-plot MDS panel with decorated terminal thumbnails."""
        return (
            self.large_panels_dir / f"epoch_{epoch:04d}"
            / f"lambda_{lam:.2f}" / "mds_large.png"
        )

    # ---- embedding-independent (always latent-mode locations) ----

    def _latent_cache_dir(self) -> Path:
        return self.results_root / "mds_cache"

    def static_image_path(self, name: str) -> Path:
        """Decoded final-step thumbnail for a static condition. Shared
        across modes — the terminal image doesn't depend on what we
        embed for the MDS."""
        return self._latent_cache_dir() / "static" / f"{name}.png"

    def probe_decoded_path(self, epoch: int, lam: float) -> Path:
        """Existing LoRA decoded probe — terminal thumbnail for the PoE+λ·R arm."""
        return (
            self.results_root / "probes" / f"epoch_{epoch:04d}"
            / f"lambda_{lam:.2f}" / "decoded.png"
        )

    def latent_cell_traj_path(self, epoch: int, lam: float) -> Path:
        """Cached *flat-latent* trajectory at (epoch, λ). Lives under
        the latent ``mds_cache`` regardless of mode — used by the
        fallback VAE-decode path that produces a terminal thumbnail
        for a LoRA cell that wasn't probed at training time."""
        return (
            self._latent_cache_dir() / "cells" / f"epoch_{epoch:04d}"
            / f"lambda_{lam:.2f}.npy"
        )

    def cached_terminal_decoded_path(self, epoch: int, lam: float) -> Path:
        """VAE-decoded fallback thumbnail for a LoRA terminal whose probe
        decoded.png does not exist on disk (e.g. λ values that weren't
        probed during training). Sits next to the cached *latent* flat
        trajectory in the latent ``mds_cache``, so semantic mode reuses
        the same decoded thumbnail without re-decoding."""
        return (
            self._latent_cache_dir() / "cells" / f"epoch_{epoch:04d}"
            / f"lambda_{lam:.2f}.decoded.png"
        )


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _flat_trajectory(out_tracker) -> np.ndarray:
    """tracker.trajectories is (T+1, 1, 4, H, W). Flatten to (T+1, D)."""
    traj = out_tracker.trajectories.cpu().float().numpy()
    T_plus_1 = traj.shape[0]
    return traj.reshape(T_plus_1, -1)


@torch.no_grad()
def _run_solo(
    *,
    label: str,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_cond: torch.Tensor,
    pool_cond: torch.Tensor,
    seq_e: torch.Tensor,
    pool_e: torch.Tensor,
    cfg_sampler: dict,
    device,
    dtype,
    save_image_to: Path | None = None,
) -> np.ndarray:
    log.info("sampling %s ...", label)
    out = run_cfg(
        init_latents=init_latents,
        models=models, scheduler=scheduler,
        seq_cond=seq_cond, pool_cond=pool_cond,
        seq_e=seq_e, pool_e=pool_e,
        guidance_scale=cfg_sampler["guidance_scale"],
        num_inference_steps=cfg_sampler["num_inference_steps"],
        height=cfg_sampler["height"], width=cfg_sampler["width"],
        euler_init_noise_sigma=cfg_sampler["euler_init_noise_sigma"],
        device=device, dtype=dtype,
    )
    if save_image_to is not None:
        save_image_to.parent.mkdir(parents=True, exist_ok=True)
        write_decoded_image(out.image, save_image_to)
        log.info("  wrote %s thumbnail -> %s", label, save_image_to)
    return _flat_trajectory(out.tracker)


@torch.no_grad()
def _run_poe_lora(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    embeddings: dict,
    lambda_value: float,
    adapter_name: str,
    cfg_sampler: dict,
    device,
    dtype,
) -> np.ndarray:
    out = run_lora_residual_inject(
        init_latents=init_latents,
        models=models, scheduler=scheduler,
        seq_a=embeddings["seq_a"], pool_a=embeddings["pool_a"],
        seq_b=embeddings["seq_b"], pool_b=embeddings["pool_b"],
        seq_j=embeddings["seq_j"], pool_j=embeddings["pool_j"],
        seq_e=embeddings["seq_e"], pool_e=embeddings["pool_e"],
        guidance_scale=cfg_sampler["guidance_scale"],
        num_inference_steps=cfg_sampler["num_inference_steps"],
        height=cfg_sampler["height"], width=cfg_sampler["width"],
        euler_init_noise_sigma=cfg_sampler["euler_init_noise_sigma"],
        device=device, dtype=dtype,
        lambda_value=float(lambda_value),
        lora_adapter_name=adapter_name,
        record_delta_at_steps=None,
        correction_max_rel_norm=None,
    )
    return _flat_trajectory(out.tracker)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _parse_lambda_list(s: str, all_lambdas: list[str]) -> list[float]:
    if s.lower() == "all":
        return [float(x) for x in all_lambdas]
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_epoch_list(s: str, all_epochs: list[int]) -> list[int]:
    if s.lower() == "all":
        return list(all_epochs)
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _lambda_str(lam: float) -> str:
    return f"{lam:.2f}"


def _checkpoint_for_epoch(
    ckpt_dir: Path, epoch: int, epoch_size: int,
) -> Path:
    """Map epoch -> lora_step_NNNNNN.pt via epoch_size."""
    step = int(epoch) * int(epoch_size)
    p = ckpt_dir / f"lora_step_{step:06d}.pt"
    if not p.is_file():
        raise FileNotFoundError(
            f"no checkpoint for epoch {epoch} (looked for {p})"
        )
    return p


# ---------------------------------------------------------------------------
# Stage 1+2: collect static + cell trajectories
# ---------------------------------------------------------------------------


def collect_trajectories(
    *,
    results_root: Path,
    layout: Layout,
    pair_slug: str,
    seed: int,
    cache_root: Path,
    epochs: list[int],
    lambdas: list[float],
    do_static: bool,
    do_cells: bool,
    overwrite: bool,
    device_arg: str | None,
    dtype_arg: str,
    model_id: str,
) -> None:
    config = json.loads((results_root / "config.json").read_text())
    cell_cfg = config["cell"]
    lora_cfg = config["lora"]
    sampler_cfg = config["sampler"]
    epoch_size = int(config["schedule"]["epoch_size"])

    device = infer_device(device_arg)
    dtype = infer_dtype(dtype_arg, device)
    log.info("device=%s dtype=%s", device, dtype)

    models = load_sdxl_models(model_id=model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(model_id)

    class _PromptShim:
        prompt_a = cell_cfg["prompt_a"]
        prompt_b = cell_cfg["prompt_b"]
        joint_prompt = cell_cfg["joint_prompt"]

    class _CfgShim:
        cell = _PromptShim()

    embeddings = encode_all_prompts(_CfgShim(), models, device, dtype)

    pinned_cell = CellPath.from_root(pair_slug, int(seed), cache_root=cache_root)
    init_latents = load_pinned_init_latents(
        pinned_cell, device=device, dtype=dtype,
        euler_init_noise_sigma=float(sampler_cfg["euler_init_noise_sigma"]),
    )

    cfg_sampler = {
        "guidance_scale": float(sampler_cfg["guidance_scale"]),
        "num_inference_steps": int(sampler_cfg["num_inference_steps"]),
        "height": int(sampler_cfg["height"]),
        "width": int(sampler_cfg["width"]),
        "euler_init_noise_sigma": float(sampler_cfg["euler_init_noise_sigma"]),
    }

    if do_static:
        ensure_dir(layout.cache_dir / "static")
        static_specs = [
            ("solo_a", embeddings["seq_a"], embeddings["pool_a"]),
            ("solo_b", embeddings["seq_b"], embeddings["pool_b"]),
            ("mono",   embeddings["seq_j"], embeddings["pool_j"]),
        ]
        for name, seq_cond, pool_cond in static_specs:
            out_path = layout.static_traj_path(name)
            img_path = layout.static_image_path(name)
            need_traj = overwrite or not out_path.is_file()
            need_img = overwrite or not img_path.is_file()
            if not need_traj and not need_img:
                log.info("[skip] static %s already cached (traj + thumbnail)", name)
                continue
            flat = _run_solo(
                label=name,
                init_latents=init_latents,
                models=models, scheduler=scheduler,
                seq_cond=seq_cond, pool_cond=pool_cond,
                seq_e=embeddings["seq_e"], pool_e=embeddings["pool_e"],
                cfg_sampler=cfg_sampler,
                device=device, dtype=dtype,
                save_image_to=img_path if need_img else None,
            )
            ensure_dir(out_path.parent)
            np.save(out_path, flat.astype(np.float16))
            log.info("  saved %s shape=%s -> %s", name, flat.shape, out_path)

    if not do_cells:
        return

    # Attach LoRA shell so we can swap states between checkpoints. Use the
    # config from the first epoch's checkpoint (all share the same shell).
    rank = int(lora_cfg["rank"])
    alpha = int(lora_cfg["alpha"])
    target_modules = tuple(lora_cfg["target_modules"])
    adapter_name = str(lora_cfg["adapter_name"])
    log.info("attaching LoRA shell: rank=%d alpha=%d adapter=%s",
             rank, alpha, adapter_name)
    from peft import LoraConfig
    unet = models["unet"]
    unet.add_adapter(
        LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
            target_modules=list(target_modules), init_lora_weights=True,
        ),
        adapter_name=adapter_name,
    )
    with torch.no_grad():
        for name_p, p in unet.named_parameters():
            if "lora_" in name_p:
                p.data = p.data.to(torch.float32)
            p.requires_grad_(False)
    unet.eval()

    ckpt_dir = results_root / "checkpoints"
    ensure_dir(layout.cache_dir / "cells")
    for epoch in epochs:
        ckpt_path = _checkpoint_for_epoch(ckpt_dir, epoch, epoch_size)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        lora_trainer.load_lora_state(unet, ckpt["lora_state"])
        log.info("loaded ckpt epoch=%d step=%d (%s)",
                 epoch, ckpt["step"], ckpt_path.name)
        for lam in lambdas:
            out_path = layout.cell_traj_path(epoch, lam)
            if out_path.is_file() and not overwrite:
                log.info("  [skip] epoch=%d λ=%.2f already cached", epoch, lam)
                continue
            ensure_dir(out_path.parent)
            flat = _run_poe_lora(
                init_latents=init_latents,
                models=models, scheduler=scheduler,
                embeddings=embeddings,
                lambda_value=lam,
                adapter_name=adapter_name,
                cfg_sampler=cfg_sampler,
                device=device, dtype=dtype,
            )
            np.save(out_path, flat.astype(np.float16))
            log.info("  saved epoch=%d λ=%.2f shape=%s", epoch, lam, flat.shape)


# ---------------------------------------------------------------------------
# Stage 3: global MDS projection
# ---------------------------------------------------------------------------


def project_global(layout: Layout, method: str = "pca") -> None:
    """Pool all cached trajectories, fit one global projection, save coords.

    method:
        "pca" — fast, deterministic; usually cleaner on diffusion latents
                because the dominant variance is the x_T → x_0 direction
                per trajectory. Default.
        "mds" — preserves pairwise euclidean distances; can be jagged
                when many near-collinear within-trajectory points compete
                with the global structure.
    """
    import warnings

    static_names = ["solo_a", "solo_b", "mono"]
    items: list[tuple[str, np.ndarray]] = []
    for name in static_names:
        p = layout.static_traj_path(name)
        if not p.is_file():
            raise FileNotFoundError(f"missing static traj: {p}")
        items.append((f"static/{name}", np.load(p).astype(np.float32)))

    cells_dir = layout.cache_dir / "cells"
    cell_keys: list[str] = []
    for epoch_dir in sorted(cells_dir.glob("epoch_*")):
        for lam_path in sorted(epoch_dir.glob("lambda_*.npy")):
            key = f"cells/{epoch_dir.name}/{lam_path.stem}"
            items.append((key, np.load(lam_path).astype(np.float32)))
            cell_keys.append(key)

    log.info("global projection (%s) pool: %d trajectories", method, len(items))
    stacked = np.concatenate([arr for _, arr in items], axis=0)
    log.info("  stacked shape=%s", stacked.shape)

    if method == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=42).fit_transform(stacked)
    elif method == "mds":
        from sklearn.manifold import MDS
        from sklearn.metrics import pairwise_distances
        dist = pairwise_distances(stacked, metric="euclidean")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proj = MDS(
                n_components=2,
                random_state=42,
                dissimilarity="precomputed",
                normalized_stress="auto",
                n_init=4,
            ).fit_transform(dist)
    else:
        raise ValueError(f"unknown projection method: {method!r}")

    coords: dict[str, list[list[float]]] = {}
    start = 0
    for key, arr in items:
        end = start + arr.shape[0]
        coords[key] = proj[start:end].tolist()
        start = end

    layout.coords_path.parent.mkdir(parents=True, exist_ok=True)
    layout.coords_path.write_text(json.dumps(coords))
    log.info("wrote %s (%d entries)", layout.coords_path, len(coords))


# ---------------------------------------------------------------------------
# Stage 4: render panels
# ---------------------------------------------------------------------------


def _style_manifold_axis(ax, prefix: str = "MDS") -> None:
    ax.set_facecolor("#FBFCFD")
    ax.grid(True, color="#E1E7EF", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.margins(x=0.12, y=0.14)
    ax.set_xlabel(f"{prefix} 1", fontsize=9.0)
    ax.set_ylabel(f"{prefix} 2", fontsize=9.0)
    ax.tick_params(axis="both", labelsize=8.0, colors="#4C566A")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#C7D0DB")


def _plot_path(ax, pts: np.ndarray, color: str, marker: str) -> None:
    if len(pts) >= 2:
        ax.plot(
            pts[:, 0], pts[:, 1],
            color=color, linewidth=2.15, alpha=0.95,
            solid_capstyle="round", zorder=2,
        )
        tail = pts[-3:] if len(pts) >= 3 else pts[-2:]
        ax.plot(
            tail[:, 0], tail[:, 1],
            color=color, linewidth=3.0, alpha=1.0,
            solid_capstyle="round", zorder=3,
        )
        if np.linalg.norm(pts[-1] - pts[-2]) > 0:
            ax.annotate(
                "",
                xy=(pts[-1, 0], pts[-1, 1]),
                xytext=(pts[-2, 0], pts[-2, 1]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.5,
                    "shrinkA": 0, "shrinkB": 0,
                    "mutation_scale": 10,
                },
                zorder=4,
            )
    mid = max(1, len(pts) // 2)
    ax.plot(
        pts[mid, 0], pts[mid, 1],
        marker=marker, linestyle="none",
        markersize=4.8, markerfacecolor="white",
        markeredgecolor=color, markeredgewidth=1.1, zorder=5,
    )
    ax.plot(
        pts[-1, 0], pts[-1, 1],
        marker=marker, color=color, linestyle="none",
        markersize=7.6, markeredgecolor="white",
        markeredgewidth=0.8, zorder=6,
    )


def _annotate_endpoint(ax, name: str, endpoint: np.ndarray, color: str) -> None:
    ax.annotate(
        COND_LABELS[name],
        xy=(endpoint[0], endpoint[1]),
        xytext=LABEL_OFFSETS[name],
        textcoords="offset points",
        ha="center", va="center",
        fontsize=7.8, fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white", "edgecolor": color,
            "linewidth": 0.9, "alpha": 0.96,
        },
        zorder=7,
    )


def render_panels(
    layout: Layout,
    *,
    overwrite: bool = False,
    epoch_subset: list[int] | None = None,
    lambda_subset: list[float] | None = None,
    method: str = "pca",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = json.loads(layout.coords_path.read_text())
    static_coords = {
        "solo_a": np.asarray(coords["static/solo_a"]),
        "solo_b": np.asarray(coords["static/solo_b"]),
        "mono":   np.asarray(coords["static/mono"]),
    }
    # Pool all coords for a stable axes extent.
    all_pts = np.concatenate([np.asarray(v) for v in coords.values()], axis=0)
    pad_x = 0.08 * (all_pts[:, 0].max() - all_pts[:, 0].min())
    pad_y = 0.10 * (all_pts[:, 1].max() - all_pts[:, 1].min())
    xlim = (all_pts[:, 0].min() - pad_x, all_pts[:, 0].max() + pad_x)
    ylim = (all_pts[:, 1].min() - pad_y, all_pts[:, 1].max() + pad_y)

    cells_dir = layout.cache_dir / "cells"
    epoch_re = re.compile(r"^epoch_(\d+)$")
    lam_re = re.compile(r"^lambda_(\d+\.\d+)$")

    rendered = 0
    for epoch_dir in sorted(cells_dir.glob("epoch_*")):
        em = epoch_re.match(epoch_dir.name)
        if not em:
            continue
        epoch = int(em.group(1))
        if epoch_subset is not None and epoch not in epoch_subset:
            continue
        for lam_path in sorted(epoch_dir.glob("lambda_*.npy")):
            lm = lam_re.match(lam_path.stem)
            if not lm:
                continue
            lam = float(lm.group(1))
            if lambda_subset is not None and not any(
                abs(lam - x) < 1e-9 for x in lambda_subset
            ):
                continue
            key = f"cells/{epoch_dir.name}/{lam_path.stem}"
            if key not in coords:
                log.warning("no coords for %s; rerun project stage", key)
                continue
            out_png = layout.panel_path(epoch, lam)
            if out_png.is_file() and not overwrite:
                continue

            cell_pts = np.asarray(coords[key])

            fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=120)
            for name in ("solo_a", "solo_b", "mono"):
                pts = static_coords[name]
                _plot_path(ax, pts, COND_COLORS[name], COND_MARKERS[name])
            _plot_path(
                ax, cell_pts, COND_COLORS["poe_lora"], COND_MARKERS["poe_lora"],
            )

            for name in ("solo_a", "solo_b", "mono"):
                _annotate_endpoint(
                    ax, name, static_coords[name][-1], COND_COLORS[name],
                )
            _annotate_endpoint(
                ax, "poe_lora", cell_pts[-1], COND_COLORS["poe_lora"],
            )

            origin = static_coords["mono"][0]
            ax.plot(
                origin[0], origin[1],
                marker="o", color="black", linestyle="none",
                markersize=5.4, markeredgecolor="white",
                markeredgewidth=0.9, zorder=6,
            )
            ax.annotate(
                r"$x_T$",
                xy=(origin[0], origin[1]),
                fontsize=8.0, fontweight="bold",
                textcoords="offset points",
                xytext=(-11, -10),
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white", "edgecolor": "#B8C2CC",
                    "linewidth": 0.8, "alpha": 0.94,
                },
                zorder=7,
            )

            _style_manifold_axis(ax, prefix="PC" if method == "pca" else "MDS")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(
                f"epoch {epoch}  ·  λ = {lam:.2f}",
                fontsize=9.5, color="#2A2F36", pad=6,
            )
            fig.tight_layout()
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=120, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)
            rendered += 1

    log.info("rendered %d MDS panels under %s", rendered, layout.panels_dir)


# ---------------------------------------------------------------------------
# Stage 4b: render LARGE single-plot panels with terminal thumbnails
# ---------------------------------------------------------------------------


# Visual constants for the large panel — tuned to read clearly in a single
# big plot rather than a 3-up row of small ones.
LARGE_FIGSIZE = (12.0, 6.2)   # inches — wider, shorter (data fans horizontally)
LARGE_DPI = 130
LARGE_PATH_LW = 2.8
LARGE_STATIC_PATH_ALPHA = 0.95
LARGE_DYN_PATH_LW = 3.4
LARGE_THUMB_SIZE = 1.1        # inches per thumbnail (source size)
LARGE_THUMB_ZOOM = 0.38       # OffsetImage zoom → ~0.42" rendered
LARGE_THUMB_BORDER = 5        # pixel border width matching path colour
LARGE_THUMB_SPREAD_IN = 1.05  # inches to push each thumbnail off its terminal
COND_NICKNAMES = {
    "solo_a":   "A",
    "solo_b":   "B",
    "mono":     "A∧B",
    "poe_lora": "LoRA",
}


def _label_font(size_px: int):
    """Best-available bold font for a PIL ImageDraw label overlay."""
    try:
        from matplotlib import font_manager as fm
        from PIL import ImageFont
        path = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
        return ImageFont.truetype(path, size_px)
    except Exception:
        try:
            from PIL import ImageFont
            return ImageFont.load_default()
        except Exception:
            return None


def _load_image_array(path: Path):
    """Read a PNG into a uint8 ndarray (H, W, 3 or 4)."""
    import matplotlib.image as mpimg
    arr = mpimg.imread(str(path))
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _bordered_thumbnail(
    image_path: Path,
    *,
    target_px: int,
    border_px: int,
    border_color: str,
    label: str | None = None,
):
    """Return a numpy thumbnail of `image_path` resized to `target_px` square
    with a solid-colour border `border_px` wide. If `label` is provided,
    a small colour-filled pill with white text is drawn in the top-left
    corner (inside the border)."""
    import matplotlib.colors as mcolors
    arr = _load_image_array(image_path)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    h, w = arr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    arr = arr[y0:y0 + side, x0:x0 + side]
    try:
        from PIL import Image
        pil = Image.fromarray(arr).resize(
            (target_px, target_px), Image.LANCZOS,
        )
        arr = np.asarray(pil, dtype=np.uint8)
    except Exception:
        log.warning("PIL not available for resize; using raw image")
    if border_px > 0:
        rgb = (np.asarray(mcolors.to_rgb(border_color)) * 255).astype(np.uint8)
        bordered = np.tile(rgb, (target_px, target_px, 1))
        bordered[border_px:-border_px, border_px:-border_px] = (
            arr[border_px:-border_px, border_px:-border_px]
        )
        arr = bordered

    if label:
        try:
            from PIL import Image, ImageDraw
            pil = Image.fromarray(arr)
            draw = ImageDraw.Draw(pil)
            font_size = max(12, target_px // 9)
            font = _label_font(font_size)
            if font is not None:
                # Use textbbox where available (Pillow ≥ 8.0); fall back to
                # textsize on older versions.
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    asc = bbox[1]
                except AttributeError:
                    text_w, text_h = draw.textsize(label, font=font)
                    asc = 0
                pad_x, pad_y = 6, 3
                rect_x0 = border_px + 3
                rect_y0 = border_px + 3
                rect_x1 = rect_x0 + text_w + 2 * pad_x
                rect_y1 = rect_y0 + text_h + 2 * pad_y
                try:
                    draw.rounded_rectangle(
                        [rect_x0, rect_y0, rect_x1, rect_y1],
                        radius=4, fill=border_color,
                        outline="white", width=1,
                    )
                except AttributeError:
                    draw.rectangle(
                        [rect_x0, rect_y0, rect_x1, rect_y1],
                        fill=border_color, outline="white", width=1,
                    )
                draw.text(
                    (rect_x0 + pad_x, rect_y0 + pad_y - asc),
                    label, fill="white", font=font,
                )
                arr = np.asarray(pil, dtype=np.uint8)
        except Exception as exc:
            log.warning("could not embed label %r: %s", label, exc)
    return arr


def _place_thumbnail_offsetbox(
    ax,
    *,
    image_path: Path,
    xy: tuple[float, float],
    offset_inches: tuple[float, float],
    color: str,
    label: str,
    zorder: int = 10,
):
    """Plant a bordered thumbnail at offset from `xy` (in data coords). The
    label is embedded as a small colour-filled pill in the thumbnail's
    top-left corner, so labels stay visible even when thumbnails partially
    overlap."""
    from matplotlib.offsetbox import (
        AnnotationBbox,
        OffsetImage,
    )
    target_px = int(LARGE_THUMB_SIZE * LARGE_DPI)
    arr = _bordered_thumbnail(
        image_path,
        target_px=target_px,
        border_px=LARGE_THUMB_BORDER,
        border_color=color,
        label=label,
    )
    oi = OffsetImage(arr, zoom=LARGE_THUMB_ZOOM)
    ox_pt = offset_inches[0] * 72.0
    oy_pt = offset_inches[1] * 72.0
    abox = AnnotationBbox(
        oi,
        xy=xy,
        xybox=(ox_pt, oy_pt),
        xycoords="data",
        boxcoords="offset points",
        frameon=False,
        pad=0.0,
        zorder=zorder,
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            lw=1.4,
            alpha=0.80,
            shrinkA=2,
            shrinkB=2,
        ),
    )
    ax.add_artist(abox)


def _spread_thumbnails(
    static_coords: dict[str, np.ndarray],
    cell_coords: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    """Pick a thumbnail offset per terminal so that the four thumbnails fan
    away from the cloud centre rather than overlapping the paths.

    Returns inch-units offsets keyed by condition name."""
    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])
    push = LARGE_THUMB_SPREAD_IN
    out: dict[str, tuple[float, float]] = {}
    terminals = {
        "solo_a":   static_coords["solo_a"][-1],
        "solo_b":   static_coords["solo_b"][-1],
        "mono":     static_coords["mono"][-1],
        "poe_lora": cell_coords[-1],
    }
    for name, pt in terminals.items():
        dx = pt[0] - cx
        dy = pt[1] - cy
        n = (dx * dx + dy * dy) ** 0.5
        if n < 1e-6:
            out[name] = (push, push)
            continue
        out[name] = (push * dx / n, push * dy / n)
    return out


def _lazy_vae_decode_terminal(
    *,
    layout: Layout,
    epoch: int,
    lam: float,
    vae_state: dict,
    model_id: str,
    device_arg: str | None,
    dtype_arg: str,
) -> Path:
    """Decode the cached trajectory's final-step latent through the VAE and
    save the result alongside the .npy. Used as a fallback when the
    on-disk LoRA decoded probe is missing for this cell.

    `vae_state` carries lazy-loaded {'models', 'device', 'dtype'}. The first
    call populates it; subsequent calls reuse the loaded VAE."""
    out_png = layout.cached_terminal_decoded_path(epoch, lam)
    if out_png.is_file():
        return out_png

    # Always read from the latent flat-trajectory cache — even in semantic
    # mode — because we need a (T+1, 4·H·W) latent to reshape and decode.
    npy_path = layout.latent_cell_traj_path(epoch, lam)
    if not npy_path.is_file():
        raise FileNotFoundError(
            f"no cached latent trajectory for epoch={epoch} λ={lam:.2f}: {npy_path}"
        )
    flat = np.load(npy_path)
    last = flat[-1]
    total = last.size
    if total % 4 != 0:
        raise RuntimeError(
            f"unexpected flat-latent size {total} at {npy_path}"
        )
    side = int(round((total / 4) ** 0.5))
    if 4 * side * side != total:
        raise RuntimeError(
            f"cannot reshape flat latent of size {total} to (1, 4, H, W) "
            f"at {npy_path}"
        )

    if "models" not in vae_state:
        log.info(
            "loading SDXL VAE for terminal fallback "
            "(first missing decoded probe encountered)..."
        )
        device = infer_device(device_arg)
        dtype = infer_dtype(dtype_arg, device)
        models = load_sdxl_models(model_id=model_id, device=device, dtype=dtype)
        vae_state.update(models=models, device=device, dtype=dtype)
    models = vae_state["models"]
    device = vae_state["device"]
    dtype = vae_state["dtype"]

    z = (
        torch.from_numpy(last.astype(np.float32))
        .reshape(1, 4, side, side)
        .to(device=device, dtype=dtype)
    )
    with torch.no_grad():
        image = decode_latents(models, z)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    write_decoded_image(image, out_png)
    log.info("  decoded fallback terminal -> %s", out_png)
    return out_png


def _resolve_lora_thumbnail(
    *,
    layout: Layout,
    epoch: int,
    lam: float,
    vae_state: dict,
    model_id: str,
    device_arg: str | None,
    dtype_arg: str,
) -> Path:
    """Return the best available decoded PNG for the LoRA terminal at
    (epoch, λ). Prefers the training-time probe; falls back to a
    VAE-decoded thumbnail of the cached trajectory."""
    probe = layout.probe_decoded_path(epoch, lam)
    if probe.is_file():
        return probe
    return _lazy_vae_decode_terminal(
        layout=layout, epoch=epoch, lam=lam,
        vae_state=vae_state, model_id=model_id,
        device_arg=device_arg, dtype_arg=dtype_arg,
    )


def render_large_panels(
    layout: Layout,
    *,
    overwrite: bool = False,
    epoch_subset: list[int] | None = None,
    lambda_subset: list[float] | None = None,
    method: str = "pca",
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device_arg: str | None = None,
    dtype_arg: str = "float16",
) -> None:
    """Render the single-plot LARGE MDS panel for each (epoch, λ) cell.

    Output: ``mds_probes_large/epoch_NNNN/lambda_X.YY/mds_large.png``.

    Decoded-image thumbnails are rendered at the terminal of each path:
        A, B, A∧B from ``mds_cache/static/{solo_a,solo_b,mono}.png``
        LoRA      from ``probes/epoch_NNNN/lambda_X.YY/decoded.png``
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = json.loads(layout.coords_path.read_text())
    static_coords = {
        "solo_a": np.asarray(coords["static/solo_a"]),
        "solo_b": np.asarray(coords["static/solo_b"]),
        "mono":   np.asarray(coords["static/mono"]),
    }
    all_pts = np.concatenate([np.asarray(v) for v in coords.values()], axis=0)
    # Tight x — small padding since the figure is now wider than tall and
    # we want the data to fill the horizontal extent. Slightly tighter y
    # so the figure isn't taller than the data warrants.
    pad_x = 0.10 * (all_pts[:, 0].max() - all_pts[:, 0].min())
    pad_y = 0.18 * (all_pts[:, 1].max() - all_pts[:, 1].min())
    xlim = (all_pts[:, 0].min() - pad_x, all_pts[:, 0].max() + pad_x)
    ylim = (all_pts[:, 1].min() - pad_y, all_pts[:, 1].max() + pad_y)

    # Verify static thumbnails exist; without them the panel is incomplete.
    missing_static = [
        name for name in ("solo_a", "solo_b", "mono")
        if not layout.static_image_path(name).is_file()
    ]
    if missing_static:
        raise FileNotFoundError(
            "missing static thumbnails: "
            + ", ".join(str(layout.static_image_path(n)) for n in missing_static)
            + " — re-run --stages collect-static (with --overwrite if needed) "
              "so decoded PNGs are produced."
        )

    cells_dir = layout.cache_dir / "cells"
    epoch_re = re.compile(r"^epoch_(\d+)$")
    lam_re = re.compile(r"^lambda_(\d+\.\d+)\.npy$")

    # Lazy VAE-state shared across all cells that need fallback decoding.
    vae_state: dict = {}

    rendered = 0
    for epoch_dir in sorted(cells_dir.glob("epoch_*")):
        em = epoch_re.match(epoch_dir.name)
        if not em:
            continue
        epoch = int(em.group(1))
        if epoch_subset is not None and epoch not in epoch_subset:
            continue
        for lam_path in sorted(epoch_dir.glob("lambda_*.npy")):
            lm = lam_re.match(lam_path.name)
            if not lm:
                continue
            lam = float(lm.group(1))
            if lambda_subset is not None and not any(
                abs(lam - x) < 1e-9 for x in lambda_subset
            ):
                continue
            key = f"cells/{epoch_dir.name}/{lam_path.stem}"
            if key not in coords:
                log.warning("no coords for %s; rerun project stage", key)
                continue
            out_png = layout.large_panel_path(epoch, lam)
            if out_png.is_file() and not overwrite:
                continue

            cell_pts = np.asarray(coords[key])
            try:
                lora_thumb = _resolve_lora_thumbnail(
                    layout=layout, epoch=epoch, lam=lam,
                    vae_state=vae_state, model_id=model_id,
                    device_arg=device_arg, dtype_arg=dtype_arg,
                )
            except Exception as exc:
                log.warning(
                    "could not resolve LoRA thumbnail for epoch=%d λ=%.2f "
                    "(probe missing + fallback failed: %s) — skipping",
                    epoch, lam, exc,
                )
                continue

            fig, ax = plt.subplots(figsize=LARGE_FIGSIZE, dpi=LARGE_DPI)
            # Static paths first (slightly thinner so the dynamic LoRA arm pops).
            for name in ("solo_a", "solo_b", "mono"):
                pts = static_coords[name]
                ax.plot(
                    pts[:, 0], pts[:, 1],
                    color=COND_COLORS[name], linewidth=LARGE_PATH_LW,
                    alpha=LARGE_STATIC_PATH_ALPHA,
                    solid_capstyle="round",
                    zorder=3,
                )
                ax.scatter(
                    pts[:, 0], pts[:, 1], s=14, color=COND_COLORS[name],
                    edgecolor="white", linewidth=0.5,
                    alpha=0.9, zorder=4,
                )
            # Dynamic path — slightly thicker / brighter.
            ax.plot(
                cell_pts[:, 0], cell_pts[:, 1],
                color=COND_COLORS["poe_lora"], linewidth=LARGE_DYN_PATH_LW,
                alpha=0.98, solid_capstyle="round",
                zorder=5,
            )
            ax.scatter(
                cell_pts[:, 0], cell_pts[:, 1], s=20,
                color=COND_COLORS["poe_lora"], edgecolor="white",
                linewidth=0.6, alpha=0.95, zorder=6,
            )

            origin = static_coords["mono"][0]
            ax.plot(
                origin[0], origin[1],
                marker="o", color="black", linestyle="none",
                markersize=8.0, markeredgecolor="white",
                markeredgewidth=1.2, zorder=7,
            )
            ax.annotate(
                r"$x_T$",
                xy=(origin[0], origin[1]),
                fontsize=12.0, fontweight="bold",
                textcoords="offset points",
                xytext=(-14, -14),
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white", "edgecolor": "#B8C2CC",
                    "linewidth": 1.0, "alpha": 0.96,
                },
                zorder=8,
            )

            # Terminal thumbnails — fan outward from the centre of the cloud.
            offsets = _spread_thumbnails(static_coords, cell_pts, xlim, ylim)
            thumb_specs = [
                ("solo_a", static_coords["solo_a"][-1],
                 layout.static_image_path("solo_a")),
                ("solo_b", static_coords["solo_b"][-1],
                 layout.static_image_path("solo_b")),
                ("mono", static_coords["mono"][-1],
                 layout.static_image_path("mono")),
                ("poe_lora", cell_pts[-1], lora_thumb),
            ]
            for name, xy, thumb_path in thumb_specs:
                _place_thumbnail_offsetbox(
                    ax,
                    image_path=thumb_path,
                    xy=(float(xy[0]), float(xy[1])),
                    offset_inches=offsets[name],
                    color=COND_COLORS[name],
                    label=COND_NICKNAMES[name],
                    zorder=10,
                )

            _style_manifold_axis(ax, prefix="PC" if method == "pca" else "MDS")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(
                f"epoch {epoch}  ·  λ = {lam:.2f}",
                fontsize=15.0, color="#2A2F36", pad=10, fontweight="bold",
            )
            fig.tight_layout()
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=LARGE_DPI, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)
            rendered += 1

    log.info(
        "rendered %d LARGE MDS panels under %s",
        rendered, layout.large_panels_dir,
    )


# ---------------------------------------------------------------------------
# Stage 5: update inspector manifest
# ---------------------------------------------------------------------------


def _scan_panel_pngs(root: Path, leaf_name: str) -> dict[str, dict[str, str]]:
    """Walk ``<root>/epoch_NNNN/lambda_X.YY/<leaf_name>`` and produce a
    {epoch_str: {lambda_str: rel_path}} mapping suitable for the manifest."""
    epoch_re = re.compile(r"^epoch_(\d+)$")
    lam_re = re.compile(r"^lambda_(\d+\.\d+)$")
    out: dict[str, dict[str, str]] = {}
    if not root.is_dir():
        return out
    for epoch_dir in sorted(root.glob("epoch_*")):
        em = epoch_re.match(epoch_dir.name)
        if not em or not epoch_dir.is_dir():
            continue
        epoch = int(em.group(1))
        by_lam: dict[str, str] = {}
        for lam_dir in sorted(epoch_dir.glob("lambda_*")):
            lm = lam_re.match(lam_dir.name)
            if not lm or not lam_dir.is_dir():
                continue
            png = lam_dir / leaf_name
            if not png.is_file():
                continue
            try:
                rel = str(png.relative_to(REPO_ROOT))
            except ValueError:
                rel = str(png)
            by_lam[lm.group(1)] = rel
        if by_lam:
            out[str(epoch)] = by_lam
    return out


def update_manifest(layout: Layout) -> None:
    """Refresh the inspector manifest with the panel maps for this layout's
    mode. Latent mode writes ``mds_cells`` / ``mds_cells_large``; semantic
    mode writes the ``_semantic`` siblings so the frontend toggle can
    switch between them without losing the other set."""
    manifest_path = layout.results_root / "inspector_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    mds_cells = _scan_panel_pngs(layout.panels_dir, "mds.png")
    mds_cells_large = _scan_panel_pngs(layout.large_panels_dir, "mds_large.png")

    if layout.mode == "latent":
        key_small = "mds_cells"
        key_large = "mds_cells_large"
    elif layout.mode == "semantic":
        key_small = "mds_cells_semantic"
        key_large = "mds_cells_large_semantic"
    else:
        raise ValueError(f"unknown Layout mode: {layout.mode!r}")

    manifest[key_small] = mds_cells
    manifest[key_large] = mds_cells_large
    manifest_path.write_text(json.dumps(manifest, indent=2))
    n_small = sum(len(v) for v in mds_cells.values())
    n_large = sum(len(v) for v in mds_cells_large.values())
    log.info(
        "updated %s with %s=%d, %s=%d",
        manifest_path, key_small, n_small, key_large, n_large,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="build_lora_inspector_mds")
    ap.add_argument(
        "--results-root", default=str(DEFAULT_RESULTS),
        help="path containing checkpoints/, config.json, probes/, ...",
    )
    ap.add_argument(
        "--pair-slug", default=DEFAULT_PAIR_SLUG,
        help="training-cache pair slug for the pinned init latents",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument(
        "--epochs", default="all",
        help="comma list of epochs, or 'all' (intersected with manifest)",
    )
    ap.add_argument(
        "--lambdas", default="all",
        help="comma list of lambdas, or 'all' (intersected with manifest)",
    )
    ap.add_argument(
        "--stages",
        default="collect-static,collect-cells,project,render,render-large,update-manifest",
        help="comma list of stages to run, in order. Stages: "
             "collect-static, collect-cells, project, render (small 3-up "
             "panels for tab 2), render-large (single-plot panels with "
             "decoded-image thumbnails at terminals, for the new MDS tab), "
             "update-manifest.",
    )
    ap.add_argument("--overwrite", action="store_true",
                    help="redo cached outputs (default: skip if present)")
    ap.add_argument(
        "--projection", default="pca", choices=("pca", "mds"),
        help="global projection method (default pca — cleaner for "
             "diffusion latents; mds preserves euclidean distances).",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "fp16", "float32", "fp32",
                             "bfloat16", "bf16"))
    ap.add_argument("--model-id",
                    default="stabilityai/stable-diffusion-xl-base-1.0")
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_argparser().parse_args(argv)

    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = REPO_ROOT / results_root
    layout = Layout.build(results_root)

    manifest_path = results_root / "inspector_manifest.json"
    if not manifest_path.is_file():
        log.error("no inspector_manifest.json at %s — run "
                  "scripts/build_lora_manifest.py first", manifest_path)
        return 2
    manifest = json.loads(manifest_path.read_text())
    all_epochs = [int(e) for e in manifest["epochs"]]
    all_lambdas = list(manifest["lambdas"])

    epochs = _parse_epoch_list(args.epochs, all_epochs)
    epochs = [e for e in epochs if e in set(all_epochs)]
    lambdas = _parse_lambda_list(args.lambdas, all_lambdas)
    lam_set_str = {f"{x:.2f}" for x in lambdas}
    lambdas = [float(x) for x in all_lambdas if x in lam_set_str]
    log.info("plan: %d epochs × %d lambdas = %d cells",
             len(epochs), len(lambdas), len(epochs) * len(lambdas))

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    cache_root = (
        Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT
    )

    do_static = "collect-static" in stages
    do_cells = "collect-cells" in stages
    if do_static or do_cells:
        collect_trajectories(
            results_root=results_root,
            layout=layout,
            pair_slug=args.pair_slug,
            seed=args.seed,
            cache_root=cache_root,
            epochs=epochs,
            lambdas=lambdas,
            do_static=do_static,
            do_cells=do_cells,
            overwrite=args.overwrite,
            device_arg=args.device,
            dtype_arg=args.dtype,
            model_id=args.model_id,
        )

    if "project" in stages:
        project_global(layout, method=args.projection)

    if "render" in stages:
        render_panels(
            layout,
            overwrite=args.overwrite,
            epoch_subset=epochs,
            lambda_subset=lambdas,
            method=args.projection,
        )

    if "render-large" in stages:
        render_large_panels(
            layout,
            overwrite=args.overwrite,
            epoch_subset=epochs,
            lambda_subset=lambdas,
            method=args.projection,
            model_id=args.model_id,
            device_arg=args.device,
            dtype_arg=args.dtype,
        )

    if "update-manifest" in stages:
        update_manifest(layout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
