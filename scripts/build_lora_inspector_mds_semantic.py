"""Plan-13 — semantic MDS over DINOv2 features of predicted-x̂₀(t).

Sibling builder to ``build_lora_inspector_mds.py``. Re-samples every
trajectory (mono, solo-A, solo-B, A∧B, and each (epoch, λ) PoE+λ·R cell)
so that we have ``tracker.trajectories`` and ``tracker.velocities`` in
memory, then:

    1. Recovers x̂₀(t) = (z_t − √(1−ᾱ_t)·ε̂_t) / √ᾱ_t at subsampled steps
       (and uses the terminal ``trajectories[-1]`` as the final point).
    2. VAE-decodes each subsampled x̂₀(t).
    3. DINOv2 ViT-S/14 forward → 384-d CLS, L2-normalised.
    4. Saves the per-trajectory CLS sequence (T_sub+1, 384) as a .npy.
    5. Globally MDS-projects on **cosine** dissimilarities — semantic
       similarity is angular, not Euclidean.
    6. Procrustes-aligns the resulting 2D coords to the latent-mode
       coords using the three static endpoints (solo_a, solo_b, mono)
       as anchors, so flipping the inspector toggle does not visually
       reshuffle the static anchors.

Outputs live under
``<results_root>/mds_cache_semantic/`` (CLS .npy files + coords json)
and ``<results_root>/mds_probes_large_semantic/`` (PNG panels). All
terminal thumbnails (static + LoRA) are re-used from the latent
``mds_cache`` — the decoded final image doesn't depend on the
embedding choice.

Stages (composable via --stages):
    collect-static  : sample solo-A, solo-B, mono; cache CLS sequences
    collect-cells   : per (epoch, λ), reload LoRA ckpt, sample, cache CLS
    project         : fit metric MDS on cosine distances, write coords
    align           : Procrustes-fit semantic coords to latent coords
                      using the 3 static endpoints
    render-large    : per cell, render the large MDS panel
    update-manifest : add mds_cells_large_semantic to inspector_manifest

A typical sequence::

    python scripts/build_lora_inspector_mds_semantic.py \\
        --epochs 0,200,800,1600 \\
        --stages collect-static,collect-cells,project,align,render-large,update-manifest

The validation gate (``scripts/cross_seed_lora_pooling/smoke_dino_distance.py``)
must PASS before running this script.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from poe_repair._sdxl.predicted_x0 import (
    alpha_bar_from_scheduler,
    predicted_x0,
)
from poe_repair.experiments.lora import trainer as lora_trainer
from poe_repair.experiments.lora.main import encode_all_prompts
from poe_repair.experiments.lora.probe import load_pinned_init_latents
from poe_repair.methods._sampling import (
    run_cfg,
    run_lora_residual_inject,
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

# Re-use the rendering + manifest helpers from the latent builder. Layout
# already has mode="semantic" support, which makes render_large_panels
# write to the right place automatically.
from scripts.build_lora_inspector_mds import (  # noqa: E402  (path-mod import)
    Layout,
    render_large_panels,
    update_manifest,
    _parse_epoch_list,
    _parse_lambda_list,
    _checkpoint_for_epoch,
)


log = logging.getLogger("build_lora_inspector_mds_semantic")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local"
DEFAULT_PAIR_SLUG = "a_cat__x__a_dog"


# ---------------------------------------------------------------------------
# DINOv2 embedder — lazy-loaded once per process
# ---------------------------------------------------------------------------


class DinoEmbedder:
    """DINOv2 ViT-S/14 CLS embedder. Loads on first call, caches on the
    instance. ``embed_decoded_batch`` accepts a tensor of decoded images
    already in the format ``decode_latents`` returns (range [0, 1],
    shape (N, 3, H, W) on CPU).

    The embedder defaults to CPU because the surrounding SDXL UNet + VAE
    in fp16 dominate the GPU budget on shared boxes — DINOv2 ViT-S/14 on
    CPU embeds ~10 images in well under a second, so the wall-clock cost
    is negligible vs. one sampler pass."""

    def __init__(self, *, device: torch.device, repo: str = "facebookresearch/dinov2",
                 entry: str = "dinov2_vits14"):
        self.device = device
        self._model = None
        self._mean = None
        self._std = None
        self._repo = repo
        self._entry = entry

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("loading DINOv2 (%s/%s) on %s ...", self._repo, self._entry, self.device)
        self._model = torch.hub.load(
            self._repo, self._entry, trust_repo=True,
        ).to(device=self.device, dtype=torch.float32)
        self._model.eval()
        self._mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=torch.float32,
        ).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed_decoded_batch(self, images: torch.Tensor) -> np.ndarray:
        """``images`` is what ``decode_latents`` returns: (N, 3, H, W) in
        [0, 1], CPU. Returns L2-normalised (N, 384) CLS as float32 numpy."""
        self._ensure_loaded()
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"expected (N, 3, H, W) decoded images, got shape={tuple(images.shape)}"
            )
        x = images.to(device=self.device, dtype=torch.float32)
        # Bicubic resize to 224 then centre-crop (already square here, so
        # the crop is a no-op but kept for safety / non-square inputs).
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bicubic", align_corners=False,
                antialias=True,
            )
        x = (x - self._mean) / self._std
        feats = self._model(x)  # (N, 384) CLS by default
        feats = feats.float().cpu().numpy()
        norms = np.linalg.norm(feats, axis=-1, keepdims=True).clip(min=1e-8)
        return feats / norms


# ---------------------------------------------------------------------------
# x̂₀(t) recovery + decode + embed — operates on a tracker returned by a
# sampler (so we have z_t AND eps_t in memory; the .npy cache only has z_t).
# ---------------------------------------------------------------------------


def _subsample_step_indices(num_steps: int, stride: int) -> list[int]:
    """Step indices [0, stride, 2·stride, ..., last-internal] for the
    intermediate timesteps. The terminal point (``trajectories[-1]``)
    is appended separately by the caller."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    idxs = list(range(0, num_steps, stride))
    return idxs


def _tracker_to_dino_cls_sequence(
    *,
    tracker,
    scheduler,
    models: dict,
    embedder: DinoEmbedder,
    stride: int,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Reconstruct x̂₀(t) at subsampled steps + the terminal latent,
    VAE-decode, and DINOv2-embed. Returns (T_sub+1, 384) CLS sequence as
    float32 numpy. Order: sampled intermediates in time order, then the
    terminal."""
    num_steps = int(tracker.num_steps)
    sub_idx = _subsample_step_indices(num_steps, stride)

    # Build a batch of x̂₀ latents for subsampled steps.
    x0_list: list[torch.Tensor] = []
    for k in sub_idx:
        z_k = tracker.trajectories[k].to(device=device, dtype=dtype)
        eps_k = tracker.velocities[k].to(device=device, dtype=dtype)
        t_k = int(tracker.timesteps[k].item())
        alpha_bar_k = alpha_bar_from_scheduler(
            scheduler, t_k, device=device, dtype=dtype,
        )
        x0_k = predicted_x0(z_k, eps_k, alpha_bar_k)
        x0_list.append(x0_k)
    # Append the final z_0 (after the last DDIM step) as the terminal point.
    z_final = tracker.trajectories[-1].to(device=device, dtype=dtype)
    x0_list.append(z_final)

    # Decode one latent at a time (VAE batch over time can blow VRAM).
    # Move each decoded image to CPU immediately and free the GPU latent.
    decoded_list: list[torch.Tensor] = []
    for x0 in x0_list:
        img = decode_latents(models, x0)  # (1, 3, H, W) in [0, 1]
        decoded_list.append(img.detach().cpu())
        del x0, img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    decoded = torch.cat(decoded_list, dim=0).cpu()

    cls_seq = embedder.embed_decoded_batch(decoded)
    return cls_seq.astype(np.float32)


# ---------------------------------------------------------------------------
# Sampling wrappers that return the in-memory tracker (so we can recover eps)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _sample_solo_tracker(
    *,
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
):
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
    return out.tracker


@torch.no_grad()
def _sample_poe_lora_tracker(
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
):
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
    return out.tracker


# ---------------------------------------------------------------------------
# Stage 1+2: collect semantic CLS sequences
# ---------------------------------------------------------------------------


def collect_semantic_sequences(
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
    stride: int,
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
    log.info("device=%s dtype=%s stride=%d", device, dtype, stride)

    models = load_sdxl_models(model_id=model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(model_id)
    # We decode ~10× more latents per trajectory than the latent-mode builder
    # (one per subsampled step, not just the terminal). Enable VAE tiling so
    # the per-decode peak memory drops well below the SDXL UNet's footprint
    # — important on shared GPUs where free VRAM is tight.
    vae = models.get("vae")
    if vae is not None and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
        # SDXL VAE defaults to tile_latent_min_size=128 with a strict ``>``
        # check, so 128×128 latents (1024² samples) never trigger tiling.
        # Lower the threshold so tiling actually fires for our setup.
        vae.tile_latent_min_size = 64
        vae.tile_sample_min_size = 512
        log.info(
            "enabled VAE tiling (tile_latent_min_size=%d) for cheaper "
            "intermediate-step decodes",
            int(vae.tile_latent_min_size),
        )
    # Co-locate DINO with the SDXL stack. ViT-S/14 (~84 MB) is dwarfed by
    # the UNet/VAE; tiling already keeps the per-decode peak modest. xFormers
    # inside DINOv2 also requires CUDA + fp16/bf16, so CPU is not an option.
    embedder = DinoEmbedder(device=device)

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
            if out_path.is_file() and not overwrite:
                log.info("[skip] static %s already cached (CLS sequence)", name)
                continue
            log.info("sampling %s ...", name)
            tracker = _sample_solo_tracker(
                init_latents=init_latents,
                models=models, scheduler=scheduler,
                seq_cond=seq_cond, pool_cond=pool_cond,
                seq_e=embeddings["seq_e"], pool_e=embeddings["pool_e"],
                cfg_sampler=cfg_sampler,
                device=device, dtype=dtype,
            )
            cls_seq = _tracker_to_dino_cls_sequence(
                tracker=tracker, scheduler=scheduler,
                models=models, embedder=embedder,
                stride=stride, device=device, dtype=dtype,
            )
            ensure_dir(out_path.parent)
            np.save(out_path, cls_seq.astype(np.float32))
            log.info("  saved %s shape=%s -> %s", name, cls_seq.shape, out_path)

    if not do_cells:
        return

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
            tracker = _sample_poe_lora_tracker(
                init_latents=init_latents,
                models=models, scheduler=scheduler,
                embeddings=embeddings,
                lambda_value=lam,
                adapter_name=adapter_name,
                cfg_sampler=cfg_sampler,
                device=device, dtype=dtype,
            )
            cls_seq = _tracker_to_dino_cls_sequence(
                tracker=tracker, scheduler=scheduler,
                models=models, embedder=embedder,
                stride=stride, device=device, dtype=dtype,
            )
            np.save(out_path, cls_seq.astype(np.float32))
            log.info("  saved epoch=%d λ=%.2f shape=%s", epoch, lam, cls_seq.shape)


# ---------------------------------------------------------------------------
# Stage 3: metric MDS on cosine distances
# ---------------------------------------------------------------------------


def project_global_semantic(layout: Layout) -> None:
    """Pool all cached CLS sequences, fit one metric MDS on cosine
    dissimilarities, save 2D coords. The CLS features are already L2-
    normalised, so cosine distance = ``1 − u·v`` and ranges [0, 2]."""
    import warnings

    static_names = ["solo_a", "solo_b", "mono"]
    items: list[tuple[str, np.ndarray]] = []
    for name in static_names:
        p = layout.static_traj_path(name)
        if not p.is_file():
            raise FileNotFoundError(f"missing static CLS sequence: {p}")
        items.append((f"static/{name}", np.load(p).astype(np.float32)))

    cells_dir = layout.cache_dir / "cells"
    for epoch_dir in sorted(cells_dir.glob("epoch_*")):
        for lam_path in sorted(epoch_dir.glob("lambda_*.npy")):
            key = f"cells/{epoch_dir.name}/{lam_path.stem}"
            items.append((key, np.load(lam_path).astype(np.float32)))

    log.info("semantic projection pool: %d trajectories", len(items))
    stacked = np.concatenate([arr for _, arr in items], axis=0)
    log.info("  stacked CLS shape=%s", stacked.shape)
    # Re-normalise just in case (CLS arrives normalised, but float32→.npy
    # round-trip can drift on the 6th decimal).
    stacked /= np.linalg.norm(stacked, axis=-1, keepdims=True).clip(min=1e-8)
    # Cosine distance from dot product. Clip tiny negatives from float
    # rounding.
    dot = stacked @ stacked.T
    dist = np.clip(1.0 - dot, 0.0, 2.0)
    # Symmetrise + zero diagonal.
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    from sklearn.manifold import MDS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proj = MDS(
            n_components=2,
            random_state=42,
            dissimilarity="precomputed",
            normalized_stress="auto",
            n_init=4,
        ).fit_transform(dist)

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
# Stage 4: Procrustes-align semantic coords to latent coords
# ---------------------------------------------------------------------------


def align_to_latent(
    layout_semantic: Layout, layout_latent: Layout,
) -> None:
    """Fit a 2D similarity transform (rotation + reflection + uniform
    scale + translation) on the three static endpoints
    {solo_a, solo_b, mono} and apply it to every semantic coord. After
    this step, flipping the inspector toggle keeps the three static
    anchor terminals in roughly the same screen position so the visual
    geometry "doesn't reshuffle"."""
    latent_coords = json.loads(layout_latent.coords_path.read_text())
    sem_coords = json.loads(layout_semantic.coords_path.read_text())

    # Endpoint coords for the three static paths.
    anchors_latent = np.asarray([
        latent_coords["static/solo_a"][-1],
        latent_coords["static/solo_b"][-1],
        latent_coords["static/mono"][-1],
    ], dtype=np.float64)
    anchors_sem = np.asarray([
        sem_coords["static/solo_a"][-1],
        sem_coords["static/solo_b"][-1],
        sem_coords["static/mono"][-1],
    ], dtype=np.float64)

    # 2D similarity transform: y = s·R·x + t, fit so that R is a rotation
    # (det=+1) or reflection (det=-1) — pick whichever minimises residual.
    src = anchors_sem
    dst = anchors_latent
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    src0 = src - src_c
    dst0 = dst - dst_c
    H = src0.T @ dst0
    U, S, Vt = np.linalg.svd(H)
    R = (Vt.T @ U.T)
    # Allow reflection — DINO-space layouts arrive in a different
    # rotation from latent-space ones; we want the closest match.
    s = (S.sum()) / max((src0 * src0).sum(), 1e-12)
    t = dst_c - s * (R @ src_c)

    def apply(p: np.ndarray) -> np.ndarray:
        return (s * (R @ p.T)).T + t

    aligned: dict[str, list[list[float]]] = {}
    for key, pts in sem_coords.items():
        arr = np.asarray(pts, dtype=np.float64)
        new = apply(arr)
        aligned[key] = new.tolist()

    layout_semantic.coords_path.write_text(json.dumps(aligned))
    log.info(
        "Procrustes-aligned semantic coords to latent anchors "
        "(scale=%.4f, |t|=%.4f) -> %s",
        s, float(np.linalg.norm(t)), layout_semantic.coords_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="build_lora_inspector_mds_semantic")
    ap.add_argument(
        "--results-root", default=str(DEFAULT_RESULTS),
        help="path containing checkpoints/, config.json, probes/, mds_cache/, ...",
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
        default="collect-static,collect-cells,project,align,render-large,update-manifest",
        help="comma list of stages to run, in order. Stages: "
             "collect-static, collect-cells, project, align (Procrustes "
             "to latent coords), render-large, update-manifest.",
    )
    ap.add_argument("--overwrite", action="store_true",
                    help="redo cached outputs (default: skip if present)")
    ap.add_argument(
        "--stride", type=int, default=5,
        help="subsample stride for intermediate x̂₀(t) embeddings "
             "(default 5 → ~10 points per 50-step trajectory + terminal).",
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
    layout_sem = Layout.build(results_root, mode="semantic")
    layout_latent = Layout.build(results_root, mode="latent")

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
        collect_semantic_sequences(
            results_root=results_root,
            layout=layout_sem,
            pair_slug=args.pair_slug,
            seed=args.seed,
            cache_root=cache_root,
            epochs=epochs,
            lambdas=lambdas,
            do_static=do_static,
            do_cells=do_cells,
            overwrite=args.overwrite,
            stride=int(args.stride),
            device_arg=args.device,
            dtype_arg=args.dtype,
            model_id=args.model_id,
        )

    if "project" in stages:
        project_global_semantic(layout_sem)

    if "align" in stages:
        if not layout_latent.coords_path.is_file():
            log.error(
                "no latent coords at %s — run build_lora_inspector_mds.py "
                "(mode=latent) first so we have static anchors to align to",
                layout_latent.coords_path,
            )
            return 2
        align_to_latent(layout_sem, layout_latent)

    if "render-large" in stages:
        render_large_panels(
            layout_sem,
            overwrite=args.overwrite,
            epoch_subset=epochs,
            lambda_subset=lambdas,
            method="mds",
            model_id=args.model_id,
            device_arg=args.device,
            dtype_arg=args.dtype,
        )

    if "update-manifest" in stages:
        update_manifest(layout_sem)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
