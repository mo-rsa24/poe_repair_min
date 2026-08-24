"""Idea 5a — CLI orchestrator.

Pure post-hoc analysis on existing veracity artefacts. Reads the saved
``x_t``, ``eps_poe``, ``eps_j`` from
``outputs/residual_diagnostics/existence/pairs/<slug>/seed_<n>/teacher_residual_const_lam<NNN>``,
reconstructs Tweedie x̂_0 at chosen step indices for chosen λ values,
decodes through SDXL's VAE, and scores against CLIP text targets.

Run:

    python -m poe_repair.experiments.residual_between_mono_and_poe.clip_window
        [--pair "a cat|a dog"]
        [--seed 42]
        [--lambdas 0.0,0.6,1.0]
        [--steps 0,5,10,15,20,25,30,35,40,45,49]
        [--targets "a cat,a dog,a cat and a dog"]
        [--skip-snapshots]   # rebuild figures from cached scores only
        [--overwrite]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from poe_repair.config import RunConfig
from poe_repair.experiments._eval_common import HEADLINE_PAIR, slugify
from poe_repair.experiments.residual_between_mono_and_poe.clip_window import analyse as A
from poe_repair.experiments.residual_between_mono_and_poe.clip_window import figures as F
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir, write_json


EXP_NAME = "residual_diagnostics/clip_window"
DEFAULT_SEED = 42


def _parse_floats(arg: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not arg:
        return default
    return tuple(float(p.strip()) for p in arg.split(",") if p.strip())


def _parse_ints(arg: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not arg:
        return default
    return tuple(int(p.strip()) for p in arg.split(",") if p.strip())


def _parse_targets(arg: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not arg:
        return default
    return tuple(s.strip() for s in arg.split(",") if s.strip())


def _label_for_lambda(lam: float) -> str:
    if lam == 0.0:
        return "PoE"
    if lam == 1.0:
        return "Mono"
    return f"λ={lam:g}"


def _veracity_run_dir(
    *, output_root: Path, pair_slug: str, seed: int, lam: float,
) -> Path:
    """Veracity uses the teacher-residual sweep at constant schedule."""
    name = f"teacher_residual_const_lam{int(round(lam * 100)):03d}"
    return (
        output_root / "residual_diagnostics" / "existence" / "pairs" / pair_slug
        / f"seed_{seed}" / name
    )


def main() -> None:
    from poe_repair.experiments import _assert_env_ok
    _assert_env_ok()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pair", type=str, default=None,
        help='Pair as "prompt_a|prompt_b". Default = HEADLINE_PAIR (a cat × a dog).',
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--lambdas", type=str, default=None,
        help="Comma-separated λ values to analyse (default: 0.0, 0.6, 1.0).",
    )
    ap.add_argument(
        "--steps", type=str, default=None,
        help="Comma-separated step indices (default: 0,5,10,...,49).",
    )
    ap.add_argument(
        "--targets", type=str, default=None,
        help="Comma-separated CLIP text targets (default: 'a cat', 'a dog', 'a cat and a dog').",
    )
    ap.add_argument(
        "--skip-snapshots", action="store_true",
        help="Skip the VAE-decode + CLIP-scoring step. Render figures from cached scores.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.pair:
        a, _, b = args.pair.partition("|")
        if not a or not b:
            raise ValueError(f"--pair must be 'A|B', got {args.pair!r}")
        prompt_a, prompt_b = a, b
    else:
        prompt_a, prompt_b = HEADLINE_PAIR

    pair_slug = slugify(prompt_a, prompt_b)
    lambdas = _parse_floats(args.lambdas, tuple(l for l, _ in A.DEFAULT_LAMBDAS))
    steps = _parse_ints(args.steps, A.DEFAULT_SNAPSHOT_STEPS)
    targets = _parse_targets(args.targets, A.DEFAULT_TEXT_TARGETS)

    cfg = RunConfig()
    out_root = cfg.paths.output_root / EXP_NAME
    metrics_dir = ensure_dir(out_root / "metrics")
    fig_dir = ensure_dir(out_root / "figures")
    snapshots_root = ensure_dir(out_root / "snapshots" / pair_slug / f"seed_{args.seed}")

    scores_json = metrics_dir / "clip_scores.json"

    snapshots_by_label: dict[str, A.TrajectorySnapshots] = {}

    if not args.skip_snapshots:
        print(f"[{EXP_NAME}] loading SDXL ctx (VAE for decode + CLIP scorer)…")
        ctx: MethodCtx = make_ctx()
        ctx.scheduler.set_timesteps(ctx.num_inference_steps)

        print(f"[{EXP_NAME}] caching CLIP text-target embeddings: {list(targets)}")
        text_embeds = A.cache_text_embeddings(targets, device=ctx.device)

        for lam in lambdas:
            run_dir = _veracity_run_dir(
                output_root=cfg.paths.output_root,
                pair_slug=pair_slug, seed=args.seed, lam=lam,
            )
            if not run_dir.exists():
                raise FileNotFoundError(
                    f"veracity run not found: {run_dir}\n"
                    "Run veracity (idea1's veracity sweep) first."
                )
            label = _label_for_lambda(lam)
            out_dir = snapshots_root / label.replace(" ", "_")
            if out_dir.exists() and not args.overwrite:
                # still re-score to refresh the JSON; PNGs survive.
                pass
            snapshots_by_label[label] = A.analyse_trajectory(
                run_dir=run_dir,
                label=label, lam=lam,
                snapshot_steps=steps,
                text_embeds=text_embeds,
                text_targets=targets,
                ctx=ctx,
                out_dir=out_dir,
            )

        A.write_clip_scores_json(snapshots_by_label, targets, scores_json)
        print(f"[{EXP_NAME}] wrote {scores_json}")
    else:
        if not scores_json.exists():
            raise FileNotFoundError(
                f"--skip-snapshots requires {scores_json} to exist; run without it first."
            )
        cache = json.loads(scores_json.read_text())
        targets = tuple(cache["text_targets"])
        for label, rec in cache["trajectories"].items():
            snap = A.TrajectorySnapshots(
                label=rec["label"],
                lam=float(rec["lam"]),
                run_dir=Path(rec["run_dir"]),
                step_indices=list(rec["step_indices"]),
                timesteps=list(rec["timesteps"]),
                decoded_paths=[Path(p) for p in rec["decoded_paths"]],
                clip_scores=list(rec["clip_scores"]),
                final_image_path=(
                    None if rec["final_image_path"] is None
                    else Path(rec["final_image_path"])
                ),
            )
            snapshots_by_label[label] = snap

    print(f"[{EXP_NAME}] rendering figures")
    title_suffix = f"{prompt_a} × {prompt_b}  |  seed {args.seed}"
    fig_paths: dict[str, Path] = {}
    fig_paths["fig01"] = F.fig01_filmstrip(
        fig_dir=fig_dir,
        snapshots_by_label=snapshots_by_label,
        title_suffix=title_suffix,
    )
    fig_paths["fig02"] = F.fig02_clip_score_trajectories(
        fig_dir=fig_dir,
        snapshots_by_label=snapshots_by_label,
        text_targets=targets,
    )
    fig_paths["fig03"] = F.fig03_cross_trajectory_comparison(
        fig_dir=fig_dir,
        snapshots_by_label=snapshots_by_label,
        text_targets=targets,
        target="a cat and a dog" if "a cat and a dog" in targets else targets[-1],
    )
    for name, p in fig_paths.items():
        print(f"[{EXP_NAME}]   wrote {name}: {p}")

    write_json(
        out_root / "summary.json",
        {
            "exp": EXP_NAME,
            "pair": [prompt_a, prompt_b],
            "pair_slug": pair_slug,
            "seed": args.seed,
            "lambdas": list(lambdas),
            "snapshot_steps": list(steps),
            "text_targets": list(targets),
            "trajectories_analysed": list(snapshots_by_label.keys()),
            "notes": (
                "Pure post-hoc diagnostic on veracity trajectories. No "
                "new sampling. CLIP scores Tweedie x̂_0 decoded via SDXL VAE."
            ),
        },
    )
    print(f"[{EXP_NAME}] done — outputs under {out_root}")


if __name__ == "__main__":
    main()
