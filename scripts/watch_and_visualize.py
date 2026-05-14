"""Watch a Method 2b student checkpoint and render seed-42 evaluation snapshots.

Polls the trainer's ``best.pt`` (and optional ``snapshots/step_*.pt`` files
written by the updated trainer). Whenever a new checkpoint lands — either
``best.pt``'s mtime changed, or a new ``snapshots/step_*.pt`` appears —
runs PoE | Mono | direct_eps(λ=1.0) on the held-out cell and writes a
side-by-side image into ``viz/step_NNNNNNN.png``.

Pure PoE + learned residual at inference (no sched-M2, no ê_J anywhere).

Usage::

    CUDA_VISIBLE_DEVICES=0 python -m scripts.watch_and_visualize \\
        --ckpt-dir /abs/path/checkpoints/students/direct_eps_overfit_catdog_hg \\
        --pair "a cat|a dog" --seed 42 \\
        --lambda-max 1.0 --window-frac 0.4 --poll-seconds 30

Stop with Ctrl-C.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import torch

from poe_repair.composers import direct_eps as cmp_direct_eps
from poe_repair.composers import mono as cmp_mono
from poe_repair.composers import poe as cmp_poe
from poe_repair.experiments._eval_common import cell_for, slugify
from poe_repair.figures._common import image_grid
from poe_repair.run import MethodCtx, make_ctx
from poe_repair.runtime import ensure_dir


def _read_step(ckpt_path: Path) -> int | None:
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        return int(sd.get("step", -1)) if isinstance(sd, dict) else None
    except Exception:
        return None


def _read_val_metrics(ckpt_path: Path) -> dict | None:
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict):
            return sd.get("val_metrics")
    except Exception:
        return None
    return None


def _render_one(
    *,
    ckpt_path: Path,
    out_path: Path,
    ctx: MethodCtx,
    prompt_a: str, prompt_b: str, seed: int,
    lambda_max: float, window_frac: float,
    exp_name: str,
) -> Path:
    """Render PoE | Mono(e_J) | direct_eps(λ) for one (pair, seed) cell."""
    cell = cell_for(prompt_a, prompt_b, seed)

    poe_path = cmp_poe.run(cell, ctx, exp_name=exp_name, overwrite=False)
    mono_path = cmp_mono.run(
        cell, ctx, anchor_source="literal",
        exp_name=exp_name, overwrite=False,
    )
    # The composer caches the result on (exp_name, METHOD_NAME, cell).
    # We use a unique exp_name per snapshot so successive snapshots don't
    # collide with each other's outputs.
    snap_exp = f"{exp_name}/snap_{ckpt_path.stem}"
    de_path = cmp_direct_eps.run(
        cell, ctx,
        window_frac=float(window_frac),
        lambda_max=float(lambda_max),
        student_ckpt=str(ckpt_path),
        exp_name=snap_exp,
        overwrite=True,
    )

    image_grid(
        [[poe_path, mono_path, de_path]],
        out_path,
        col_labels=["PoE", "Mono (e_J)", f"direct_eps (λ={lambda_max:g})"],
        row_labels=[f"seed {seed}"],
        title=(
            f"snapshot {ckpt_path.name} — {prompt_a} × {prompt_b}\n"
            "pure PoE + learned residual (no sched-M2, no ê_J)"
        ),
        panel_size=2.4,
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True,
                    help="checkpoints/students/<output_name>/ directory.")
    ap.add_argument("--pair", default="a cat|a dog",
                    help='"prompt_a|prompt_b"')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda-max", type=float, default=1.0)
    ap.add_argument("--window-frac", type=float, default=0.4)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument(
        "--max-iterations", type=int, default=0,
        help="Stop after this many renders (0 = run forever).",
    )
    args = ap.parse_args()

    a, _, b = args.pair.partition("|")
    if not a or not b:
        raise ValueError(f"--pair must be 'A|B', got {args.pair!r}")
    prompt_a, prompt_b = a, b

    ckpt_dir = args.ckpt_dir.resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt dir not found: {ckpt_dir}")
    snapshots_dir = ckpt_dir / "snapshots"
    best_path = ckpt_dir / "best.pt"

    output_root = ckpt_dir / "viz"
    ensure_dir(output_root)
    archive_dir = ensure_dir(output_root / "checkpoints")

    print(f"[viz] watching {ckpt_dir}")
    print(f"[viz] writing snapshots to {output_root}")
    print(f"[viz] eval cell: {prompt_a} × {prompt_b}, seed {args.seed}, "
          f"λ_max={args.lambda_max}, window_frac={args.window_frac}")
    print("[viz] loading SDXL context (one-time)...")
    ctx = make_ctx(output_root=ckpt_dir.parent.parent.parent / "outputs")
    print(f"[viz] device={ctx.device} dtype={ctx.dtype}")

    seen_best_mtime: float | None = None
    seen_snapshots: set[str] = set()
    iterations = 0

    exp_name = f"viz_{ckpt_dir.name}"
    try:
        while True:
            new_paths: list[Path] = []

            # Poll best.pt: archive a copy when mtime changes so we can
            # render against a stable snapshot that won't be overwritten
            # mid-render by the trainer.
            if best_path.exists():
                m = best_path.stat().st_mtime
                if seen_best_mtime is None or m > seen_best_mtime:
                    seen_best_mtime = m
                    step = _read_step(best_path)
                    tag = f"best_step_{step:07d}" if step else f"best_{int(m)}"
                    archived = archive_dir / f"{tag}.pt"
                    if not archived.exists():
                        shutil.copy2(best_path, archived)
                    new_paths.append(archived)

            # Poll snapshots/ (populated only by the updated trainer).
            if snapshots_dir.exists():
                for snap in sorted(snapshots_dir.glob("step_*.pt")):
                    if snap.name not in seen_snapshots:
                        seen_snapshots.add(snap.name)
                        new_paths.append(snap)

            for ckpt in new_paths:
                step = _read_step(ckpt)
                vm = _read_val_metrics(ckpt) or {}
                tag = f"step_{step:07d}" if step else ckpt.stem
                out_path = output_root / f"{tag}.png"
                if out_path.exists():
                    continue
                print(f"[viz] rendering {ckpt.name}  step={step}  "
                      f"val_rmse={vm.get('val_rmse', '—')}  "
                      f"val_rel_err={vm.get('val_rel_err', '—')}")
                t0 = time.time()
                try:
                    _render_one(
                        ckpt_path=ckpt,
                        out_path=out_path,
                        ctx=ctx,
                        prompt_a=prompt_a, prompt_b=prompt_b, seed=args.seed,
                        lambda_max=args.lambda_max,
                        window_frac=args.window_frac,
                        exp_name=exp_name,
                    )
                except Exception as exc:
                    print(f"[viz] render failed for {ckpt.name}: {exc}")
                    continue
                print(f"[viz] wrote {out_path}  ({time.time() - t0:.1f}s)")
                iterations += 1
                if args.max_iterations and iterations >= args.max_iterations:
                    print(f"[viz] reached --max-iterations={args.max_iterations}; exiting")
                    return

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("[viz] stopped (Ctrl-C)")


if __name__ == "__main__":
    main()
