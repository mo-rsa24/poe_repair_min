"""Run the Plan-10 per-group cross-seed LoRA pipeline end-to-end.

Resolves ``--group`` to the representative pair from
``scripts.build_group_cache.GROUP_REGISTRY``, writes the per-pair
``seed_pool.yaml`` if it doesn't exist, then runs:

  Task A   — load the pool (leak guard sanity)
  Step 0   — inference-time r̄_t pre-screen
  Task B   — pooled training (per k ∈ KSET) + sample_heldout
  Task D   — Δ̄_t bridge against the k=8 pooled run
  Contact  — Task-B contact sheet

Any step can be skipped with the corresponding ``--skip-*`` flag.
Defaults to running everything except Task C (per-seed ceiling), which
is its own multi-GPU run.

All artefacts land under
``outputs/cross_seed_lora_pooling/<pair_slug>/`` so pairs don't
collide.

Examples
--------
Full pipeline for G4 on GPU 0::

    CUDA_VISIBLE_DEVICES=0 python -m scripts.cross_seed_lora_pooling.run_group --group g4

Train only k=8 and skip Task D / Step 0 (useful for re-runs)::

    python -m scripts.cross_seed_lora_pooling.run_group --group g6 \
        --kset 8 --skip-step0 --skip-task-d
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Sequence

from scripts.build_group_cache import DEFERRED_GROUPS, GROUP_REGISTRY, GroupSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get("PY", sys.executable)

KSET_DEFAULT = ["1a", "1b", "4", "8"]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "outputs" / "cross_seed_lora_pooling"


def _pair_slug(spec: GroupSpec) -> str:
    """Mirror scripts.build_training_cache.slugify."""
    import re

    def _clean(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

    return f"{_clean(spec.prompt_a)}__x__{_clean(spec.prompt_b)}"


def _group_root(slug: str, base: Path) -> Path:
    return base / slug


def _ensure_pool_yaml(group: str, spec: GroupSpec, base: Path) -> Path:
    slug = _pair_slug(spec)
    out_dir = _group_root(slug, base)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "seed_pool.yaml"
    if pool_path.exists():
        return pool_path
    pool_path.write_text(
        textwrap.dedent(
            f"""\
            pair_slug: {slug}
            train_pool:  [1, 2, 3, 4, 5, 6, 7, 8]
            held_out:    [9, 10, 11, 12]
            generated_by: scripts/cross_seed_lora_pooling/run_group.py
            group: {group}
            notes: |
              Plan-10 cross-seed pool for {group} ({spec.label}).
              train_pool = 8 seeds; held_out = 4 seeds (disjoint).
            """
        )
    )
    print(f"[pool] wrote {pool_path}")
    return pool_path


def _run(cmd: Sequence[str], *, cwd: Path = REPO_ROOT) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n[run] {printable}\n")
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def task_a_leak_guard(pool_path: Path) -> None:
    _run([
        PY, "-m", "poe_repair.experiments.held_out_seeds.seed_pool",
        "--seed-pool", str(pool_path),
    ])


def step0_prescreen(
    spec: GroupSpec,
    pool_path: Path,
    out_root: Path,
    *,
    cache_root: Path | None,
) -> None:
    cmd = [
        PY, "-m", "poe_repair.experiments.held_out_seeds.step0_prescreen",
        "--prompt-a", spec.prompt_a,
        "--prompt-b", spec.prompt_b,
        "--joint-prompt", spec.joint_prompt,
        "--seed-pool-path", str(pool_path),
        "--out-dir", str(out_root / "step0_prescreen"),
    ]
    if cache_root is not None:
        cmd.extend(["--cache-root", str(cache_root)])
    _run(cmd)


def _resolve_ckpt(run_dir: Path) -> Path:
    latest = run_dir / "checkpoints" / "latest.json"
    if latest.exists():
        return Path(json.loads(latest.read_text())["path"])
    # fallback: highest step_*.pt
    ckpts = sorted((run_dir / "checkpoints").glob("lora_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/checkpoints/")
    return ckpts[-1]


def train_pooled(
    *,
    spec: GroupSpec,
    pool_path: Path,
    out_root: Path,
    k_tag: str,
    epochs: int,
    batch: int,
    xformers: bool,
    wandb_mode: str,
    wandb_project: str,
    sample_every_epochs: int,
    sample_num_inference_steps: int,
    log_every_epochs: int,
    ckpt_every_epochs: int,
    cache_root: Path | None,
) -> Path:
    """Train one pooled LoRA for the given k tag and return its run dir."""
    task_b_root = out_root / "task_b_learning_curve"
    task_b_root.mkdir(parents=True, exist_ok=True)

    if k_tag in {"1a", "1b"}:
        k = 1
        pick = 1 if k_tag == "1a" else 5
        run_id = f"k01_pick{pick}__ep{epochs}"
        extra = ["--single-seed-pick", str(pick)]
    elif k_tag.isdigit():
        k = int(k_tag)
        run_id = f"k{k:02d}__ep{epochs}"
        extra = []
    else:
        raise ValueError(f"unknown k tag {k_tag!r}")

    cmd = [
        PY, "-m", "poe_repair.experiments.held_out_seeds.train_pooled",
        "--k", str(k),
        "--total-epochs", str(epochs),
        "--seed-pool-path", str(pool_path),
        "--output-root", str(task_b_root),
        "--run-id", run_id,
        "--train-batch-size", str(batch),
        "--wandb-mode", wandb_mode,
        "--wandb-project", wandb_project,
        "--sample-every-epochs", str(sample_every_epochs),
        "--sample-num-inference-steps", str(sample_num_inference_steps),
        "--log-every-epochs", str(log_every_epochs),
        "--ckpt-every-epochs", str(ckpt_every_epochs),
    ]
    if xformers:
        cmd.append("--xformers")
    if cache_root is not None:
        cmd.extend(["--cache-root", str(cache_root)])
    cmd.extend(extra)
    _run(cmd)

    run_dir = task_b_root / run_id
    ckpt = _resolve_ckpt(run_dir)
    sample_dir = run_dir / "samples" / "heldout"
    sample_cmd = [
        PY, "-m", "poe_repair.experiments.held_out_seeds.sample_heldout",
        "--checkpoint", str(ckpt),
        "--out-dir", str(sample_dir),
        "--prompt-a", spec.prompt_a,
        "--prompt-b", spec.prompt_b,
        "--joint-prompt", spec.joint_prompt,
        "--seed-pool-path", str(pool_path),
        "--record-eps",
    ]
    if cache_root is not None:
        sample_cmd.extend(["--cache-root", str(cache_root)])
    _run(sample_cmd)
    return run_dir


def task_d_bridge(
    pool_path: Path,
    k8_run_dir: Path,
    *,
    cache_root: Path | None,
) -> None:
    if not k8_run_dir.exists():
        print(f"[skip] task D: no k=8 run dir at {k8_run_dir}")
        return
    cmd = [
        PY, "-m", "poe_repair.experiments.held_out_seeds.task_d_bridge",
        "--pooled-run", str(k8_run_dir),
        "--seed-pool-path", str(pool_path),
    ]
    if cache_root is not None:
        cmd.extend(["--cache-root", str(cache_root)])
    _run(cmd)


def contact_sheet_task_b(out_root: Path) -> None:
    _run([
        PY, "-m", "poe_repair.experiments.held_out_seeds.contact_sheet",
        "--task", "B",
        "--out-root", str(out_root / "task_b_learning_curve"),
    ])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--group", required=True,
        help="Taxonomy group label: g1, g2, g3, g4, or g6. G5 aborts.",
    )
    ap.add_argument(
        "--kset", default=",".join(KSET_DEFAULT),
        help=f"Comma-separated k tags. Default: {','.join(KSET_DEFAULT)}.",
    )
    ap.add_argument("--total-epochs", type=int, default=2000,
                    help="Forwarded to train_pooled. Default 2000.")
    ap.add_argument("--train-batch-size", type=int, default=2,
                    help="Forwarded to train_pooled. Default 2.")
    ap.add_argument("--sample-every-epochs", type=int, default=100,
                    help="Mid-training sampling cadence. 0 disables. "
                         "Default 100 (spaced so it doesn't dominate wall time).")
    ap.add_argument("--sample-num-inference-steps", type=int, default=20,
                    help="DDIM steps for mid-training samples. Default 20.")
    ap.add_argument("--log-every-epochs", type=int, default=10)
    ap.add_argument("--ckpt-every-epochs", type=int, default=50,
                    help="Checkpoint cadence. Lower = more disk. Default 50.")
    ap.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_BASE,
        help=f"Base output directory. Per-pair subdir is appended. "
             f"Default: {DEFAULT_OUTPUT_BASE} (repo). Override to "
             f"/datasets/... for space-constrained runs.",
    )
    ap.add_argument(
        "--cache-root", type=Path, default=None,
        help="Forwarded to every module as --cache-root. Default: env "
             "$POE_REPAIR_TRAINING_CACHE.",
    )
    ap.add_argument("--no-xformers", action="store_true")
    ap.add_argument("--wandb-mode", default="online",
                    choices=("online", "offline", "disabled"))
    ap.add_argument("--wandb-project", default="poe-repair-cross-seed")
    ap.add_argument("--skip-task-a", action="store_true")
    ap.add_argument("--skip-step0", action="store_true")
    ap.add_argument("--skip-task-b", action="store_true",
                    help="Skip pooled training + sampling for all k.")
    ap.add_argument("--skip-task-d", action="store_true")
    ap.add_argument("--skip-contact-sheet", action="store_true")
    args = ap.parse_args()

    group = args.group.strip().lower()
    if group in DEFERRED_GROUPS:
        print(f"[error] {DEFERRED_GROUPS[group]}", file=sys.stderr)
        return 2
    if group not in GROUP_REGISTRY:
        print(
            f"[error] unknown group {args.group!r}. "
            f"Known: {sorted(GROUP_REGISTRY)}; deferred: {sorted(DEFERRED_GROUPS)}.",
            file=sys.stderr,
        )
        return 2
    spec = GROUP_REGISTRY[group]
    slug = _pair_slug(spec)
    out_root = _group_root(slug, args.output_root)

    print(
        f"[plan] group={group} pair={slug} "
        f"epochs={args.total_epochs} kset={args.kset} "
        f"batch={args.train_batch_size} "
        f"sample_every={args.sample_every_epochs} "
        f"ckpt_every={args.ckpt_every_epochs} "
        f"out_root={out_root}"
    )

    pool_path = _ensure_pool_yaml(group, spec, args.output_root)

    if not args.skip_task_a:
        task_a_leak_guard(pool_path)
    if not args.skip_step0:
        step0_prescreen(spec, pool_path, out_root, cache_root=args.cache_root)

    k_tags = [t.strip() for t in args.kset.split(",") if t.strip()]
    k8_run_dir: Path | None = None
    if not args.skip_task_b:
        for tag in k_tags:
            run_dir = train_pooled(
                spec=spec,
                pool_path=pool_path,
                out_root=out_root,
                k_tag=tag,
                epochs=args.total_epochs,
                batch=args.train_batch_size,
                xformers=not args.no_xformers,
                wandb_mode=args.wandb_mode,
                wandb_project=args.wandb_project,
                sample_every_epochs=args.sample_every_epochs,
                sample_num_inference_steps=args.sample_num_inference_steps,
                log_every_epochs=args.log_every_epochs,
                ckpt_every_epochs=args.ckpt_every_epochs,
                cache_root=args.cache_root,
            )
            if tag == "8":
                k8_run_dir = run_dir

    if not args.skip_task_d:
        if k8_run_dir is None:
            k8_run_dir = out_root / "task_b_learning_curve" / f"k08__ep{args.total_epochs}"
        task_d_bridge(pool_path, k8_run_dir, cache_root=args.cache_root)

    if not args.skip_contact_sheet:
        contact_sheet_task_b(out_root)

    print(f"\n[done] group={group} → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
