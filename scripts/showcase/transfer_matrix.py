#!/usr/bin/env python
"""The transfer matrix at the credible tier (plan 04, 01-showcase-the-trained-lora).

`phase1_r8_100k` is the adapter under study for this scope (decisions-taken-here.md):
a rank-8 cross-attention LoRA trained on 11 animal pairs x 8 seeds (88 runs), pooling
across concepts, not just seeds. This script evaluates it on the 8 pairs its own
pair_pool.json lists as held out, none of which share a token with any of the 11
training pairs (audited below, not assumed). That is the harder generalisation axis:
a new pair, not just a new seed of a trained pair.

(An earlier version of this script used the G6 cross_seed_lora_pooling checkpoint —
trained on 8 seeds of one single pair, "a cat" x "a dog". decisions-taken-here.md
calls that run "the shelved cross-seed adapter", kept only as a free replication
datum. It answers a narrower question than this plan asks and is not used here.)

Pre-registered window: LoRA active over steps 0-10 of 50, lambda=1, per
decisions-taken-here.md's "The generalization demo ships at the reviewer-credible
tier". Init latents and text embeddings are read straight from each pair's cached
training-cache cell (embeddings.pt) rather than re-encoded, and the plain-PoE
baseline reuses that cell's own cached poe.png rather than re-rendering it. Eval
seeds are phase1_r8_100k's own held-out seed pool, 9-16 (seed_pool.json).

Usage:
    python scripts/showcase/transfer_matrix.py --census
    python scripts/showcase/transfer_matrix.py --fill [--pairs a_leopard__x__a_jaguar,...]
    python scripts/showcase/transfer_matrix.py --score
    python scripts/showcase/transfer_matrix.py --render
    python scripts/showcase/transfer_matrix.py --strip
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig
from poe_repair.methods._sampling import run_lora_residual_inject_masked, write_decoded_image
from poe_repair.run import make_ctx
from poe_repair.training_cache import DEFAULT_CACHE_ROOT

POOLED_LORA_ROOT = Path("artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora")
DEFAULT_CHECKPOINT_RUN = "phase1_r8_100k"


def resolve_checkpoint(run_name: str) -> tuple[Path, Path]:
    """run_name is a subdirectory of POOLED_LORA_ROOT, e.g. phase1_r8_100k, or (once
    Experiments A/B land) a 200k-resume or rank-16/32 run under the same root.

    Resolves the highest lora_step_*.pt actually present under run_dir/checkpoints/,
    not the path recorded in checkpoints/latest.json: phase1_r8_100k's latest.json
    still records the pre-archival absolute path
    (outputs/animals_compose_transfer/pooled_lora/...), from before the run was
    archived to artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/; the
    file itself lives beside latest.json, latest.json just wasn't updated when the
    run moved. What's on disk here is ground truth; a recorded path elsewhere may not be.
    """
    run_dir = POOLED_LORA_ROOT / run_name
    candidates = sorted(
        (run_dir / "checkpoints").glob("lora_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"no lora_step_*.pt under {run_dir}/checkpoints")
    return run_dir, candidates[-1]


CHECKPOINT_RUN_DIR, CHECKPOINT_PATH = resolve_checkpoint(DEFAULT_CHECKPOINT_RUN)

OUT_ROOT_BASE = Path(
    "/datasets/mmolefe/poe_repair_min/outputs/showcase_the_trained_lora/transfer_matrix"
)
OUT_ROOT = OUT_ROOT_BASE / DEFAULT_CHECKPOINT_RUN

LORA_TARGET_MODULES = ("attn2.to_q", "attn2.to_k", "attn2.to_v")
LORA_ADAPTER_NAME = "lora"

# CHECKPOINT_RUN_DIR/pair_pool.json, verbatim.
TRAIN_PAIRS = [
    "a_wolf__x__a_husky", "a_lion__x__a_tiger", "a_cheetah__x__a_cougar",
    "a_horse__x__a_zebra", "a_donkey__x__a_pony", "a_crocodile__x__an_alligator",
    "a_rabbit__x__a_hare", "a_dolphin__x__a_porpoise", "a_crow__x__a_raven",
    "a_gorilla__x__a_chimpanzee", "a_turtle__x__a_tortoise",
]
EVAL_POOL = [
    "a_leopard__x__a_jaguar", "a_frog__x__a_toad", "an_eagle__x__a_hawk",
    "a_seal__x__a_walrus", "a_goose__x__a_swan", "a_cow__x__a_buffalo",
    "a_cat__x__a_dog", "an_elephant__x__a_penguin",
]

WINDOW_ON_STEPS = 10
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
LAMBDA = 1.0
# CHECKPOINT_RUN_DIR/seed_pool.json's "held_out" list.
EVAL_SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)

STOP_WORDS = {"a", "an"}


def tokenize_slug(slug: str) -> set[str]:
    return {
        w for part in slug.lower().split("__x__")
        for w in re.findall(r"[a-z']+", part)
        if w not in STOP_WORDS
    }


def train_tokens() -> set[str]:
    tok = set()
    for slug in TRAIN_PAIRS:
        tok |= tokenize_slug(slug)
    return tok


def resolve_cell_dir(pair_slug: str) -> tuple[str, list[int]]:
    """Find the training-cache split holding this pair and its cached eval seeds."""
    for split in ("heldout", "train"):
        root = DEFAULT_CACHE_ROOT / split / pair_slug
        if root.exists():
            cached = {int(p.name.split("_")[1]) for p in root.glob("seed_*")
                      if (p / "embeddings.pt").exists()}
            seeds = [s for s in EVAL_SEEDS if s in cached]
            if not seeds:
                raise FileNotFoundError(
                    f"{pair_slug}: none of eval seeds {EVAL_SEEDS} cached under {split}"
                )
            return split, seeds
    raise FileNotFoundError(f"{pair_slug}: no training-cache cell under heldout/ or train/")


def census() -> list[dict]:
    train_tok = train_tokens()
    squares = []
    for pair_slug in EVAL_POOL:
        shared = tokenize_slug(pair_slug) & train_tok
        rec = {"checkpoint_run": CHECKPOINT_RUN_DIR.name, "eval_pair": pair_slug,
               "disjoint": len(shared) == 0, "shared_tokens": sorted(shared)}
        if rec["disjoint"]:
            split, seeds = resolve_cell_dir(pair_slug)
            rec.update({"split": split, "seeds": seeds,
                        "cached": (OUT_ROOT / pair_slug / "fill_manifest.json").exists()})
        squares.append(rec)

    n_disjoint = sum(1 for s in squares if s["disjoint"])
    n_within = len(squares) - n_disjoint
    print(f"[transfer_matrix] census: {len(squares)} candidate pairs (phase1_r8_100k's own "
          f"held-out list), {n_disjoint} disjoint (scored), {n_within} within (excluded)")
    for s in squares:
        if not s["disjoint"]:
            print(f"  EXCLUDED {s['eval_pair']}: shares {s['shared_tokens']} with a training pair")
        else:
            print(f"  {s['eval_pair']}: split={s['split']} seeds={s['seeds']}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "matrix_manifest.json").write_text(json.dumps(squares, indent=2))
    return squares


def _attach_and_load_lora(unet: torch.nn.Module) -> dict:
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    state = ckpt.get("lora_state")
    if state is None:
        raise KeyError(f"{CHECKPOINT_PATH} has no 'lora_state' key (found: {list(ckpt.keys())})")

    # Read rank/alpha/target-modules from the checkpoint's own config rather than
    # assuming rank 8 — required once Experiment B's rank-16/32 checkpoints exist.
    lora_meta = ckpt.get("config", {}).get("lora", {})
    rank = int(lora_meta.get("rank", 8))
    alpha = int(lora_meta.get("alpha", rank))
    target_modules = tuple(lora_meta.get("target_modules", LORA_TARGET_MODULES))
    adapter_name = str(lora_meta.get("adapter_name", LORA_ADAPTER_NAME))

    lora_cfg = LoRAConfig(
        rank=rank, alpha=alpha, dropout=0.0,
        target_modules=target_modules, init="gaussian",
        adapter_name=adapter_name,
    )
    fake_cfg = SimpleNamespace(lora=lora_cfg)
    attach_info = lora_trainer.attach_lora(unet, fake_cfg)

    lora_trainer.load_lora_state(unet, state)
    attach_info["n_loaded"] = len(state)
    print(f"[transfer_matrix] loaded {CHECKPOINT_RUN_DIR.name} LoRA: rank={rank} alpha={alpha} "
          f"targets={target_modules} n_loaded={attach_info['n_loaded']}")
    return attach_info


def fill(pair_slugs: list[str] | None) -> None:
    squares = json.loads((OUT_ROOT / "matrix_manifest.json").read_text())
    todo = [s for s in squares if s["disjoint"] and (pair_slugs is None or s["eval_pair"] in pair_slugs)]
    if not todo:
        print("[transfer_matrix] nothing to fill")
        return

    ctx = make_ctx(num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE)
    attach_info = _attach_and_load_lora(ctx.models["unet"])
    mask = [True] * WINDOW_ON_STEPS + [False] * (NUM_INFERENCE_STEPS - WINDOW_ON_STEPS)

    for sq in todo:
        pair_slug, split, seeds = sq["eval_pair"], sq["split"], sq["seeds"]
        out_dir = OUT_ROOT / pair_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        cell_dir = DEFAULT_CACHE_ROOT / split / pair_slug
        samples = []
        for seed in seeds:
            emb = torch.load(cell_dir / f"seed_{seed}" / "embeddings.pt",
                              map_location=ctx.device, weights_only=False)
            out = run_lora_residual_inject_masked(
                init_latents=emb["init_latents"].to(device=ctx.device, dtype=ctx.dtype),
                models=ctx.models, scheduler=ctx.scheduler,
                seq_a=emb["seq_a"].to(device=ctx.device, dtype=ctx.dtype),
                pool_a=emb["pool_a"].to(device=ctx.device, dtype=ctx.dtype),
                seq_b=emb["seq_b"].to(device=ctx.device, dtype=ctx.dtype),
                pool_b=emb["pool_b"].to(device=ctx.device, dtype=ctx.dtype),
                seq_e=emb["seq_uncond"].to(device=ctx.device, dtype=ctx.dtype),
                pool_e=emb["pool_uncond"].to(device=ctx.device, dtype=ctx.dtype),
                guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
                cfg_mask=mask, composition_mode="with_prompt",
                height=1024, width=1024,
                euler_init_noise_sigma=float(emb["euler_init_noise_sigma"]),
                device=ctx.device, dtype=ctx.dtype,
                lambda_value=LAMBDA, lora_adapter_name=attach_info["adapter_name"],
            )
            img_path = out_dir / f"seed_{seed}.png"
            write_decoded_image(out.image, img_path)
            samples.append({
                "seed": seed, "png": str(img_path),
                "max_delta_norm": max(out.extras["delta_norm_per_step"]),
            })
            print(f"[transfer_matrix] {pair_slug} seed={seed} -> {img_path.name} "
                  f"max||r_hat||={samples[-1]['max_delta_norm']:.3f}")

        (out_dir / "fill_manifest.json").write_text(json.dumps({
            "eval_pair": pair_slug, "split": split, "checkpoint": str(CHECKPOINT_PATH),
            "window_on_steps": WINDOW_ON_STEPS, "num_inference_steps": NUM_INFERENCE_STEPS,
            "lambda_value": LAMBDA, "samples": samples,
        }, indent=2))


def score() -> None:
    from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
        instance_score_to_dict, score_output_instances,
    )

    squares = json.loads((OUT_ROOT / "matrix_manifest.json").read_text())
    rows = []
    for sq in squares:
        if not sq["disjoint"]:
            continue
        pair_slug, split, seeds = sq["eval_pair"], sq["split"], sq["seeds"]
        prompt_a, prompt_b = pair_slug.split("__x__")
        prompt_a = prompt_a.replace("_", " ")
        prompt_b = prompt_b.replace("_", " ")
        cell_dir = DEFAULT_CACHE_ROOT / split / pair_slug
        manifest_path = OUT_ROOT / pair_slug / "fill_manifest.json"
        if not manifest_path.exists():
            print(f"[transfer_matrix] skip {pair_slug}: not filled yet")
            continue
        fill_manifest = json.loads(manifest_path.read_text())
        png_by_seed = {s["seed"]: Path(s["png"]) for s in fill_manifest["samples"]}

        corrected, baseline = [], []
        for seed in seeds:
            c = score_output_instances(png_by_seed[seed], prompt_a, prompt_b)
            b = score_output_instances(cell_dir / f"seed_{seed}" / "poe.png", prompt_a, prompt_b)
            corrected.append({"seed": seed, **instance_score_to_dict(c)})
            baseline.append({"seed": seed, **instance_score_to_dict(b)})

        corrected_rate = sum(1 for r in corrected if r["n_instances"] >= 2) / len(corrected)
        baseline_rate = sum(1 for r in baseline if r["n_instances"] >= 2) / len(baseline)
        row = {
            "eval_pair": pair_slug, "split": split, "seeds": seeds,
            "corrected_compose_rate": corrected_rate, "plain_poe_compose_rate": baseline_rate,
            "corrected_scores": corrected, "plain_poe_scores": baseline,
        }
        rows.append(row)
        print(f"[transfer_matrix] {pair_slug}: corrected={corrected_rate:.2f} "
              f"plain_poe={baseline_rate:.2f}")

    (OUT_ROOT / "transfer_matrix.json").write_text(json.dumps({
        "checkpoint_run": CHECKPOINT_RUN_DIR.name, "train_pairs": TRAIN_PAIRS,
        "window_on_steps": WINDOW_ON_STEPS, "num_inference_steps": NUM_INFERENCE_STEPS,
        "lambda_value": LAMBDA, "rows": rows,
    }, indent=2))
    print(f"[transfer_matrix] wrote {len(rows)} scored rows -> transfer_matrix.json")


def render() -> None:
    import matplotlib.pyplot as plt

    data = json.loads((OUT_ROOT / "transfer_matrix.json").read_text())
    rows = sorted(data["rows"], key=lambda r: -r["corrected_compose_rate"])

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 4))
    x = range(len(rows))
    ax.bar([i - 0.2 for i in x], [r["corrected_compose_rate"] for r in rows],
           width=0.4, label=f"{data['checkpoint_run']} LoRA (window 0-{data['window_on_steps']}, "
                             f"lambda={data['lambda_value']})", color="#2ca02c")
    ax.bar([i + 0.2 for i in x], [r["plain_poe_compose_rate"] for r in rows],
           width=0.4, label="plain PoE baseline", color="#d62728")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["eval_pair"].replace("__x__", " x ").replace("_", " ")
                         for r in rows], rotation=45, ha="right")
    ax.set_ylabel("compose rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Transfer matrix: phase1_r8_100k (11 training pairs) on its 8 held-out pairs")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_ROOT / "step-34_transfer-matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[transfer_matrix] wrote {out_path}")


def qual_strip() -> None:
    """2 best / 2 worst squares, each as (corrected, plain-PoE) at their first seed,
    captioned with the actual compose rate. Instruction 2.2's four-square eyeball
    check, saved as a figure rather than done ad hoc."""
    import matplotlib.pyplot as plt
    from PIL import Image

    data = json.loads((OUT_ROOT / "transfer_matrix.json").read_text())
    rows = sorted(data["rows"], key=lambda r: -r["corrected_compose_rate"])
    if len(rows) < 4:
        print(f"[transfer_matrix] only {len(rows)} scored rows, need >=4 for the strip")
        return
    picks = [("best", rows[0]), ("best", rows[1]), ("worst", rows[-1]), ("worst", rows[-2])]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, (tag, row) in enumerate(picks):
        pair_slug, seed = row["eval_pair"], row["seeds"][0]
        corrected_png = OUT_ROOT / pair_slug / f"seed_{seed}.png"
        poe_png = None
        for split in ("heldout", "train"):
            candidate = DEFAULT_CACHE_ROOT / split / pair_slug / f"seed_{seed}" / "poe.png"
            if candidate.exists():
                poe_png = candidate
                break

        label = pair_slug.replace("__x__", " x ").replace("_", " ")
        for r, img_path in enumerate([corrected_png, poe_png]):
            ax = axes[r, col]
            ax.imshow(Image.open(img_path).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{tag}: {label}\ncorrected rate {row['corrected_compose_rate']:.2f}",
                              fontsize=9)
            else:
                ax.set_xlabel(f"plain PoE rate {row['plain_poe_compose_rate']:.2f}", fontsize=9)

    fig.suptitle("phase1_r8_100k transfer: 2 best / 2 worst held-out pairs "
                 "(seed shown = first eval seed)")
    fig.tight_layout()
    out_path = OUT_ROOT / "step-34_transfer-matrix-qual-strip.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[transfer_matrix] wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--strip", action="store_true")
    ap.add_argument("--pairs", default=None, help="comma-separated eval-pair slugs to restrict --fill to")
    args = ap.parse_args(argv)

    if not (args.census or args.fill or args.score or args.render or args.strip):
        ap.error("pass at least one of --census / --fill / --score / --render / --strip")

    if args.census:
        census()
    if args.fill:
        if not (OUT_ROOT / "matrix_manifest.json").exists():
            print("[transfer_matrix] no matrix_manifest.json — run --census first.", file=sys.stderr)
            return 2
        pairs = args.pairs.split(",") if args.pairs else None
        fill(pairs)
    if args.score:
        score()
    if args.render:
        render()
    if args.strip:
        qual_strip()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
