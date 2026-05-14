"""Run D4-A / D4-A-t: for each (seed, condition, window) cell, drive
``run_delta_override`` and grade the decoded image via the §4 protocol.

Decisions:

* ``oracle``, ``shared_mean``, ``shuffle``, ``zero`` × N seeds × windows.
* Init latents and DDIM step sequence are pinned per seed across all
  (condition, window) cells — this is the wiring requirement spelled out
  in §7b's D4-A-t section.
* Grading: GroundingDINO regime + VQAScore-min (3 questions). The grader
  module is lazy-loaded; a missing detector / VQA backbone is reported in
  the verdict as ``"grader_unavailable"`` rather than crashing the run.

The runner emits a structured JSON (``D4aResult.to_dict``) that the figure
module consumes directly — keeping rendering free of any IO with the
sampler stack.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from poe_repair.experiments._eval_common import slugify
from poe_repair.experiments.thread_c_structure.d4a.overrides import (
    Condition, OverrideBuilder, build_overrides_for_seed,
)
from poe_repair.experiments.thread_c_structure.loader import CellPath
from poe_repair.methods._sampling import run_delta_override, write_decoded_image


WindowSpec = tuple[str, tuple[int, int]]


DEFAULT_WINDOWS: tuple[WindowSpec, ...] = (
    ("pre_commit",  (0, 5)),
    ("commit",      (5, 25)),
    ("post_commit", (25, 49)),
    ("all",         (0, 49)),
)

DEFAULT_VQA_QUESTIONS_TEMPLATE = (
    "Is there a {a} in the image?",
    "Is there a {b} in the image?",
    "Is the {a} clearly separate from the {b}?",
)


@dataclass
class D4aGradeRecord:
    detection_regime: str
    detection_confidences: dict[str, float]
    vqa_min: float
    vqa_mean: float
    vqa_per_question: list[float]
    clip_score: float | None = None
    grader_error: str | None = None


@dataclass
class D4aSeedRow:
    seed: int
    window_label: str
    window: tuple[int, int]
    condition: Condition
    image_path: str
    grade: D4aGradeRecord


@dataclass
class D4aResult:
    pair_slug: str
    seeds: list[int]
    windows: list[WindowSpec]
    shuffle_pairing: dict[int, int]
    rows: list[D4aSeedRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pair_slug": self.pair_slug,
            "seeds": self.seeds,
            "windows": [{"label": lab, "lo": lo, "hi": hi} for lab, (lo, hi) in self.windows],
            "shuffle_pairing": {str(k): v for k, v in self.shuffle_pairing.items()},
            "rows": [
                {
                    "seed": r.seed,
                    "window_label": r.window_label,
                    "window": list(r.window),
                    "condition": r.condition.value,
                    "image_path": r.image_path,
                    "grade": asdict(r.grade),
                }
                for r in self.rows
            ],
        }

    def by_window(self, label: str) -> list[D4aSeedRow]:
        return [r for r in self.rows if r.window_label == label]


# ---------------------------------------------------------------------------
# Sampler driver
# ---------------------------------------------------------------------------


def _run_one(
    ctx,
    cell,                          # PairSeedCell (run.py's runtime cell)
    delta_tensor: torch.Tensor,    # (T, B, C, H, W)
    window: tuple[int, int] | None,
    image_path: Path,
):
    """Lazy-import sampler stack so the post-hoc plots stay CPU-only."""
    from poe_repair.composers._helpers import encode_pair, init_latents_for_cell

    init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
    emb = encode_pair(cell, ctx)
    out = run_delta_override(
        init_latents=init_latents,
        models=ctx.models,
        scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"],
        seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale,
        num_inference_steps=ctx.num_inference_steps,
        height=cell.height, width=cell.width,
        euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        delta_override=delta_tensor.to(dtype=ctx.dtype),
        window=window,
    )
    write_decoded_image(out.image, image_path)
    return out


# ---------------------------------------------------------------------------
# Grader (§4 protocol)
# ---------------------------------------------------------------------------


def _grade_image(
    image_path: Path,
    *,
    prompt_a: str,
    prompt_b: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    vqa_questions: tuple[str, ...] | None = None,
) -> D4aGradeRecord:
    """Run GroundingDINO + LLaVA (if installed) and return a grade record."""
    queries = (prompt_a.strip().lower(), prompt_b.strip().lower())
    detection_regime = "unavailable"
    detection_confs: dict[str, float] = {}
    vqa_min = float("nan")
    vqa_mean = float("nan")
    vqa_per_q: list[float] = []
    err_parts: list[str] = []
    try:
        from poe_repair.experiments.veracity.metrics import (
            classify_detection_regime, detect_boxes,
        )
        dets = detect_boxes(
            image_path, list(queries),
            box_threshold=box_threshold, text_threshold=text_threshold,
        )
        detection_regime = classify_detection_regime(
            dets, queries=queries, threshold=box_threshold,
        )
        for q in queries:
            confs = [d["confidence"] for d in dets if d.get("label", "").strip().lower() == q]
            detection_confs[q] = float(max(confs)) if confs else 0.0
    except Exception as exc:  # noqa: BLE001
        err_parts.append(f"detection: {type(exc).__name__}: {exc}")

    questions_template = vqa_questions or DEFAULT_VQA_QUESTIONS_TEMPLATE
    questions = [
        q.format(a=queries[0], b=queries[1]) for q in questions_template
    ]
    try:
        from poe_repair.experiments.veracity.metrics import vqascore_yesno
        vqa_per_q = vqascore_yesno(image_path, questions)
        if vqa_per_q:
            vqa_min = float(min(vqa_per_q))
            vqa_mean = float(sum(vqa_per_q) / len(vqa_per_q))
    except Exception as exc:  # noqa: BLE001
        err_parts.append(f"vqa: {type(exc).__name__}: {exc}")

    return D4aGradeRecord(
        detection_regime=detection_regime,
        detection_confidences=detection_confs,
        vqa_min=vqa_min,
        vqa_mean=vqa_mean,
        vqa_per_question=vqa_per_q,
        grader_error="; ".join(err_parts) or None,
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_d4a(
    *,
    cells: list[CellPath],
    prompt_a: str,
    prompt_b: str,
    out_dir: Path,
    ctx=None,
    windows: tuple[WindowSpec, ...] = DEFAULT_WINDOWS,
    conditions: tuple[Condition, ...] = (
        Condition.ORACLE, Condition.SHARED_MEAN,
        Condition.SHUFFLE, Condition.ZERO,
    ),
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    overwrite: bool = False,
) -> D4aResult:
    """Run all (seed, condition, window) cells and grade the outputs.

    When ``ctx`` is None the sampler stack is loaded via
    ``poe_repair.run.make_ctx()``. The caller can pass in an existing
    ``MethodCtx`` to avoid re-loading SDXL for repeated invocations.
    """
    from poe_repair.experiments._eval_common import cell_for
    builder = build_overrides_for_seed(cells)
    pair_slug = slugify(prompt_a, prompt_b)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ctx is None:
        from poe_repair.run import make_ctx
        ctx = make_ctx()

    result = D4aResult(
        pair_slug=pair_slug,
        seeds=builder.seeds,
        windows=list(windows),
        shuffle_pairing={int(k): int(v) for k, v in builder.shuffle_pairing.items()},
    )

    for seed in builder.seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        cell = cell_for(prompt_a, prompt_b, seed)
        for window_label, window in windows:
            window_dir = seed_dir / window_label
            window_dir.mkdir(parents=True, exist_ok=True)
            for cond in conditions:
                image_path = window_dir / f"{cond.value}.png"
                if not (image_path.exists() and not overwrite):
                    delta_tensor = builder.override_for(seed, cond)
                    _run_one(
                        ctx, cell, delta_tensor,
                        window=tuple(window), image_path=image_path,
                    )
                grade = _grade_image(
                    image_path,
                    prompt_a=prompt_a, prompt_b=prompt_b,
                    box_threshold=box_threshold, text_threshold=text_threshold,
                )
                result.rows.append(
                    D4aSeedRow(
                        seed=int(seed),
                        window_label=window_label,
                        window=tuple(window),
                        condition=cond,
                        image_path=str(image_path),
                        grade=grade,
                    )
                )
    (out_dir / "d4a.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result
