#!/usr/bin/env python
"""Log the four rungs of "what the correction is made of" to W&B: every figure as an image,
every sidecar as one artifact, the headline numbers as the run summary. Prints the run id,
which goes into the review file's Runs table.

Project prime_lab/poe-repair-animals-compose. W&B owns the numbers; the plan tree owns the verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correction_span_common as C  # noqa: E402

FIGURES = (
    "orthogonal-share-over-steps.png", "seed-15-experts-tweedie-strip.png",
    "adapter-span-share-over-steps.png", "seed-15-adapter-vs-correction-norm-maps.png",
    "track-kinetic-energy.png", "which-animal-over-steps.png",
)
SIDECARS = ("orthogonal-share-over-steps.json", "same-place-share.json",
            "adapter-span-share.json", "track-energy-and-which-animal.json")


def main() -> int:
    missing = [f for f in FIGURES + SIDECARS if not (C.RESULTS / f).exists()]
    if missing:
        raise SystemExit(f"missing before logging: {missing}")
    span = json.loads((C.RESULTS / "orthogonal-share-over-steps.json").read_text())["summary"]
    same = json.loads((C.RESULTS / "same-place-share.json").read_text())["summary"]
    adapt = json.loads((C.RESULTS / "adapter-span-share.json").read_text())["summary"]
    track = json.loads((C.RESULTS / "track-energy-and-which-animal.json").read_text())["summary"]

    run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab",
                     name="what-the-correction-is-made-of_cat-x-dog_r32-030050",
                     job_type="cache-analysis", tags=["scope-05", "plan-07", "cache-only", "a_cat__x__a_dog"],
                     config={"pair": C.PAIR, "seeds": list(C.SEEDS), "guidance": C.GUIDANCE,
                             "adapter": "rank 32, alpha 32, step 30050, lambda read at 1.0",
                             "bar": C.bar_block(),
                             "plan": "plans/05-when-does-the-outcome-lock-in/plans/tests/07-what-the-correction-is-made-of.md"})
    run.log({f"figures/{f.removesuffix('.png')}": wandb.Image(str(C.RESULTS / f)) for f in FIGURES})
    art = wandb.Artifact("what-the-correction-is-made-of-sidecars", type="results",
                         description="sidecars for the four rungs; every number in the finding is read from these")
    for f in SIDECARS:
        art.add_file(str(C.RESULTS / f))
    run.log_artifact(art)
    run.summary.update({
        "rung2/early_window_orthogonal_share": span["early_window_orthogonal_share_mean_over_seeds"],
        "rung2/verdict": span["verdict"],
        "rung1/same_place_share_step5": same["same_place_share_mean_per_step"][5],
        "rung1/same_place_share_step10": same["same_place_share_mean_per_step"][10],
        "rung1/cos_on_shared_positions_step5": same["cos_on_shared_positions_mean_per_step"][5],
        "rung3/early_window_ortho_share_adapter": adapt["early_window"]["ortho_share_adapter_mean"],
        "rung3/early_window_cos_orth_adapter_vs_r": adapt["early_window"]["cos_orth_adapter_vs_r_mean"],
        "rung3/sanity_min_cos_live_vs_cached": adapt["sanity_cos_live_vs_cached_poe_min"],
        "rung4/kinetic_energy_from_step_10_median_poe": track["poe"]["kinetic_energy_from_step_10_median"],
        "rung4/kinetic_energy_from_step_10_median_joint": track["joint"]["kinetic_energy_from_step_10_median"],
        "rung4/kinetic_energy_from_step_10_median_lora_1.2": track["lora_1.2"]["kinetic_energy_from_step_10_median"],
        "rung4/runs_with_a_real_flip_poe": track["poe"]["runs_with_a_real_flip_after_step_10"],
    })
    print("run id:", run.id, "url:", run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
