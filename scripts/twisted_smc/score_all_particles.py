"""Score every saved particle of a twisted-SMC run with the validated instance-count scorer.

The run scored only the shown (largest-weight) particle per render cell, so its compose
fraction was over 3 images per checkpoint. This reads all K particles of every cell for the
Mono and PoE references and for the SMC render at every checkpoint, and writes
``all_particles_scores.json`` into the run directory:

    per_image:  {path: {"count": n, "compose": 0/1}}
    per_panel:  {"mono": f, "poe": f, "smc": {"step_000000": f, ...}}   over all K x cells
    per_cell:   the same split by render cell

Usage:
    <co3 python> scripts/twisted_smc/score_all_particles.py <run_dir> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available on the pinned device (known failure poe-launch-002)")

    from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances

    pat = re.compile(r"^(?P<stem>.+)__(?P<panel>mono|poe|smc)__p(?P<k>\d+)\.png$")
    files: list[tuple[str, str, str, int, Path]] = []      # (panel_key, panel, cell, k, path)
    for p in sorted((args.run_dir / "references").glob("*__p*.png")):
        m = pat.match(p.name)
        if m:
            files.append((m["panel"], m["panel"], m["stem"], int(m["k"]), p))
    for step_dir in sorted((args.run_dir / "samples").glob("step_*")):
        for p in sorted(step_dir.glob("*__smc__p*.png")):
            m = pat.match(p.name)
            if m:
                files.append((f"smc/{step_dir.name}", "smc", m["stem"], int(m["k"]), p))

    per_image: dict[str, dict] = {}
    per_panel: dict[str, list[float]] = defaultdict(list)
    per_cell: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for i, (pkey, panel, cell, k, p) in enumerate(files):
        n, _ = count_instances(p, device=device)
        c = 1.0 if n >= 2 else 0.0
        rel = str(p.relative_to(args.run_dir))
        per_image[rel] = {"count": int(n), "compose": c, "panel": panel, "cell": cell, "particle": k}
        per_panel[pkey].append(c)
        per_cell[cell][pkey].append(c)
        print(f"[{i + 1}/{len(files)}] {rel}: count={n}", flush=True)

    def frac(v: list[float]) -> dict:
        return {"compose_fraction": sum(v) / len(v), "n_images": len(v), "n_compose": int(sum(v))}

    out = {
        "run_dir": str(args.run_dir),
        "scorer": "poe_repair.experiments.compose_scorer_validation.detection_scorer.count_instances, compose iff count >= 2",
        "per_panel": {k: frac(v) for k, v in sorted(per_panel.items())},
        "per_cell": {c: {k: frac(v) for k, v in sorted(d.items())} for c, d in sorted(per_cell.items())},
        "per_image": per_image,
    }
    dst = args.run_dir / "all_particles_scores.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}")
    for k, v in out["per_panel"].items():
        print(f"{k}: {v['n_compose']}/{v['n_images']} = {v['compose_fraction']:.3f}")


if __name__ == "__main__":
    main()
