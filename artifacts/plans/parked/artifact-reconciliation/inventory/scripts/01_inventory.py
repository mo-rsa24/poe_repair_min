#!/usr/bin/env python3
"""Re-runnable artifact inventory for poe_repair_min.

Walks outputs/ and poe_repair/experiments/, finds every local W&B run directory,
extracts (project, run-id, step, runtime, exit-state) from each, and prints one
reconciliation table: experiment -> runs -> status -> surviving artifact path.

Read-only. Does not modify or move any file. Re-run after runs change:
    python inventory/scripts/01_inventory.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
EXPERIMENTS = ROOT / "poe_repair" / "experiments"


def human(nbytes: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if nbytes < 1024:
            return f"{nbytes:.0f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.0f}P"


def dir_size(p: Path) -> int:
    total = 0
    for dp, _, fs in os.walk(p):
        for f in fs:
            fp = Path(dp) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def wandb_runs():
    """Yield dicts for each local W&B run directory."""
    for run_dir in sorted(OUTPUTS.rglob("wandb/run-*")):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name.split("-")[-1]
        files = run_dir / "files"
        meta = read_json(files / "wandb-metadata.json")
        summ = read_json(files / "wandb-summary.json")
        dbg = run_dir / "logs" / "debug.log"
        dbgi = run_dir / "logs" / "debug-internal.log"

        # project name lives in the debug logs
        project = ""
        for logf in (dbg, dbgi):
            if logf.exists():
                for line in logf.read_text(errors="ignore").splitlines():
                    if "'project':" in line or '"project":' in line:
                        seg = line.split("project", 1)[1]
                        # seg looks like: ': 'poe-repair-lora', ...  -> value is 3rd token
                        project = seg.split("'")[2] if "'" in seg else seg.split('"')[2]
                        break
            if project:
                break

        # exit state: a clean finish reaches the run footer ("restore done");
        # an early death stops at "Redirects installed"; a benign W&B upload
        # failure logs a filestream "fatal" but still wrote a checkpoint.
        tail = dbg.read_text(errors="ignore")[-2000:] if dbg.exists() else ""
        idbg = dbgi.read_text(errors="ignore") if dbgi.exists() else ""
        if "restore done" in tail or "Reached EOF" in tail:
            state = "finished"
        elif "Redirects installed" in tail and "restore done" not in tail:
            state = "died-early"
        else:
            state = "unknown"
        sync_fatal = "filestream: fatal error" in idbg

        yield {
            "project": project,
            "run_id": run_id,
            "program": (meta.get("program") or meta.get("codePath") or "").strip(),
            "step": summ.get("_step"),
            "runtime_s": summ.get("_runtime"),
            "state": state,
            "sync_fatal": sync_fatal,
            "dir": str(run_dir.relative_to(ROOT)),
        }


def surviving_checkpoints():
    """Map experiment top-dir -> list of surviving checkpoint files."""
    out = {}
    for pt in sorted(OUTPUTS.rglob("*.pt")):
        name = pt.name
        if not any(k in name for k in ("lora_step", "best.pt", "last.pt", "step_")):
            continue
        exp = pt.relative_to(OUTPUTS).parts[0]
        out.setdefault(exp, []).append(str(pt.relative_to(ROOT)))
    return out


def main():
    print("=" * 78)
    print("W&B RUNS (local run directories)")
    print("=" * 78)
    runs = list(wandb_runs())
    for r in runs:
        flag = "  [SYNC-FATAL]" if r["sync_fatal"] else ""
        rt = f"{r['runtime_s']:.0f}s" if isinstance(r["runtime_s"], (int, float)) else "-"
        print(f"[{r['project'] or '?':22}] {r['run_id']:10} step={str(r['step']):>7} "
              f"rt={rt:>8} {r['state']:11}{flag}")
        print(f"    {r['dir']}")

    print()
    print("=" * 78)
    print("SURVIVING CHECKPOINTS (per experiment top-dir)")
    print("=" * 78)
    for exp, cks in sorted(surviving_checkpoints().items()):
        print(f"{exp:28} {len(cks):5} ckpt files   e.g. {cks[-1]}")

    print()
    print("=" * 78)
    print("EXPERIMENT DISK FOOTPRINT")
    print("=" * 78)
    if OUTPUTS.exists():
        for d in sorted(OUTPUTS.iterdir()):
            if d.is_dir():
                print(f"{human(dir_size(d)):>7}  outputs/{d.name}")

    print()
    print("Projects seen locally:",
          sorted({r["project"] for r in runs if r["project"]}))


if __name__ == "__main__":
    main()
