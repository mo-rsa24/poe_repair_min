"""Last logged value of every eval metric, per cell, for each run folder named on the command line.

Reads history.json only; nothing is averaged, recomputed or carried over from another run.
"""
import json, sys
from pathlib import Path

out = {}
for p in sys.argv[1:]:
    d = Path(p)
    h = d / "history.json"
    if not h.exists():
        continue
    rows = json.loads(h.read_text())
    rows = rows if isinstance(rows, list) else rows.get("rows", [])
    last, last_step = {}, None
    for r in rows:
        s = r.get("train/optimizer_step")
        if s is not None:
            last_step = s
        for k, v in r.items():
            if k.startswith("eval/") and v is not None:
                last[k] = [v, s]
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    att = json.loads((d / "lora_attach.json").read_text()) if (d / "lora_attach.json").exists() else {}
    wb = sorted((d / "wandb").glob("run-*")) if (d / "wandb").exists() else []
    out[d.name] = {
        "path": str(d),
        "wandb": wb[-1].name.split("-")[-1] if wb else None,
        "rank": cfg.get("lora", {}).get("rank"),
        "targets": cfg.get("lora", {}).get("target_modules"),
        "rank_pattern": cfg.get("lora", {}).get("rank_pattern"),
        "loss_space": cfg.get("loss", {}).get("space") or cfg.get("loss_space"),
        "total_epochs": cfg.get("schedule", {}).get("total_epochs"),
        "trainable": att.get("trainable_params"),
        "total_params": att.get("total_params"),
        "n_matched": att.get("n_matched"),
        "last_step": last_step,
        "eval": last,
    }
print(json.dumps(out, indent=1))
