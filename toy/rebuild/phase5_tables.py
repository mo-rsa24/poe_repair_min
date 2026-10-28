"""Print phase 5's results as markdown tables, from rebuild/out/phase5/overlap-dial.json and finetune.json.

    python -m rebuild.phase5_tables
"""
import json
import os

import numpy as np

OUT = "rebuild/out/phase5"


def dial_tables():
    d = json.load(open(os.path.join(OUT, "overlap-dial.json")))
    q = d["quadrant"]
    print("| Overlap | Exact, 0.15 inside | Tuned MALA | UHA | Reverse diffusion | Exact, whole quadrant | TV tuned MALA / UHA / reverse diffusion (floor) |")
    print("|---|---|---|---|---|---|---|")
    for f in sorted({r["overlap"] for r in q}):
        rr = [r for r in q if r["overlap"] == f]
        m = lambda g: f"{np.mean([g(r) for r in rr]):.2f} ± {np.std([g(r) for r in rr]):.2f}"
        tv = lambda c: f"{np.mean([r[c]['tv'] for r in rr]):.2f}"
        print(f"| {f:.0%} | {m(lambda r: r['exact_interior'])} | {m(lambda r: r['MALA, tuned']['interior'])} | "
              f"{m(lambda r: r['UHA']['interior'])} | {m(lambda r: r['reverse diffusion']['interior'])} | "
              f"{m(lambda r: r['exact_quadrant'])} | {tv('MALA, tuned')} / {tv('UHA')} / {tv('reverse diffusion')} "
              f"({np.mean([r['floor_tv'] for r in rr]):.2f}) |")
    print()
    print("| Background variance | Overlap | Exact A∩B | Tuned MALA | UHA | Reverse diffusion | TV tuned MALA / UHA / reverse diffusion (floor) |")
    print("|---|---|---|---|---|---|---|")
    for r in d["parts_world"]:
        print(f"| {r['background_variance']} | {r['overlap']:g} | {r['exact_weight_ab']:.3f} | {r['MALA, tuned']['ab']:.3f} | "
              f"{r['UHA']['ab']:.3f} | {r['reverse diffusion']['ab']:.3f} | {r['MALA, tuned']['tv']:.2f} / {r['UHA']['tv']:.2f} / "
              f"{r['reverse diffusion']['tv']:.2f} ({r['floor_tv']:.2f}) |")


def finetune_table():
    path = os.path.join(OUT, "finetune.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    print()
    print("| Fine-tune steps | Exact, 0.15 inside | Tuned MALA | UHA | Exact, whole quadrant | Tuned MALA | UHA | TV tuned MALA / UHA |")
    print("|---|---|---|---|---|---|---|---|")
    for r in d["checkpoints"] + [dict(d["from_scratch"], steps="from scratch, 10000")]:
        print(f"| {r['steps']} | {r['exact']['interior']:.3f} | {r['MALA, tuned']['interior']:.3f} | {r['UHA']['interior']:.3f} | "
              f"{r['exact']['quadrant']:.3f} | {r['MALA, tuned']['quadrant']:.3f} | {r['UHA']['quadrant']:.3f} | "
              f"{r['MALA, tuned']['tv']:.2f} / {r['UHA']['tv']:.2f} |")


if __name__ == "__main__":
    dial_tables()
    finetune_table()
