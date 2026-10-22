#!/usr/bin/env python
"""Search a slice of Re-LAION-2B-en on the laptop, and lay out the pictures for an eye check.

The full mirror (128 files, 2,097,693,557 captions) lives on the cluster. This works on the first few
files, downloaded from Hugging Face with your own token, so the search and the look can happen without
the cluster. Every count says how many files and captions it covers.

    python scripts/census/laptop_slice.py download --files 4    # needs HF_TOKEN or ~/.cache/huggingface/token
    python scripts/census/laptop_slice.py count                 # whole-word hits, timed per file
    python scripts/census/laptop_slice.py fetch                 # pictures behind four buckets of hits
    python scripts/census/laptop_slice.py sheets                # numbered contact sheets, captions beside
    python scripts/census/laptop_slice.py tally                 # your calls -> table and figure

Run with /home/molef/miniconda3/bin/python3 -I (pyarrow, requests, PIL, matplotlib; nothing else).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
SLICE = Path("/home/molef/datasets/relaion2b-en-slice")
LOOK = REPO / "artifacts/results/what-the-training-captions-show"
DATASET = "laion/relaion2B-en-research-safe"
FILE = "part-{:05d}-339dc23d-0869-4dc0-9390-b4036fcc80c2-c000.snappy.parquet"
SEED = 20261022

SUBJECTS = {"camel": r"\b(camel|camels|dromedary)\b", "dog": r"\b(dog|dogs|puppy)\b"}
FAMILIES = {
    "wet": r"\b(wet|soaked|drenched|soaking|dripping|damp)\b",
    "weather": r"\b(rain|raining|rainy|storm|downpour|puddle)\b",
    "water": r"\b(swimming|bath|river|lake|splash|beach)\b",
}
ANY_WET = r"\b(wet|soaked|drenched|soaking|dripping|damp|rain|raining|rainy|storm|downpour|puddle|swimming|bath|river|lake|splash|beach)\b"
# what gets looked at: an unbiased draw of each animal, and its captions that use a wet, weather or water word
BUCKETS = {
    "camel-any": ("camel", None, 120),
    "camel-wet-words": ("camel", ANY_WET, 400),
    "dog-any": ("dog", None, 120),
    "dog-wet-words": ("dog", ANY_WET, 120),
}


def token() -> str:
    t = os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").read_text().strip()
    return t


def download(n_files: int) -> None:
    import requests
    SLICE.mkdir(parents=True, exist_ok=True)
    head = {"Authorization": f"Bearer {token()}"}
    for i in range(n_files):
        dst = SLICE / FILE.format(i)
        if dst.exists():
            print(dst.name, "already here"); continue
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{FILE.format(i)}"
        t0, got = time.time(), 0
        with requests.get(url, headers=head, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = dst.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk); got += len(chunk)
            tmp.rename(dst)
        dt = time.time() - t0
        print(f"{dst.name}: {got / 1e9:.2f} GB in {dt:.0f} s ({got / 1e6 / dt:.1f} MB/s)", flush=True)


def files() -> list[Path]:
    return sorted(SLICE.glob("part-*.parquet"))


def count() -> None:
    LOOK.mkdir(parents=True, exist_ok=True)
    totals = {"files": 0, "captions": 0, "seconds": 0.0, "per_file": []}
    hits = {s: [] for s in SUBJECTS}
    for path in files():
        t0, n = time.time(), 0
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):  # one row group at a time: a whole file does not fit beside a render
            tab = pf.read_row_group(g, columns=["url", "caption", "key"])
            low = pc.utf8_lower(pc.fill_null(tab["caption"], ""))
            for s, pat in SUBJECTS.items():
                sub = tab.filter(pc.match_substring_regex(low, pat))
                sub = sub.append_column("file", pa.array([path.name] * len(sub), pa.string()))
                hits[s].append(sub)
            n += len(tab)
        dt = time.time() - t0
        totals["files"] += 1; totals["captions"] += n; totals["seconds"] += dt
        totals["per_file"].append({"file": path.name, "captions": n, "seconds": round(dt, 1)})
        print(f"{path.name}: {n:,} captions searched in {dt:.1f} s", flush=True)
    out = {"slice": totals}
    for s in SUBJECTS:
        t = pa.concat_tables(hits[s])
        pq.write_table(t, SLICE / f"hits-{s}.parquet")
        low = pc.utf8_lower(t["caption"])
        row = {"all": len(t)}
        for fam, pat in FAMILIES.items():
            row[fam] = int(pc.sum(pc.match_substring_regex(low, pat)).as_py() or 0)
        row["any wet, weather or water word"] = int(pc.sum(pc.match_substring_regex(low, ANY_WET)).as_py() or 0)
        out[s] = row
    out["full_mirror"] = {"files": 128, "captions": 2_097_693_557, "camel": 784_797, "dog": 14_396_194,
                          "source": "caption-counts.md, searched on the cluster 2026-10-21"}
    (LOOK / "counts.json").write_text(json.dumps(out, indent=1))
    # every camel caption with a wet, weather or water word, as text to read
    t = pq.read_table(SLICE / "hits-camel.parquet")
    keep = pc.match_substring_regex(pc.utf8_lower(t["caption"]), ANY_WET)
    rows = t.filter(keep).to_pylist()
    lines = [f"# Camel captions with a wet, weather or water word\n",
             f"{len(rows)} captions, from {totals['files']} files ({totals['captions']:,} captions). "
             f"Family patterns: {json.dumps(FAMILIES)}\n"]
    lines += [f"{i + 1}. {r['caption'].strip()[:200]}  ([link]({r['url']}))" for i, r in enumerate(rows)]
    (LOOK / "camel-captions-with-wet-words.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "slice"}, indent=1))
    print(f"slice: {totals['files']} files, {totals['captions']:,} captions, {totals['seconds']:.0f} s")


def _fetch(row: dict, dst: Path) -> str:
    import requests
    from PIL import Image
    try:
        r = requests.get(row["url"], timeout=8, headers={"User-Agent": "Mozilla/5.0 (research)"})
        if r.status_code != 200:
            return f"http {r.status_code}"
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        if min(im.size) < 64:
            return "too small"
        im.thumbnail((384, 384)); im.save(dst, quality=88)
        return "ok"
    except Exception as e:
        return type(e).__name__


def fetch() -> None:
    rng = random.Random(SEED)
    for name, (subject, pat, n) in BUCKETS.items():
        t = pq.read_table(SLICE / f"hits-{subject}.parquet")
        if pat:
            t = t.filter(pc.match_substring_regex(pc.utf8_lower(t["caption"]), pat))
        rows = t.to_pylist(); rng.shuffle(rows); rows = rows[:n]
        d = LOOK / name / "images"; d.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(32) as ex:
            status = list(ex.map(lambda ir: _fetch(ir[1], d / f"{ir[0]:04d}.jpg"), enumerate(rows)))
        rec = [{"i": i, "status": s, "caption": r["caption"], "url": r["url"]} for i, (r, s) in enumerate(zip(rows, status))]
        (LOOK / name / "fetch.json").write_text(json.dumps(rec, indent=1))
        print(f"{name}: {sum(s == 'ok' for s in status)} of {len(rows)} pictures loaded ({len(t):,} hits in the slice)")


def sheets(per: int = 30, cols: int = 6, side: int = 220) -> None:
    from PIL import Image, ImageDraw
    for name in BUCKETS:
        rec = [r for r in json.loads((LOOK / name / "fetch.json").read_text()) if r["status"] == "ok"]
        out = LOOK / name / "sheets"; out.mkdir(exist_ok=True)
        for s in range(math.ceil(len(rec) / per)):
            chunk = rec[s * per:(s + 1) * per]
            rows_n = math.ceil(len(chunk) / cols)
            sheet = Image.new("RGB", (cols * side, rows_n * (side + 4)), "white")
            dr = ImageDraw.Draw(sheet)
            md = [f"# {name}, sheet {s + 1:02d}\n", "| Tile | Caption | Link |", "|---|---|---|"]
            for k, r in enumerate(chunk):
                im = Image.open(LOOK / name / "images" / f"{r['i']:04d}.jpg"); im.thumbnail((side - 6, side - 6))
                x, y = (k % cols) * side, (k // cols) * (side + 4)
                sheet.paste(im, (x + 3, y + 3))
                dr.rectangle([x + 3, y + 3, x + 33, y + 25], fill="black"); dr.text((x + 8, y + 7), str(k + 1), fill="white")
                cap = r["caption"].replace("|", "/").replace("\n", " ").strip()[:160]
                md.append(f"| {k + 1} | {cap} | [open]({r['url']}) |")
            sheet.save(out / f"sheet-{s + 1:02d}.jpg", quality=88)
            (out / f"sheet-{s + 1:02d}.md").write_text("\n".join(md) + "\n")
        print(f"{name}: {len(rec)} pictures on {math.ceil(len(rec) / per)} sheets in {out}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tally() -> None:
    """calls.json per bucket: {"sheet-01": {"wet": [4, 17], "none": [2, 9], "unsure": [21]}}; the rest are dry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = {}
    for name in BUCKETS:
        cf = LOOK / name / "calls.json"
        if not cf.exists():
            continue
        calls = json.loads(cf.read_text())
        shown = wet = none = unsure = 0
        for sheet, c in calls.items():
            n = len([l for l in (LOOK / name / "sheets" / f"{sheet}.md").read_text().splitlines() if l.startswith("| ") and l[2].isdigit()])
            shown += n; wet += len(c.get("wet", [])); none += len(c.get("none", [])); unsure += len(c.get("unsure", []))
        animals = shown - none - unsure; dry = animals - wet
        lo, hi = wilson(wet, animals)
        table[name] = {"pictures looked at": shown, "no animal": none, "can't tell": unsure, "animal, dry": dry,
                       "animal, wet": wet, "wet share of animals": round(wet / animals, 4) if animals else None,
                       "95% interval": [round(lo, 4), round(hi, 4)],
                       "rule of three (if 0 wet)": round(3 / animals, 4) if animals and wet == 0 else None}
    (LOOK / "tally.json").write_text(json.dumps(table, indent=1))
    lines = ["| Bucket | Looked at | No animal | Can't tell | Animal, dry | Animal, wet | Wet share (95% interval) |",
             "|---|---|---|---|---|---|---|"]
    for k, v in table.items():
        ct = v["can't tell"]
        lines.append(f"| {k} | {v['pictures looked at']} | {v['no animal']} | {ct} | {v['animal, dry']} | "
                     f"{v['animal, wet']} | {v['wet share of animals']} ({v['95% interval'][0]}–{v['95% interval'][1]}) |")
    (LOOK / "tally.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if table:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        names = list(table); x = range(len(names))
        b1 = [table[n]["animal, wet"] for n in names]; b2 = [table[n]["animal, dry"] for n in names]
        b3 = [table[n]["no animal"] + table[n]["can't tell"] for n in names]
        ax.bar(x, b1, color="#2a72b8", label="animal, wet")
        ax.bar(x, b2, bottom=b1, color="#b8862b", label="animal, dry")
        ax.bar(x, b3, bottom=[a + b for a, b in zip(b1, b2)], color="#c9c4b8", label="no animal or can't tell")
        for i, n in enumerate(names):
            ax.text(i, sum(v for v in (b1[i], b2[i], b3[i])) + 2, f"wet {b1[i]} of {b1[i] + b2[i]}", ha="center", fontsize=10)
        ax.set_xticks(list(x), names); ax.set_ylabel("pictures, called by eye"); ax.legend(fontsize=9)
        ax.set_title("Training pictures behind camel and dog captions (Re-LAION-2B-en slice)", fontsize=11)
        fig.tight_layout(); fig.savefig(LOOK / "tally.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["download", "count", "fetch", "sheets", "tally"])
    ap.add_argument("--files", type=int, default=4)
    a = ap.parse_args()
    {"download": lambda: download(a.files), "count": count, "fetch": fetch, "sheets": sheets, "tally": tally}[a.stage]()
