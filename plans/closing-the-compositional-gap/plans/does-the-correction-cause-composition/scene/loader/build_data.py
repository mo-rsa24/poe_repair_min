#!/usr/bin/env python3
"""Build the scene's data file from the artifacts and the paperwork.

Read-only on everything it touches. It writes exactly two things: the typed data file
the app imports, and downscaled thumbnails under the app's public folder.

Every value it emits carries the file it came from. Where a claim's number exists only
as a sentence in the review file, it is emitted as a quote with its line number and the
state `quoted`, never as a number the app can plot.

Run:
  PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
  $PY loader/build_data.py                      # thumbnails included
  $PY loader/build_data.py --no-thumbs          # data file only, seconds
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

SCENE = Path(__file__).resolve().parent.parent
SCOPE = SCENE.parent
REPO = Path("/home-mscluster/mmolefe/Playground/PhD/poe_repair_min")

sys.path.insert(0, str(REPO))
from poe_repair import paths  # noqa: E402

DEFAULTS = {
    "curves": str(paths.resolve(paths.HOW_MUCH_CORRECTION_IS_NEEDED) / "dose_curves.json"),
    "images": paths.resolve(paths.HOW_MUCH_CORRECTION_IS_NEEDED) / "pairs",
    "figures": str(paths.resolve(paths.HOW_MUCH_CORRECTION_IS_NEEDED)),
}

REVIEW = SCOPE / "review/hypothesis-02-more-correction-more-composition.md"
PLAN = SCOPE / "plans/hypothesis-02-more-correction-more-composition.md"
PROCEDURE = SCOPE / "procedures/hypothesis-02-recheck-the-headline-numbers.md"
SCORER = REPO / "poe_repair/experiments/compose_scorer_validation/detection_scorer.py"
PLOTTER = REPO / "scripts/plot_dose_curves.py"
SWEEP_SH = REPO / "scripts/mechanism_study/run_dose_sweep.sh"
CANARIES = REPO / "tests/test_interaction_term_canaries.py"
PAIR_POOL = paths.resolve(paths.DOES_THE_FIX_REACH_UNSEEN_PAIRS) / "pair_pool.yaml"
PAPER_FIGS = REPO / "paper/iclr/figures"
MAKE_F2 = REPO / "scripts/compose_rate_vs_correction.py"

ROW_DIRS = {"oracle": "", "random": "_random", "wrong_pair": "_wrong_pair"}
ROW_LABELS = {
    "oracle": "the real correction",
    "random": "a random vector, same size",
    "wrong_pair": "another pair's correction",
}


class Missing(Exception):
    """A source the ledger depends on is not where it should be."""


def rel(p) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def stamp(p) -> dict:
    p = Path(p)
    if not p.exists():
        raise Missing(f"{p} does not exist")
    st = p.stat()
    return {
        "path": rel(p),
        "abs": str(p),
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "bytes": st.st_size,
    }


def find_line(path: Path, needle: str) -> dict:
    """Locate a sentence in a file and return it with its line number.

    Fails loudly when the anchor has drifted, so a reworded review file breaks the
    build instead of silently emitting a stale quote.
    """
    text = path.read_text().splitlines()
    for i, line in enumerate(text, 1):
        if needle in line:
            return {"path": rel(path), "line": i, "text": line.strip()}
    raise Missing(f"anchor not found in {rel(path)}: {needle!r}")


STOP = re.compile(r"^\s*(-\s*\[[ x]\]|#{2,}\s|\|)")


def block_after(path: Path, needle: str, n: int, include_anchor: bool = False) -> dict:
    """An anchor's own answer, quoted verbatim with its line range.

    n is a maximum, not a count: the block stops early at the next checkbox question,
    the next heading, or a table, so one claim's quote can never run into the next one's.
    """
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if needle in line:
            start = i if include_anchor else i + 1
            end = i + 1 + n
            for j in range(i + 1, min(end, len(lines))):
                if STOP.match(lines[j]):
                    end = j
                    break
            body = [x.strip() for x in lines[start:end]]
            while body and not body[-1]:
                body.pop()
            return {
                "path": rel(path),
                "lines": [start + 1, start + len(body)],
                "body": body,
            }
    raise Missing(f"anchor not found in {rel(path)}: {needle!r}")


def read_constant(path: Path, name: str) -> dict:
    for i, line in enumerate(path.read_text().splitlines(), 1):
        m = re.match(rf"^{name}\s*=\s*(.+?)\s*(#.*)?$", line)
        if m:
            return {"name": name, "raw": m.group(1), "path": rel(path), "line": i}
    raise Missing(f"{name} not found in {rel(path)}")


# --------------------------------------------------------------------------- cells


def cell_image(images_root: Path, row: str, pair: str, seed: int, lam: float) -> dict:
    """The PNG behind one scored record, or an honest miss.

    At lambda 0 nothing is injected, so the three rows share one image. That sharing is
    recorded on the cell rather than hidden, because it is why the three rows read the
    same rate at the left edge.
    """
    tag = f"lam{int(round(lam * 100)):03d}"
    suffix = "" if lam == 0.0 else ROW_DIRS[row]
    stem = f"teacher_residual_const_{tag}{suffix}"
    png = images_root / pair / f"seed_{seed}" / stem / f"{stem}.png"
    return {
        "relPath": str(png.relative_to(images_root)),
        "exists": png.exists(),
        "sharedAtZero": lam == 0.0,
    }


def build_cells(curves_doc: dict, images_root: Path) -> list[dict]:
    cells = []
    for s in curves_doc["scores"]:
        img = cell_image(images_root, s["row"], s["pair"], s["seed"], s["lambda"])
        cells.append(
            {
                "row": s["row"],
                "pair": s["pair"],
                "seed": s["seed"],
                "lam": s["lambda"],
                "nInstances": s["n_instances"],
                "compose": s["compose"],
                "image": img,
            }
        )
    return cells


# -------------------------------------------------------------------------- claims

# id, type, state, and the anchor that locates the claim's own words in the review file.
# The text is never typed here; it is read from the file at build time.
CLAIM_ANCHORS = [
    ("C1", "dose-response", "measured",
     "Does the oracle compose-rate rise with", 7),
    ("C2", "threshold", "measured",
     "Do the curves hold when only this sweep's own cells are scored?", 5),
    ("C3", "threshold", "quoted",
     "Does the scorer's instance count mean what the rule says it means?", 14),
    ("C4", "invariance", "quoted",
     "Does the harness leave plain PoE untouched", 5),
    ("C5", "comparison", "quoted",
     "Do the two control rows actually inject something different from the oracle?", 4),
    ("C6", "existence", "measured",
     "Does the eyeball agree with the scorer on the smoke cell?", 4),
    ("C7", "existence", "measured",
     "Does the five-image strip read the same on complete cells?", 16),
    ("C8", "null", "measured",
     "A control the pool does not actually have", 6),
    ("C9", "environment", "measured",
     "Did the output land where the plan said it would?", 5),
]

# What each quoted claim needs before it can become a measured one. Commands are taken
# from the procedure file where it holds one; where it does not, that is said plainly.
GAP_FILLS = {
    "C3": {
        "what": "the before-and-after box counts on the known cells: the control's false "
                "positive dropping 2 to 1, and the oracle staying at 2 or above",
        "command": "the step 3 snippet in "
                   "procedures/hypothesis-02-recheck-the-headline-numbers.md, "
                   "writing its two counts to JSON instead of printing them",
        "output": "a small JSON beside dose_curves.json",
        "cost": "about a minute on CPU, no GPU needed",
        "hasCommand": True,
    },
    "C4": {
        "what": "a stored pass record for the eight canary tests, rather than the "
                "assertions read from source",
        "command": "pytest tests/test_interaction_term_canaries.py "
                   "--junitxml=plans/.../scene/data/canaries.xml",
        "output": "canaries.xml beside the data file",
        "cost": "seconds, CPU",
        "hasCommand": True,
    },
    "C5": {
        "what": "the latent distances between the three rows and the per-step size of "
                "the real correction, so direction-not-magnitude is a loaded number",
        "command": "none. No procedure in this scope names the command that produced "
                   "2.33, 2.21, 2.67 or 9.53",
        "output": "unknown until a procedure is written",
        "cost": "unknown",
        "hasCommand": False,
    },
}


def build_claims() -> list[dict]:
    out = []
    for cid, ctype, state, anchor, span in CLAIM_ANCHORS:
        loc = find_line(REVIEW, anchor)
        body = block_after(REVIEW, anchor, span)
        q = re.sub(r"^-\s*\[[ x]\]\s*", "", loc["text"]).strip()
        mark = None
        m = re.match(r"^(✅|❌|🟡|⚠️)\s*", q)
        if m:
            mark, q = m.group(1), q[m.end():]
        # Some claims are a bold statement rather than a question, with the sentence
        # continuing on the same line. Take the bolded part as the heading; the rest
        # belongs to the answer.
        lead = re.match(r"^\*\*(.+?)\*\*\s*(.*)$", q)
        carry = ""
        if lead:
            q, carry = lead.group(1), lead.group(2)
        out.append(
            {
                "id": cid,
                "type": ctype,
                "state": state,
                "mark": mark,
                "question": q,
                # Blank lines are kept: they are the paragraph breaks in the review file,
                # and dropping them turns a quoted answer into a stack of fragments.
                "answer": ([carry] if carry else []) + body["body"],
                "source": {"path": loc["path"], "lines": [loc["line"], body["lines"][1]]},
                "fill": GAP_FILLS.get(cid),
            }
        )
    return out


def parse_rate_table(path: Path, anchor: str) -> dict:
    """Pull a markdown percentage table out of the review file, verbatim.

    Used only for the pre-floor read, whose results file was overwritten. The rows are
    kept as the strings they are in the file so nothing is silently re-derived.
    """
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if anchor in l)
    rows, first, last = [], None, None
    for i in range(start, min(start + 25, len(lines))):
        if lines[i].strip().startswith("|"):
            first = i + 1 if first is None else first
            last = i + 1
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            rows.append(cells)
        elif rows:
            break
    return {"path": rel(path), "lines": [first, last], "rows": rows}


# --------------------------------------------------------------------- environment


def du_bytes(path: Path) -> int:
    """Disk usage in bytes, the same quantity `du -sh` reports.

    Not apparent size (`du -sb`), which is smaller and would read as a contradiction of
    the figure in the review file.
    """
    out = subprocess.run(["du", "-s", "-B1", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])


def filesystem_of(path: Path) -> str:
    out = subprocess.run(["df", "--output=target", str(path)], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[-1].strip()


def build_environment(images_root: Path) -> dict:
    out_line = find_line(SWEEP_SH, "OUT=$REPO/")
    guard_line = find_line(SWEEP_SH, "df --output=pcent")
    return {
        "outputBytes": du_bytes(images_root),
        "outputFilesystem": filesystem_of(images_root),
        "scriptWrites": out_line,
        "scriptGuardChecks": guard_line,
        "guardFilesystem": "/datasets",
        "planTask": find_line(PLAN, "Move the output off /home-mscluster"),
    }


# ------------------------------------------------------------------------ thumbnails


def make_thumbs(images_root: Path, cells: list[dict], size: int) -> dict:
    from PIL import Image

    dest = SCENE / "public/thumbs"
    made = skipped = 0
    seen = set()
    for c in cells:
        rp = c["image"]["relPath"]
        if rp in seen or not c["image"]["exists"]:
            continue
        seen.add(rp)
        target = (dest / rp).with_suffix(".jpg")
        src = images_root / rp
        if target.exists() and target.stat().st_mtime >= src.stat().st_mtime:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im.thumbnail((size, size))
            im.convert("RGB").save(target, "JPEG", quality=82)
        made += 1
    return {"dir": "public/thumbs", "px": size, "written": made, "upToDate": skipped,
            "unique": len(seen)}


def link_full(images_root: Path, figures_root: Path) -> dict:
    pub = SCENE / "public"
    pub.mkdir(parents=True, exist_ok=True)
    links = {}
    for name, target in (("full", images_root), ("figures", figures_root),
                         ("paperfigs", PAPER_FIGS)):
        link = pub / name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        links[name] = str(target)
    return links


# ------------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default=DEFAULTS["curves"])
    ap.add_argument("--images", default=str(DEFAULTS["images"]))
    ap.add_argument("--figures", default=DEFAULTS["figures"])
    ap.add_argument("--thumb-px", type=int, default=320)
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--out", default=str(SCENE / "src/data/result.json"))
    args = ap.parse_args()

    curves_path = Path(args.curves)
    images_root = Path(args.images)
    figures_root = Path(args.figures)

    curves_doc = json.loads(curves_path.read_text())
    cells = build_cells(curves_doc, images_root)
    missing = [c for c in cells if not c["image"]["exists"]]

    # The assembled paper figures live with the paper; the diagnostics live beside the
    # sweep output. Both are served, from two different roots.
    figures = {}
    for key, root, public, fname in [
        ("f2", PAPER_FIGS, "/paperfigs", "compose-rate-as-correction-rises.png"),
        ("f2b", PAPER_FIGS, "/paperfigs", "compose-rate-as-correction-rises-for-a-dissimilar-pair.png"),
        ("f2Halfwidth", PAPER_FIGS, "/paperfigs", "compose-rate-as-correction-rises-with-a-random-control.png"),
        ("stripBoxes", figures_root, "/figures",
         "dose_strip_an_elephant__x__a_penguin_seed10_boxes.png"),
        ("stripDissimilar", figures_root, "/figures",
         "dose_strip_an_elephant__x__a_penguin_seed10.png"),
        ("stripSupplementary", figures_root, "/figures",
         "dose_strip_a_leopard__x__a_jaguar_seed9.png"),
        ("curvesPng", figures_root, "/figures", "dose_curves.png"),
    ]:
        p = Path(root) / fname
        figures[key] = (stamp(p) | {"publicPath": f"{public}/{fname}"}) if p.exists() else None

    data = {
        "meta": {
            "builtFrom": {
                "curves": stamp(curves_path),
                "review": stamp(REVIEW),
                "plan": stamp(PLAN),
                "procedure": stamp(PROCEDURE),
                "scorer": stamp(SCORER),
                "plotter": stamp(PLOTTER),
                "canaries": stamp(CANARIES),
                "sweepScript": stamp(SWEEP_SH),
                "pairPool": stamp(PAIR_POOL),
                "figureScript": stamp(MAKE_F2),
            },
            "imagesRoot": str(images_root),
            "figuresRoot": str(figures_root),
            "runIdNote": "The sweep ran outside Slurm on the session node, so there is no "
                         "job id. The log path is its identity.",
            "runLog": rel(REPO / "results/mechanism_study/dose_sweep.log"),
            "missingImages": len(missing),
        },
        "verdict": block_after(REVIEW, "Where it stands in one line", 4, include_anchor=True),
        "runKind": find_line(REVIEW, "**Tests the claim.**"),
        "constants": {
            "minBoxFraction": read_constant(SCORER, "MIN_BOX_FRACTION"),
            "sweepSeeds": read_constant(PLOTTER, "SWEEP_SEEDS"),
        },
        "rows": [{"key": k, "label": ROW_LABELS[k], "isControl": k != "oracle"}
                 for k in ("oracle", "random", "wrong_pair")],
        "lambdas": curves_doc["lambdas"],
        "cells": cells,
        "fileSummary": {
            "curves": curves_doc["curves"],
            "auc": curves_doc["auc"],
            "nCells": curves_doc["n_cells"],
            "scorer": curves_doc["scorer"],
            "note": "The app recomputes all of these from the cells and shows the "
                    "comparison. These are the file's own values, kept for that check.",
        },
        "claims": build_claims(),
        "supersededTable": {
            **parse_rate_table(REVIEW, "Do the curves hold when only this sweep's own cells"),
            "why": "Its results file was overwritten by the re-score, so these figures "
                   "exist only as text in the review file.",
        },
        "environment": build_environment(images_root),
        "figures": figures,
    }

    if args.no_thumbs:
        # Report what is already on disk, so the page never shows an unknown count just
        # because this run skipped the resize pass.
        have = list((SCENE / "public/thumbs").rglob("*.jpg"))
        data["meta"]["thumbs"] = {"dir": "public/thumbs", "px": args.thumb_px, "written": 0,
                                  "upToDate": len(have), "unique": len(have)}
    else:
        data["meta"]["thumbs"] = make_thumbs(images_root, cells, args.thumb_px)
    data["meta"]["publicLinks"] = link_full(images_root, figures_root)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))

    print(f"wrote {rel(out)}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"  cells        {len(cells)}   images missing: {len(missing)}")
    print(f"  claims       {len(data['claims'])}  "
          f"({sum(c['state']=='measured' for c in data['claims'])} measured, "
          f"{sum(c['state']=='quoted' for c in data['claims'])} quoted)")
    print(f"  constants    {data['constants']['minBoxFraction']['name']}="
          f"{data['constants']['minBoxFraction']['raw']} from "
          f"{data['constants']['minBoxFraction']['path']}:"
          f"{data['constants']['minBoxFraction']['line']}")
    if "thumbs" in data["meta"]:
        t = data["meta"]["thumbs"]
        print(f"  thumbnails   {t['written']} written, {t['upToDate']} already current, "
              f"{t['unique']} unique cells at {t['px']}px")
    if missing:
        print("  MISSING IMAGES (first 5):")
        for c in missing[:5]:
            print("   ", c["row"], c["pair"], c["seed"], c["lam"], c["image"]["relPath"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Missing as e:
        print(f"loader stopped: {e}", file=sys.stderr)
        sys.exit(2)
