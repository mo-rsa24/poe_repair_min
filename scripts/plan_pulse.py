#!/usr/bin/env python3
"""Report-only check of the plan tree against real system state.

Reads. Never writes. Four checks, in the order they cost you time:

1. STALE      a task line says a run is in flight, and nothing is in flight.
2. UNHARVESTED  a run's output is newer than the plan file that owns it.
3. ORPHAN     a markdown file in a scope that no task names.
4. DEBRIS     narration a plan file is not supposed to carry.

Usage:
    python3 scripts/plan_pulse.py                 # whole tree
    python3 scripts/plan_pulse.py interaction-term  # one scope
    python3 scripts/plan_pulse.py --checks 1,2    # only the expensive ones

Exit code is 0 always. This tells you things; it does not gate anything.
"""

import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANS = os.path.join(REPO, "plans")

# A task line claiming work is in flight. Case-sensitive on RUNNING on purpose:
# "worth running" and "Re-running" are prose, not a claim about the queue.
IN_FLIGHT = re.compile(
    r"\bRUNNING\b|\brunning as job\b|\bqueued as\b|\bjob \d{4,}\b|\bin flight\b|⧗"
)
# Narration a plan file should not carry (history belongs in CHANGELOG.md).
DEBRIS = re.compile(
    r"Corrected \d{4}|Superseded|Updated 20\d\d-|this used to say|"
    r"Deviation from the original|retained as the record|Added 20\d\d-\d\d-\d\d while",
    re.I,
)
# A path inside a ## Runs table cell or a task line.
# At least three segments, so a bare /datasets/mmolefe does not match everything.
PATHLIKE = re.compile(
    r"[`\s(](/datasets(?:/[\w.-]+){3,}|outputs(?:/[\w.-]+){2,}|results(?:/[\w.-]+){2,})"
)

SKIP_DIRS = {".git", "__pycache__", "shelved", "phases"}


def walk_md(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if fn.endswith(".md"):
                yield os.path.join(dirpath, fn)


def live_jobs():
    """Everything Slurm knows about, plus bare processes on this node."""
    live = []
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j %T"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        live += [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["pgrep", "-af", r"sweep\|train\|sbatch\|\.sbatch"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        for line in out.splitlines():
            if "plan_pulse" in line or "pgrep" in line:
                continue
            live.append(line.strip())
    except Exception:
        pass
    return live


_MTIME_CACHE = {}
MAX_FILES_SCANNED = 4000  # output trees hold hundreds of thousands of images


def newest_mtime(path):
    """Newest mtime under path, or None if it does not exist.

    Capped: a cache directory can hold 100k files and the answer does not get
    truer after the first few thousand.
    """
    if path in _MTIME_CACHE:
        return _MTIME_CACHE[path]
    result = None
    if os.path.exists(path):
        if os.path.isfile(path):
            result = os.path.getmtime(path)
        else:
            newest, seen = 0.0, 0
            for dirpath, dirnames, filenames in os.walk(path):
                for fn in filenames:
                    try:
                        newest = max(newest, os.path.getmtime(os.path.join(dirpath, fn)))
                    except OSError:
                        continue
                    seen += 1
                if seen > MAX_FILES_SCANNED:
                    break
            result = newest or None
    _MTIME_CACHE[path] = result
    return result


def check_stale(files, live):
    hits = []
    blob = " ".join(live)
    for path in files:
        for n, line in enumerate(open(path, errors="replace"), 1):
            if not IN_FLIGHT.search(line):
                continue
            if "[x]" in line:  # already closed out
                continue
            # If any live job or process shares a word with the line, believe it.
            words = {w for w in re.findall(r"[\w.]{6,}", line) if not w.isdigit()}
            if any(w in blob for w in words):
                continue
            hits.append((path, n, line.strip()[:110]))
    return hits


def last_commit_time(path):
    """When this plan file was last committed. Falls back to mtime if untracked.

    Commit time, not mtime, because editing a plan for any reason resets mtime
    and would hide every unharvested result underneath it.
    """
    try:
        out = subprocess.run(
            ["git", "-C", REPO, "log", "-1", "--format=%ct", "--", path],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        if out:
            return float(out)
    except Exception:
        pass
    return os.path.getmtime(path)


def check_unharvested(files):
    hits = []
    for path in files:
        text = open(path, errors="replace").read()
        plan_mtime = last_commit_time(path)
        seen = set()
        for m in PATHLIKE.finditer(text):
            p = m.group(1).rstrip("`.,)")
            if p in seen:
                continue
            seen.add(p)
            full = p if p.startswith("/") else os.path.join(REPO, p)
            out_mtime = newest_mtime(full)
            if out_mtime and out_mtime > plan_mtime + 60:
                age = (out_mtime - plan_mtime) / 3600
                hits.append((path, p, age))
    return hits


def check_orphans(files):
    by_scope = {}
    for path in files:
        by_scope.setdefault(os.path.dirname(path), []).append(path)
    corpus = {p: open(p, errors="replace").read() for p in files}
    hits = []
    for scope, members in by_scope.items():
        for path in members:
            name = os.path.basename(path)
            if name in ("MASTER_PLAN.md", "CLAUDE.md", "README.md", "CHANGELOG.md"):
                continue
            stem = name[:-3]
            named = any(
                (name in text or stem in text)
                for other, text in corpus.items()
                if other != path
            )
            if not named:
                hits.append(path)
    return hits


def check_debris(files):
    hits = []
    for path in files:
        for n, line in enumerate(open(path, errors="replace"), 1):
            if DEBRIS.search(line):
                hits.append((path, n, line.strip()[:110]))
    return hits


def rel(p):
    return os.path.relpath(p, REPO)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    checks = {1, 2, 3, 4}
    for a in sys.argv[1:]:
        if a.startswith("--checks"):
            checks = {int(x) for x in a.split("=")[-1].replace(",", " ").split()}

    root = os.path.join(PLANS, args[0]) if args else PLANS
    if not os.path.isdir(root):
        print(f"no such scope: {root}")
        return
    files = list(walk_md(root))
    print(f"plan pulse: {len(files)} plan files under {rel(root)}\n")

    found = 0
    if 1 in checks:
        hits = check_stale(files, live_jobs())
        found += len(hits)
        print(f"STALE: task lines claiming a run is in flight with nothing in flight ({len(hits)})")
        for path, n, line in hits:
            print(f"  {rel(path)}:{n}  {line}")
        print()

    if 2 in checks:
        hits = check_unharvested(files)
        found += len(hits)
        print(f"UNHARVESTED: output newer than the plan that owns it ({len(hits)})")
        for path, p, age in sorted(hits, key=lambda h: -h[2]):
            print(f"  {rel(path)}  <- {p}  ({age:.0f}h newer)")
        print()

    if 3 in checks:
        hits = check_orphans(files)
        found += len(hits)
        print(f"ORPHAN: markdown no task names ({len(hits)})")
        for path in hits:
            print(f"  {rel(path)}")
        print()

    if 4 in checks:
        hits = check_debris(files)
        found += len(hits)
        print(f"DEBRIS: narration a plan should not carry ({len(hits)})")
        for path, n, line in hits:
            print(f"  {rel(path)}:{n}  {line}")
        print()

    if found == 0:
        print("nothing to report.")
    else:
        thing = "thing" if found == 1 else "things"
        print(f"{found} {thing} to look at. Nothing was changed.")


if __name__ == "__main__":
    main()
