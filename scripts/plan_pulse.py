#!/usr/bin/env python3
"""Report-only check of the plan tree against real system state.

Reads. Never writes. Four checks, in the order they cost you time:

1. STALE      a task line says a run is in flight, and nothing is in flight.
2. UNHARVESTED  a run's output is newer than the plan file that owns it.
3. ORPHAN     a markdown file in a scope that no task names.
4. DEBRIS     narration a plan file is not supposed to carry.
5. RUNSTATE   a design plan carrying run state that belongs in its review file.
6. UNJUDGED   a review file whose run finished with questions still unanswered.
7. JARGON     a design plan with the mechanical tells of prose that will not read cold.
8. UNLISTED   a live plan in none of the root master plan's four lists, or in two.
9. SLOTS      a reserved figure slot whose review question got answered (fillable or broken).

    python3 scripts/plan_pulse.py --brief   the three orientation lines, derived

Checks 5, 6 and 7 apply only to a scope that has a review/ folder, so they switch
themselves on as scopes are converted and stay quiet on the ones that are not.

Usage:
    python3 scripts/plan_pulse.py                 # whole tree
    python3 scripts/plan_pulse.py does-the-correction-cause-composition  # one scope
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
            if name in ("MASTER_PLAN.md", "CLAUDE.md", "README.md", "CHANGELOG.md",
                        "parked.md", "INDEX.md"):
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


# Run state: belongs in a review file, never in a design plan.
RUN_STATE = re.compile(
    r"✓ verified \(|^## Runs\b|\bwandb-[a-z0-9]{6,}\b|\bRUNNING\b|\bjob \d{4,}\b",
    re.M,
)


def scope_of(path):
    """The scope directory for a design plan, or None if this is not one.

    A design plan lives at <scope>/plans/NN-name.md. Files under review/ and
    procedures/ are allowed to carry run state and are not design plans.
    """
    parent = os.path.dirname(path)
    if os.path.basename(parent) != "plans":
        return None
    return os.path.dirname(parent)


def opted_in(scope):
    """A scope has opted into the design/review split once review/ exists."""
    return bool(scope) and os.path.isdir(os.path.join(scope, "review"))


def check_runstate(files):
    hits = []
    for path in files:
        scope = scope_of(path)
        if not opted_in(scope):
            continue
        text = open(path, errors="replace").read()
        for n, line in enumerate(text.splitlines(), 1):
            if RUN_STATE.search(line):
                hits.append((path, n, line.strip()[:100]))
    return hits


def check_unjudged(files):
    """A review file recording a finished run while questions stay unanswered."""
    hits = []
    for path in files:
        if os.path.basename(os.path.dirname(path)) != "review":
            continue
        text = open(path, errors="replace").read()
        rows_done = [
            l for l in text.splitlines()
            if l.lstrip().startswith("|") and re.search(r"\bdone\b", l)
        ]
        unanswered = [
            (n, l.strip()[:100]) for n, l in enumerate(text.splitlines(), 1)
            if re.match(r"\s*- \[ \]", l)
        ]
        if rows_done and unanswered:
            hits.append((path, len(rows_done), unanswered))
    return hits


# Terms that mean nothing to a cold reader without a gloss. Mechanical on purpose:
# this check finds the tells, it does not judge whether prose reads well.
JARGON = [
    "λ", "delta_norm", "PMI", "norm-matched", "dose-response", "compose-rate",
    "AUC", "canary", "r_t", "eps_poe", "eps_j", "eps_a", "eps_b", "eps_uncond",
    "relative_norm", "oracle", "wrong_pair", "crossbar", "triptych", "SVD",
    "MDS", "elbow", "held-out", "Mono", "PoE", "chimera", "logSNR",
]
DOUBLE_MARKER = re.compile(r"- \[[ x]\] *(✅|❌|🟡|◑|⚠️|➖|⏸|⧗)")
LONG_SENTENCE_WORDS = 40


def glossary_terms(scope):
    """Terms defined in the root master plan plus this scope's own glossary."""
    terms = set()
    for path in (os.path.join(REPO, "MASTER_PLAN.md"),
                 os.path.join(scope or "", "MASTER_PLAN.md")):
        if not os.path.isfile(path):
            continue
        for m in re.finditer(r"^- \*\*(.+?)\*\*", open(path, errors="replace").read(), re.M):
            # Lowercased and colon-stripped: a glossary writes "Chimera:" where a
            # plan writes "chimera", and that is the same term.
            terms.add(m.group(1).rstrip(":").lower())
    return terms


def check_jargon(files):
    """Four mechanical tells that a design plan will not read cold."""
    hits = []
    for path in files:
        scope = scope_of(path)
        if not opted_in(scope):
            continue
        text = open(path, errors="replace").read()
        defined = " ".join(glossary_terms(scope)).lower()
        problems = []

        body = text.split("\n", 1)[1] if "\n" in text else ""
        opener = body[:600]
        if not re.search(r"^## What this asks", body, re.M):
            problems.append("no plain 'What this asks, in one line' opener")

        lowered = text.lower()
        ungloss = [t for t in JARGON if t.lower() in lowered and t.lower() not in defined]
        if ungloss:
            problems.append("unglossed: " + ", ".join(sorted(ungloss)[:6]))

        doubles = len(DOUBLE_MARKER.findall(text))
        if doubles:
            problems.append(f"{doubles} task line(s) with both a checkbox and an outcome marker")

        # Prose only. Code fences, headings, and blockquotes are excluded: the
        # image-generation prompt is a long instruction to an image model on
        # purpose, and counting it as an unreadable sentence is noise.
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        # Group into blocks first. A bullet does not end in a period, so
        # sentence-splitting raw text welds consecutive bullets into one
        # enormous pseudo-sentence: that was a checker bug, not bad prose.
        blocks, cur = [], []
        for l in prose.splitlines():
            s = l.strip()
            starts_item = bool(re.match(r"[-*+]\s|\d+\.\s", s))
            if not s or starts_item or s.startswith((">", "#", "|", "*(", "---")):
                if cur:
                    blocks.append(" ".join(cur)); cur = []
                if starts_item:
                    cur = [s]
                continue
            cur.append(s)
        if cur:
            blocks.append(" ".join(cur))
        long_ones = []
        for b in blocks:
            # A block whose length is a pasteable command chain is an anchor.
            if re.search(r"`/[a-z-]+", b):
                continue
            for s in re.split(r"(?<=[.!?])\s", b):
                if len(s.split()) > LONG_SENTENCE_WORDS:
                    long_ones.append(s.strip()[:60])
        if long_ones:
            problems.append(f"{len(long_ones)} sentence(s) over {LONG_SENTENCE_WORDS} words")

        if problems:
            hits.append((path, problems))
    return hits


ROOT_PLAN = os.path.join(REPO, "MASTER_PLAN.md")
# The four lists in the root master plan. Every live design plan belongs to
# exactly one. The first is an order; the rest are pools with no order.
LISTS = [
    "## The paper: what has to land",
    "## Reading, in the background",
    "## Experiments running in the background",
    "## Standing jobs",
]


def root_sections():
    """The root master plan split into its four lists."""
    if not os.path.isfile(ROOT_PLAN):
        return {}
    text = open(ROOT_PLAN, errors="replace").read()
    out, current = {}, None
    for line in text.splitlines():
        if line.strip() in LISTS:
            current = line.strip()
            out[current] = []
        elif current and line.startswith("## "):
            current = None
        elif current:
            out[current].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def paper_rows():
    """The paper table as (step, plan, status, waits_on) tuples, in order."""
    section = root_sections().get(LISTS[0], "")
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].isdigit():
            rows.append((int(cells[0]), cells[1], cells[3], cells[4]))
    return rows


def check_one_table(files):
    """A live design plan must appear in exactly one of the four lists.

    Live means it still has an open task. A plan with every task ticked is
    finished, and the paper table deliberately drops finished rows so it stays
    readable; flagging those would punish that choice.

    Two spellings count as a mention, because the tables and the standing list
    write a plan differently: `scope/NN-name` in a table cell, and the full
    link path `plans/scope/plans/NN-name.md` in the standing list.
    """
    sections = root_sections()
    if not sections:
        return []
    hits = []
    for path in files:
        scope = scope_of(path)
        if not scope:
            continue
        if "- [ ]" not in open(path, errors="replace").read():
            continue  # finished, not live
        name, stem = os.path.basename(scope), os.path.basename(path)[:-3]
        spellings = (f"{name}/{stem}", f"{name}/plans/{stem}")
        where = [
            title for title, body in sections.items()
            if any(s in body for s in spellings)
        ]
        if len(where) != 1:
            hits.append((f"{name}/{stem}", where))
    return hits


FIGURES_MD = os.path.join(REPO, "paper", "iclr", "figures.md")
REVIEW_PATH_IN_CELL = re.compile(r"plans/[\w/-]+/review/[\w-]+\.md")


def figure_slots():
    """Rows of the figure register as (fig_id, status, answered_by_cell, file_cell)."""
    if not os.path.isfile(FIGURES_MD):
        return []
    rows = []
    for line in open(FIGURES_MD, errors="replace"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and re.match(r"F\d+$", cells[0]):
            rows.append((cells[0], cells[1], cells[4], cells[5]))
    return rows


def check_slots():
    """Watch the reserved figure slots against their review files.

    A slot names the review file that must answer before its caption may claim
    anything. When that file's pre-registered bar section carries an answer,
    the slot is fillable (✅) or broken (❌). Broken means the result does not
    fit the slot's claim: do not build the figure, run the diagnosis procedure
    in context/research-guidelines.md.
    """
    hits = []
    for fig, status, answered_by, file_cell in figure_slots():
        m = REVIEW_PATH_IN_CELL.search(answered_by)
        if status in ("reserved", "fillable") and m:
            rpath = os.path.join(REPO, m.group(0))
            if not os.path.isfile(rpath):
                continue
            text = open(rpath, errors="replace").read()
            bar = text.split("## The pre-registered bar", 1)
            section = bar[1].split("\n## ", 1)[0] if len(bar) == 2 else ""
            if "❌" in section:
                hits.append((fig, f"BROKEN: bar answered against the claim, see {m.group(0)}"))
            elif "- [x] ✅" in section and status == "reserved":
                hits.append((fig, f"fillable: bar answered ✅ in {m.group(0)}, figure not built"))
        if status == "built":
            path_m = re.search(r"[\w./-]+\.(png|pdf)", file_cell)
            if path_m and not os.path.exists(os.path.join(REPO, "paper", "iclr", path_m.group(0))) \
               and not os.path.exists(os.path.join(REPO, path_m.group(0))):
                hits.append((fig, f"built but its file is missing: {path_m.group(0)}"))
    return hits


def brief():
    """The three orientation lines, derived rather than remembered."""
    live = [l for l in live_jobs() if l.strip()]
    if live:
        going_on = f"{len(live)} thing(s) in flight:\n" + "\n".join(
            f"      {l[:110]}" for l in live[:6]
        )
    else:
        going_on = "nothing is running. No queued jobs, no sweep processes on this node."

    try:
        last = subprocess.run(
            ["git", "-C", REPO, "log", "-1", "--format=%h %s"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception:
        last = "(git unavailable)"

    rows = paper_rows()
    ready = [r for r in rows if r[2] != "✅" and not r[3]]
    if ready:
        step, plan, status, _ = sorted(ready)[0]
        nxt = f"step {step}, {plan} (status {status})"
        blocked = len(rows) - len(ready)
        nxt += f". {len(ready)} row(s) unblocked, {blocked} waiting on something."
    elif rows:
        nxt = "every paper row is blocked. Read the Waits on column and find what died."
    else:
        nxt = "no paper table found in MASTER_PLAN.md."

    # The time split: writing needs no GPU and no queue, a run does. Unblocked
    # runs should be launched first, so they finish while the writing happens.
    writable = [f"step {s} {p}" for s, p, st, w in sorted(ready) if p.startswith("writing-the-paper")]
    launchable = [f"step {s} {p}" for s, p, st, w in sorted(ready) if not p.startswith("writing-the-paper")]

    print("Where things stand\n")
    print(f"  What is going on:      {going_on}")
    print(f"  The last thing we did: {last}")
    print(f"  Do this next:          {nxt}")
    if launchable:
        print(f"  Launch first:          {'; '.join(launchable[:4])} (runs; start them so they finish while you write)")
    if writable:
        print(f"  Write meanwhile:       {'; '.join(writable[:4])} (no GPU, no queue)")
    slots = check_slots()
    if slots:
        print(f"  Figure slots:          " + "; ".join(f"{f} {msg}" for f, msg in slots[:3]))


def rel(p):
    return os.path.relpath(p, REPO)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--brief" in sys.argv:
        brief()
        return
    checks = {1, 2, 3, 4, 5, 6, 7, 8, 9}
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

    if 5 in checks:
        hits = check_runstate(files)
        found += len(hits)
        print(f"RUNSTATE: run state in a design plan, belongs in its review file ({len(hits)})")
        for path, n, line in hits:
            print(f"  {rel(path)}:{n}  {line}")
        print()

    if 6 in checks:
        hits = check_unjudged(files)
        found += len(hits)
        print(f"UNJUDGED: a finished run with review questions still unanswered ({len(hits)})")
        for path, n_done, unanswered in hits:
            print(f"  {rel(path)}  {n_done} run(s) done, {len(unanswered)} question(s) open")
            for n, line in unanswered:
                print(f"      :{n}  {line}")
        print()

    if 7 in checks:
        hits = check_jargon(files)
        found += len(hits)
        print(f"JARGON: design plans that will not read cold ({len(hits)})")
        for path, problems in hits:
            print(f"  {rel(path)}")
            for p in problems:
                print(f"      {p}")
        print()

    if 8 in checks:
        hits = check_one_table(files)
        found += len(hits)
        print(f"UNLISTED: a live plan in none of the four root lists, or in more than one ({len(hits)})")
        for key, where in hits:
            place = "no list" if not where else f"{len(where)} lists"
            print(f"  {key}  ({place})")
        print()

    if 9 in checks:
        hits = check_slots()
        found += len(hits)
        print(f"SLOTS: figure slots whose state changed under them ({len(hits)})")
        for fig, msg in hits:
            print(f"  {fig}  {msg}")
        print()

    if found == 0:
        print("nothing to report.")
    else:
        thing = "thing" if found == 1 else "things"
        print(f"{found} {thing} to look at. Nothing was changed.")


if __name__ == "__main__":
    main()
