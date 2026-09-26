#!/usr/bin/env python3
"""Launch queued runs on biggpu cards as they free up, and keep a status table and an event log.

Runs on the laptop and reaches the cluster only through scripts/cluster.sh, so it stops when the
laptop sleeps. Every cycle it:

  1. reads every card on the biggpu nodes, Blackwell first, and who is on each one;
  2. checks every running run: process alive, `ok` line seen, finished, or crashed;
  3. launches at most one queued run, on the first card that passes the shared-device rules;
  4. rewrites run_queue_status.md and appends launches, crashes and finishes to run_queue_events.log.

A card is usable when it is not faulted, has the run's mem_gb plus 2 GB free, carries no foreign
process that has been active in the last three checks, and has had none of our runs launched on it
in the last 15 minutes (a new run's memory climbs for minutes after start).

    nohup python3 scripts/local/run_queue.py > scripts/local/run_queue.out 2>&1 &
    python3 scripts/local/run_queue.py --once        # one cycle, then exit
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CLUSTER = REPO / "scripts" / "cluster.sh"
QUEUE = HERE / "run_queue.yaml"
STATE = HERE / "run_queue_state.json"
STATUS = HERE / "run_queue_status.md"
EVENTS = HERE / "run_queue_events.log"

RDIR = "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min"
NODES = ["mscluster112", "mscluster110", "mscluster109", "mscluster106", "mscluster107"]
BLACKWELL = {"mscluster110", "mscluster111", "mscluster112"}
PY = {True: "/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python",
      False: "/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python"}
ME = "mmolefe"
LOG_DIR = "/datasets/mmolefe/poe_repair_min/outputs/reward_finetune/logs"
ERRORS = re.compile(r"Traceback|OutOfMemoryError|out of memory|ERROR|Killed")
STEP = re.compile(r"step (\d+)/(\d+)")
IDLE_CHECKS, IDLE_UTIL, SETTLE_S, MARGIN_MB = 3, 5, 15 * 60, 2048


def remote(cmd: str, timeout: int = 150) -> str:
    try:
        r = subprocess.run([str(CLUSTER), "sh", cmd], capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except subprocess.TimeoutExpired:
        return ""


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def event(kind: str, name: str, detail: str) -> None:
    line = f"{now()} {kind} {name}: {detail}"
    with EVENTS.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def probe() -> tuple[dict, dict]:
    """Every card as {(node, idx): {...}} and every process on them as {(node, idx): [(pid, user, mb)]}."""
    cmd = "; ".join(
        f"timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 {n} bash {RDIR}/scripts/gpu_probe.sh 2>/dev/null"
        for n in NODES)
    cards, apps = {}, {}
    for line in remote(cmd).splitlines():
        p = line.split()
        if p[:1] == ["GPU"] and len(p) >= 6:
            faulted = not p[3].isdigit() or not p[5].isdigit()
            cards[(p[1], int(p[2]))] = {"used": int(p[3]) if p[3].isdigit() else 0,
                                        "total": int(p[4]) if p[4].isdigit() else 0,
                                        "util": int(p[5]) if p[5].isdigit() else -1,
                                        "faulted": faulted}
        elif p[:1] == ["APP"] and len(p) >= 6 and p[2].isdigit():
            apps.setdefault((p[1], int(p[2])), []).append((p[3], p[4], p[5]))
    return cards, apps


def check_run(name: str, run: dict, spec: dict) -> None:
    # The bracket keeps pgrep from matching the shell that runs it, whose own command line
    # carries the same text and would read every run as alive.
    pat = f"[{spec['match'][0]}]{spec['match'][1:]}"
    out = remote(f"ssh -o BatchMode=yes -o ConnectTimeout=8 {run['node']} "
                 f"\"pgrep -f -- '{pat}' >/dev/null && echo ALIVE || echo GONE\"; "
                 f"tail -c 20000 {run['log']} 2>/dev/null")
    alive = "ALIVE" in out.splitlines()[:1]
    steps = STEP.findall(out)
    if steps:
        run["last_step"] = f"{steps[-1][0]}/{steps[-1][1]}"
    err = [l for l in out.splitlines() if ERRORS.search(l)]
    if spec["done"] in out:
        run["state"] = "done"
        event("DONE", name, f"finished on {run['node']} GPU {run['gpu']}; bar: {spec['bar']}")
    elif not alive:
        run["state"] = "crashed"
        event("CRASH", name, (err[-1] if err else "process gone, no error in the log")[:240])
    elif spec["ok"] in out and not run.get("ok"):
        run["ok"] = True
        run["state"] = "running"
        event("HEALTHY", name, f"reached '{spec['ok']}' on {run['node']} GPU {run['gpu']}")


def usable(key, card, apps, util_hist, state, need_mb, specs) -> bool:
    # Our own runs are counted at their peak, never at the reading: a training run's memory swings
    # by tens of GB between steps, and a launch sized to a low reading once put a third run on a
    # card two runs already fill at their peaks.
    own_now = sum(int(mb) for _, u, mb in apps.get(key, []) if u == ME and mb.isdigit())
    own_peak = sum(specs[n]["mem_gb"] * 1024 for n, r in state["runs"].items()
                   if n in specs and (r.get("node"), r.get("gpu")) == key
                   and r["state"] in ("starting", "running"))
    used = card["used"] - own_now + max(own_now, own_peak)
    if card["faulted"] or card["total"] - used < need_mb + MARGIN_MB:
        return False
    # A foreign job seen in any of the last three checks counts, not only one on the card now: a
    # worker loop that runs in bursts read as a free card between bursts and got K2 put beside it.
    hist = util_hist.get(f"{key[0]}:{key[1]}", [])
    seen = state.get("foreign", {}).get(f"{key[0]}:{key[1]}", [])
    if any(seen[-IDLE_CHECKS:]) and (len(hist) < IDLE_CHECKS or max(hist[-IDLE_CHECKS:]) > IDLE_UTIL):
        return False
    for r in state["runs"].values():
        if (r.get("node"), r.get("gpu")) == key and time.time() - r.get("launched_ts", 0) < SETTLE_S:
            return False
    return True


def launch(name: str, spec: dict, key) -> dict:
    node, gpu = key
    log = f"{LOG_DIR}/{name}-{node}.log"
    py = PY[node in BLACKWELL]
    cmd = (f"ssh -o BatchMode=yes {node} \"ALLOW_SHARE=1 PY={py} GPU={gpu} nohup bash "
           f"{RDIR}/{spec['launcher']} {spec['args']} > {log} 2>&1 & echo PID=\\$!\"")
    out = remote(cmd)
    m = re.search(r"PID=(\d+)", out)
    event("LAUNCH", name, f"{node} GPU {gpu}, log {log}, pid {m.group(1) if m else '?'}")
    return {"state": "starting", "node": node, "gpu": gpu, "log": log,
            "pid": m.group(1) if m else None, "launched_ts": time.time(), "launched": now()}


def write_status(state: dict, specs: dict, cards: dict, apps: dict) -> None:
    rows = ["| Run | State | Where | Last step | Launched | Log |", "|---|---|---|---|---|---|"]
    for name in specs:
        r = state["runs"].get(name, {"state": "queued"})
        where = f"{r['node']} GPU {r['gpu']}" if r.get("node") else ""
        rows.append(f"| {name} | {r['state']} | {where} | {r.get('last_step', '')} | "
                    f"{r.get('launched', '')} | `{r.get('log', '')}` |")
    cardrows = ["| Card | Used / total GB | Utilisation | On it |", "|---|---|---|---|"]
    for (n, i), c in sorted(cards.items(), key=lambda kv: NODES.index(kv[0][0])):
        who = ", ".join(sorted({u for _, u, _ in apps.get((n, i), [])})) or "nobody"
        util = "faulted" if c["faulted"] else f"{c['util']}%"
        cardrows.append(f"| {n} GPU {i} | {c['used'] / 1024:.1f} / {c['total'] / 1024:.1f} | {util} | {who} |")
    STATUS.write_text(f"# Run queue\n\nChecked {now()}. Written by `run_queue.py` every cycle; the "
                      f"queue itself is `run_queue.yaml`, and launches, crashes and finishes are in "
                      f"`run_queue_events.log`.\n\n## Runs\n\n" + "\n".join(rows) +
                      "\n\n## Cards\n\n" + "\n".join(cardrows) + "\n")


def cycle() -> None:
    specs = {r["name"]: r for r in yaml.safe_load(QUEUE.read_text())["runs"]}
    for name, s in specs.items():
        if not s.get("bar"):
            sys.exit(f"{name} has no pass bar in {QUEUE.name}; a run without one is not queued")
    state = json.loads(STATE.read_text()) if STATE.exists() else {"runs": {}, "util": {}}
    for name, s in specs.items():
        if name not in state["runs"] and s.get("adopt"):
            a = s["adopt"]
            state["runs"][name] = {"state": "starting", "node": a["node"], "gpu": a["gpu"],
                                   "log": a["log"], "launched": "by hand", "launched_ts": 0}
            event("ADOPT", name, f"tracking the run already on {a['node']} GPU {a['gpu']}")

    cards, apps = probe()
    for key, c in cards.items():
        h = state["util"].setdefault(f"{key[0]}:{key[1]}", [])
        h.append(c["util"] if c["util"] >= 0 else 100)
        del h[:-IDLE_CHECKS]
        fs = state.setdefault("foreign", {}).setdefault(f"{key[0]}:{key[1]}", [])
        fs.append(any(u != ME for _, u, _ in apps.get(key, [])))
        del fs[:-IDLE_CHECKS]

    for name, r in state["runs"].items():
        if r["state"] in ("starting", "running") and name in specs:
            check_run(name, r, specs[name])

    for name, s in specs.items():
        if name in state["runs"]:
            continue
        for key in sorted(cards, key=lambda k: (NODES.index(k[0]), k[1])):
            if usable(key, cards[key], apps, state["util"], state, s["mem_gb"] * 1024, specs):
                state["runs"][name] = launch(name, s, key)
                break
        break                      # one launch per cycle: the next run waits for this one's memory

    STATE.write_text(json.dumps(state, indent=1))
    write_status(state, specs, cards, apps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=240, help="seconds between cycles")
    a = ap.parse_args()
    while True:
        try:
            cycle()
        except Exception as exc:          # a dropped SSH must not end the poller
            event("POLLER-ERROR", "-", f"{type(exc).__name__}: {exc}"[:240])
        if a.once:
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
