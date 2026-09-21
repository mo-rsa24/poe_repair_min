"""Print the current state of the training pool, read off the filesystem every time.

Nothing here is remembered between runs. Seeds come from the cache directory, verdicts come from
verdicts.json beside the sheets, so the table cannot drift from what is actually on disk.
"""
import json, os, sys

CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
G = "artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from"
V = os.path.join(G, "verdicts.json")
EVAL_SEEDS = set(range(9, 17))


def seeds_on_disk(pair):
    out = set()
    for split in ("train", "heldout"):
        d = os.path.join(CACHE, split, pair)
        if os.path.isdir(d):
            for s in os.listdir(d):
                if s.startswith("seed_") and s[5:].isdigit():
                    out.add(int(s[5:]))
    return sorted(out)


def fmt(seeds):
    return ",".join(str(s) for s in seeds) if seeds else "-"


def main():
    v = json.load(open(V))
    kept = dict(v["kept"])
    pending = dict(v["kept_pending_confirmation"])
    rejected = set(v["rejected"])
    barred = set(sum(v["never_eligible"].values(), []))

    pairs = set()
    for split in ("train", "heldout"):
        d = os.path.join(CACHE, split)
        if os.path.isdir(d):
            pairs |= set(os.listdir(d))
    pairs -= barred

    rows = []
    for p in sorted(pairs):
        disk = seeds_on_disk(p)
        usable = [s for s in disk if s not in EVAL_SEEDS]
        if p in kept:
            state, keep = "in", kept[p]
        elif p in pending:
            state, keep = "in?", pending[p]
        elif p in rejected:
            state, keep = "out", []
        elif not usable:
            continue
        else:
            state, keep = "unjudged", []
        rows.append((state, p, usable, keep))

    order = {"in": 0, "in?": 1, "unjudged": 2, "out": 3}
    rows.sort(key=lambda r: (order[r[0]], -len(r[3]), r[1]))

    w = max(len(r[1]) for r in rows)
    print(f"{'':4s} {'pair':{w}s} {'rendered (seeds 1-8, 17+)':>26s} {'kept':>14s}")
    print("-" * (w + 48))
    last = None
    for state, p, usable, keep in rows:
        if state != last:
            print()
            last = state
        print(f"{state:4s} {p:{w}s} {fmt(usable):>26s} {fmt(keep):>14s}")

    n_in = sum(len(s) for s in kept.values())
    n_pend = sum(len(s) for s in pending.values())
    n_unj = sum(len(r[2]) for r in rows if r[0] == "unjudged")
    print()
    print(f"  pool, confirmed          {n_in:4d} cells across {len(kept)} pairs")
    print(f"  pool, awaiting a word    {n_pend:4d} cells across {len(pending)} pairs")
    print(f"  rendered, never judged   {n_unj:4d} cells across "
          f"{sum(1 for r in rows if r[0]=='unjudged')} pairs")
    print(f"  rejected                 {sum(1 for r in rows if r[0]=='out'):4d} pairs")
    print(f"\n  {v['seed_policy']}")


if __name__ == "__main__":
    main()
