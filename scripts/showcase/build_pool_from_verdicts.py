"""Turn the verdict file into the three configs the pooled trainer reads, plus an explicit cell list.

The trainer takes a pair pool, a prompt registry and a seed pool, and trains on every pair crossed
with every training seed. This pool is not a cross product: each pair keeps only the seeds a person
judged good, so the cell list is written separately and the trainer is pointed at it. The seed pool
is still written because the trainer requires one, but with an explicit cell list it decides nothing.

Prompts come from each cell's own meta.json rather than being re-derived from the slug, so the
prompts trained on are byte-identical to the ones that produced the cached targets.
"""
import json, os, sys
import yaml

CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
G = "artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from"
OUT = "artifacts/_shared/cross_pair_pool_configs"
HELDOUT = ["a_cat__x__a_dog", "an_elephant__x__a_penguin"]


def cell_dir(pair, seed):
    for sp in ("train", "heldout"):
        d = os.path.join(CACHE, sp, pair, "seed_%d" % seed)
        if os.path.exists(os.path.join(d, "mono.png")):
            return d
    return None


def main(tag, include_pending):
    v = json.load(open(os.path.join(G, "verdicts.json")))
    kept = dict(v["kept"])
    if include_pending:
        kept.update(v["kept_pending_confirmation"])

    cells, prompts, missing = {}, {}, []
    for pair in sorted(kept):
        seeds = sorted(kept[pair])
        good = []
        for s in seeds:
            d = cell_dir(pair, s)
            if d is None:
                missing.append(f"{pair} seed {s}")
                continue
            good.append(s)
            if pair not in prompts:
                m = json.load(open(os.path.join(d, "meta.json")))
                prompts[pair] = {"prompt_a": m["pair"][0], "prompt_b": m["pair"][1],
                                 "joint_prompt": m["joint_prompt"]}
        if good:
            cells[pair] = good

    if missing:
        print("REFUSING: kept cells with nothing on disk:", missing)
        return 1

    os.makedirs(OUT, exist_ok=True)
    pool_p = os.path.join(OUT, f"pair_pool_{tag}.yaml")
    prom_p = os.path.join(OUT, f"pair_prompts_{tag}.yaml")
    seed_p = os.path.join(OUT, f"seed_pool_{tag}.yaml")
    cells_p = os.path.join(OUT, f"cells_{tag}.json")

    yaml.safe_dump({"train": sorted(cells), "heldout": HELDOUT}, open(pool_p, "w"), sort_keys=False)
    yaml.safe_dump({p: prompts[p] for p in sorted(cells)} |
                   {h: {"prompt_a": h.split("__x__")[0].replace("_", " "),
                        "prompt_b": h.split("__x__")[1].replace("_", " "),
                        "joint_prompt": h.replace("__x__", " and ").replace("_", " ")}
                    for h in HELDOUT},
                   open(prom_p, "w"), sort_keys=False)
    all_seeds = sorted({s for ss in cells.values() for s in ss})
    yaml.safe_dump({"train_pool": all_seeds, "held_out": list(range(9, 17))},
                   open(seed_p, "w"), sort_keys=False)
    json.dump(cells, open(cells_p, "w"), indent=1)

    n = sum(len(s) for s in cells.values())
    print(f"{n} cells across {len(cells)} pairs")
    print(f"  pair pool   {pool_p}")
    print(f"  prompts     {prom_p}")
    print(f"  seed pool   {seed_p}   (superseded by the cell list; written because the trainer needs one)")
    print(f"  cell list   {cells_p}")
    print()
    for p in sorted(cells, key=lambda k: -len(cells[k])):
        print(f"  {len(cells[p]):2d}  {p:34s} {cells[p]}")
    return 0


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    sys.exit(main(tag, "--with-pending" in sys.argv))
