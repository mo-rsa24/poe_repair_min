"""How much do the two experts' cross-attention maps land on the same pixels?

Attend-and-Excite raises a neglected subject's attention peak, which is the right move when both
subjects share one prompt and one competes the other out. Under product-of-experts they do not
share a prompt: the cat lives in branch A and the dog in branch B, each the only subject its branch
has, so each already attends to its own animal and there is no neglected token to excite. Measured
on cat x dog seed 9, the Attend-and-Excite loss starts at 0.33, meaning peak attention is already
0.67 before any intervention.

The fused animal is not a neglected subject. It is both experts painting the same pixels. So the
quantity worth measuring, and the one worth steering on, is how much the two maps overlap.

Two numbers per step, both on maps normalised to sum 1 so they are spatial distributions:

  overlap   sum of elementwise min   1.0 if the two experts want identical pixels, 0.0 if disjoint
  cosine    cos(cat_map, dog_map)    the same idea without the mass normalisation

This reads the maps ``capture_attention.py`` already wrote and touches no GPU.

  python3 scripts/mechanism_study/attention_overlap.py --regimes plain_poe lora_lambda1 --out <json>
"""
import argparse, json, os, re, sys

import torch

ROOT = "/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism"
STEP_RE = re.compile(r"step_(\d+)_token_(cat|dog)_branch_poe\.pt$")


def maps_for(regime, pair, seed):
    """{step: {'cat': map, 'dog': map}} for one captured cell."""
    d = os.path.join(ROOT, regime, pair, "seed_%d" % seed, "attn_maps")
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in os.listdir(d):
        m = STEP_RE.match(f)
        if m:
            t = torch.load(os.path.join(d, f), map_location="cpu")
            if isinstance(t, dict):                     # tolerate a wrapped save
                t = t.get("map", t.get("attn", next(iter(t.values()))))
            out.setdefault(int(m.group(1)), {})[m.group(2)] = t.float().reshape(-1)
    return {k: v for k, v in out.items() if "cat" in v and "dog" in v}


def pair_scores(cat, dog):
    """Overlap of the two maps as spatial distributions, and their cosine."""
    c = cat.clamp_min(0)
    g = dog.clamp_min(0)
    cs = float((c * g).sum() / (c.norm() * g.norm() + 1e-12))
    c = c / (c.sum() + 1e-12)
    g = g / (g.sum() + 1e-12)
    return float(torch.minimum(c, g).sum()), cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=["plain_poe", "lora_lambda1"])
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(1, 13)))
    ap.add_argument("--early", nargs=2, type=int, default=[0, 10],
                    metavar=("LO", "HI"), help="the commit window to average over")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lo, hi = args.early
    res = {}
    for reg in args.regimes:
        res[reg] = {}
        for s in args.seeds:
            mm = maps_for(reg, args.pair, s)
            if not mm:
                continue
            per = {st: pair_scores(v["cat"], v["dog"]) for st, v in sorted(mm.items())}
            early = [ov for st, (ov, _) in per.items() if lo <= st < hi]
            allst = [ov for _, (ov, _) in per.items()]
            ecos = [cs for st, (_, cs) in per.items() if lo <= st < hi]
            res[reg][s] = {
                "overlap_early": sum(early) / len(early) if early else None,
                "overlap_all": sum(allst) / len(allst) if allst else None,
                "cosine_early": sum(ecos) / len(ecos) if ecos else None,
                "per_step_overlap": {str(st): round(ov, 4) for st, (ov, _) in per.items()},
            }

    print("overlap of the cat and dog attention maps, steps %d to %d" % (lo, hi))
    print("1.0 = the two experts want identical pixels, 0.0 = disjoint\n")
    print("%-6s %s" % ("seed", "  ".join("%-22s" % r for r in args.regimes)))
    common = [s for s in args.seeds if all(s in res[r] for r in args.regimes)]
    for s in common:
        cells = []
        for r in args.regimes:
            v = res[r][s]
            cells.append("%-22s" % ("overlap %.3f cos %.3f" % (v["overlap_early"], v["cosine_early"])))
        print("%-6d %s" % (s, "  ".join(cells)))
    print()
    for r in args.regimes:
        vals = [res[r][s]["overlap_early"] for s in common]
        print("%-14s mean early overlap %.4f over %d seeds" % (r, sum(vals) / len(vals), len(vals)))

    if args.out:
        json.dump({"window": [lo, hi], "regimes": res}, open(args.out, "w"), indent=2, sort_keys=True)
        print("\n-> %s" % args.out)


if __name__ == "__main__":
    main()
