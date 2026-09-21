"""Every fix side by side, for one adapter, at one checkpoint.

Columns are the fixes, rows are seeds, and the two leftmost columns pin the references: the
joint-prompt target this seed is nominally scored against, and what the product of the two
single-animal predictions draws with no adapter at all. Everything else in a row is the same
adapter, the same noise and the same sampler, so the only thing that changes across a row is the
fix. No tile uses the joint prompt.

Seeds whose own target does not contain both animals are labelled as broken on their row. On those
rows the target column is a reference in name only, and the tiles have to be judged against the
words instead.

Directories are read as ``<adapter>-<checkpoint>-<fix>``, which is the convention these runs write.

  python3 scripts/mechanism_study/fix_grid.py --adapter orth3-20k --seeds 9 10 \
      --fix "no fix=no-fix" --fix "cat boost 50=cat-boost-50" \
      --out artifacts/results/<question> --prefix every-fix-orth3-20k
"""
import argparse, json, os, sys

from PIL import Image, ImageDraw, ImageFont

ATTN = "/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism"
CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
QUALITY = "/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/target_quality.json"

CELL, PAD, LEFT, TOP = 250, 10, 230, 160
INK, MUTED, BG, RED, BLUE, GREEN = ((17, 17, 17), (110, 110, 110), (255, 255, 255),
                                    (200, 42, 42), (23, 92, 176), (36, 138, 61))


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/truetype/liberation/LiberationSans%s.ttf" % ("-Bold" if bold else "")):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def cell_dir(pair, seed):
    for split in ("train", "heldout"):
        d = os.path.join(CACHE, split, pair, "seed_%d" % seed)
        if os.path.isdir(d):
            return d
    return None


def targets_ok(pair):
    """{seed: does its joint-prompt render actually show both animals}."""
    out = {}
    if not os.path.exists(QUALITY):
        return out
    for r in json.load(open(QUALITY)):
        if r.get("pair") != pair:
            continue
        parts = str(r.get("seed", "")).split("_")
        if len(parts) > 1 and parts[1].isdigit():
            out[int(parts[1])] = bool(r.get("both_present"))
    return out


def tile(path):
    im = Image.new("RGB", (CELL, CELL), (240, 240, 240))
    if path and os.path.exists(path):
        im.paste(Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS), (0, 0))
    else:
        ImageDraw.Draw(im).text((8, 8), "not rendered", fill=RED, font=font(14, True))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="e.g. orth3-20k; prefixes every directory")
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--fix", action="append", required=True, metavar="LABEL=SUFFIX",
                    help="repeatable, in column order; reads <adapter>-<suffix>")
    ap.add_argument("--note", default="", help="one extra line under the title")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    fixes = []
    for spec in args.fix:
        if "=" not in spec:
            sys.exit("--fix needs LABEL=SUFFIX, got %r" % spec)
        fixes.append(tuple(spec.split("=", 1)))

    ok = targets_ok(args.pair)
    ncol = len(fixes) + 2
    W = LEFT + ncol * (CELL + PAD) + PAD
    H = TOP + len(args.seeds) * (CELL + 46) + PAD
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((PAD, 16), "%s: every fix, %s" % (args.adapter,
            "seeds " + " and ".join(str(s) for s in args.seeds)), fill=INK, font=font(28, True))
    dr.text((PAD, 56), "Same adapter, same noise, same sampler in every tile. Only the fix differs, "
                       "and no tile uses the joint prompt.", fill=MUTED, font=font(15))
    if args.note:
        dr.text((PAD, 78), args.note, fill=MUTED, font=font(15))

    for r, seed in enumerate(args.seeds):
        y = TOP + r * (CELL + 46)
        cd = cell_dir(args.pair, seed)
        good = ok.get(seed)
        dr.text((PAD, y + CELL // 2 - 24), "seed %d" % seed, fill=INK, font=font(22, True))
        dr.text((PAD, y + CELL // 2 + 4),
                "target sound" if good else "target BROKEN" if good is False else "target unscored",
                fill=GREEN if good else RED if good is False else MUTED, font=font(13, True))
        cols = [("joint target", cd and os.path.join(cd, "mono.png")),
                ("product alone", cd and os.path.join(cd, "poe.png"))]
        cols += [(lbl, os.path.join(ATTN, "%s-%s" % (args.adapter, suf), args.pair,
                                    "seed_%d" % seed, "sample_seed_%d.png" % seed))
                 for lbl, suf in fixes]
        for c, (lbl, p) in enumerate(cols):
            x = LEFT + c * (CELL + PAD)
            im.paste(tile(p), (x, y))
            if r == 0:
                dr.text((x, y - 26), lbl,
                        fill=RED if (c == 0 and good is False) else BLUE, font=font(16, True))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "%s.png" % args.prefix)
    im.save(path)
    print("%d seeds x %d columns -> %s" % (len(args.seeds), ncol, path))


if __name__ == "__main__":
    main()
