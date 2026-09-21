"""One seed, checkpoints across, one correction method per row.

The adapter's render is a step function of its weights: at fixed starting noise it jumps between a
few discrete outcomes as training proceeds rather than converging on one. This lays that out for a
single cell so the question "does this correction stop the jumping" is answered by looking along a
row, and "what did training do on its own" by looking along the top row.

Every row is the same sampler and the same noise. Only the correction differs. No row uses the
joint prompt: the branches are always the two single-subject prompts and the unconditional.

Rows are given as ``label=<regime directory template>`` where the template carries ``{ck}`` for the
zero-padded checkpoint step, e.g. ``s10_excite_cat_{ck}``. That keeps the figure agnostic about
which experiment wrote which directory.

  python3 scripts/mechanism_study/intervention_grid.py --seed 10 \
      --checkpoints 010000 020000 030000 \
      --row "adapter alone=ckpt{ck}_none" --row "excite the cat=s10_excite_cat_{ck}" \
      --out artifacts/results/<question> --prefix seed10-corrections
"""
import argparse, json, os, sys

from PIL import Image, ImageDraw, ImageFont

ATTN = "/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism"
CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
QUALITY = "/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/target_quality.json"

CELL, PAD, LEFT, TOP = 260, 12, 360, 260
INK, MUTED, BG = (17, 17, 17), (110, 110, 110), (255, 255, 255)
RED, BLUE = (200, 42, 42), (23, 92, 176)


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


def target_ok(pair, seed):
    """Does this seed's joint-prompt target actually show both subjects?"""
    if not os.path.exists(QUALITY):
        return None
    for r in json.load(open(QUALITY)):
        if r.get("pair") != pair:
            continue
        parts = str(r.get("seed", "")).split("_")
        if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) == seed:
            return bool(r.get("both_present"))
    return None


def tile(path, size):
    im = Image.new("RGB", (size, size), (238, 238, 238))
    if path and os.path.exists(path):
        im.paste(Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS), (0, 0))
    else:
        ImageDraw.Draw(im).text((8, 8), "missing", fill=RED, font=font(15, True))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="zero-padded step strings")
    ap.add_argument("--row", action="append", required=True, metavar="LABEL=TEMPLATE",
                    help="regime directory template with {ck}; repeatable, in figure order")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="intervention-grid")
    args = ap.parse_args()

    rows = []
    for spec in args.row:
        if "=" not in spec:
            sys.exit("--row needs LABEL=TEMPLATE, got %r" % spec)
        lbl, tmpl = spec.split("=", 1)
        rows.append((lbl, tmpl))

    ok = target_ok(args.pair, args.seed)
    cd = cell_dir(args.pair, args.seed)
    spoken = args.pair.replace("__x__", " and ").replace("_", " ")

    title = "%s, seed %d: does any correction stop the flipping?%s" % (
        spoken, args.seed, (" " + args.label) if args.label else "")
    lines = ["Same seed and same sampler in every cell. Only the correction differs, and no row "
             "uses the joint prompt.",
             "Read along a row: a correction that works gives a cat and a dog at every checkpoint, "
             "not just at one."]
    if ok is False:
        lines.append("This seed's own joint-prompt target does not contain both subjects, so judge "
                     "these by eye against the words, never against that target.")

    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    need = int(max([probe.textlength(title, font=font(28, True))]
                   + [probe.textlength(t, font=font(16)) for t in lines])) + 2 * PAD
    W = max(LEFT + len(args.checkpoints) * (CELL + PAD) + PAD, need)
    H = TOP + len(rows) * (CELL + PAD + 30) + PAD
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((PAD, 20), title, fill=INK, font=font(28, True))
    for i, t in enumerate(lines):
        dr.text((PAD, 62 + i * 24), t, fill=RED if (ok is False and i == len(lines) - 1) else MUTED,
                font=font(16))

    # The two fixed references, sitting in the label gutter above the first row.
    for i, (lbl, fn) in enumerate((("joint target", "mono.png"), ("PoE alone", "poe.png"))):
        x = PAD + i * 130
        im.paste(tile(cd and os.path.join(cd, fn), 120), (x, TOP - 148))
        dr.text((x, TOP - 24), lbl, fill=RED if (i == 0 and ok is False) else BLUE, font=font(14, True))

    for r, (lbl, tmpl) in enumerate(rows):
        y = TOP + r * (CELL + PAD + 30)
        f19 = font(19, True)
        words, wrapped, cur = lbl.split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if dr.textlength(t, font=f19) <= LEFT - 2 * PAD:
                cur = t
            else:
                wrapped.append(cur); cur = w
        if cur:
            wrapped.append(cur)
        y0 = y + CELL // 2 - 12 * len(wrapped)
        for t in wrapped:
            dr.text((PAD, y0), t, fill=BLUE, font=f19); y0 += 24
        for c, ck in enumerate(args.checkpoints):
            x = LEFT + c * (CELL + PAD)
            p = os.path.join(ATTN, tmpl.format(ck=ck), args.pair,
                             "seed_%d" % args.seed, "sample_seed_%d.png" % args.seed)
            im.paste(tile(p, CELL), (x, y))
            if r == 0:
                dr.text((x, y - 30), "checkpoint %s" % format(int(ck), ","),
                        fill=INK, font=font(19, True))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "%s.png" % args.prefix)
    im.save(path)
    print("%d rows x %d checkpoints -> %s" % (len(rows), len(args.checkpoints), path))


if __name__ == "__main__":
    main()
