"""Grids of exactly the cells a run trained on, for judging by eye whether the data is clean.

One row per pair, one tile per seed that pair actually contributed. The seed sets are ragged (one
pair gives a single seed, another gives six), so rows have different lengths and the tile carries
its own seed number rather than relying on a shared column header.

Each tile is that cell's `mono.png`, the joint-prompt render, which is the picture the adapter is
trained to reproduce. The border is the automatic both-present check, green for pass and red for
fail, so the same sheet that shows the data also shows where the automatic read disagrees with
the eye. Every row is captioned with that pair's own joint prompt, because a target is judged
against the prompt that produced it.

  python3 scripts/showcase/training_pool_grids.py \
      --cells artifacts/_shared/cross_pair_pool_configs/cells_v54.json \
      --label "the 54-cell pool" --rows-per-figure 6 --out <dir>

With --compare pointing at a second cell list, a tile present only in the second list is marked
"extra", which is how the 54-cell pool and the 182-cell unfiltered pool are read on one sheet.
"""
import argparse, glob, json, os, sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disallowed_subjects import check as check_subject

CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
QUALITY = "/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/target_quality.json"

TILE = int(os.environ.get("TILE", "300"))
PAD, LEFT, TOP, ROWCAP, BORDER = 14, 430, 128, 34, 6
GREEN, RED, INK, MUTED, BG = (36, 138, 61), (200, 42, 42), (17, 17, 17), (110, 110, 110), (255, 255, 255)
AMBER = (183, 121, 31)


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/truetype/liberation/LiberationSans%s.ttf" % ("-Bold" if bold else "")):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def load_quality():
    """The automatic both-present verdict, keyed (pair, seed). Missing means never scored."""
    verdict = {}
    if not os.path.exists(QUALITY):
        return verdict
    for r in json.load(open(QUALITY)):
        parts = r["seed"].split("_")
        if len(parts) > 1 and parts[1].isdigit():
            verdict[(r["pair"], int(parts[1]))] = bool(r.get("both_present"))
    return verdict


CO3_ROOT = None   # set from --co3-root; when set, tiles come from CO3 instead of the cache


def co3_png(pair, seed):
    """The CO3 render for one cell. sample_co3.py nests its output as
    <root>/<pair>/algo-<algo>_/<joint prompt with underscores>/seed-<n>/<name>.png, and the leaf
    name varies with the prompt, so the path is matched rather than constructed."""
    if not CO3_ROOT:
        return None
    hits = sorted(glob.glob(os.path.join(CO3_ROOT, pair, "*", "*", "seed-%d" % seed, "*.png")))
    return hits[0] if hits else None


def cell_dir(pair, seed):
    for split in ("train", "heldout"):
        d = os.path.join(CACHE, split, pair, "seed_%d" % seed)
        if os.path.isdir(d):
            return d
    return None


def joint_prompt(pair, seeds):
    for s in seeds:
        d = cell_dir(pair, s)
        if d and os.path.exists(os.path.join(d, "meta.json")):
            return json.load(open(os.path.join(d, "meta.json"))).get("joint_prompt", "")
    return ""


def wrap(dr, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if dr.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def build(rows, out_path, title, subtitle, verdict, extra, max_cols=8):
    # A pair with fifteen seeds would make a sheet so wide every tile shrinks past the point of
    # being judgeable, so a long row folds onto further lines of at most max_cols tiles.
    banded = [(pair, [seeds[i:i + max_cols] for i in range(0, len(seeds), max_cols)])
              for pair, seeds in rows]
    ncol = max(len(b) for _, bands in banded for b in bands)
    nband = sum(len(bands) for _, bands in banded)
    # A pair contributing one seed makes a sheet narrower than its own heading, so the title runs
    # off the edge. The header text sets a floor on the width.
    W = max(LEFT + ncol * (TILE + PAD) + PAD, 1560)
    H = TOP + nband * (TILE + ROWCAP + PAD) + PAD
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((PAD, 22), title, fill=INK, font=font(30, True))
    dr.text((PAD, 62), subtitle, fill=MUTED, font=font(16))
    legend = ("these are CO3 renders, so the border repeats the joint-prompt scorer's verdict on the "
              "same cell and is not a judgement of the CO3 picture."
              if CO3_ROOT else
              "green border = the automatic check says both animals are present, red = it says they are not. "
              "Judge against the row's prompt and mark where you disagree.")
    dr.text((PAD, 88), legend, fill=MUTED, font=font(15))

    band_i = 0
    for pair, bands in banded:
        seeds = [s for b in bands for s in b]
        for bi, band in enumerate(bands):
            y = TOP + band_i * (TILE + ROWCAP + PAD)
            band_i += 1
            if bi == 0:
                jp = joint_prompt(pair, seeds)
                lines = wrap(dr, '"%s"' % jp, font(21, True), LEFT - 3 * PAD)
                y0 = y + TILE // 2 - 30 - 12 * (len(lines) - 1)
                for ln in lines:
                    dr.text((PAD, y0), ln, fill=INK, font=font(21, True)); y0 += 26
                dr.text((PAD, y0 + 4), pair, fill=MUTED, font=font(13))
                npass = sum(1 for s in seeds if verdict.get((pair, s)) is True)
                dr.text((PAD, y0 + 24), "%d cells, automatic: %d of %d pass"
                        % (len(seeds), npass, len(seeds)), fill=MUTED, font=font(13))
            else:
                dr.text((PAD, y + TILE // 2 - 10), "… same pair, continued",
                        fill=MUTED, font=font(16))
            _band(dr, im, pair, band, y, verdict, extra)
    im.save(out_path)
    return out_path


def _band(dr, im, pair, seeds, y, verdict, extra):
    for c, s in enumerate(seeds):
            x = LEFT + c * (TILE + PAD)
            v = verdict.get((pair, s))
            col = GREEN if v is True else RED if v is False else MUTED
            dr.rectangle([x - BORDER, y - BORDER, x + TILE + BORDER, y + TILE + BORDER], fill=col)
            if CO3_ROOT:
                p = co3_png(pair, s)
            else:
                d = cell_dir(pair, s)
                p = os.path.join(d, "mono.png") if d else None
            if p and os.path.exists(p):
                im.paste(Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS), (x, y))
            else:
                dr.rectangle([x, y, x + TILE, y + TILE], fill=(238, 238, 238))
                dr.text((x + 10, y + 10), "not rendered yet" if CO3_ROOT else "cell missing",
                        fill=RED, font=font(15, True))
            tag = "  extra" if (pair, s) in extra else ""
            dr.text((x, y + TILE + 9), "seed %d%s" % (s, tag),
                    fill=AMBER if tag else INK, font=font(15, True))
            dr.text((x, y + TILE + 27),
                    "pass" if v is True else "fail" if v is False else "not scored",
                    fill=col, font=font(13))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="JSON {pair_slug: [seeds]} — the cells a run trained on")
    ap.add_argument("--compare", default=None, help="a second cell list; cells only in --cells are marked extra")
    ap.add_argument("--label", required=True, help="what this pool is, e.g. 'the 54-cell pool'")
    ap.add_argument("--rows-per-figure", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="training-pool")
    ap.add_argument("--co3-root", default=None,
                    help="draw tiles from CO3 renders under this root instead of the cache's mono.png")
    ap.add_argument("--max-cols", type=int, default=8,
                    help="tiles per line; a longer seed list folds onto further lines")
    args = ap.parse_args()

    global CO3_ROOT
    CO3_ROOT = args.co3_root

    cells = {k: sorted(int(x) for x in v) for k, v in json.load(open(args.cells)).items()}
    extra = set()
    if args.compare:
        base = {k: set(int(x) for x in v) for k, v in json.load(open(args.compare)).items()}
        for p, ss in cells.items():
            for s in ss:
                if s not in base.get(p, set()):
                    extra.add((p, s))

    for pair, seeds in cells.items():
        check_subject(joint_prompt(pair, seeds) or pair.replace("__x__", " and ").replace("_", " "))

    verdict = load_quality()
    os.makedirs(args.out, exist_ok=True)
    rows = sorted(cells.items())
    total = sum(len(v) for v in cells.values())
    written = []
    for i in range(0, len(rows), args.rows_per_figure):
        chunk = rows[i:i + args.rows_per_figure]
        n = i // args.rows_per_figure + 1
        name = "%s-%02d.png" % (args.prefix, n)
        sub = ("%d pairs, %d cells in this pool. Sheet %d of %d. Each tile is the joint-prompt render "
               "the adapter is trained to copy."
               % (len(rows), total, n, (len(rows) + args.rows_per_figure - 1) // args.rows_per_figure))
        p = build(chunk, os.path.join(args.out, name),
                  "training data: %s" % args.label, sub, verdict, extra, max_cols=args.max_cols)
        written.append(p)
        print("  %2d  %-34s %s" % (n, name, ", ".join(k for k, _ in chunk)))
    print("\n%d sheets -> %s" % (len(written), args.out))


if __name__ == "__main__":
    main()
