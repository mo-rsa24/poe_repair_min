"""Mono-vs-PoE grids, for choosing which pairs to add to the filtered training list.

Rows are pairs, columns are seeds, capped at 8 of each per sheet. Each cell is that cell's own
mono.png (left half) beside its own poe.png (right half): the picture the adapter is trained to
copy, next to the picture PoE would draw without it. A pair worth adding is one where the left
half is clean and the right half is visibly wrong, which is the training-data selection criterion
itself, made visible instead of read off a scorer.

  python3 scripts/showcase/mono_poe_grids.py \
      --cells artifacts/_shared/cross_pair_pool_configs/cells_v55.json \
      --label "the v55 pool" --out <dir> --prefix pool-v55

  python3 scripts/showcase/mono_poe_grids.py --pairs a_typewriter__x__a_cactus ... \
      --label "object belt round 1" --out <dir> --prefix objects-r1

Cell layout is fixed at 8 seeds x 8 pairs per sheet ("happy with an 8x8"); more pairs or seeds
spill onto further sheets with the same prefix, and a pair with fewer than 8 cached seeds leaves
its remaining columns blank rather than reusing another pair's seeds.
"""
import argparse, json, os, sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disallowed_subjects import check as check_subject

CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
QUALITY = "/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/target_quality.json"

HALF = int(os.environ.get("HALF", "170"))   # each of mono/poe half-tile, side by side
GAP_MP = 3                                  # hairline between mono and poe within one cell
PAD, LEFT, TOP, ROWCAP, BORDER = 12, 380, 118, 30, 5
GREEN, RED, INK, MUTED, BG = (36, 138, 61), (200, 42, 42), (17, 17, 17), (110, 110, 110), (255, 255, 255)
BLUE = (23, 92, 176)   # the seed already approved for this pair
MONO_ONLY = False      # set by --mono-only: show the target alone, so more seeds fit


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/truetype/liberation/LiberationSans%s.ttf" % ("-Bold" if bold else "")):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def load_quality():
    verdict = {}
    if not os.path.exists(QUALITY):
        return verdict
    for r in json.load(open(QUALITY)):
        parts = r["seed"].split("_")
        if len(parts) > 1 and parts[1].isdigit():
            verdict[(r["pair"], int(parts[1]))] = bool(r.get("both_present"))
    return verdict


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


def cell_tile(w, h, path, tag):
    """One half of a cell: the image if it exists, else a grey placeholder saying why not."""
    im = Image.new("RGB", (w, h), (238, 238, 238))
    if path and os.path.exists(path):
        src = Image.open(path).convert("RGB")
        im.paste(src.resize((w, h), Image.LANCZOS), (0, 0))
    else:
        d = ImageDraw.Draw(im)
        d.text((6, 6), "no %s" % tag, fill=RED, font=font(12, True))
    return im


def build(pairs_seeds, out_path, title, subtitle, verdict, max_seeds=8, approved=None):
    tile_w = HALF if MONO_ONLY else 2 * HALF + GAP_MP
    ncol = max(min(len(seeds), max_seeds) for _, seeds in pairs_seeds)
    W = LEFT + ncol * (tile_w + PAD) + PAD
    H = TOP + len(pairs_seeds) * (HALF + ROWCAP + PAD) + PAD
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((PAD, 20), title, fill=INK, font=font(28, True))
    dr.text((PAD, 58), subtitle, fill=MUTED, font=font(15))
    legend = ("each tile is the joint-prompt render, the picture the adapter is trained to copy. "
              "Pick the seeds where both things are clearly present."
              if MONO_ONLY else
              "each cell: mono (left, the training target) next to poe (right, what PoE draws "
              "without the adapter). Green/red border repeats the automatic both-present check "
              "on the mono half only.")
    dr.text((PAD, 80), legend, fill=MUTED, font=font(14))

    for r, (pair, seeds) in enumerate(pairs_seeds):
        y = TOP + r * (HALF + ROWCAP + PAD)
        cols = sorted(seeds)[:max_seeds]
        jp = joint_prompt(pair, seeds)
        lines = wrap(dr, '"%s"' % jp, font(19, True), LEFT - 3 * PAD)
        y0 = y + HALF // 2 - 26 - 11 * (len(lines) - 1)
        for ln in lines:
            dr.text((PAD, y0), ln, fill=INK, font=font(19, True)); y0 += 24
        dr.text((PAD, y0 + 4), pair, fill=MUTED, font=font(12))

        for c, s in enumerate(cols):
            x = LEFT + c * (tile_w + PAD)
            v = verdict.get((pair, s))
            col = GREEN if v is True else RED if v is False else MUTED
            dr.rectangle([x - BORDER, y - BORDER, x + tile_w + BORDER, y + HALF + BORDER], fill=col)
            d = cell_dir(pair, s)
            im.paste(cell_tile(HALF, HALF, d and os.path.join(d, "mono.png"), "mono"), (x, y))
            if not MONO_ONLY:
                im.paste(cell_tile(HALF, HALF, d and os.path.join(d, "poe.png"), "poe"),
                         (x + HALF + GAP_MP, y))
            is_ok = s in (approved or {}).get(pair, [])
            dr.text((x, y + HALF + 6),
                    "seed %d%s" % (s, "  <- your pick" if is_ok else ""),
                    fill=BLUE if is_ok else INK, font=font(13, True))
    im.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cells", help="JSON {pair_slug: [seeds]}")
    src.add_argument("--pairs", nargs="*", help="pair slugs; every cached seed (up to 8) is used")
    ap.add_argument("--label", required=True)
    ap.add_argument("--pairs-per-figure", type=int, default=8, help="rows per sheet (the 8x8 default)")
    ap.add_argument("--max-seeds", type=int, default=8, help="columns per sheet (the 8x8 default)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="mono-poe")
    ap.add_argument("--mono-only", action="store_true",
                    help="show only the joint-prompt target, no PoE half, so more seeds fit a row")
    ap.add_argument("--approved", default=None,
                    help="JSON {pair: [seeds]} of cells already accepted; they get a blue label so "
                         "a candidate seed is judged against the one you already chose")
    args = ap.parse_args()

    if args.cells:
        cells = {k: sorted(int(x) for x in v) for k, v in json.load(open(args.cells)).items()}
    else:
        cells = {}
        for p in args.pairs:
            # The belt draws random seeds from 1-8 and 17-60, so a fixed 1..22 scan misses most of
            # them and silently produces an empty sheet. Read the directory instead of guessing.
            seeds = []
            for split in ("train", "heldout"):
                d = os.path.join(CACHE, split, p)
                if os.path.isdir(d):
                    seeds = sorted(int(x.split("_")[1]) for x in os.listdir(d)
                                   if x.startswith("seed_") and x.split("_")[1].isdigit()
                                   and os.path.exists(os.path.join(d, x, "mono.png")))
                    break
            cells[p] = [s for s in seeds if not (9 <= s <= 16)]

    for pair, seeds in cells.items():
        check_subject(joint_prompt(pair, seeds) or pair.replace("__x__", " and ").replace("_", " "))

    global MONO_ONLY
    MONO_ONLY = args.mono_only
    approved = json.load(open(args.approved)) if args.approved else {}
    approved = {k: [int(x) for x in v] for k, v in approved.items()}
    verdict = load_quality()
    os.makedirs(args.out, exist_ok=True)
    rows = sorted(cells.items())
    written = []
    for i in range(0, len(rows), args.pairs_per_figure):
        chunk = rows[i:i + args.pairs_per_figure]
        n = i // args.pairs_per_figure + 1
        name = "%s-%02d.png" % (args.prefix, n)
        total_sheets = (len(rows) + args.pairs_per_figure - 1) // args.pairs_per_figure
        sub = "%d pairs in %s. Sheet %d of %d." % (len(rows), args.label, n, total_sheets)
        p = build(chunk, os.path.join(args.out, name),
                  ("joint-prompt targets: %s" if args.mono_only else "mono vs PoE: %s") % args.label, sub, verdict, max_seeds=args.max_seeds, approved=approved)
        written.append(p)
        print("  %2d  %-30s %s" % (n, name, ", ".join(k for k, _ in chunk)))
    print("\n%d sheets -> %s" % (len(written), args.out))


if __name__ == "__main__":
    main()
