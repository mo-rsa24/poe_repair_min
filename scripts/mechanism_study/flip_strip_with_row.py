"""One row per checkpoint strip: the adapter alone on top, one fix underneath.

The top row is every checkpoint the training run sampled with no fix applied, which is where the
flip shows: the same starting noise draws a different picture each time. The bottom row is the same
adapter with one fix, and it can only be filled where saved weights exist. The run wrote weights
every 10,000 steps while it sampled every 1,250, so most columns have a picture on top and nothing
to re-render underneath. Those cells say "no saved checkpoint" rather than being dropped, because
the gap is why the comparison is thin.

One file per strength, so strengths can be flipped through instead of squeezed onto one sheet:

  python3 scripts/mechanism_study/flip_strip_with_row.py --seed 10 --doses 20 50 100 \
      --out artifacts/results/<question> --prefix cat-boost

writes `cat-boost-<n>-every-checkpoint.png` for each. `--regime-template` picks which fix's
directories to read; it takes `{d}` for the strength and `{ck}` for the padded checkpoint.
"""
import argparse, os, re, sys

from PIL import Image, ImageDraw, ImageFont

ATTN = "/datasets/mmolefe/poe_repair_min/outputs/attn_mechanism"
CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
PANELS = ("/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/pool43-all50-orth3/wandb/"
          "run-20260909_210345-9nlg4v85/files/media/images/samples/out_out")

TILE, PAD, LEFT, TOP = 150, 8, 300, 250
INK, MUTED, BG, RED, BLUE, GREEN = ((17, 17, 17), (110, 110, 110), (255, 255, 255),
                                    (200, 42, 42), (23, 92, 176), (36, 138, 61))
# The logged panel is three 256-squares side by side; the adapter's own render is the third.
PANEL_SIZE, LORA_TILE = (800, 320), (536, 48, 792, 304)


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/truetype/liberation/LiberationSans%s.ttf" % ("-Bold" if bold else "")):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def logged_panels(pair, seed):
    root = os.path.join(PANELS, pair)
    out = {}
    if not os.path.isdir(root):
        return out
    for f in os.listdir(root):
        m = re.match(r"seed_%d_(\d+)_[0-9a-f]+\.png$" % seed, f)
        if m:
            out[int(m.group(1))] = os.path.join(root, f)
    return out


def cell_dir(pair, seed):
    for split in ("train", "heldout"):
        d = os.path.join(CACHE, split, pair, "seed_%d" % seed)
        if os.path.isdir(d):
            return d
    return None


def paste(im, dr, path, x, y, outline=None):
    if path and os.path.exists(path):
        src = Image.open(path).convert("RGB")
        if src.size == PANEL_SIZE:
            src = src.crop(LORA_TILE)
        im.paste(src.resize((TILE, TILE), Image.LANCZOS), (x, y))
        if outline:
            dr.rectangle([x - 3, y - 3, x + TILE + 3, y + TILE + 3], outline=outline, width=3)
        return True
    im.paste(Image.new("RGB", (TILE, TILE), (246, 246, 246)), (x, y))
    dr.text((x + 8, y + TILE // 2 - 16), "no saved", fill=(170, 170, 170), font=font(12))
    dr.text((x + 8, y + TILE // 2 - 2), "checkpoint", fill=(170, 170, 170), font=font(12))
    return False


def build(seed, pair, dose, tmpl, saved, out_path, row_label):
    logged = logged_panels(pair, seed)
    steps = sorted(set(logged) | set(saved))
    cd = cell_dir(pair, seed)
    W = LEFT + len(steps) * (TILE + PAD) + PAD
    H = TOP + 2 * (TILE + 58) + PAD
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    spoken = pair.replace("__x__", " and ").replace("_", " ")
    dr.text((PAD, 18), "%s, seed %d: the adapter alone, and the same adapter with %s" % (spoken, seed, row_label),
            fill=INK, font=font(30, True))
    dr.text((PAD, 60), "Top: every checkpoint the run sampled, adapter alone. The picture jumps "
                       "between pictures instead of settling.", fill=MUTED, font=font(16))
    dr.text((PAD, 84), "Bottom: the same adapter with the fix applied on steps 5 to 25. It "
                       "exists only in the green columns, because the run wrote", fill=MUTED, font=font(16))
    dr.text((PAD, 108), "weights every 10,000 steps while it sampled every 1,250.",
            fill=MUTED, font=font(16))
    dr.text((PAD, 136), "Seed %d's own joint prompt draws two dogs and no cat. No joint prompt is "
                        "used in any tile below." % seed, fill=RED, font=font(16))

    for i, (lbl, fn) in enumerate((("joint target", "mono.png"), ("PoE alone", "poe.png"))):
        x = PAD + i * 130
        p = cd and os.path.join(cd, fn)
        if p and os.path.exists(p):
            im.paste(Image.open(p).convert("RGB").resize((120, 120), Image.LANCZOS), (x, TOP - 130))
        dr.text((x, TOP - 8), lbl, fill=RED if i == 0 else BLUE, font=font(14, True))

    rows = [("adapter alone",
             lambda st: logged.get(st) or os.path.join(
                 ATTN, "ckpt%06d_none" % st, pair, "seed_%d" % seed, "sample_seed_%d.png" % seed)),
            ("+ %s" % row_label,
             lambda st: os.path.join(ATTN, tmpl.format(d=dose, ck="%06d" % st), pair,
                                     "seed_%d" % seed, "sample_seed_%d.png" % seed))]
    for r, (lbl, g) in enumerate(rows):
        y = TOP + r * (TILE + 58)
        dr.text((PAD, y + TILE // 2 - 10), lbl, fill=BLUE, font=font(18, True))
        for c, st in enumerate(steps):
            x = LEFT + c * (TILE + PAD)
            paste(im, dr, g(st), x, y, outline=GREEN if st in saved else None)
            if r == 0:
                dr.text((x, y + TILE + 6), "%d" % st,
                        fill=GREEN if st in saved else INK, font=font(13, True))
    im.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--doses", "--strengths", dest="doses", nargs="+", type=int,
                    required=True, help="one figure per strength")
    ap.add_argument("--saved", nargs="+", type=int, default=[10000, 20000, 30000],
                    help="checkpoint steps whose weights exist on disk")
    ap.add_argument("--regime-template", default="s10_excat{d}_{ck}",
                    help="{d} is the strength, {ck} the zero-padded checkpoint step")
    ap.add_argument("--row-label", default="cat boost")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for d in args.doses:
        path = os.path.join(args.out, "%s-%d-every-checkpoint.png" % (args.prefix, d))
        build(args.seed, args.pair, d, args.regime_template, set(args.saved), path,
              "%s, strength %d" % (args.row_label, d))
        have = sum(os.path.exists(os.path.join(
            ATTN, args.regime_template.format(d=d, ck="%06d" % st), args.pair,
            "seed_%d" % args.seed, "sample_seed_%d.png" % args.seed)) for st in args.saved)
        print("  strength %-4d %d/%d saved checkpoints -> %s" % (d, have, len(args.saved),
                                                             os.path.basename(path)))


if __name__ == "__main__":
    main()
