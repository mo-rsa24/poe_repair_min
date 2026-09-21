"""Lay every inline-sampled checkpoint of one pair out as a strip, so basin flips are visible.

A training run samples the same held-out cell every N epochs and logs a three-tile panel: the
joint-prompt target, what PoE draws alone, and what the adapter draws at that checkpoint. Clicking
through those panels in W&B one at a time hides the thing worth seeing, which is that the adapter's
render does not converge on one picture. It jumps between a few discrete outcomes as training
proceeds, because the sampler commits early to one basin and a small weight change flips which.

This crops the adapter tile out of every panel and lays them in one row per seed, with that seed's
target and PoE tiles pinned at the left as the fixed reference. Under each tile sits the step and
the early-bucket cosine between the learned and the actual correction, so the picture and the
number that is supposed to describe it are read together.

A seed whose target does not show both subjects is labelled as such on its row. Any statistic
comparing a render to that target is meaningless on that row, and the row exists to show why.

  python3 scripts/showcase/checkpoint_filmstrip.py \
      --run-dir <the wandb run dir> --pair a_cat__x__a_dog \
      --out artifacts/results/<question> --prefix seed-flips
"""
import argparse, json, os, re, sys

from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
QUALITY = "/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/target_quality.json"

# Panel geometry, measured off the logged panels rather than assumed: three 256-square tiles at
# these offsets, captioned above. Re-measure if the inline sampler's layout ever changes.
TILE, TILE_Y = [(8, 264), (272, 528), (536, 792)], (48, 304)

CELL = 170                      # each tile in the strip
PAD, GAP_REF, TOP, CAP = 8, 34, 150, 40
LEFT = 150                      # row label gutter
INK, MUTED, BG = (17, 17, 17), (110, 110, 110), (255, 255, 255)
RED, GREEN, BLUE = (200, 42, 42), (36, 138, 61), (23, 92, 176)


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/truetype/liberation/LiberationSans%s.ttf" % ("-Bold" if bold else "")):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def panels(run_dir, pair):
    """{seed: [(step, path), ...]} for every logged panel of this pair, in step order."""
    root = os.path.join(run_dir, "files", "media", "images", "samples", "out_out", pair)
    if not os.path.isdir(root):
        sys.exit("no sampled panels under %s" % root)
    out = {}
    for f in os.listdir(root):
        m = re.match(r"seed_(\d+)_(\d+)_[0-9a-f]+\.png$", f)
        if m:
            out.setdefault(int(m.group(1)), []).append((int(m.group(2)), os.path.join(root, f)))
    for s in out:
        out[s].sort()
    return out


def tile(path, idx):
    """Crop one of the three tiles out of a logged panel and square it to CELL."""
    im = Image.open(path).convert("RGB")
    x0, x1 = TILE[idx]
    return im.crop((x0, TILE_Y[0], x1, TILE_Y[1])).resize((CELL, CELL), Image.LANCZOS)


def target_verdict(pair):
    """{seed: both_present} from the instance-count scan of the joint-prompt targets."""
    if not os.path.exists(QUALITY):
        return {}
    v = {}
    for r in json.load(open(QUALITY)):
        if r.get("pair") != pair:
            continue
        parts = str(r.get("seed", "")).split("_")
        if len(parts) > 1 and parts[1].isdigit():
            v[int(parts[1])] = bool(r.get("both_present"))
    return v


def cosines(entity, project, run_id, pair, seeds):
    """{(seed, step): early-bucket cosine} from W&B, empty if it cannot be reached."""
    try:
        import wandb
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        keys = ["eval/tracking/learned_actual_cosine/bucket_early/out_out/%s/seed_%02d" % (pair, s)
                for s in seeds]
        out = {}
        for row in run.scan_history(keys=["_step"] + keys, page_size=2000):
            st = row.get("_step")
            if st is None:
                continue
            for s, k in zip(seeds, keys):
                if row.get(k) is not None:
                    out[(s, st)] = float(row[k])
        return out
    except Exception as e:                       # offline, or the run is mid-flight; not fatal
        print("  (no cosines: %s)" % e)
        return {}


def nearest(cos, seed, step):
    """The cosine logged at this checkpoint, or at the nearest step within half a sample gap."""
    if (seed, step) in cos:
        return cos[(seed, step)]
    cand = [(abs(st - step), v) for (sd, st), v in cos.items() if sd == seed]
    if not cand:
        return None
    d, v = min(cand)
    return v if d <= 40 else None


def build(rows, out_path, title, subtitle, verdict, cos):
    ncol = max(len(ck) for _, ck in rows)
    W = LEFT + 2 * (CELL + PAD) + GAP_REF + ncol * (CELL + PAD) + PAD
    H = TOP + len(rows) * (CELL + CAP + PAD + 26) + PAD
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((PAD, 24), title, fill=INK, font=font(34, True))
    for i, line in enumerate(subtitle):
        dr.text((PAD, 70 + i * 24), line, fill=MUTED, font=font(17))

    for r, (seed, cks) in enumerate(rows):
        y = TOP + r * (CELL + CAP + PAD + 26)
        ok = verdict.get(seed)
        dr.text((PAD, y + CELL // 2 - 26), "seed %d" % seed, fill=INK, font=font(26, True))
        dr.text((PAD, y + CELL // 2 + 6),
                "target sound" if ok else "target BROKEN" if ok is False else "target unscored",
                fill=GREEN if ok else RED if ok is False else MUTED, font=font(15, True))
        if ok is False:
            dr.text((PAD, y + CELL // 2 + 26), "no cat in it,", fill=RED, font=font(13))
            dr.text((PAD, y + CELL // 2 + 42), "so distances", fill=RED, font=font(13))
            dr.text((PAD, y + CELL // 2 + 58), "mean nothing", fill=RED, font=font(13))

        first = cks[0][1]
        for i, (lbl, idx) in enumerate((("target", 0), ("PoE alone", 1))):
            x = LEFT + i * (CELL + PAD)
            im.paste(tile(first, idx), (x, y))
            dr.rectangle([x - 2, y - 2, x + CELL + 2, y + CELL + 2], outline=BLUE, width=2)
            dr.text((x, y + CELL + 6), lbl, fill=BLUE, font=font(15, True))

        x0 = LEFT + 2 * (CELL + PAD) + GAP_REF
        for i, (step, path) in enumerate(cks):
            x = x0 + i * (CELL + PAD)
            im.paste(tile(path, 2), (x, y))
            dr.text((x, y + CELL + 6), "step %d" % step, fill=INK, font=font(15, True))
            c = nearest(cos, seed, step)
            dr.text((x, y + CELL + 24),
                    "cos %.3f" % c if c is not None else "cos n/a", fill=MUTED, font=font(14))
    im.save(out_path)
    return out_path


def main():
    global CELL
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="the wandb run directory holding the panels")
    ap.add_argument("--pair", default="a_cat__x__a_dog")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="default: every seed found")
    ap.add_argument("--entity", default="prime_lab")
    ap.add_argument("--project", default="poe-repair-animals-compose")
    ap.add_argument("--run-id", default=None, help="default: parsed off the run directory name")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="checkpoint-filmstrip")
    ap.add_argument("--cell", type=int, default=CELL, help="tile size in the strip")
    args = ap.parse_args()
    CELL = args.cell

    found = panels(args.run_dir, args.pair)
    seeds = sorted(args.seeds if args.seeds else found)
    rows = [(s, found[s]) for s in seeds if found.get(s)]
    if not rows:
        sys.exit("no panels for seeds %s" % seeds)

    run_id = args.run_id or os.path.basename(args.run_dir.rstrip("/")).split("-")[-1]
    cos = cosines(args.entity, args.project, run_id, args.pair, [s for s, _ in rows])
    verdict = target_verdict(args.pair)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "%s.png" % args.prefix)
    spoken = args.pair.replace("__x__", " and ").replace("_", " ")
    build(rows, path,
          "every checkpoint of %s, %s" % (spoken, args.label),
          ["Left, in blue: the joint-prompt target this seed is scored against, and what PoE draws "
           "with no adapter. Both are fixed for the whole row.",
           "Right: what the adapter draws at each checkpoint, same starting noise every time. "
           "cos is the early-bucket cosine between the learned and the actual correction.",
           "The render is a step function of the weights, so it jumps between a few outcomes "
           "instead of converging on one. Read along a row to see which."],
          verdict, cos)
    for s, cks in rows:
        print("  seed %-3d %2d checkpoints, steps %d to %d" % (s, len(cks), cks[0][0], cks[-1][0]))
    print("\n-> %s" % path)


if __name__ == "__main__":
    main()
