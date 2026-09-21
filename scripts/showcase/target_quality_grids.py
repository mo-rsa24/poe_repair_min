"""Grids of joint-prompt target renders, for judging by eye which cells the adapter may learn from.

One figure per group of pairs. Rows are pairs, columns are seeds 1 to 8, each tile is that cell's
joint-prompt render (`mono.png`), which is the picture the adapter is trained to reproduce.

Every tile carries the automatic verdict as its border colour, green for pass and red for fail,
so the same pass that picks the training cells also says how far the automatic read can be
trusted. Each row is captioned with that pair's own joint prompt, because a target is judged
against the prompt that produced it and not against a generic description.
"""
import argparse, json, os, sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disallowed_subjects import check as check_subject, is_allowed

CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
QUALITY = "/datasets/mmolefe/poe_repair_min/outputs/showcase/target_quality/target_quality.json"
EVAL_PAIRS = {"a_cat__x__a_dog", "an_elephant__x__a_penguin"}
SEEDS = [int(x) for x in os.environ.get('SEEDS','1 2 3 4 5 6 7 8').split()]

TILE = int(os.environ.get('TILE', '200'))
PAD, LEFT, TOP, ROWCAP, BORDER = 14, 400, 132, 30, 6
GREEN, RED, INK, MUTED, BG = (36, 138, 61), (200, 42, 42), (17, 17, 17), (110, 110, 110), (255, 255, 255)


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/truetype/liberation/LiberationSans%s.ttf" % ("-Bold" if bold else "")):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def load_quality():
    verdict = {}
    for r in json.load(open(QUALITY)):
        seed = r["seed"]
        n = int(seed.split("_")[1]) if seed.split("_")[1].isdigit() else None
        if n is not None:
            verdict[(r["pair"], n)] = bool(r.get("both_present"))
    return verdict


def find_cell(pair, seed):
    for split in ("train", "heldout"):
        d = os.path.join(CACHE, split, pair, "seed_%d" % seed)
        if os.path.exists(os.path.join(d, "mono.png")):
            return d
    return None


def joint_prompt(pair):
    for seed in SEEDS:
        d = find_cell(pair, seed)
        if d and os.path.exists(os.path.join(d, "meta.json")):
            return json.load(open(os.path.join(d, "meta.json"))).get("joint_prompt", pair)
    return pair.replace("__x__", " and ").replace("_", " ")


def candidates():
    out = []
    for split in ("train", "heldout"):
        d = os.path.join(CACHE, split)
        if not os.path.isdir(d):
            continue
        for pair in os.listdir(d):
            if pair in EVAL_PAIRS or pair in out:
                continue
            if not is_allowed(pair):
                continue
            if all(find_cell(pair, s) for s in SEEDS):
                out.append(pair)
    return sorted(set(out))


def build(pairs, path, title, verdict):
    W = LEFT + len(SEEDS) * (TILE + PAD) + PAD
    H = TOP + len(pairs) * (TILE + ROWCAP + PAD) + PAD + 40
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((PAD, 20), title, fill=INK, font=font(30, True))
    dr.text((PAD, 58), "each tile is the picture the joint prompt drew on that seed, which is what the adapter is trained to copy.",
            fill=MUTED, font=font(15))
    dr.text((PAD, 80), "green border = the automatic check says both things are present.   red = it says they are not.   "
                       "judge against the row's own prompt, and mark where you disagree with the border.",
            fill=MUTED, font=font(15))
    for c, s in enumerate(SEEDS):
        x = LEFT + c * (TILE + PAD)
        dr.text((x + TILE // 2 - 24, TOP - 22), "seed %d" % s, fill=MUTED, font=font(16, True))
    for r, pair in enumerate(pairs):
        y = TOP + r * (TILE + ROWCAP + PAD)
        prompt = joint_prompt(pair)
        f19 = font(19, True)
        words, lines, cur = ('"%s"' % prompt).split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if dr.textlength(t, font=f19) > LEFT - 2 * PAD and cur:
                lines.append(cur); cur = w
            else:
                cur = t
        lines.append(cur)
        y0 = y + TILE // 2 - 24 - 11 * (len(lines) - 1)
        for i, ln in enumerate(lines):
            dr.text((PAD, y0 + 22 * i), ln, fill=INK, font=f19)
        base = y0 + 22 * len(lines)
        dr.text((PAD, base + 2), pair, fill=MUTED, font=font(12))
        npass = sum(1 for s in SEEDS if verdict.get((pair, s)) is True)
        dr.text((PAD, base + 22), "automatic: %d of 8 pass" % npass, fill=MUTED, font=font(13))
        for c, s in enumerate(SEEDS):
            x = LEFT + c * (TILE + PAD)
            d = find_cell(pair, s)
            v = verdict.get((pair, s))
            col = GREEN if v is True else RED if v is False else MUTED
            dr.rectangle([x - BORDER, y - BORDER, x + TILE + BORDER, y + TILE + BORDER], fill=col)
            if d:
                t = Image.open(os.path.join(d, "mono.png")).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
                im.paste(t, (x, y))
            else:
                dr.rectangle([x, y, x + TILE, y + TILE], fill=(240, 240, 240))
            dr.text((x, y + TILE + 8), "seed %d   %s" % (s, "pass" if v else "fail" if v is False else "not scored"),
                    fill=col, font=font(13, True))
    im.save(path)
    return path


def short(pair):
    """The pair as a filename coordinate: articles dropped, one hyphenated token per side."""
    a, _, b = pair.partition("__x__")
    strip = lambda q: q[3:] if q.startswith("an_") else q[2:] if q.startswith("a_") else q
    return (strip(a) + "-" + strip(b)).replace("_", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from/"
                                     "joint-prompt-targets-vs-pair-and-seed")
    ap.add_argument("--per-figure", type=int, default=5)
    ap.add_argument("--pairs", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    verdict = load_quality()
    pairs = args.pairs or candidates()
    for p in pairs:
        check_subject(p)   # refuses before a single tile is drawn
    print("%d candidate pairs, all 8 training seeds rendered, the two evaluation pairs excluded" % len(pairs))
    made = []
    for i in range(0, len(pairs), args.per_figure):
        grp = pairs[i:i + args.per_figure]
        n = i // args.per_figure + 1
        coord = "%s-to-%s" % (short(grp[0]), short(grp[-1]))
        p = os.path.join(args.out, coord + ".png")
        build(grp, p, "joint-prompt targets: %s to %s" % (short(grp[0]), short(grp[-1])), verdict)
        made.append(p)
        print("  %2d  %-52s %s" % (n, coord + ".png", grp[0] + " .. " + grp[-1]))
    idx = os.path.join(args.out, "joint-prompt-targets-vs-pair-and-seed.json")
    json.dump({"pairs": pairs, "per_figure": args.per_figure, "sheets": made,
               "seeds": SEEDS, "cache": CACHE, "automatic_verdict_source": QUALITY,
               "excluded_pairs": sorted(EVAL_PAIRS)}, open(idx, "w"), indent=1)
    print("\n%d sheets -> %s" % (len(made), args.out))


if __name__ == "__main__":
    main()
