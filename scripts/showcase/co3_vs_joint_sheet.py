"""Put CO3's render beside the plain joint prompt's, for the same pair and seed index.

Two rows per pair: the plain joint prompt on top, CO3 underneath. CO3 seeds its own noise, so the
same column is not the same starting draw; the comparison is how often each method draws both
things across eight seeds, never a per-tile match.
"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disallowed_subjects import check as check_subject

CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
CO3 = os.environ.get("CO3_DIR", "/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/co3_targets")
CO3_LABEL = os.environ.get("CO3_LABEL", "CO3")
SEEDS = list(range(1, 9))
TILE, PAD, LEFT, TOP, CAP = 190, 12, 330, 120, 26
INK, MUTED, BG = (17, 17, 17), (110, 110, 110), (255, 255, 255)
JOINT, CO3C = (60, 90, 180), (190, 110, 30)


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def mono(pair, seed):
    for sp in ("train", "heldout"):
        p = os.path.join(CACHE, sp, pair, "seed_%d" % seed, "mono.png")
        if os.path.exists(p):
            return p


def co3(pair, seed):
    base = os.path.join(CO3, pair)
    if not os.path.isdir(base):
        return None
    for algo in os.listdir(base):
        for pt in os.listdir(os.path.join(base, algo)):
            d = os.path.join(base, algo, pt, "seed-%d" % seed)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".png"):
                        return os.path.join(d, f)


def prompt_of(pair):
    for sp in ("train", "heldout"):
        m = os.path.join(CACHE, sp, pair, "seed_1", "meta.json")
        if os.path.exists(m):
            return json.load(open(m)).get("joint_prompt", pair)
    return pair


def main(pairs, out):
    for p in pairs:
        check_subject(p)
    W = LEFT + len(SEEDS) * (TILE + PAD) + PAD
    H = TOP + len(pairs) * (2 * (TILE + CAP) + 3 * PAD + 26) + PAD
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((PAD, 22), "the plain joint prompt against CO3, same pair, eight seeds each",
            fill=INK, font=font(28, True))
    dr.text((PAD, 60), "CO3 draws its own noise, so a column is not a shared starting draw. Read the rate across "
                       "the row, never one tile against the tile below it.", fill=MUTED, font=font(15))
    dr.text((PAD, 82), os.environ.get("CO3_NOTE","CO3: Dutta et al, arXiv 2509.25940."), fill=MUTED, font=font(15))
    for c, s in enumerate(SEEDS):
        dr.text((LEFT + c * (TILE + PAD) + TILE // 2 - 22, TOP - 24), "seed %d" % s, fill=MUTED, font=font(15, True))
    y = TOP
    for pair in pairs:
        dr.text((PAD, y + 4), '"%s"' % prompt_of(pair), fill=INK, font=font(18, True))
        for r, (label, getter, col) in enumerate((("joint prompt", mono, JOINT), (CO3_LABEL, co3, CO3C))):
            ry = y + 30 + r * (TILE + CAP + PAD)
            dr.text((PAD, ry + TILE // 2 - 8), label, fill=col, font=font(17, True))
            for c, s in enumerate(SEEDS):
                x = LEFT + c * (TILE + PAD)
                f = getter(pair, s)
                dr.rectangle([x - 3, ry - 3, x + TILE + 3, ry + TILE + 3], fill=col)
                if f:
                    im.paste(Image.open(f).convert("RGB").resize((TILE, TILE), Image.LANCZOS), (x, ry))
                else:
                    dr.rectangle([x, ry, x + TILE, ry + TILE], fill=(238, 238, 238))
                    dr.text((x + 40, ry + TILE // 2), "not rendered", fill=MUTED, font=font(13))
        y += 2 * (TILE + CAP) + 3 * PAD + 26
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main(["a_giraffe__x__a_zebra", "a_giraffe__x__a_hippo"], sys.argv[1])
