"""SDXL's joint-prompt render against SD 3.5 medium's, same pair and seed index.

Both are seeded per image but the models differ in architecture and latent space, so a column is not
a shared starting draw. Read the rate along a row, never one tile against the tile below it.
"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disallowed_subjects import check as check_subject

CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
SD35 = "/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/sd35_joint"
SEEDS = list(range(1, 9))
TILE, PAD, LEFT, TOP, CAP = 175, 11, 320, 120, 22
INK, MUTED, BG = (17, 17, 17), (110, 110, 110), (255, 255, 255)
XL, S35 = (60, 90, 180), (30, 140, 90)


def font(sz, bold=False):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


def sdxl(pair, seed):
    for sp in ("train", "heldout"):
        p = os.path.join(CACHE, sp, pair, "seed_%d" % seed, "mono.png")
        if os.path.exists(p):
            return p


def sd35(pair, seed):
    p = os.path.join(SD35, pair, "seed_%d.png" % seed)
    return p if os.path.exists(p) else None


def prompt_of(pair):
    for sp in ("train", "heldout"):
        m = os.path.join(CACHE, sp, pair, "seed_1", "meta.json")
        if os.path.exists(m):
            return json.load(open(m))["joint_prompt"]
    return pair


def main(pairs, out):
    for p in pairs:
        check_subject(p)
    W = LEFT + len(SEEDS) * (TILE + PAD) + PAD
    H = TOP + len(pairs) * (2 * (TILE + CAP) + 2 * PAD + 24) + PAD
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((PAD, 22), "the same joint prompt on SDXL and on SD 3.5 medium", fill=INK, font=font(28, True))
    dr.text((PAD, 60), "both at 50 steps; SDXL at guidance 7.5 from the cache, SD 3.5 at 7.0. Architectures and latent "
                       "spaces differ, so a column is not a shared draw.", fill=MUTED, font=font(15))
    dr.text((PAD, 82), "read the rate along a row: how often does each model put both named things in the frame?",
            fill=MUTED, font=font(15))
    for c, s in enumerate(SEEDS):
        dr.text((LEFT + c * (TILE + PAD) + TILE // 2 - 20, TOP - 22), "seed %d" % s, fill=MUTED, font=font(14, True))
    y = TOP
    for pair in pairs:
        dr.text((PAD, y + 2), '"%s"' % prompt_of(pair), fill=INK, font=font(17, True))
        for r, (label, get, col) in enumerate((("SDXL", sdxl, XL), ("SD 3.5", sd35, S35))):
            ry = y + 28 + r * (TILE + CAP + PAD)
            dr.text((PAD, ry + TILE // 2 - 8), label, fill=col, font=font(16, True))
            for c, s in enumerate(SEEDS):
                x = LEFT + c * (TILE + PAD)
                f = get(pair, s)
                dr.rectangle([x - 3, ry - 3, x + TILE + 3, ry + TILE + 3], fill=col)
                if f:
                    im.paste(Image.open(f).convert("RGB").resize((TILE, TILE), Image.LANCZOS), (x, ry))
                else:
                    dr.rectangle([x, ry, x + TILE, ry + TILE], fill=(238, 238, 238))
        y += 2 * (TILE + CAP) + 2 * PAD + 24
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main(["a_giraffe__x__a_zebra", "a_rhino__x__a_zebra", "a_bear__x__a_salmon",
          "a_monkey__x__a_parrot", "a_horse__x__a_goat", "a_giraffe__x__a_hippo"], sys.argv[1])
