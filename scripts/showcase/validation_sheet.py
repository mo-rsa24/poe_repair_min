"""The sheet a person labels to build ground truth, with no machine verdict anywhere on it.

Every other sheet in this grouping shows the automatic check's verdict as a border, on purpose, so a
reader can see where it disagrees. This one must not: it is the reference the screeners are measured
against, and a verdict on the tile would make the labels a judgement of the verdict.
"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disallowed_subjects import check as check_subject

TILE, PAD, LEFT, TOP, CAP = 200, 14, 360, 132, 30
INK, MUTED, BG, EDGE = (17, 17, 17), (110, 110, 110), (255, 255, 255), (205, 205, 205)


def font(sz, bold=False):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


def main(cells_json, out):
    cells = json.load(open(cells_json))
    for c in cells:
        check_subject(c["pair"])
    pairs = []
    for c in cells:
        if c["pair"] not in [p for p, _, _ in pairs]:
            pairs.append((c["pair"], c["name_a"], c["name_b"]))
    ncol = max(len([c for c in cells if c["pair"] == p]) for p, _, _ in pairs)
    W = LEFT + ncol * (TILE + PAD) + PAD
    H = TOP + len(pairs) * (TILE + CAP + PAD) + PAD
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((PAD, 22), "ground truth: does each picture show both named things?", fill=INK, font=font(30, True))
    dr.text((PAD, 62), "no machine verdict appears on this sheet, on purpose. Your labels are the reference the "
                       "screeners get measured against,", fill=MUTED, font=font(15))
    dr.text((PAD, 84), "so a border would turn them into a judgement of the border. Mark each tile good, bad or "
                       "unsure in the capture sheet.", fill=MUTED, font=font(15))
    dr.text((PAD, 106), "good = both things the row's prompt names are present and each is clearly itself.",
            fill=MUTED, font=font(15))
    y = TOP
    for pair, a, b in pairs:
        rows = sorted([c for c in cells if c["pair"] == pair], key=lambda c: c["seed"])
        dr.text((PAD, y + TILE // 2 - 30), '"%s and %s"' % (a, b), fill=INK, font=font(19, True))
        dr.text((PAD, y + TILE // 2 + 0), pair, fill=MUTED, font=font(12))
        kind = "subject + scene" if any(w in b for w in ("field", "iceberg", "meadow", "pond", "landscape")) \
            else "object + object" if "mug" in a else "animal + animal"
        dr.text((PAD, y + TILE // 2 + 20), kind, fill=MUTED, font=font(13))
        for c, cell in enumerate(rows):
            x = LEFT + c * (TILE + PAD)
            dr.rectangle([x - 2, y - 2, x + TILE + 2, y + TILE + 2], fill=EDGE)
            im.paste(Image.open(cell["path"]).convert("RGB").resize((TILE, TILE), Image.LANCZOS), (x, y))
            dr.text((x, y + TILE + 8), "seed %d" % cell["seed"], fill=MUTED, font=font(14, True))
        y += TILE + CAP + PAD
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im.save(out)
    print("wrote", out, "|", len(cells), "tiles")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
