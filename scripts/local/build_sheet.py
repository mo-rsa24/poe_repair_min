"""A contact sheet of whatever has landed: a row per pair, a column per seed, empty where missing.

Used by conveyor.sh while renders are still arriving, so the sheet can be opened at any moment.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
root, out = Path(sys.argv[1]), Path(sys.argv[2])
seeds = [2, 9, 11, 42]
pairs = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "rejected")
if not pairs: raise SystemExit("nothing yet")
T, G, L, TOP = 300, 6, 260, 30
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
c = Image.new("RGB", (L + len(seeds) * (T + G), TOP + len(pairs) * (T + G)), "white")
d = ImageDraw.Draw(c)
for j, s in enumerate(seeds):
    d.text((L + j * (T + G) + 4, 8), f"seed {s}", fill="black", font=f)
for i, name in enumerate(pairs):
    y = TOP + i * (T + G)
    d.text((6, y + T // 2 - 10), name.replace("__x__", " + ").replace("_", " "), fill="black", font=f)
    for j, s in enumerate(seeds):
        p = root / name / f"seed{s:02d}" / "mono.png"
        x = L + j * (T + G)
        if p.exists():
            c.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (x, y))
        else:
            d.rectangle((x, y, x + T, y + T), outline="#cccccc")
c.save(out)
print(out)
