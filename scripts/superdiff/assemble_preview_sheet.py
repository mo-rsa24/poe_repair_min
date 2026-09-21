"""One early-look sheet from preview_sdlora.py: rows seeds 9-12, columns lambda_value
{0, 0.25, 0.5, 0.75, 1}; the lambda=0 column is plan 05's no-adapter kappa 0.50 render.
Usage: python assemble_preview_sheet.py <tag> <rank> <step> [pair_slug short]"""
import sys, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
tag, rank, step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pair = sys.argv[4] if len(sys.argv) > 4 else "a_cat__x__a_dog"; short = sys.argv[5] if len(sys.argv) > 5 else "cat_dog"
SD = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff")
BASE = SD / "lambda_sweep" / "pairs"; PREV = SD / "preview_sdlora" / tag / "pairs"; OUT = SD / "preview_sdlora"
SEEDS = [9, 10, 11, 12]; LVS = [0.0, 0.25, 0.5, 0.75, 1.0]; TILE, LEFT, TOP = 256, 90, 60
def font(s):
    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(c).exists(): return ImageFont.truetype(c, s)
    return ImageFont.load_default()
FB, FS = font(22), font(18)
def tile(seed, lv):
    if lv == 0.0:
        n = "superdiff_200steps_kappa0.50"; return BASE / pair / f"seed_{seed}" / n / f"{n}.png"
    n = f"superdiff_200steps_kappa0.50_lora{tag}_lv{lv:.2f}"; return PREV / pair / f"seed_{seed}" / n / f"{n}.png"
stem = f"{short}_grid_200_steps_kappa_050_sdlora_r{rank}_step{step//1000}k_preview"
W, H = LEFT + TILE * len(LVS), TOP + TILE * len(SEEDS)
sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet); tiles, missing = [], []
for r, seed in enumerate(SEEDS):
    d.text((8, TOP + r * TILE + TILE // 2 - 10), f"seed {seed}", fill="black", font=FS)
    for c, lv in enumerate(LVS):
        if r == 0: d.text((LEFT + c * TILE + TILE // 2 - 30, 18), f"λ = {lv:.2f}", fill="black", font=FB)
        p = tile(seed, lv); x, y = LEFT + c * TILE, TOP + r * TILE
        if p.exists():
            sheet.paste(Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS), (x, y)); tiles.append(str(p))
        else:
            d.rectangle([x, y, x + TILE, y + TILE], outline="red", width=3); d.text((x + 20, y + TILE // 2), "missing", fill="red", font=FS); missing.append(str(p))
sheet.save(OUT / f"{stem}.png")
json.dump({"figure": f"{stem}.png", "what": f"Early look: rank {rank} SuperDiff-residual LoRA at optimizer step {step} of 100000, injected into SuperDiff on every step, eps_M(off)+lambda*(eps_M(on)-eps_M(off)), 200 steps, kappa 0.5. Not a plan deliverable.", "pair": pair, "rows_seeds": SEEDS, "cols_lambda_value": LVS, "tiles": tiles, "missing": missing}, open(OUT / f"{stem}.json", "w"), indent=2)
print(OUT / f"{stem}.png", "tiles", len(tiles), "missing", len(missing))
