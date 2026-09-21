"""One contact sheet per (pair, kappa setting) at 200 steps: rows seeds 9-12, columns lambda.

Named the way the figure set was asked for: cat_dog_grid_200_steps_kappa_050.png,
cat_dog_grid_200_steps_kappa_balanced.png, butterfly_meadow_grid_200_steps_kappa_050.png.
Labels are drawn on the sheet. A missing tile is drawn as a red box, never silently skipped,
so a partial sweep is visible as partial. Each sheet gets a .json sidecar.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SWEEP = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/lambda_sweep")
OUT = Path("/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/paper/iclr/figures/how-much-is-added/across-composition-rules")
OUT.mkdir(parents=True, exist_ok=True)

STEPS = 200
SEEDS = [9, 10, 11, 12]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FIGURES = [("a_cat__x__a_dog", "cat_dog", k) for k in [None, 0.0, 0.25, 0.5, 0.75, 1.0]] + \
          [("a_butterfly__x__a_flower_meadow", "butterfly_meadow", 0.5)]
TILE, LEFT, TOP = 256, 90, 60

def font(size):
    for cand in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()
F_BIG, F_SMALL = font(22), font(18)

def cell_name(kappa, lam):
    ktag = "unclamped" if kappa is None else f"kappa{kappa:.2f}"
    name = f"superdiff_{STEPS}steps_{ktag}"
    if lam != 0.0:
        name += f"_lam{lam:.2f}"
    return name

for pair_slug, short, kappa in FIGURES:
    ktag_file = "balanced" if kappa is None else f"{int(round(kappa*100)):03d}"
    stem = f"{short}_grid_{STEPS}_steps_kappa_{ktag_file}"
    W, H = LEFT + TILE * len(LAMBDAS), TOP + TILE * len(SEEDS)
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    tiles, missing = [], []
    for r, seed in enumerate(SEEDS):
        draw.text((8, TOP + r * TILE + TILE // 2 - 10), f"seed {seed}", fill="black", font=F_SMALL)
        for c, lam in enumerate(LAMBDAS):
            if r == 0:
                draw.text((LEFT + c * TILE + TILE // 2 - 30, 18), f"λ = {lam:.2f}", fill="black", font=F_BIG)
            name = cell_name(kappa, lam)
            p = SWEEP / "pairs" / pair_slug / f"seed_{seed}" / name / f"{name}.png"
            x, y = LEFT + c * TILE, TOP + r * TILE
            if p.exists():
                sheet.paste(Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS), (x, y))
                tiles.append({"row_seed": seed, "col_lambda": lam, "source": str(p)})
            else:
                draw.rectangle([x, y, x + TILE, y + TILE], outline="red", width=3)
                draw.text((x + 20, y + TILE // 2), "missing", fill="red", font=F_SMALL)
                missing.append(str(p))
    sheet.save(OUT / f"{stem}.png")
    json.dump({
        "figure": f"{stem}.png",
        "what": "SuperDiff (AND) with a fixed kappa per sheet; columns add back a fraction lambda of "
                "r_t^SD = eps_J - eps_M (lambda 0 = SuperDiff alone, lambda 1 = the joint prompt's "
                "prediction exactly); rows are seeds. Tiles are 1024x1024 renders shown at 256px.",
        "pair": pair_slug, "num_inference_steps": STEPS, "guidance_scale": 7.5,
        "kappa": "balanced (pipeline's own, unclamped)" if kappa is None else kappa,
        "rows_seeds": SEEDS, "cols_lambda": LAMBDAS,
        "composer": "poe_repair/composers/superdiff.py, kappa_override=<kappa>, lam=<col value>",
        "tiles": tiles, "missing": missing,
    }, open(OUT / f"{stem}.json", "w"), indent=2)
    print(f"{stem}.png  tiles={len(tiles)} missing={len(missing)}")
print("sheets in", OUT)
