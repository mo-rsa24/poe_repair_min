"""Plan 07 task 2.2: six transfer sheets, one per (pair, rank), 200 steps, kappa 0.5.

Rows seeds 9-12, columns lambda_value in {0, 0.25, 0.5, 0.75, 1}. The lambda=0 column is plan
05's no-adapter kappa0.50 render (lambda_sweep tree); the rest come from the transfer tree.
Missing tiles are drawn as red boxes. Named to sit beside plan 05's kappa_050 sheets.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SD = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff")
BASE = SD / "lambda_sweep" / "pairs"      # lambda=0 column: superdiff_200steps_kappa0.50
XFER = SD / "transfer" / "grid" / "pairs"  # superdiff_200steps_kappa0.50_lorar{rank}_lv{lv}
OUT = Path("/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/paper/iclr/figures/how-much-is-added/across-composition-rules")
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = [("a_cat__x__a_dog", "cat_dog"), ("a_butterfly__x__a_flower_meadow", "butterfly_meadow")]
RANKS = [8, 16, 32]
SEEDS = [9, 10, 11, 12]
LVS = [0.0, 0.25, 0.5, 0.75, 1.0]
TILE, LEFT, TOP = 256, 90, 60

def font(size):
    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()
F_BIG, F_SMALL = font(22), font(18)

def tile_path(pair, seed, rank, lv):
    if lv == 0.0:
        name = "superdiff_200steps_kappa0.50"
        return BASE / pair / f"seed_{seed}" / name / f"{name}.png"
    name = f"superdiff_200steps_kappa0.50_lorar{rank}_lv{lv:.2f}"
    return XFER / pair / f"seed_{seed}" / name / f"{name}.png"

summary = json.load(open(SD / "transfer" / "transfer_summary.json")) if (SD / "transfer" / "transfer_summary.json").exists() else []
ckpt_by_rank = {r["rank"]: (r["checkpoint"], r["checkpoint_is_final"]) for r in summary}

for pair, short in PAIRS:
    for rank in RANKS:
        stem = f"{short}_grid_200_steps_kappa_050_lora_r{rank}"
        W, H = LEFT + TILE * len(LVS), TOP + TILE * len(SEEDS)
        sheet = Image.new("RGB", (W, H), "white"); draw = ImageDraw.Draw(sheet)
        tiles, missing = [], []
        for r, seed in enumerate(SEEDS):
            draw.text((8, TOP + r * TILE + TILE // 2 - 10), f"seed {seed}", fill="black", font=F_SMALL)
            for c, lv in enumerate(LVS):
                if r == 0:
                    draw.text((LEFT + c * TILE + TILE // 2 - 30, 18), f"λ = {lv:.2f}", fill="black", font=F_BIG)
                p = tile_path(pair, seed, rank, lv); x, y = LEFT + c * TILE, TOP + r * TILE
                if p.exists():
                    sheet.paste(Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS), (x, y))
                    tiles.append({"row_seed": seed, "col_lambda_value": lv, "source": str(p)})
                else:
                    draw.rectangle([x, y, x + TILE, y + TILE], outline="red", width=3)
                    draw.text((x + 20, y + TILE // 2), "missing", fill="red", font=F_SMALL); missing.append(str(p))
        sheet.save(OUT / f"{stem}.png")
        ck = ckpt_by_rank.get(rank, ("unknown", None))
        json.dump({
            "figure": f"{stem}.png",
            "what": "The PoE-trained LoRA injected into SuperDiff on every one of 200 steps at kappa 0.5: "
                    "eps_M(off) + lambda * (eps_M(on) - eps_M(off)). lambda 0 is the no-adapter render "
                    "from plan 05's kappa_050 sheet. Rows are seeds. Tiles are 1024x1024 shown at 256px.",
            "pair": pair, "rank": rank, "alpha": rank, "checkpoint": ck[0], "checkpoint_is_final": ck[1],
            "num_inference_steps": 200, "guidance_scale": 7.5, "kappa": 0.5,
            "rows_seeds": SEEDS, "cols_lambda_value": LVS,
            "read_against": f"{short}_grid_200_steps_kappa_050.png",
            "composer": "poe_repair/composers/superdiff.py, kappa_override=0.5, lora_adapter_name='lora', lambda_value=<col>",
            "tiles": tiles, "missing": missing,
        }, open(OUT / f"{stem}.json", "w"), indent=2)
        print(f"{stem}.png  tiles={len(tiles)} missing={len(missing)}")
print("sheets in", OUT)
