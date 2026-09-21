"""Plan 08 task 3.2: six sheets, one per (pair, rank), for the SuperDiff-trained adapters.

Same layout as plan 05's kappa_050 sheets and plan 07's lora_r* sheets: rows seeds 9-12,
columns lambda_value in {0, 0.25, 0.5, 0.75, 1}. The lambda=0 column is plan 05's no-adapter
kappa0.50 render. Named *_sdlora_r{rank}.png so the three families sit side by side in the
figures folder. Missing tiles are drawn as red boxes. Run on the node that rendered the cells
(the session node's NFS view can lag); OUT may be overridden for an early look.
"""
import json, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SD = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff")
BASE = SD / "lambda_sweep" / "pairs"
XFER = SD / "transfer_sdlora" / "grid" / "pairs"
OUT = Path(os.environ.get("OUT", "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/paper/iclr/figures/how-much-is-added/across-composition-rules"))
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = [("a_cat__x__a_dog", "cat_dog"), ("a_butterfly__x__a_flower_meadow", "butterfly_meadow")]
RANKS = [8, 16, 32]; SEEDS = [9, 10, 11, 12]; LVS = [0.0, 0.25, 0.5, 0.75, 1.0]
TILE, LEFT, TOP = 256, 90, 60

def font(size):
    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()
F_BIG, F_SMALL = font(22), font(18)

def tile_path(pair, seed, rank, lv):
    if lv == 0.0:
        name = "superdiff_200steps_kappa0.50"; return BASE / pair / f"seed_{seed}" / name / f"{name}.png"
    name = f"superdiff_200steps_kappa0.50_lorasdr{rank}_lv{lv:.2f}"
    return XFER / pair / f"seed_{seed}" / name / f"{name}.png"

def final_ckpt(rank):
    d = SD / f"sdlora_r{rank}_100k" / "checkpoints"
    c = sorted(d.glob("lora_step_1000*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    return str(c[-1]) if c else None

for pair, short in PAIRS:
    for rank in RANKS:
        stem = f"{short}_grid_200_steps_kappa_050_sdlora_r{rank}"
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
        json.dump({
            "figure": f"{stem}.png",
            "what": "A LoRA trained on SuperDiff's own residual (eps_J - eps_M at kappa 0.5, along "
                    "SuperDiff's 200-step trajectories) injected into SuperDiff on every step: "
                    "eps_M(off) + lambda * (eps_M(on) - eps_M(off)). lambda 0 is the no-adapter render. "
                    "Rows are seeds. Tiles are 1024x1024 shown at 256px.",
            "pair": pair, "rank": rank, "alpha": rank, "checkpoint": final_ckpt(rank),
            "num_inference_steps": 200, "guidance_scale": 7.5, "kappa": 0.5,
            "rows_seeds": SEEDS, "cols_lambda_value": LVS,
            "read_against": [f"{short}_grid_200_steps_kappa_050.png", f"{short}_grid_200_steps_kappa_050_lora_r{rank}.png"],
            "composer": "poe_repair/composers/superdiff.py, kappa_override=0.5, lora_adapter_name='lora', lambda_value=<col>",
            "tiles": tiles, "missing": missing,
        }, open(OUT / f"{stem}.json", "w"), indent=2)
        print(f"{stem}.png  tiles={len(tiles)} missing={len(missing)}")
print("sheets in", OUT)
