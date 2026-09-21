"""One timeline sheet per (rank, pair): rows = Mono target, SuperDiff kappa 0.5 alone, then one row
per rendered checkpoint (LoRA at lambda 1 on every step); columns = seeds 9-12. Built from the
tiles the sample watcher already rendered. Output: <SD>/timelines/<pair>_r<rank>_timeline.png"""
import sys, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
SD = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff")
BASE = SD / "lambda_sweep" / "pairs"; SAMP = SD / "samples_sdlora"; OUT = SD / "timelines"; OUT.mkdir(exist_ok=True)
PAIRS = [("a_cat__x__a_dog", "cat_dog"), ("a_butterfly__x__a_flower_meadow", "butterfly_meadow")]
SEEDS = [9, 10, 11, 12]; T = 220; LEFT = 150; TOP = 50
def font(s):
    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(c).exists(): return ImageFont.truetype(c, s)
    return ImageFont.load_default()
F = font(18)
for rank in (8, 16, 32):
    steps = sorted({int(p.name.split("_")[1]) for p in (SD / f"sdlora_r{rank}_100k" / "samples" / "superdiff").glob("step_*_done.json")})
    for pair, short in PAIRS:
        rows = [("Mono (target)", lambda s: BASE / pair / f"seed_{s}" / "superdiff_200steps_kappa0.50_lam1.00" / "superdiff_200steps_kappa0.50_lam1.00.png"),
                ("SuperDiff κ=0.5", lambda s: BASE / pair / f"seed_{s}" / "superdiff_200steps_kappa0.50" / "superdiff_200steps_kappa0.50.png")]
        for st in steps:
            n = f"superdiff_200steps_kappa0.50_lorasdr{rank}s{st}_lv1.00"
            rows.append((f"LoRA λ=1 @ {st//1000}k", lambda s, n=n, st=st: SAMP / f"sdr{rank}s{st}" / "pairs" / pair / f"seed_{s}" / n / f"{n}.png"))
        W, H = LEFT + T * len(SEEDS), TOP + T * len(rows)
        sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
        d.text((8, 8), f"{pair}   rank {rank}   200 steps, κ 0.5", fill="black", font=F)
        for c, s in enumerate(SEEDS): d.text((LEFT + c * T + T // 2 - 30, 28), f"seed {s}", fill="black", font=F)
        for r, (label, fn) in enumerate(rows):
            d.text((8, TOP + r * T + T // 2 - 10), label, fill="black", font=F)
            for c, s in enumerate(SEEDS):
                p = fn(s); x, y = LEFT + c * T, TOP + r * T
                if p.exists(): sheet.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (x, y))
                else: d.rectangle([x, y, x + T, y + T], outline="red", width=3)
        fp = OUT / f"{short}_r{rank}_timeline.png"; sheet.save(fp); print(fp, "rows", len(rows))
