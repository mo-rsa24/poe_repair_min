"""Quantitative figures for the three SuperDiff findings in report/is-the-gap-the-samplers-or-the-models/.

1. superdiff-baseline-eight-cells.png: the 8-cell grid (2 pairs x 2 step counts x 2 kappa-clamp
   settings, seed 9) from corrector/superdiff/parity/grid/.
2. poe-adapter-in-superdiff-cosine-profile.png: per-step cosine between the rank-8 PoE-trained
   adapter's correction and SuperDiff's missing residual on one cell, with both norms, from the
   transfer_diag sidecar.
3. superdiff-lora-separation-by-checkpoint.png: seeds (of 4) that separate per checkpoint per rank,
   eye reads recorded in plan 08's review file; stated, not detector-scored.
Every number drawn is written to superdiff-report-figures.json beside the images.
"""
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
SD = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff")
A = Path("/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/results/is-the-gap-the-samplers-or-the-models")
out = {}
# dataviz reference palette, first categorical slots
C = ["#4C6EF5", "#F76707", "#2F9E44", "#AE3EC9"]

# 1. eight cells
def font(s):
    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(c).exists(): return ImageFont.truetype(c, s)
    return ImageFont.load_default()
F = font(18); T, LEFT, TOP = 256, 200, 50
cols = [("200 steps, clamped", "superdiff_200steps_clamped"), ("200 steps, unclamped", "superdiff_200steps_unclamped"),
        ("50 steps, clamped", "superdiff_50steps_clamped"), ("50 steps, unclamped", "superdiff_50steps_unclamped")]
rows = [("a_cat__x__a_dog", "cat × dog, seed 9"), ("a_butterfly__x__a_flower_meadow", "butterfly × meadow, seed 9")]
sheet = Image.new("RGB", (LEFT + T * 4, TOP + T * 2), "white"); d = ImageDraw.Draw(sheet)
tiles = []
for c, (lab, name) in enumerate(cols): d.text((LEFT + c * T + 10, 20), lab, fill="black", font=F)
for r, (pair, lab) in enumerate(rows):
    d.text((8, TOP + r * T + T // 2 - 10), lab, fill="black", font=F)
    for c, (_, name) in enumerate(cols):
        p = SD / "parity" / "grid" / "pairs" / pair / "seed_9" / name / f"{name}.png"
        sheet.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (LEFT + c * T, TOP + r * T)); tiles.append(str(p))
sheet.save(A / "superdiff-baseline-eight-cells.png"); out["eight_cells_tiles"] = tiles

# 2. cosine profile
sc = json.load(open(SD / "transfer_diag/cosine/pairs/a_cat__x__a_dog/seed_9/superdiff_200steps_kappa0.50_with_rt_lorar8_lv1.00/superdiff_200steps_kappa0.50_with_rt_lorar8_lv1.00.json"))
cos = np.array(sc["delta_hat_cos_r_t"]); dh = np.array(sc["delta_hat_norms"]); rt = np.array(sc["r_t_sd_norms"]); steps = np.arange(len(cos))
buckets = [(0, 20), (20, 100), (100, 180), (180, 200)]
out["cosine"] = {"checkpoint": sc.get("lora_checkpoint") or sc.get("checkpoint") or "lora_step_420000.pt (see review)",
                 "median_cos_by_bucket": {f"{a}-{b-1}": float(np.median(cos[a:b])) for a, b in buckets},
                 "min_cos": float(cos.min()), "max_cos": float(cos.max()),
                 "median_delta_hat_norm_by_bucket": {f"{a}-{b-1}": float(np.median(dh[a:b])) for a, b in buckets},
                 "median_r_t_sd_norm_by_bucket": {f"{a}-{b-1}": float(np.median(rt[a:b])) for a, b in buckets}, "n_steps": int(len(cos))}
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(steps, cos, color=C[0], lw=1); ax[0].axhline(0, color="grey", lw=0.8)
for a, b in buckets: ax[0].hlines(np.median(cos[a:b]), a, b - 1, color=C[1], lw=2.5)
ax[0].set_ylim(-1, 1); ax[0].set_xlabel("SuperDiff step (0 = pure noise, 199 = image)"); ax[0].set_ylabel("cosine(Δ̂, r_t^SD)")
ax[0].set_title("Direction: adapter correction vs SuperDiff's missing residual\n(orange: median per step bucket)")
ax[1].plot(steps, dh, color=C[0], label="‖Δ̂‖, the rank-8 PoE adapter's correction"); ax[1].plot(steps, rt, color=C[2], label="‖r_t^SD‖ = ‖ε_J − ε_M‖, what SuperDiff is missing")
ax[1].set_yscale("log"); ax[1].set_xlabel("SuperDiff step"); ax[1].set_ylabel("L2 norm of the per-step prediction difference"); ax[1].set_title("Size: right order of magnitude throughout"); ax[1].legend(fontsize=8)
fig.suptitle("cat × dog, seed 9, κ 0.5, 200 steps, λ 1, from the transfer_diag sidecar", fontsize=10); fig.tight_layout()
fig.savefig(A / "poe-adapter-in-superdiff-cosine-profile.png", dpi=120)

# 3. separation by checkpoint (eye reads from plan 08's review, seeds of 4)
reads = {"cat_dog_two_bodies": {8: {10: 3, 20: 4, 30: 4, 40: 4, 50: 3, 60: 3}, 16: {10: 4, 20: 4, 30: 4, 40: 3, 50: 0, 60: 0}, 32: {10: 4, 20: 0, 30: 0}},
         "butterfly_clear":    {8: {10: 2, 20: 2, 30: 4, 40: 0, 50: 0, 60: 1}, 16: {10: 1, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0}, 32: {10: 0, 20: 0, 30: 0}}}
out["separation_by_checkpoint_eye_reads"] = reads
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for k, (key, title) in enumerate([("cat_dog_two_bodies", "cat × dog: seeds showing two bodies"), ("butterfly_clear", "butterfly × meadow: seeds with a clear butterfly")]):
    for i, rank in enumerate((8, 16, 32)):
        xs = sorted(reads[key][rank]); ax[k].plot(xs, [reads[key][rank][x] for x in xs], "o-", color=C[i], label=f"rank {rank}")
    ax[k].axhline(0, color="grey", lw=0.5); ax[k].set_ylim(-0.3, 4.3); ax[k].set_yticks([0, 1, 2, 3, 4]); ax[k].set_xlabel("checkpoint, thousand optimizer steps"); ax[k].set_title(title, fontsize=10)
ax[0].set_ylabel("seeds of 4 (seeds 9 to 12), eye read"); ax[0].legend()
fig.suptitle("SuperDiff-residual LoRA at λ 1 on every step, κ 0.5, 200 steps: how many seeds separate, per checkpoint", fontsize=10); fig.tight_layout()
fig.savefig(A / "superdiff-lora-separation-by-checkpoint.png", dpi=120)
json.dump(out, open(A / "superdiff-report-figures.json", "w"), indent=2)
print(json.dumps(out["cosine"], indent=1))
