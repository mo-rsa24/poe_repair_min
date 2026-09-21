"""The kappa x lambda figure set, 200 steps only.

cat x dog: six kappa settings (balanced = pipeline's own kappa, unclamped; then forced 0, 0.25,
0.5, 0.75, 1). butterfly x meadow: kappa 0.5 only. Every figure is seeds 9-12 (rows) by
lambda in {0, 0.25, 0.5, 0.75, 1} (columns), lambda being the fraction of r_t^SD added back.
140 renders. Ordered so the most informative figures land first. Every cell is cached by
run(), so a restart resumes.
"""
import sys, time, json
sys.path.insert(0, "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min")
from pathlib import Path
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, infer_device, infer_dtype
from poe_repair.composers import superdiff

device = infer_device(None)
dtype = infer_dtype("fp16", device)
output_root = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff")
STEPS = 200
SEEDS = [9, 10, 11, 12]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CAT_DOG = ("a_cat__x__a_dog", "a cat", "a dog")
BUTTERFLY = ("a_butterfly__x__a_flower_meadow", "a butterfly", "a flower meadow")

# (pair, kappa setting) in the order the figures should land. None = balanced.
FIGURES = [
    (CAT_DOG, None), (CAT_DOG, 0.5), (BUTTERFLY, 0.5),
    (CAT_DOG, 0.0), (CAT_DOG, 0.25), (CAT_DOG, 0.75), (CAT_DOG, 1.0),
]

results = []
total = len(FIGURES) * len(SEEDS) * len(LAMBDAS)
n = 0
t_start = time.perf_counter()
for (pair_slug, prompt_a, prompt_b), kappa in FIGURES:
    for seed in SEEDS:
        for lam in LAMBDAS:
            n += 1
            cell = PairSeedCell(pair_dir=None, pair_slug=pair_slug, prompt_a=prompt_a,
                                prompt_b=prompt_b, seed=seed, regime="lambda_sweep",
                                height=1024, width=1024, grid_assets={})
            ctx = MethodCtx(models={}, scheduler=None, output_root=output_root,
                            device=device, dtype=dtype, guidance_scale=7.5,
                            num_inference_steps=STEPS)
            t0 = time.perf_counter()
            path = superdiff.run(cell, ctx, exp_name="lambda_sweep",
                                 kappa_clamp=False, kappa_override=kappa, lam=lam)
            wall = time.perf_counter() - t0
            ktag = "balanced" if kappa is None else f"{kappa:.2f}"
            print(f"[{n:3d}/{total}] {pair_slug} kappa={ktag} seed={seed} lam={lam:.2f} "
                  f"wall={wall:.1f}s elapsed={(time.perf_counter()-t_start)/3600:.2f}h", flush=True)
            results.append({"pair_slug": pair_slug, "kappa": kappa, "seed": seed, "lam": lam,
                            "steps": STEPS, "wall_time_s": wall, "image_path": str(path)})

summary_path = output_root / "lambda_sweep" / "sweep_summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(summary_path, "w"), indent=2)
print("ALL", total, "CELLS DONE. Summary at", summary_path, flush=True)
