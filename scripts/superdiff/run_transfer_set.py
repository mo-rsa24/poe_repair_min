"""Plan 07 task 2.1: the 96-cell transfer set.

rank in {8, 16, 32} x pair in {cat x dog, butterfly x meadow} x seed in {9..12} x
lambda_value in {0.25, 0.5, 0.75, 1}, 200 steps, kappa 0.5, adapter on for every step.
lambda_value = 0 is plan 05's kappa_050 render, reused by the assembler. Every cell is cached,
so a restart resumes. One adapter attached at a time; attaching a new rank re-attaches on the
same UNet, so the loop is rank-outermost.
"""
import sys, time, json
sys.path.insert(0, "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min")
from pathlib import Path
from types import SimpleNamespace
import torch
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, infer_device, infer_dtype
from poe_repair.composers import superdiff
from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig
from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer

def _latest(run_dir, final_name):
    """The final checkpoint if it exists, else the newest on disk; returns (path, is_final)."""
    from pathlib import Path
    d = Path(run_dir) / "checkpoints"
    final = d / final_name
    if final.exists():
        return str(final), True
    cands = sorted(d.glob("lora_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    return (str(cands[-1]), False) if cands else (None, False)

_SHOW = "/datasets/mmolefe/poe_repair_min/outputs/showcase"
CKPTS = {}
CKPT_IS_FINAL = {}
for rank, run, final in ((8, "phase1_r8_450k", "lora_step_450000.pt"),
                         (16, "phase1_r16_100k", "lora_step_100000.pt"),
                         (32, "phase1_r32_100k", "lora_step_100000.pt")):
    CKPTS[rank], CKPT_IS_FINAL[rank] = _latest(f"{_SHOW}/{run}", final)
PAIRS = [("a_cat__x__a_dog", "a cat", "a dog"),
         ("a_butterfly__x__a_flower_meadow", "a butterfly", "a flower meadow")]
SEEDS = [9, 10, 11, 12]
LAMBDA_VALUES = [0.25, 0.5, 0.75, 1.0]
STEPS, KAPPA = 200, 0.5
output_root = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/transfer")

missing = [r for r, p in CKPTS.items() if p is None]
if missing:
    sys.exit(f"no checkpoint at all for ranks {missing}")
for r in CKPTS: print(f"rank {r}: {CKPTS[r]}  final={CKPT_IS_FINAL[r]}", flush=True)

device = infer_device(None); dtype = infer_dtype("fp16", device)
models = superdiff._load_superdiff_models(device, dtype)
unet = models["unet"]

results, n, t_start = [], 0, time.perf_counter()
total = len(CKPTS) * len(PAIRS) * len(SEEDS) * len(LAMBDA_VALUES)
for rank, path in CKPTS.items():
    cfg = SimpleNamespace(lora=LoRAConfig(rank=rank, alpha=rank, dropout=0.0,
        target_modules=("attn2.to_q", "attn2.to_k", "attn2.to_v"), init="gaussian", adapter_name="lora"))
    # a previous rank's adapter (same name 'lora') must go before the next attaches
    if hasattr(unet, 'delete_adapters'):
        try: unet.delete_adapters('lora')
        except Exception: pass
    info = lora_trainer.attach_lora(unet, cfg)
    state = torch.load(path, map_location="cpu", weights_only=False)["lora_state"]
    lora_trainer.load_lora_state(unet, state)
    if info["n_matched"] == 0:
        sys.exit(f"rank {rank}: zero matched modules, refusing to render the baseline under a new name")
    print(f"rank {rank}: attached, matched={info['n_matched']}, tensors={len(state)}", flush=True)
    for pair_slug, prompt_a, prompt_b in PAIRS:
        for seed in SEEDS:
            for lv in LAMBDA_VALUES:
                n += 1
                cell = PairSeedCell(pair_dir=None, pair_slug=pair_slug, prompt_a=prompt_a, prompt_b=prompt_b,
                                    seed=seed, regime="transfer", height=1024, width=1024, grid_assets={})
                ctx = MethodCtx(models={}, scheduler=None, output_root=output_root, device=device,
                                dtype=dtype, guidance_scale=7.5, num_inference_steps=STEPS)
                t0 = time.perf_counter()
                p = superdiff.run(cell, ctx, exp_name="grid", kappa_clamp=False, kappa_override=KAPPA,
                                  lora_adapter_name="lora", lambda_value=lv, lora_tag=f"r{rank}")
                wall = time.perf_counter() - t0
                print(f"[{n:2d}/{total}] r{rank} {pair_slug} seed={seed} lv={lv:.2f} wall={wall:.1f}s "
                      f"elapsed={(time.perf_counter()-t_start)/3600:.2f}h", flush=True)
                results.append({"rank": rank, "checkpoint": path, "checkpoint_is_final": CKPT_IS_FINAL[rank], "pair_slug": pair_slug, "seed": seed, "lambda_value": lv,
                                "wall_time_s": wall, "image_path": str(p)})

summary = output_root / "transfer_summary.json"
summary.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(summary, "w"), indent=2)
print("ALL", total, "CELLS DONE. Summary at", summary, flush=True)
