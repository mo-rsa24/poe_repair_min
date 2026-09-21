"""Early look at a SuperDiff-trained adapter from an intermediate checkpoint: one pair, one rank,
seeds 9-12 x lambda_value {0.25,0.5,0.75,1}, 200 steps, kappa 0.5. Not a plan deliverable."""
import sys, time, json, argparse
sys.path.insert(0, "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min")
from pathlib import Path
from types import SimpleNamespace
import torch
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, infer_device, infer_dtype
from poe_repair.composers import superdiff
from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig
from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, required=True); ap.add_argument("--checkpoint", required=True)
ap.add_argument("--pair", default="a_cat__x__a_dog"); ap.add_argument("--prompt-a", default="a cat"); ap.add_argument("--prompt-b", default="a dog")
ap.add_argument("--tag", required=True); a = ap.parse_args()
SD = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff")
device = infer_device(None); dtype = infer_dtype("fp16", device)
models = superdiff._load_superdiff_models(device, dtype); unet = models["unet"]
cfg = SimpleNamespace(lora=LoRAConfig(rank=a.rank, alpha=a.rank, dropout=0.0, target_modules=("attn2.to_q","attn2.to_k","attn2.to_v"), init="gaussian", adapter_name="lora"))
info = lora_trainer.attach_lora(unet, cfg); lora_trainer.load_lora_state(unet, torch.load(a.checkpoint, map_location="cpu", weights_only=False)["lora_state"])
print("attached", info["n_matched"], "from", a.checkpoint, flush=True)
t0 = time.perf_counter(); n = 0
for seed in (9, 10, 11, 12):
    for lv in (0.25, 0.5, 0.75, 1.0):
        n += 1
        cell = PairSeedCell(pair_dir=None, pair_slug=a.pair, prompt_a=a.prompt_a, prompt_b=a.prompt_b, seed=seed, regime="preview", height=1024, width=1024, grid_assets={})
        ctx = MethodCtx(models={}, scheduler=None, output_root=SD / "preview_sdlora", device=device, dtype=dtype, guidance_scale=7.5, num_inference_steps=200)
        p = superdiff.run(cell, ctx, exp_name=a.tag, kappa_clamp=False, kappa_override=0.5, lora_adapter_name="lora", lambda_value=lv, lora_tag=a.tag)
        print(f"[{n}/16] seed={seed} lv={lv:.2f} elapsed={(time.perf_counter()-t0)/60:.1f}min", flush=True)
print("PREVIEW DONE", flush=True)
