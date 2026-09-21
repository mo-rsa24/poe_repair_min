"""One cell with the adapter AND r_t^SD formed, so the sidecar carries cosine(delta_hat, r_t^SD)
per step. cat x dog seed 9, rank 8 (latest checkpoint), lambda_value=1, kappa 0.5, 200 steps."""
import sys, json
sys.path.insert(0, "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min")
from pathlib import Path
from types import SimpleNamespace
import torch
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, infer_device, infer_dtype
from poe_repair.composers import superdiff
from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig
from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
d = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r8_450k/checkpoints")
ck = sorted(d.glob("lora_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[-1]
device = infer_device(None); dtype = infer_dtype("fp16", device)
models = superdiff._load_superdiff_models(device, dtype); unet = models["unet"]
cfg = SimpleNamespace(lora=LoRAConfig(rank=8, alpha=8, dropout=0.0, target_modules=("attn2.to_q","attn2.to_k","attn2.to_v"), init="gaussian", adapter_name="lora"))
lora_trainer.attach_lora(unet, cfg); lora_trainer.load_lora_state(unet, torch.load(ck, map_location="cpu", weights_only=False)["lora_state"])
cell = PairSeedCell(pair_dir=None, pair_slug="a_cat__x__a_dog", prompt_a="a cat", prompt_b="a dog", seed=9, regime="diag", height=1024, width=1024, grid_assets={})
ctx = MethodCtx(models={}, scheduler=None, output_root=Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/transfer_diag"), device=device, dtype=dtype, guidance_scale=7.5, num_inference_steps=200)
p = superdiff.run(cell, ctx, exp_name="cosine", kappa_clamp=False, kappa_override=0.5, lora_adapter_name="lora", lambda_value=1.0, lora_tag="r8", form_r_t=True, overwrite=True)
s = json.load(open(p.with_suffix(".json"))); c = s["delta_hat_cos_r_t"]
import statistics as st
print("checkpoint:", ck.name, "| n:", len(c))
for name, sl in (("0-19", slice(0,20)), ("20-99", slice(20,100)), ("100-179", slice(100,180)), ("180-199", slice(180,200))):
    print(f"steps {name:8s} median cos(delta_hat, r_t^SD) = {st.median(c[sl]):+.3f}   min {min(c[sl]):+.3f} max {max(c[sl]):+.3f}")
print("DONE", p, flush=True)
