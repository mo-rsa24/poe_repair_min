"""Plan 07 task 1.2's done-when, run once the GPU is free.

(a) attach each available checkpoint to SuperDiff's UNet and print the matched count;
(b) with the rank-8 adapter attached and lambda_value=0, the render must be byte-identical to
    plan 05's superdiff_200steps_kappa0.50 render for the same pair/seed (the off pass alone is
    the baseline);
(c) with lambda_value=1, ||delta_hat|| must be non-zero on every step.
Uses 20 steps for (b)/(c) against a fresh 20-step no-adapter render, so it costs ~1 min; the
200-step byte check is repeated by the driver's first cell.
"""
import sys, json, hashlib
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
FALLBACK_R8 = "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/results/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt"

device = infer_device(None); dtype = infer_dtype("fp16", device)
models = superdiff._load_superdiff_models(device, dtype)
unet = models["unet"]

def attach(rank, path):
    cfg = SimpleNamespace(lora=LoRAConfig(rank=rank, alpha=rank, dropout=0.0,
        target_modules=("attn2.to_q", "attn2.to_k", "attn2.to_v"), init="gaussian", adapter_name="lora"))
    # a previous rank's adapter (same name 'lora') must go before the next attaches
    if hasattr(unet, 'delete_adapters'):
        try: unet.delete_adapters('lora')
        except Exception: pass
    info = lora_trainer.attach_lora(unet, cfg)
    state = torch.load(path, map_location="cpu", weights_only=False)["lora_state"]
    lora_trainer.load_lora_state(unet, state)
    return info["n_matched"], len(state)

# (a) counts, on whichever checkpoints exist
for rank, path in CKPTS.items():
    print(f"rank {rank}: matched={attach(rank, path)[0]}  ({path}, final={CKPT_IS_FINAL[rank]})")

# (b)/(c) with rank 8 (final if present, else the 100k fallback)
path8 = CKPTS[8]
n, _ = attach(8, path8)
print("rank 8 attached for the inertness check, matched =", n, "from", Path(path8).name)

cell = PairSeedCell(pair_dir=None, pair_slug="a_cat__x__a_dog", prompt_a="a cat", prompt_b="a dog",
                    seed=9, regime="test", height=1024, width=1024, grid_assets={})
out = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff/inert_check")
ctx = MethodCtx(models={}, scheduler=None, output_root=out, device=device, dtype=dtype,
                guidance_scale=7.5, num_inference_steps=20)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]

base = superdiff.run(cell, ctx, exp_name="t", kappa_clamp=False, kappa_override=0.5, overwrite=True)
lv0 = superdiff.run(cell, ctx, exp_name="t", kappa_clamp=False, kappa_override=0.5,
                    lora_adapter_name="lora", lambda_value=0.0, lora_tag="r8", overwrite=True)
lv1 = superdiff.run(cell, ctx, exp_name="t", kappa_clamp=False, kappa_override=0.5,
                    lora_adapter_name="lora", lambda_value=1.0, lora_tag="r8", overwrite=True)
print("baseline sha:", sha(base), "| lambda_value=0 sha:", sha(lv0), "| identical:", sha(base) == sha(lv0))
d = json.load(open(lv1.with_suffix(".json")))["delta_hat_norms"]
print("lambda_value=1 ||delta_hat|| per step: n=", len(d), "min=", round(min(d), 3), "max=", round(max(d), 3), "any zero:", any(v == 0 for v in d))
