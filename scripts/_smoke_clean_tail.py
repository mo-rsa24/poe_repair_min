"""Smoke for clean_tail_k_sweep: the real model, the real checkpoint, the real sampler call,
cut to 10 steps and one cell. Proves attach, the lambda_window flag, decode and the scorers."""
import sys, time, gc
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "showcase"))
import torch
import clean_tail_k_sweep as S
from poe_repair.composers._helpers import encode_pair, init_latents_for_cell, get_joint_embeds
from poe_repair.experiments.interaction_term.cell import cell_from_slug
from poe_repair.methods._sampling import run_cfg, write_decoded_image
from poe_repair.methods._poe_langevin import run_lora_langevin_windowed_poe
from poe_repair.run import make_ctx
import lambda_boundary_probe as lbp

S._device_guard(); S._disk_guard(S.OUT)
out = Path("/datasets/mmolefe/poe_repair_min/outputs/clean_tail_k_sweep/_smoke"); out.mkdir(parents=True, exist_ok=True)
STEPS = 10
ctx = make_ctx(num_inference_steps=STEPS)
pair, seed = "an_elephant__x__a_penguin", 10
cell = cell_from_slug(pair, seed)
print("cell", cell.pair_slug, cell.height, cell.width, "prompts", cell.prompt_a, "/", cell.prompt_b, flush=True)
init_latents, euler_sigma = init_latents_for_cell(cell, ctx)
emb = encode_pair(cell, ctx)

# 1. a plain-CFG reference, with no adapter attached in this process yet
t0 = time.time()
seq_j, pool_j = get_joint_embeds(cell, ctx)
o = run_cfg(init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
            seq_cond=seq_j, pool_cond=pool_j, seq_e=emb["seq_e"], pool_e=emb["pool_e"],
            guidance_scale=ctx.guidance_scale, num_inference_steps=STEPS,
            height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
            device=ctx.device, dtype=ctx.dtype)
mono = out / "mono.png"; write_decoded_image(o.image, mono); del o; gc.collect(); torch.cuda.empty_cache()
print(f"reference ok ({time.time()-t0:.0f}s) -> {mono}", flush=True)

# 2. attach the rank-16 checkpoint and render one windowed condition
ck, rank = S.CHECKPOINTS["v1_freeze_null_r16_s0_25_30k"]
lbp.LORA_RANK = lbp.LORA_ALPHA = rank
info = lbp._attach_and_load_lora(ctx.models["unet"], Path(ck))
print("attach", {k: v for k, v in info.items() if k in ("n_matched", "n_loaded", "checkpoint_step")}, flush=True)
assert int(info["n_matched"]) > 0 and int(info["n_loaded"]) > 0, "adapter matched or loaded nothing"

for cutoff in (5, STEPS):
    t0 = time.time()
    o = run_lora_langevin_windowed_poe(
        init_latents=init_latents, models=ctx.models, scheduler=ctx.scheduler,
        seq_a=emb["seq_a"], pool_a=emb["pool_a"], seq_b=emb["seq_b"], pool_b=emb["pool_b"],
        seq_e=emb["seq_e"], pool_e=emb["pool_e"],
        guidance_scale=ctx.guidance_scale, num_inference_steps=STEPS,
        height=cell.height, width=cell.width, euler_init_noise_sigma=euler_sigma,
        device=ctx.device, dtype=ctx.dtype,
        lambda_value=S.LAMBDA_VALUE, k=0, c=0.0, corrector_window=None,
        noise_seed=seed, lora_adapter_name=lbp.LORA_ADAPTER_NAME,
        lambda_window=(0, cutoff), corrector_score="frozen")
    on = sum(1 for r in o.extras["per_step"] if r["adapter_on"])
    print(f"cutoff {cutoff}: adapter on for {on} of {STEPS} steps ({time.time()-t0:.0f}s)", flush=True)
    assert on == cutoff, f"lambda_window selected {on} steps, not {cutoff}"
    p = out / f"cut{cutoff}.png"; write_decoded_image(o.image, p); del o; gc.collect(); torch.cuda.empty_cache()

# 3. the three measurements
print("mem peak GB", torch.cuda.max_memory_allocated()/1e9, flush=True)
for p in (mono, out / "cut5.png", out / f"cut{STEPS}.png"):
    s = S.score_png(p, pair)
    print(p.name, "n=", s["n_instances"], "conf", s["concept_conf"], "lap",
          round(S.laplacian_var(p), 1), flush=True)
S.REF_DIR.joinpath(pair, f"seed_{seed}").mkdir(parents=True, exist_ok=True)
import shutil; shutil.copy(mono, S.mono_path(pair, seed))
print("dino dist cut5 vs mono:", S.dino_dist_to_mono(out / "cut5.png", pair, seed), flush=True)
S.mono_path(pair, seed).unlink()
print("SMOKE OK", flush=True)
