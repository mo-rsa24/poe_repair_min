"""Plan 08's replacement for the trainer's inline sampler, which renders with the PoE sampler
and so cannot show a SuperDiff-residual adapter. Every checkpoint at a multiple of 10k
optimizer steps (and the final one) is rendered through SuperDiff itself at 200 steps,
kappa 0.5, guidance 7.5, with the adapter injected on every step as
eps_M(off) + lambda * (eps_M(on) - eps_M(off)).

One strip per (checkpoint, pair), the trainer's inline-sample layout: rows seeds 9-12, columns
  Mono (target): the joint prompt through SuperDiff's integrator, plan 05's lam=1 render at kappa 0.5 |
  SuperDiff kappa 0.5, no adapter (default) | LoRA at lambda 1 on every step, this checkpoint.
The two baseline columns are plan 05's renders (lambda_sweep/), rendered here only if missing. Strips go to <run>/samples/superdiff/step_XXXXXX_<pair>.png and are logged to a
companion W&B run sdlora_r{R}_100k_samples (same project) against ckpt_step. A companion run
rather than the training run itself: two processes writing one run at once is unsupported,
and the training run's step counter is already ahead of any checkpoint's step.
Polls every 5 minutes; exits once every rank has its final checkpoint rendered.
"""
import sys, time, json, os
sys.path.insert(0, "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min")
from pathlib import Path
from types import SimpleNamespace
import torch, wandb
from PIL import Image, ImageDraw, ImageFont
from poe_repair.run import MethodCtx
from poe_repair.runtime import PairSeedCell, infer_device, infer_dtype
from poe_repair.composers import superdiff
from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig
from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer

SD = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/corrector/superdiff")
RANKS = [int(r) for r in os.environ.get("RANKS", "8,16,32").split(",")]
EVERY = int(os.environ.get("EVERY", "10000"))
PAIRS = [("a_cat__x__a_dog", "a cat", "a dog", "cat_dog"),
         ("a_butterfly__x__a_flower_meadow", "a butterfly", "a flower meadow", "butterfly_meadow")]
SEEDS = [9, 10, 11, 12]; LVS = [0.5, 1.0]; STEPS, KAPPA = 200, 0.5
TILE, LEFT, TOP = 320, 90, 60
device = infer_device(None); dtype = infer_dtype("fp16", device)
models = superdiff._load_superdiff_models(device, dtype); unet = models["unet"]

def font(s):
    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(c).exists(): return ImageFont.truetype(c, s)
    return ImageFont.load_default()
FB, FS = font(20), font(17)

def cell(pair, seed):
    return PairSeedCell(pair_dir=None, pair_slug=pair[0], prompt_a=pair[1], prompt_b=pair[2], seed=seed,
                        regime="samples", height=1024, width=1024, grid_assets={})
def ctx(root):
    return MethodCtx(models={}, scheduler=None, output_root=root, device=device, dtype=dtype,
                     guidance_scale=7.5, num_inference_steps=STEPS)

def baseline(pair, seed, kappa):
    # plan 05's renders live at SD/lambda_sweep/pairs/...; the composer returns the cached file
    return superdiff.run(cell(pair, seed), ctx(SD), exp_name="lambda_sweep", kappa_clamp=False, kappa_override=kappa)

def attach(rank, ckpt):
    cfg = SimpleNamespace(lora=LoRAConfig(rank=rank, alpha=rank, dropout=0.0,
        target_modules=("attn2.to_q", "attn2.to_k", "attn2.to_v"), init="gaussian", adapter_name="lora"))
    if hasattr(unet, "delete_adapters"):
        try: unet.delete_adapters("lora")
        except Exception: pass
    info = lora_trainer.attach_lora(unet, cfg)
    lora_trainer.load_lora_state(unet, torch.load(ckpt, map_location="cpu", weights_only=False)["lora_state"])
    assert info["n_matched"] > 0

def mono(pair, seed):
    # the joint prompt through SuperDiff's own integrator: eps_M + 1.0 * (eps_J - eps_M) = eps_J
    return superdiff.run(cell(pair, seed), ctx(SD), exp_name="lambda_sweep", kappa_clamp=False, kappa_override=KAPPA, lam=1.0)

def strip(rank, step, pair, tag):
    cols = [("Mono (target)", lambda s: mono(pair, s)), ("SuperDiff κ=0.5 (default)", lambda s: baseline(pair, s, 0.5)),
            (f"LoRA λ=1 @ step {step}", lambda s: superdiff.run(cell(pair, s), ctx(SD / "samples_sdlora"), exp_name=tag,
             kappa_clamp=False, kappa_override=KAPPA, lora_adapter_name="lora", lambda_value=1.0, lora_tag=tag))]
    W, H = LEFT + TILE * len(cols), TOP + TILE * len(SEEDS)
    sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
    d.text((8, 8), f"{pair[0]}  rank {rank}", fill="black", font=FS)
    for r, seed in enumerate(SEEDS):
        d.text((8, TOP + r * TILE + TILE // 2 - 10), f"seed {seed}", fill="black", font=FS)
        for c, (label, fn) in enumerate(cols):
            if r == 0: d.text((LEFT + c * TILE + 6, 34), label, fill="black", font=FB)
            p = fn(seed)
            sheet.paste(Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS), (LEFT + c * TILE, TOP + r * TILE))
    out = SD / f"sdlora_r{rank}_100k" / "samples" / "superdiff"; out.mkdir(parents=True, exist_ok=True)
    fp = out / f"step_{step:06d}_{pair[3]}.png"; sheet.save(fp); return fp

def pending(rank):
    d = SD / f"sdlora_r{rank}_100k" / "checkpoints"; done = SD / f"sdlora_r{rank}_100k" / "samples" / "superdiff"
    out = []
    for p in sorted(d.glob("lora_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1])):
        s = int(p.stem.split("_")[-1])
        if (s % EVERY == 0 or s >= 100000) and not (done / f"step_{s:06d}_done.json").exists():
            out.append((s, p))
    return out

def final_done(rank):
    done = SD / f"sdlora_r{rank}_100k" / "samples" / "superdiff"
    return any(int(p.stem.split("_")[1]) >= 100000 for p in done.glob("step_*_done.json"))

def wb_log(rank, step, fps):
    # One short-lived run handle per log call: a process holds one active W&B run at a time, and
    # opening a second (another rank) finishes the first, so each rank's run is reopened with
    # resume="allow", written, and closed.
    run = wandb.init(project="poe-repair-animals-compose", entity="prime_lab", id=f"sdlora_r{rank}_samples",
                     name=f"sdlora_r{rank}_100k_samples", resume="allow",
                     config={"rank": rank, "steps": STEPS, "kappa": KAPPA, "lambda_value": 1.0, "seeds": SEEDS,
                             "training_run": f"sdlora_r{rank}_100k"})
    run.define_metric("ckpt_step"); run.define_metric("samples/*", step_metric="ckpt_step")
    run.log({"ckpt_step": step, **{f"samples/{k}": wandb.Image(str(v)) for k, v in fps.items()}})
    run.finish()

while True:
    did = False
    for rank in RANKS:
        for step, ckpt in pending(rank):
            t0 = time.perf_counter(); attach(rank, ckpt); tag = f"sdr{rank}s{step}"
            fps = {pair[3]: strip(rank, step, pair, tag) for pair in PAIRS}
            wb_log(rank, step, fps)
            json.dump({"checkpoint": str(ckpt), "strips": {k: str(v) for k, v in fps.items()}, "wall_s": time.perf_counter() - t0},
                      open(fps["cat_dog"].parent / f"step_{step:06d}_done.json", "w"), indent=2)
            print(f"rank {rank} step {step}: {list(fps.values())} in {(time.perf_counter()-t0)/60:.1f} min", flush=True); did = True
    if all(final_done(r) for r in RANKS):
        print("ALL FINALS RENDERED", flush=True); break
    if not did: time.sleep(300)
