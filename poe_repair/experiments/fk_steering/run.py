"""Runner for Feynman-Kac steering on the detector reward (scope 06, plan 10).

Per pair and seed: the two eta-0 references (Mono, plain PoE), the unweighted K-particle
controls and the steered runs at every K in ``--particles``, every finished particle scored,
a sheet per pair, ``summary.json`` and ``verdict.json``, W&B.

    <co3 python> -m poe_repair.experiments.fk_steering.run --smoke
    <co3 python> -m poe_repair.experiments.fk_steering.run --pairs a_cat__x__a_dog a_butterfly__x__a_flower_meadow --seeds 9 10 11 12 13 14 15 16
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from poe_repair._sdxl.runtime import (encode_prompt_sdxl, infer_device, infer_dtype,
                                      load_ddim_scheduler, load_sdxl_models)
from poe_repair.experiments.fk_steering import steer
from poe_repair.experiments.twisted_smc.sampler import decode_one, run_particles
from poe_repair.experiments.twisted_smc.train import WandB
from poe_repair.methods._sampling import write_decoded_image

log = logging.getLogger("fk_steering")

DEFAULT_OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/fk_steering")
DEFAULT_CACHE = Path("/datasets/mmolefe/poe_repair_min/outputs/training_cache")
PROMPTS = {
    "a_cat__x__a_dog": ("a cat", "a dog", "a cat and a dog"),
    "a_butterfly__x__a_flower_meadow": ("a butterfly", "a flower meadow", "a butterfly and a flower meadow"),
}
JUDGED_PAIR = "a_cat__x__a_dog"
CONTROL_PAIR = "a_butterfly__x__a_flower_meadow"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--pairs", nargs="+", default=[JUDGED_PAIR, CONTROL_PAIR])
    p.add_argument("--seeds", nargs="+", type=int, default=[9, 10, 11, 12, 13, 14, 15, 16])
    p.add_argument("--particles", nargs="+", type=int, default=list(steer.K_VALUES))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--resample-steps", nargs="+", type=int, default=list(steer.RESAMPLE_STEP_INDICES))
    p.add_argument("--lam", type=float, default=steer.LAMBDA)
    p.add_argument("--eta", type=float, default=steer.ETA)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--unet-chunk", type=int, default=2)
    p.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--thumb", type=int, default=256)
    p.add_argument("--run-id", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", default="poe-repair-animals-compose")
    p.add_argument("--wandb-entity", default="prime_lab")
    p.add_argument("--adapter-checkpoint", type=Path, default=None,
                   help="attach this LoRA and use eps_frozen + lambda*(eps_lora - eps_frozen) as the proposal (plan 11)")
    p.add_argument("--adapter-rank", type=int, default=steer.ADAPTER_RANK)
    p.add_argument("--adapter-alpha", type=int, default=None, help="defaults to the rank, the showcase trainings' setting")
    p.add_argument("--adapter-lambdas", nargs="+", type=float, default=list(steer.ADAPTER_LAMBDAS))
    p.add_argument("--control-pair-lambdas", nargs="+", type=float, default=None,
                   help="lambdas to run on the control pair (default: the same as --adapter-lambdas)")
    p.add_argument("--fidelity", default="none", choices=["none", "imagereward"],
                   help="tie-breaker within a count level: r = min(count,2) + FIDELITY_WEIGHT * sigmoid(score)")
    p.add_argument("--smoke", action="store_true", help="cat x dog seed 9, K 2, 10 steps, 512², resample at 0,2,4,6,8")
    a = p.parse_args(argv)
    if a.smoke:
        a.pairs, a.seeds, a.particles, a.steps = [JUDGED_PAIR], [9], [2], 10
        a.resample_steps, a.size, a.wandb_mode = [0, 2, 4, 6, 8], 512, "disabled"
    if a.adapter_alpha is None:
        a.adapter_alpha = a.adapter_rank
    if a.control_pair_lambdas is None:
        a.control_pair_lambdas = list(a.adapter_lambdas)
    return a


def attach_adapter(unet, checkpoint: Path, rank: int, alpha: int) -> dict:
    """Attach the showcase LoRA (cross-attention q, k, v) and load its weights; leave it disabled."""
    from types import SimpleNamespace
    from poe_repair.experiments.one_pair_one_seed import trainer as lora_trainer
    from poe_repair.experiments.one_pair_one_seed.config import LoRAConfig
    cfg = LoRAConfig(rank=rank, alpha=alpha, dropout=0.0, target_modules=("attn2.to_q", "attn2.to_k", "attn2.to_v"),
                     init="gaussian", adapter_name="lora")
    info = lora_trainer.attach_lora(unet, SimpleNamespace(lora=cfg))
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt["lora_state"]
    lora_trainer.load_lora_state(unet, state)
    unet.disable_adapters()
    return {"n_loaded": len(state), "checkpoint_step": int(ckpt.get("step", -1)), "matched": info.get("matched_modules"),
            "rank": rank, "alpha": alpha, "checkpoint": str(checkpoint)}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def pinned_init(cache_root: Path, pair: str, seed: int, h: int, w: int) -> torch.Tensor:
    """The cache's pinned initial noise for (pair, seed), or the identical formula when the
    cache has no cell (checked byte-identical on cat x dog seeds 9 and 13, 2026-09-05)."""
    cell = cache_root / "heldout" / pair / f"seed_{seed}" / "embeddings.pt"
    if cell.exists():
        e = torch.load(cell, map_location="cpu", weights_only=False)
        init = e["init_latents"].reshape(4, *e["init_latents"].shape[-2:]).float()
        if tuple(init.shape[-2:]) == (h, w):
            return init / float(e.get("euler_init_noise_sigma", 1.0)), "cache"
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(1, 4, h, w, generator=g, dtype=torch.float16)[0].float(), "formula"


def particle_init(base: torch.Tensor, K: int, seed: int) -> torch.Tensor:
    """Particle 0 is the pinned noise; particles 1..K-1 are fresh draws seeded from the seed."""
    g = torch.Generator(device="cpu").manual_seed(10_000 * seed + 7)
    rest = torch.randn((K, *base.shape), generator=g)
    rest[0] = base
    return rest


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class Detector:
    """The validated instance count, optionally paired with ImageReward as the fidelity tie-breaker."""

    def __init__(self, device, fidelity: str = "none"):
        from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances
        self.fn, self.device, self.fidelity = count_instances, device, fidelity
        self.rm = None
        if fidelity == "imagereward":
            import ImageReward as RM
            self.rm = RM.load("ImageReward-v1.0", device=str(device))
        self.prompt = None

    def count(self, path: Path) -> int:
        n, _ = self.fn(Path(path), device=self.device)
        return int(n)

    def score(self, path: Path) -> float | None:
        if self.rm is None:
            return None
        return float(self.rm.score(self.prompt, str(path)))

    def __call__(self, path: Path):
        n = self.count(path)
        return n if self.rm is None else (n, self.score(path))


# ---------------------------------------------------------------------------
# Sheet
# ---------------------------------------------------------------------------


def _font(size):
    for cand in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def draw_sheet(rows: list[dict], columns: list[tuple[str, str]], title: str, out_png: Path, thumb: int) -> dict:
    """rows: [{"seed": 9, "tiles": {col_key: {"path": ..., "count": n}}}]. Missing tiles are red."""
    left, top, gap = 90, 70, 6
    W = left + len(columns) * (thumb + gap)
    H = top + len(rows) * (thumb + gap)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    f_big, f_small, f_tag = _font(18), _font(14), _font(20)
    d.text((8, 6), title, fill="black", font=f_big)
    for c, (key, label) in enumerate(columns):
        x = left + c * (thumb + gap)
        for i, line in enumerate(label.split("\n")):
            d.text((x + 2, 30 + 15 * i), line, fill="black", font=f_small)
    side = {"tiles": [], "missing": []}
    for r, row in enumerate(rows):
        y = top + r * (thumb + gap)
        d.text((8, y + thumb // 2 - 8), f"seed {row['seed']}", fill="black", font=f_small)
        for c, (key, _) in enumerate(columns):
            x = left + c * (thumb + gap)
            t = row["tiles"].get(key)
            if t and Path(t["path"]).exists():
                sheet.paste(Image.open(t["path"]).convert("RGB").resize((thumb, thumb), Image.LANCZOS), (x, y))
                n = t["count"]
                col = (20, 140, 40) if n >= 2 else (200, 40, 40)
                d.rectangle([x + 4, y + 4, x + 34, y + 30], fill=col)
                d.text((x + 10, y + 4), str(n), fill="white", font=f_tag)
                fid = t.get("fidelity")
                if fid is not None:
                    d.rectangle([x + 38, y + 4, x + 118, y + 30], fill=(40, 40, 40))
                    d.text((x + 42, y + 8), f"IR {fid:+.2f}", fill="white", font=f_small)
                side["tiles"].append({"row_seed": row["seed"], "column": key, "source": t["path"], "count": n, "compose": int(n >= 2), "fidelity": fid})
            else:
                d.rectangle([x, y, x + thumb, y + thumb], outline="red", width=3)
                d.text((x + 20, y + thumb // 2), "missing", fill="red", font=f_small)
                side["missing"].append({"row_seed": row["seed"], "column": key})
    sheet.save(out_png)
    return side


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    a = parse_args(argv)
    device = infer_device(a.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            sys.exit("torch.cuda.is_available() is False on the pinned device (poe-launch-002); aborting")
        log.info("device %s = %s", device, torch.cuda.get_device_name(device))
    dtype = infer_dtype(a.dtype, device)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    adapter_on = a.adapter_checkpoint is not None
    prefix = ("smoke_" if a.smoke else "") + ("fka_" if adapter_on else "fk_")
    # the host and pid keep two jobs launched in the same second out of one directory
    run_id = a.run_id or (prefix + f"K{'-'.join(map(str, a.particles))}_s{a.steps}_{stamp}_{os.uname().nodename.split('.')[0]}-{os.getpid()}")
    run_dir = a.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**vars(a), "out_root": str(a.out_root), "cache_root": str(a.cache_root),
                                                    "run_id": run_id, "node": os.uname().nodename, "pid": os.getpid(),
                                                    "constants": {k: getattr(steer, k) for k in
                                                                  ("PASS_MARGIN", "NULL_MARGIN", "MAX_REWARD_DISAGREEMENT",
                                                                   "REWARD_BLIND_STEP", "LAMBDA", "REWARD_CLIP",
                                                                   "RESAMPLE_STEP_INDICES", "K_VALUES", "ETA")}}, indent=1, default=str))
    log.info("run dir %s (node %s, pid %d)", run_dir, os.uname().nodename, os.getpid())

    models = load_sdxl_models(model_id=a.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(a.model_id)
    adapter_info = None
    if adapter_on:
        adapter_info = attach_adapter(models["unet"], a.adapter_checkpoint, a.adapter_rank, a.adapter_alpha)
        (run_dir / "adapter.json").write_text(json.dumps(adapter_info, indent=1, default=str))
        log.info("adapter attached: %s", adapter_info)
    lambdas_for = lambda pair: ([None] if not adapter_on else  # noqa: E731
                                (list(a.adapter_lambdas) if pair == JUDGED_PAIR else list(a.control_pair_lambdas)))
    ltag = lambda lam: "" if lam is None else f"_lam{lam:g}"  # noqa: E731
    det = Detector(device, fidelity=a.fidelity)
    wb = WandB(mode=a.wandb_mode, project=a.wandb_project, entity=a.wandb_entity, name=run_id, run_dir=run_dir,
               config=json.loads((run_dir / "config.json").read_text()), tags=["fk-steering", "scope-06", "plan-10"],
               group="fk-steering")
    if wb.run is not None:
        log.info("wandb run id %s url %s", wb.run.id, wb.run.url)
    h = w = a.size // 8
    enc = lambda t: encode_prompt_sdxl(t, models=models, device=device, dtype=dtype)  # noqa: E731
    seq_e, pool_e = enc("")
    common = dict(models=models, scheduler=scheduler, seq_e=seq_e, pool_e=pool_e, guidance_scale=a.guidance_scale,
                  num_inference_steps=a.steps, height=a.size, width=a.size, device=device, dtype=dtype)

    summary: dict = {"run_id": run_id, "pairs": {}, "constants": json.loads((run_dir / "config.json").read_text())["constants"]}
    step_counter = 0
    for pair in a.pairs:
        pa, pb, pj = PROMPTS[pair]
        det.prompt = pj                      # the fidelity model scores against the joint prompt
        seq_a, pool_a = enc(pa); seq_b, pool_b = enc(pb); seq_j, pool_j = enc(pj)
        pair_dir = run_dir / pair
        rows, per_seed = [], {}
        for seed in a.seeds:
            t0 = time.time()
            sd = pair_dir / f"seed_{seed}"
            sd.mkdir(parents=True, exist_ok=True)
            base, origin = pinned_init(a.cache_root, pair, seed, h, w)
            cell: dict = {"seed": seed, "init_noise_from": origin, "tiles": {}, "runs": {}}
            # eta-0 references, K 1: Mono and plain PoE from the pinned noise
            for comp, tag in (("mono", "mono_eta0"), ("poe", "poe_eta0")):
                png = sd / f"{tag}.png"
                if not png.exists():
                    out = run_particles(init_latents=base[None], composition=comp, seq_a=seq_a, pool_a=pool_a, seq_b=seq_b,
                                        pool_b=pool_b, seq_j=seq_j, pool_j=pool_j, eta=0.0, generator=None, twist=None,
                                        unet_chunk=a.unet_chunk, **common)
                    write_decoded_image(decode_one(models, out.latents[0]), png)
                cell["tiles"][tag] = {"path": str(png), "count": det.count(png), "fidelity": det.score(png)}
            # adapter-alone eta-0 reference per lambda (the showcase render at that lambda)
            for lam in lambdas_for(pair):
                if lam is None:
                    continue
                png = sd / f"adapter_eta0{ltag(lam)}.png"
                if not png.exists():
                    out = steer.run_fk_steering(init_latents=base[None], seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                                                eta=0.0, generator=None, reward_fn=None, unet_chunk=a.unet_chunk,
                                                adapter_lambda=lam, **common)
                    write_decoded_image(decode_one(models, out.latents[0]), png)
                cell["tiles"][f"adapter_eta0{ltag(lam)}"] = {"path": str(png), "count": det.count(png), "fidelity": det.score(png)}
            # controls and steered runs per K (and per lambda when the adapter is the proposal)
            for K in a.particles:
              for lam in lambdas_for(pair):
                init = particle_init(base, K, seed)
                for steered in (False, True):
                    name = f"{'fk' if steered else 'ctrl'}_K{K}{ltag(lam)}"
                    rd = sd / name
                    rd.mkdir(exist_ok=True)
                    done = rd / "run.json"
                    if done.exists():
                        rec = json.loads(done.read_text())
                    else:
                        g = torch.Generator(device="cpu").manual_seed(seed)
                        g_rs = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
                        res = steer.run_fk_steering(
                            init_latents=init, seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                            eta=a.eta, generator=g, resample_generator=g_rs, reward_fn=det if steered else None,
                            resample_steps=tuple(a.resample_steps), lam=a.lam,
                            xhat_dir=rd / "xhat" if steered else None, unet_chunk=a.unet_chunk,
                            adapter_lambda=lam, **common)
                        finals, fids = [], []
                        for k in range(K):
                            p = rd / f"p{k}.png"
                            write_decoded_image(decode_one(models, res.latents[k]), p)
                            finals.append(det.count(p)); fids.append(det.score(p))
                        final_r = [steer.clipped_reward(c, f) for c, f in zip(finals, fids)]
                        if steered:
                            lm = res.lineage_max.tolist()
                            pick = max(range(K), key=lambda k: (final_r[k], lm[k], -k))
                        else:
                            pick = 0
                        best = max(range(K), key=lambda k: (final_r[k], -k))
                        agreement = {}
                        if steered:
                            for s, rws in res.rewards_by_step.items():
                                agree = 0
                                for k in range(K):
                                    anc = steer.trace_ancestor(res.ancestry, res.resampled_at, k, s)
                                    agree += int((rws[anc] >= 2) == (finals[k] >= 2))
                                agreement[str(s)] = agree / K
                        rec = {"K": K, "steered": steered, "final_counts": finals, "final_fidelity": fids, "picked": pick, "best": best,
                               "all_fraction": sum(c >= 2 for c in finals) / K,
                               "picked_compose": int(finals[pick] >= 2), "p0_compose": int(finals[0] >= 2),
                               "best_compose": int(finals[best] >= 2), "agreement_by_step": agreement,
                               "informative_resamples": sum(1 for wv in res.weights_by_step.values() if max(wv) - min(wv) > 1e-6),
                               "final_pngs": [str(rd / f"p{k}.png") for k in range(K)]}
                        if steered and rec["informative_resamples"] == 0:
                            # no weight vector ever separated the particles, so the FK run must equal the control
                            import hashlib
                            ctrl_pngs = json.loads((sd / f"ctrl_K{K}{ltag(lam)}" / "run.json").read_text())["final_pngs"]
                            same = [hashlib.md5(Path(p).read_bytes()).hexdigest() == hashlib.md5(Path(q).read_bytes()).hexdigest()
                                    for p, q in zip(ctrl_pngs, rec["final_pngs"])]
                            rec["identical_to_control_when_uninformative"] = all(same)
                            if not all(same):
                                log.warning("FK run with no informative resample differs from the control: %s", same)
                        steer.dump(res, rd / "steer.json", extra=rec)
                        done.write_text(json.dumps(rec, indent=1))
                        del res
                        torch.cuda.empty_cache()
                    cell["runs"][name] = rec
                    fidl = rec.get("final_fidelity") or [None] * K
                    if steered:
                        cell["tiles"][f"fk_K{K}{ltag(lam)}"] = {"path": rec["final_pngs"][rec["picked"]], "count": rec["final_counts"][rec["picked"]], "fidelity": fidl[rec["picked"]]}
                    else:
                        cell["tiles"][f"ctrl_p0_K{K}{ltag(lam)}"] = {"path": rec["final_pngs"][0], "count": rec["final_counts"][0], "fidelity": fidl[0]}
                        cell["tiles"][f"best_K{K}{ltag(lam)}"] = {"path": rec["final_pngs"][rec["best"]], "count": rec["final_counts"][rec["best"]], "fidelity": fidl[rec["best"]]}
            (sd / "cell.json").write_text(json.dumps(cell, indent=1))
            per_seed[seed] = cell
            rows.append(cell)
            log.info("[%s] seed %d done in %.0f s: %s", pair, seed, time.time() - t0,
                     {k: v["count"] for k, v in cell["tiles"].items()})
            step_counter += 1
            wb.log({f"progress/{pair}/seed": seed}, step=step_counter)

        Kmax = max(a.particles)
        for lam in lambdas_for(pair):
            lt = ltag(lam)
            columns = [("mono_eta0", "Mono, joint prompt\nDDIM eta 0"), ("poe_eta0", "plain PoE\nDDIM eta 0")]
            if lam is not None:
                columns.append((f"adapter_eta0{lt}", f"adapter alone, lambda {lam:g}\nDDIM eta 0"))
            columns += [(f"ctrl_p0_K{Kmax}{lt}", f"control, particle 0\nK {Kmax}, eta {a.eta:g}, no weights" + (f", adapter {lam:g}" if lam is not None else "")),
                        (f"best_K{Kmax}{lt}", f"best of {Kmax} unweighted\nby final count")]
            columns += [(f"fk_K{K}{lt}", f"FK steering K {K}" + (f" on adapter {lam:g}" if lam is not None else "") +
                         f"\nlambda {a.lam:g}, resample at {','.join(map(str, a.resample_steps))}") for K in a.particles]
            sheet_png = run_dir / f"fk_steering_{pair}{lt}_sheet.png"
            side = draw_sheet(rows, columns, f"{pair.replace('__x__', ' x ').replace('_', ' ')}{'' if lam is None else f', rank-32 correction at lambda {lam:g} in the proposal'}: "
                              f"{a.steps} DDIM steps, guidance {a.guidance_scale:g}, {a.size}², count of animal instances on each tile, compose = 2 or more",
                              sheet_png, a.thumb)
            rates = {key: sum(r["tiles"][key]["count"] >= 2 for r in rows if key in r["tiles"]) / len(rows) for key, _ in columns}
            extras = {}
            if a.fidelity != "none":
                extras["mean_fidelity_by_column"] = {key: sum(r["tiles"][key].get("fidelity") or 0.0 for r in rows if key in r["tiles"]) / len(rows) for key, _ in columns}
            for K in a.particles:
                fk, ct = f"fk_K{K}{lt}", f"ctrl_K{K}{lt}"
                extras[f"fk_K{K}_all_fraction"] = sum(r["runs"][fk]["all_fraction"] for r in rows) / len(rows)
                extras[f"ctrl_K{K}_all_fraction"] = sum(r["runs"][ct]["all_fraction"] for r in rows) / len(rows)
                extras[f"ctrl_K{K}_p0"] = sum(r["runs"][ct]["p0_compose"] for r in rows) / len(rows)
                extras[f"ctrl_K{K}_best"] = sum(r["runs"][ct]["best_compose"] for r in rows) / len(rows)
                extras[f"fk_K{K}_informative_resamples_mean"] = sum(r["runs"][fk]["informative_resamples"] for r in rows) / len(rows)
                steps_seen = sorted({s for r in rows for s in r["runs"][fk]["agreement_by_step"]}, key=int)
                extras[f"fk_K{K}_agreement_by_step"] = {s: sum(r["runs"][fk]["agreement_by_step"][s] for r in rows) / len(rows) for s in steps_seen}
            side.update({"columns": columns, "compose_rate_by_column": rates, "n_seeds": len(rows), "secondary": extras,
                         "settings": {"steps": a.steps, "eta_particles": a.eta, "eta_references": 0.0, "guidance": a.guidance_scale,
                                      "size": a.size, "lambda": a.lam, "resample_steps": a.resample_steps, "particles": a.particles,
                                      "adapter_lambda": lam, "adapter": adapter_info}})
            (run_dir / f"fk_steering_{pair}{lt}_sheet.json").write_text(json.dumps(side, indent=1, default=str))
            summary["pairs"][f"{pair}{lt}"] = {"compose_rate_by_column": rates, "secondary": extras, "adapter_lambda": lam}
            wb.log_image(f"sheets/{pair}{lt}", sheet_png, step=step_counter, caption=f"compose rates {rates}")
            payload = {f"compose/{pair}{lt}/{k}": v for k, v in rates.items()}
            for K in a.particles:
                for s, v in extras[f"fk_K{K}_agreement_by_step"].items():
                    payload[f"agreement/{pair}{lt}/K{K}/step_{s}"] = v
            wb.log(payload, step=step_counter)
            log.info("[%s%s] compose rates %s", pair, lt, rates)

    # verdict on the judged pair
    v = {"judged_pair": JUDGED_PAIR, "constants": summary["constants"]}
    Ks = sorted(a.particles)
    if not adapter_on and JUDGED_PAIR in summary["pairs"]:
        pr = summary["pairs"][JUDGED_PAIR]
        Khi, Klo = Ks[-1], Ks[0]
        fk_hi, fk_lo = pr["compose_rate_by_column"].get(f"fk_K{Khi}"), pr["compose_rate_by_column"].get(f"fk_K{Klo}")
        c_hi, c_lo = pr["secondary"][f"ctrl_K{Khi}_p0"], pr["secondary"][f"ctrl_K{Klo}_p0"]
        agr = pr["secondary"][f"fk_K{Khi}_agreement_by_step"].get(str(steer.REWARD_BLIND_STEP))
        dis = None if agr is None else 1.0 - agr
        v.update({"K_high": Khi, "K_low": Klo, "fk_high": fk_hi, "fk_low": fk_lo, "control_high": c_hi, "control_low": c_lo,
                  "disagreement_at_blind_step": dis,
                  "verdict": steer.verdict(fk16=fk_hi, fk4=fk_lo, ctrl16=c_hi, ctrl4=c_lo, disagreement=dis,
                                           k_high=Khi, k_low=Klo)})
        if CONTROL_PAIR in summary["pairs"]:
            cp = summary["pairs"][CONTROL_PAIR]["compose_rate_by_column"]
            v["control_pair"] = {"poe_eta0": cp.get("poe_eta0"), **{f"fk_K{K}": cp.get(f"fk_K{K}") for K in Ks}}
    elif adapter_on:
        # plan 11: judged at ADAPTER_JUDGED_LAMBDA against the adapter-alone control at the same lambda, K = max
        Khi = Ks[-1]
        per_lam = {}
        for lam in a.adapter_lambdas:
            key = f"{JUDGED_PAIR}{ltag(lam)}"
            if key not in summary["pairs"]:
                continue
            pr = summary["pairs"][key]
            agr = pr["secondary"][f"fk_K{Khi}_agreement_by_step"].get(str(steer.REWARD_BLIND_STEP))
            per_lam[f"{lam:g}"] = {"fk": pr["compose_rate_by_column"].get(f"fk_K{Khi}{ltag(lam)}"),
                                   "control_p0": pr["secondary"][f"ctrl_K{Khi}_p0"],
                                   "adapter_eta0": pr["compose_rate_by_column"].get(f"adapter_eta0{ltag(lam)}"),
                                   "best_of_K": pr["secondary"][f"ctrl_K{Khi}_best"],
                                   "disagreement_at_blind_step": None if agr is None else 1.0 - agr}
        jl = f"{steer.ADAPTER_JUDGED_LAMBDA:g}"
        v.update({"K": Khi, "judged_lambda": steer.ADAPTER_JUDGED_LAMBDA, "per_lambda": per_lam})
        if jl in per_lam:
            j = per_lam[jl]
            gaps = [per_lam[l]["fk"] - per_lam[l]["control_p0"] for l in per_lam]
            if j["disagreement_at_blind_step"] is not None and j["disagreement_at_blind_step"] > steer.MAX_REWARD_DISAGREEMENT:
                v["verdict"] = "inconclusive: the reward on x0hat at step %d disagrees with the final verdict on %.2f of the final particles (bar %.2f)" % (
                    steer.REWARD_BLIND_STEP, j["disagreement_at_blind_step"], steer.MAX_REWARD_DISAGREEMENT)
            elif j["fk"] - j["control_p0"] >= steer.PASS_MARGIN:
                v["verdict"] = "support: FK steering on the adapter at lambda %s composes %.3f more often than the adapter alone (bar %.2f)" % (jl, j["fk"] - j["control_p0"], steer.PASS_MARGIN)
            elif all(abs(g) <= steer.NULL_MARGIN for g in gaps):
                v["verdict"] = "null: FK steering on the adapter is within %.2f of the adapter alone at every lambda %s" % (steer.NULL_MARGIN, list(per_lam))
            else:
                v["verdict"] = "neither bar met: gaps by lambda %s" % {l: round(per_lam[l]["fk"] - per_lam[l]["control_p0"], 3) for l in per_lam}
        for lam in a.control_pair_lambdas:
            key = f"{CONTROL_PAIR}{ltag(lam)}"
            if key in summary["pairs"]:
                v.setdefault("control_pair", {})[f"{lam:g}"] = summary["pairs"][key]["compose_rate_by_column"]
    summary["verdict"] = v
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "verdict.json").write_text(json.dumps(v, indent=1))
    art_dir = run_dir / "_artifact"
    art_dir.mkdir(exist_ok=True)
    for f in list(run_dir.glob("*.png")) + list(run_dir.glob("*.json")):
        (art_dir / f.name).write_bytes(f.read_bytes())
    wb.log_artifact_dir(art_dir, name=f"fk-steering-{run_id}", kind="sheets", aliases=["latest"])
    wb.finish()
    log.info("verdict: %s", v.get("verdict"))
    log.info("[done] %s", run_dir)


if __name__ == "__main__":
    main()
