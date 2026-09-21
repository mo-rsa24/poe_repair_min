"""Train the twist head contrastively and render Mono | PoE | twisted-SMC strips to W&B.

Every ``--render-every`` optimizer steps (default 10k) each render cell is sampled
three ways from the same K noise draws at the same ``eta``:

    Mono  : joint prompt, K particles, no twist      (the target)
    PoE   : product of experts, K particles, no twist (the default)
    SMC   : product of experts, K particles, twist on (the solution under test)

and the three particle-0 images are pasted into one strip, logged as a
``wandb.Image`` under ``samples/<split>/<pair>/seed_NN`` and bundled into a
``wandb.Artifact`` named ``strips-step<N>``. Mono and PoE are rendered once at
the start and reused, since nothing in them changes with training.

Run::

    python -m poe_repair.experiments.twisted_smc.train --wandb-mode online ...
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from poe_repair._sdxl.runtime import infer_device, infer_dtype, load_ddim_scheduler, load_sdxl_models
from poe_repair.methods._sampling import write_decoded_image
from poe_repair.experiments.twisted_smc.data import (
    BankCell, ContrastiveSampler, build_or_load_bank, discover_bank_cells,
)
from poe_repair.experiments.twisted_smc.sampler import decode_one, run_particles
from poe_repair.experiments.twisted_smc.twist import TwistHead

log = logging.getLogger("twisted_smc.train")

DEFAULT_CACHE_ROOT = Path(os.environ.get(
    "POE_REPAIR_TRAINING_CACHE", "/datasets/mmolefe/poe_repair_min/outputs/training_cache"))
DEFAULT_OUTPUT_ROOT = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/twisted_smc")

# Bars, fixed before any run (plan 09 of scope 06). Read at the final render.
#   pass:  compose fraction of the SMC panel exceeds the PoE control's by PASS_MARGIN or more
#   null:  the two are within NULL_MARGIN of each other
#   inconclusive: the twist never learned (validation accuracy under MIN_VAL_ACC)
PASS_MARGIN = 0.25
NULL_MARGIN = 0.10
MIN_VAL_ACC = 0.60


# ---------------------------------------------------------------------------
# W&B shim (same shape as one_pair_one_seed.main.WandBLogger, without RunConfig)
# ---------------------------------------------------------------------------


class WandB:
    def __init__(self, *, mode: str, project: str, entity: str | None, name: str,
                 run_dir: Path, config: dict, tags: list[str], group: str = "twisted-smc"):
        self.run = None
        self.history: list[dict] = []
        self.history_path = run_dir / "history.json"
        if mode == "disabled":
            return
        try:
            import wandb
        except Exception as exc:
            log.warning("wandb unavailable (%s); history.json only", exc)
            return
        if mode == "offline":
            os.environ.setdefault("WANDB_MODE", "offline")
        self.run = wandb.init(project=project, entity=entity, name=name, group=group,
                              tags=tags, mode=mode, dir=str(run_dir), config=config, reinit=True)

    def log(self, payload: dict, *, step: int) -> None:
        rec = dict(payload); rec["_step"] = int(step)
        self.history.append(rec)
        if self.run is not None:
            try:
                self.run.log(payload, step=step)
            except Exception as exc:
                log.warning("wandb.log failed: %s", exc)

    def log_image(self, key: str, path: Path, *, step: int, caption: str = "") -> None:
        if self.run is None:
            return
        try:
            import wandb
            self.run.log({key: wandb.Image(str(path), caption=caption)}, step=step)
        except Exception as exc:
            log.warning("wandb.log_image failed: %s", exc)

    def log_artifact_dir(self, directory: Path, *, name: str, kind: str, aliases: list[str]) -> None:
        if self.run is None:
            return
        try:
            import wandb
            art = wandb.Artifact(name=name, type=kind)
            art.add_dir(str(directory))
            self.run.log_artifact(art, aliases=aliases)
        except Exception as exc:
            log.warning("wandb.log_artifact failed: %s", exc)

    def finish(self) -> None:
        try:
            self.history_path.write_text(json.dumps(self.history, indent=1))
        except Exception as exc:
            log.warning("history.json dump failed: %s", exc)
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _strip(panels: list[tuple[str, Path]], title: str, thumb: int) -> Image.Image:
    pad, label_h, title_h = 8, 22, 26
    W = len(panels) * thumb + (len(panels) + 1) * pad
    H = title_h + label_h + thumb + 2 * pad
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 6), title, fill=(255, 255, 255))
    for i, (label, p) in enumerate(panels):
        x = pad + i * (thumb + pad)
        draw.text((x, title_h), label, fill=(200, 200, 200))
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb, thumb))
        canvas.paste(im, (x + (thumb - im.width) // 2, title_h + label_h + (thumb - im.height) // 2))
    return canvas


class Renderer:
    """Holds the SDXL models and the per-cell embeddings and noise draws."""

    def __init__(self, args, cells: list[BankCell], models: dict, device, dtype):
        self.args = args
        self.cells = cells
        self.models = models
        self.device, self.dtype = device, dtype
        self.scheduler = load_ddim_scheduler(args.model_id)
        self.emb: dict[str, dict] = {}
        self.init: dict[str, torch.Tensor] = {}
        self.refs: dict[str, dict[str, Path]] = {}
        self.detector = None
        h, w = args.render_height // 8, args.render_width // 8
        for c in cells:
            key = self._key(c)
            emb = torch.load(c.cell_dir / "embeddings.pt", map_location="cpu", weights_only=False)
            self.emb[key] = {k: v.to(device=device, dtype=dtype) for k, v in emb.items()
                             if k.startswith(("seq_", "pool_"))}
            g = torch.Generator(device="cpu").manual_seed(10_000 * c.seed + 7)
            base = torch.randn((args.particles, 4, h, w), generator=g)
            init0 = emb.get("init_latents")
            if init0 is not None and tuple(init0.shape[-2:]) == (h, w):
                base[0] = init0.reshape(4, h, w).float()   # particle 0 is the cache's own noise
            sigma = float(emb.get("euler_init_noise_sigma", 1.0))
            self.init[key] = base / sigma

    @staticmethod
    def _key(c: BankCell) -> str:
        return f"{c.split}/{c.pair_slug}/seed_{c.seed:02d}"

    def _run(self, c: BankCell, composition: str, twist, step_seed: int):
        e = self.emb[self._key(c)]
        g = torch.Generator(device="cpu").manual_seed(step_seed)
        return run_particles(
            init_latents=self.init[self._key(c)], models=self.models, scheduler=self.scheduler,
            composition=composition,
            seq_a=e["seq_a"], pool_a=e["pool_a"], seq_b=e["seq_b"], pool_b=e["pool_b"],
            seq_j=e["seq_j"], pool_j=e["pool_j"], seq_e=e["seq_uncond"], pool_e=e["pool_uncond"],
            guidance_scale=self.args.guidance_scale, num_inference_steps=self.args.render_steps,
            height=self.args.render_height, width=self.args.render_width,
            device=self.device, dtype=self.dtype, eta=self.args.eta, generator=g,
            twist=twist, twist_temperature=self.args.twist_temperature,
            ess_threshold=self.args.ess_threshold, unet_chunk=self.args.unet_chunk,
        )

    def _pick(self, out) -> int:
        return int(torch.argmax(out.log_weights).item())

    def _save_particles(self, out, directory: Path, stem: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        pick = self._pick(out)
        for k in range(out.latents.shape[0]):
            write_decoded_image(decode_one(self.models, out.latents[k]), directory / f"{stem}__p{k}.png")
        chosen = directory / f"{stem}.png"
        Image.open(directory / f"{stem}__p{pick}.png").save(chosen)
        return chosen

    def render_references(self, run_dir: Path) -> None:
        ref_dir = run_dir / "references"
        for c in self.cells:
            key = self._key(c)
            stem = key.replace("/", "__")
            paths = {}
            for comp in ("mono", "poe"):
                target = ref_dir / f"{stem}__{comp}.png"
                if not target.exists():
                    out = self._run(c, comp, None, step_seed=c.seed)
                    self._save_particles(out, ref_dir, f"{stem}__{comp}")
                paths[comp] = target
            self.refs[key] = paths
            log.info("references ready for %s", key)

    def _compose(self, path: Path) -> float | None:
        if self.args.no_detector:
            return None
        try:
            if self.detector is None:
                from poe_repair.experiments.compose_scorer_validation.detection_scorer import count_instances
                self.detector = count_instances
            n, _ = self.detector(path, device=self.device)
            return 1.0 if n >= 2 else 0.0
        except Exception as exc:
            log.warning("detector failed on %s: %s", path, exc)
            return None

    def render_step(self, twist: TwistHead, step: int, run_dir: Path, wb: WandB) -> dict:
        twist.eval()
        sub = run_dir / "samples" / f"step_{step:06d}"
        sub.mkdir(parents=True, exist_ok=True)
        agg: dict[str, list[float]] = {"ess": [], "n_resample": [], "compose_mono": [],
                                       "compose_poe": [], "compose_smc": []}
        for c in self.cells:
            key = self._key(c)
            stem = key.replace("/", "__")
            out = self._run(c, "poe", twist, step_seed=c.seed)
            smc_png = self._save_particles(out, sub, f"{stem}__smc")
            refs = self.refs[key]
            strip = _strip(
                [("Mono (target)", refs["mono"]), ("PoE (default)", refs["poe"]),
                 (f"twisted SMC @ step {step}", smc_png)],
                f"{c.pair_slug}  seed {c.seed:02d}  K={self.args.particles} eta={self.args.eta}",
                self.args.thumb,
            )
            strip_path = sub / f"{stem}__strip.png"
            strip.save(strip_path)
            payload: dict[str, float] = {}
            ess_mean = float(sum(out.ess) / max(1, len(out.ess)))
            payload[f"eval/ess_mean/{key}"] = ess_mean
            payload[f"eval/n_resamples/{key}"] = float(len(out.resampled_at))
            agg["ess"].append(ess_mean); agg["n_resample"].append(float(len(out.resampled_at)))
            for name, p in (("mono", refs["mono"]), ("poe", refs["poe"]), ("smc", smc_png)):
                s = self._compose(p)
                if s is not None:
                    payload[f"eval/compose/{name}/{key}"] = s
                    agg[f"compose_{name}"].append(s)
            wb.log(payload, step=step)
            wb.log_image(f"samples/{key}", strip_path, step=step,
                         caption=f"resampled at steps {out.resampled_at}")
            (sub / f"{stem}__smc.json").write_text(json.dumps({
                "ess": out.ess, "resampled_at": out.resampled_at, "ancestry": out.ancestry,
                "log_weights": out.log_weights.tolist(),
                "log_psi_final": None if out.log_psi_final is None else out.log_psi_final.tolist(),
                "picked": self._pick(out),
            }, indent=1))
        summary = {f"eval/{k}_mean": sum(v) / len(v) for k, v in agg.items() if v}
        wb.log(summary, step=step)
        wb.log_artifact_dir(sub, name=f"strips-step{step:06d}", kind="strips",
                            aliases=["latest", f"step{step}"])
        twist.train()
        return summary


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-id", default="auto")
    p.add_argument("--train-max-pairs", type=int, default=None)
    p.add_argument("--train-max-seeds", type=int, default=None)
    p.add_argument("--val-max-pairs", type=int, default=8)
    p.add_argument("--val-max-seeds", type=int, default=2)
    p.add_argument("--render-train-pairs", type=int, default=2)
    p.add_argument("--render-heldout-pairs", type=int, default=1)
    p.add_argument("--render-pairs", nargs="*", default=None,
                   help="explicit pair slugs to render (searched in train then heldout)")
    # twist head
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimisation
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=16, help="positives per batch; negatives match")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--t-min", type=int, default=1)
    p.add_argument("--t-max", type=int, default=999)
    p.add_argument("--no-hflip", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--ckpt-every", type=int, default=10_000)
    p.add_argument("--render-every", type=int, default=10_000)
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--no-detector", action="store_true")
    # sampler
    p.add_argument("--particles", type=int, default=4)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--ess-threshold", type=float, default=0.5)
    p.add_argument("--twist-temperature", type=float, default=1.0)
    p.add_argument("--render-steps", type=int, default=20)
    p.add_argument("--render-height", type=int, default=1024)
    p.add_argument("--render-width", type=int, default=1024)
    p.add_argument("--unet-chunk", type=int, default=2)
    p.add_argument("--thumb", type=int, default=320)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--torch-seed", type=int, default=42)
    # wandb
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", default="poe-repair-animals-compose")
    p.add_argument("--wandb-entity", default="prime_lab")
    p.add_argument("--smoke", action="store_true", help="tiny settings end to end")
    args = p.parse_args(argv)
    if args.smoke:
        args.train_max_pairs = args.train_max_pairs or 2
        args.train_max_seeds = args.train_max_seeds or 2
        args.val_max_pairs, args.val_max_seeds = 1, 1
        args.render_train_pairs, args.render_heldout_pairs = 1, 1
        args.total_steps = min(args.total_steps, 40)
        args.log_every, args.val_every, args.ckpt_every, args.render_every = 5, 10, 20, 20
        args.particles = min(args.particles, 2)
        args.render_steps = min(args.render_steps, 4)
        args.render_height = args.render_width = 512
        args.batch_size = min(args.batch_size, 4)
    return args


def _timestep_buckets(t: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"late_noise": t >= 700, "mid": (t >= 300) & (t < 700), "near_clean": t < 300}


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    torch.manual_seed(args.torch_seed)
    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype, device)

    if args.run_id == "auto":
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_id = f"twist_w{args.width}_b{args.batch_size}_lr{args.lr:.0e}_s{args.total_steps}_{ts}"
        if args.smoke:
            args.run_id = "smoke_" + args.run_id
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                     for k, v in vars(args).items()}, indent=2))
    log.info("run dir %s  device %s  dtype %s", run_dir, device, dtype)

    # ---- cells ------------------------------------------------------------
    train_cells = discover_bank_cells(args.cache_root, "train", max_pairs=args.train_max_pairs,
                                      max_seeds_per_pair=args.train_max_seeds)
    train_pairs = {c.pair_slug for c in train_cells}
    val_cells = discover_bank_cells(args.cache_root, "heldout", max_pairs=args.val_max_pairs,
                                    max_seeds_per_pair=args.val_max_seeds, exclude_pairs=train_pairs)
    if args.render_pairs:
        render_cells = []
        for slug in args.render_pairs:
            found = [c for c in train_cells + val_cells if c.pair_slug == slug]
            if not found:
                found = discover_bank_cells(args.cache_root, "heldout", pair_filter=[slug], max_seeds_per_pair=1)
            if found[0].split == "heldout" and slug in train_pairs:
                raise SystemExit(f"render pair {slug} is a training pair; its heldout copy is not held out")
            render_cells.append(found[0])
    else:
        seen: dict[str, BankCell] = {}
        for c in train_cells:
            seen.setdefault(c.pair_slug, c)
        render_cells = list(seen.values())[: args.render_train_pairs]
        seen = {}
        for c in val_cells:
            seen.setdefault(c.pair_slug, c)
        render_cells += list(seen.values())[: args.render_heldout_pairs]
    log.info("train cells %d (pairs %d), val cells %d, render cells %s",
             len(train_cells), len({c.pair_slug for c in train_cells}), len(val_cells),
             [f"{c.split}/{c.pair_slug}/seed_{c.seed}" for c in render_cells])
    (run_dir / "cells.json").write_text(json.dumps({
        "train": [f"{c.split}/{c.pair_slug}/seed_{c.seed}" for c in train_cells],
        "val": [f"{c.split}/{c.pair_slug}/seed_{c.seed}" for c in val_cells],
        "render": [f"{c.split}/{c.pair_slug}/seed_{c.seed}" for c in render_cells],
    }, indent=1))

    # ---- models and banks ---------------------------------------------------
    models = load_sdxl_models(model_id=args.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(args.model_id)
    bank_dir = args.output_root / "latent_bank"
    train_bank = build_or_load_bank(train_cells, bank_dir=bank_dir, vae=models["vae"], device=device)
    val_bank = build_or_load_bank(val_cells, bank_dir=bank_dir, vae=models["vae"], device=device)
    train_sampler = ContrastiveSampler(train_bank, alphas_cumprod=scheduler.alphas_cumprod,
                                       batch_size=args.batch_size, device=device, seed=args.torch_seed,
                                       t_min=args.t_min, t_max=args.t_max, hflip=not args.no_hflip)
    val_sampler = ContrastiveSampler(val_bank, alphas_cumprod=scheduler.alphas_cumprod,
                                     batch_size=args.batch_size, device=device, seed=999,
                                     t_min=args.t_min, t_max=args.t_max, hflip=False)
    val_batches = [val_sampler.next_batch() for _ in range(8)]

    twist = TwistHead(width=args.width, num_blocks=args.num_blocks, dropout=args.dropout).to(device)
    log.info("twist head: %.2fM parameters", twist.num_parameters() / 1e6)
    opt = torch.optim.AdamW(twist.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    wb = WandB(mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_entity,
               name=args.run_id, run_dir=run_dir, tags=["twisted-smc", "twist-head"],
               config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})

    renderer = None
    if not args.skip_render:
        renderer = Renderer(args, render_cells, models, device, dtype)
        t0 = time.time()
        renderer.render_references(run_dir)
        log.info("references rendered in %.0fs", time.time() - t0)

    def checkpoint(step: int) -> Path:
        ck = run_dir / "checkpoints"
        ck.mkdir(exist_ok=True)
        path = ck / f"twist_step_{step:06d}.pt"
        torch.save({"twist_state": twist.state_dict(), "step": step, "opt_state": opt.state_dict(),
                    "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, path)
        (ck / "latest.json").write_text(json.dumps({"path": str(path), "step": step}))
        return path

    @torch.no_grad()
    def validate(step: int) -> None:
        twist.eval()
        losses, accs = [], []
        for b in val_batches:
            logit = twist(b["x_t"], b["timestep"], b["cond"])
            losses.append(F.binary_cross_entropy_with_logits(logit, b["y"]).item())
            accs.append(((logit > 0).float() == b["y"]).float().mean().item())
        val_state["acc"] = sum(accs) / len(accs)
        wb.log({"val/loss": sum(losses) / len(losses), "val/acc": val_state["acc"]}, step=step)
        twist.train()

    val_state: dict[str, float] = {"acc": float("nan")}

    def verdict(summary: dict, step: int) -> dict:
        smc = summary.get("eval/compose_smc_mean")
        poe = summary.get("eval/compose_poe_mean")
        mono = summary.get("eval/compose_mono_mean")
        v = {"step": step, "compose_smc": smc, "compose_poe": poe, "compose_mono": mono,
             "val_acc": val_state["acc"], "pass_margin": PASS_MARGIN, "null_margin": NULL_MARGIN,
             "min_val_acc": MIN_VAL_ACC}
        if smc is None or poe is None:
            v["verdict"] = "no detector read"
        elif not (val_state["acc"] >= MIN_VAL_ACC):
            v["verdict"] = "inconclusive: twist did not learn"
        elif smc - poe >= PASS_MARGIN:
            v["verdict"] = "pass: resampling on the twist lifts compose over the PoE control"
        elif abs(smc - poe) <= NULL_MARGIN:
            v["verdict"] = "null: the twist finds nothing the product proposes"
        else:
            v["verdict"] = "between the bars"
        (run_dir / "verdict.json").write_text(json.dumps(v, indent=2))
        log.info("verdict: %s", v)
        return v

    # ---- loop ---------------------------------------------------------------
    twist.train()
    t_start = time.time()
    try:
        if renderer is not None:
            renderer.render_step(twist, 0, run_dir, wb)
        for step in range(1, args.total_steps + 1):
            b = train_sampler.next_batch()
            logit = twist(b["x_t"], b["timestep"], b["cond"])
            loss = F.binary_cross_entropy_with_logits(logit, b["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(twist.parameters(), 5.0)
            opt.step()
            if step % args.log_every == 0 or step == 1:
                with torch.no_grad():
                    per = F.binary_cross_entropy_with_logits(logit, b["y"], reduction="none")
                    acc = ((logit > 0).float() == b["y"]).float().mean().item()
                    payload = {"train/loss": loss.item(), "train/acc": acc, "train/grad_norm": float(gn),
                               "train/lr": args.lr, "train/logit_pos_mean": logit[b["y"] > 0.5].mean().item(),
                               "train/logit_neg_mean": logit[b["y"] < 0.5].mean().item(),
                               "train/steps_per_sec": step / max(1e-6, time.time() - t_start)}
                    for name, mask in _timestep_buckets(b["timestep"]).items():
                        if mask.any():
                            payload[f"train/loss_bucket/{name}"] = per[mask].mean().item()
                wb.log(payload, step=step)
                log.info("step %d loss %.4f acc %.3f", step, loss.item(), acc)
            if step % args.val_every == 0:
                validate(step)
            if step % args.ckpt_every == 0:
                checkpoint(step)
            if renderer is not None and step % args.render_every == 0:
                t0 = time.time()
                summary = renderer.render_step(twist, step, run_dir, wb)
                log.info("render at step %d in %.0fs: %s", step, time.time() - t0, summary)
                if step == args.total_steps:
                    verdict(summary, step)
    finally:
        checkpoint(step if "step" in locals() else 0)
        wb.finish()
    log.info("done: %s", run_dir)


if __name__ == "__main__":
    main()
