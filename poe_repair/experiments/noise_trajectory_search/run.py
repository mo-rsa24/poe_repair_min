"""Runner for the noise trajectory search on the corrected sampler (scope 01, plan 16).

Per pair and seed, five renders from the seed's pinned initial noise:

    mono_eta0       the joint prompt, plain CFG, deterministic DDIM        (reference)
    poe_eta0        plain product-of-experts, deterministic DDIM           (reference)
    adapter_eta0    PoE + lambda x the rank-32 correction, deterministic   (the shipped render)
    control_eta1    the same corrected sampler at eta 1, one noise draw per step, no search
    search_eta1     the same run, the same draws, except inside SEARCH_STEPS where each draw
                    is the pivot of a greedy search over candidates judged one step ahead

Every render is scored (instance count, ImageReward, sharpness, both-ness and which-animal on
the landing finding's DINOv2 axes, distance to the seed's joint-prompt render), the running
estimates at FRAME_STEPS are saved and embedded, and the pair gets a sheet, a three-column
strip, four figures, ``summary.json`` and ``verdict.json``, all logged to W&B.

    <co3 python> -m poe_repair.experiments.noise_trajectory_search.run --smoke
    <co3 python> -m poe_repair.experiments.noise_trajectory_search.run --pairs a_cat__x__a_dog a_butterfly__x__a_flower_meadow
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

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from poe_repair._sdxl.runtime import (encode_prompt_sdxl, infer_device, infer_dtype,
                                      load_ddim_scheduler, load_sdxl_models)
from poe_repair.experiments.fk_steering.run import attach_adapter, pinned_init
from poe_repair.experiments.noise_trajectory_search import search
from poe_repair.experiments.twisted_smc.sampler import decode_one, run_particles
from poe_repair.experiments.twisted_smc.train import WandB
from poe_repair.methods._sampling import write_decoded_image

log = logging.getLogger("noise_trajectory_search")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path("/datasets/mmolefe/poe_repair_min/outputs/showcase/noise_trajectory_search")
DEFAULT_CACHE = Path("/datasets/mmolefe/poe_repair_min/outputs/training_cache")
CLOUD_FEATS = REPO_ROOT / "artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space-dino-feats.npy"
PROMPTS = {
    "a_cat__x__a_dog": ("a cat", "a dog", "a cat and a dog", "cat", "dog"),
    "a_butterfly__x__a_flower_meadow": ("a butterfly", "a flower meadow", "a butterfly and a flower meadow",
                                        "butterfly", "flower meadow"),
}
JUDGED_PAIR = "a_cat__x__a_dog"
CONTROL_PAIR = "a_butterfly__x__a_flower_meadow"
CONDITIONS = ("mono_eta0", "poe_eta0", "adapter_eta0", "control_eta1", "search_eta1")
LABELS = {"mono_eta0": "Mono, joint prompt\nDDIM eta 0",
          "poe_eta0": "plain PoE\nDDIM eta 0",
          "adapter_eta0": "PoE + correction\nDDIM eta 0 (shipped)",
          "control_eta1": "PoE + correction\neta 1, no search",
          "search_eta1": "PoE + correction\neta 1 + noise search"}
COLORS = {"poe_eta0": "#111111", "adapter_eta0": "#e7298a", "control_eta1": "#7570b3", "search_eta1": "#d95f02",
          "mono_eta0": "#1b9e77"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--pairs", nargs="+", default=[JUDGED_PAIR, CONTROL_PAIR])
    p.add_argument("--seeds", nargs="+", type=int, default=[9, 10, 11, 12, 13, 14, 15, 16])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=search.ETA)
    p.add_argument("--adapter-lambda", type=float, default=search.LAMBDA_ADAPTER)
    p.add_argument("--adapter-checkpoint", type=Path, default=Path(search.ADAPTER_CHECKPOINT))
    p.add_argument("--adapter-rank", type=int, default=search.ADAPTER_RANK)
    p.add_argument("--search-steps", nargs="+", type=int, default=list(search.SEARCH_STEPS))
    p.add_argument("--num-candidates", type=int, default=search.NUM_CANDIDATES)
    p.add_argument("--num-rounds", type=int, default=search.NUM_ROUNDS)
    p.add_argument("--epsilon", type=float, default=search.EPSILON)
    p.add_argument("--local-sigma", type=float, default=search.LOCAL_SIGMA)
    p.add_argument("--frame-steps", nargs="+", type=int, default=list(search.FRAME_STEPS))
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--thumb", type=int, default=256)
    p.add_argument("--run-id", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", default="poe-repair-animals-compose")
    p.add_argument("--wandb-entity", default="prime_lab")
    p.add_argument("--smoke", action="store_true", help="cat x dog seed 9, 10 steps, 512², search steps 2 to 5, 1 round of 2")
    a = p.parse_args(argv)
    if a.smoke:
        a.pairs, a.seeds, a.steps, a.size, a.wandb_mode = [JUDGED_PAIR], [9], 10, 512, "disabled"
        a.search_steps, a.num_candidates, a.num_rounds = [2, 3, 4, 5], 2, 1
        a.frame_steps = [0, 2, 4, 6, 8, 9]
    return a


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def laplacian_var(png: Path) -> float:
    img = np.asarray(Image.open(png).convert("L"), dtype=np.float32)
    gy, gx = np.gradient(img)
    return float(np.var(gx[1:] - gx[:-1]) + np.var(gy[:, 1:] - gy[:, :-1]))


def cloud_axes():
    """The landing finding's plane: 48 x 384 DINOv2 features (solo_a, solo_b, joint, poe, lora_1.0,
    lora_1.2; 8 seeds each). x is cat centroid to dog centroid, y is the solo midpoint toward the
    joint-prompt centroid, orthogonalised against x."""
    feats = np.load(CLOUD_FEATS)
    ca, cb, cj = feats[0:8].mean(0), feats[8:16].mean(0), feats[16:24].mean(0)
    origin = 0.5 * (ca + cb)
    u1 = cb - ca; u1 /= np.linalg.norm(u1)
    u2 = cj - origin; u2 -= (u2 @ u1) * u1; u2 /= np.linalg.norm(u2)
    clouds = {"solo_a": feats[0:8], "solo_b": feats[8:16], "joint": feats[16:24], "poe_ref": feats[24:32],
              "lora_1.2_ref": feats[40:48]}
    proj = {k: [(float((f - origin) @ u1), float((f - origin) @ u2)) for f in v] for k, v in clouds.items()}
    return origin, u1, u2, proj


class Scorer:
    """Instance count (the validated compose scorer), per-query confidences, ImageReward against
    the joint prompt, sharpness, and DINOv2 CLS features."""

    def __init__(self, device: torch.device):
        from poe_repair.experiments.compose_scorer_validation.detection_scorer import (
            count_instances, instance_score_to_dict, score_output_instances)
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.build_lora_inspector_mds_semantic import DinoEmbedder
        import ImageReward as RM
        self.device = device
        self._count, self._per_query, self._to_dict = count_instances, score_output_instances, instance_score_to_dict
        self.rm = RM.load("ImageReward-v1.0", device=str(device))
        self.dino = DinoEmbedder(device=device)   # on the GPU: on CPU DINOv2 hits the CUDA-only xformers kernel (poe-mem-002)
        self.prompt = None

    def count(self, png: Path) -> int:
        n, _ = self._count(Path(png), device=self.device)
        return int(n)

    def fidelity(self, png: Path) -> float:
        return float(self.rm.score(self.prompt, str(png)))

    def reward(self, png: Path):
        return self.count(png), self.fidelity(png)

    def embed(self, png: Path) -> np.ndarray:
        im = torch.from_numpy(np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        return self.dino.embed_decoded_batch(im)[0]

    def embed_many(self, pngs: list[Path]) -> np.ndarray:
        ims = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1) for p in pngs]
        return self.dino.embed_decoded_batch(torch.stack(ims))

    def final(self, png: Path, qa: str, qb: str) -> dict:
        d = self._to_dict(self._per_query(Path(png), qa, qb, device=self.device))
        return {"count": int(d["n_instances"]), "compose": int(d["n_instances"] >= 2),
                "conf_a": float(d["conf_a"]), "conf_b": float(d["conf_b"]),
                "presence_a": int(d["conf_a"] >= search.BUTTERFLY_PRESENT_CONF),
                "imagereward": self.fidelity(png), "sharpness": laplacian_var(png)}


# ---------------------------------------------------------------------------
# Sheets and strips
# ---------------------------------------------------------------------------


def _font(size):
    for cand in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def draw_sheet(rows: list[dict], columns: list[tuple[str, str]], title: str, out_png: Path, thumb: int,
               judged: bool, caption_lines: int = 2) -> None:
    left, top, gap, cap = 90, 74, 6, 16 * caption_lines
    W = left + len(columns) * (thumb + gap)
    H = top + len(rows) * (thumb + cap + gap)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    f_big, f_small, f_tag = _font(18), _font(13), _font(20)
    d.text((8, 6), title, fill="black", font=f_big)
    for c, (key, label) in enumerate(columns):
        x = left + c * (thumb + gap)
        for i, line in enumerate(label.split("\n")):
            d.text((x + 2, 32 + 15 * i), line, fill="black", font=f_small)
    for r, row in enumerate(rows):
        y = top + r * (thumb + cap + gap)
        d.text((8, y + thumb // 2 - 8), f"seed {row['seed']}", fill="black", font=f_small)
        for c, (key, _) in enumerate(columns):
            x = left + c * (thumb + gap)
            t = row["renders"].get(key)
            if t and Path(t["png"]).exists():
                sheet.paste(Image.open(t["png"]).convert("RGB").resize((thumb, thumb), Image.LANCZOS), (x, y))
                n = t["count"]
                col = (20, 140, 40) if n >= 2 else (200, 40, 40)
                d.rectangle([x + 4, y + 4, x + 34, y + 30], fill=col)
                d.text((x + 10, y + 4), str(n), fill="white", font=f_tag)
                l1 = f"sharp {t['sharpness']:.0f}  IR {t['imagereward']:+.2f}"
                l2 = (f"both {t['both_ness']:.2f}  to-mono {t['dist_to_mono']:.2f}" if judged
                      else f"butterfly conf {t['conf_a']:.2f}")
                d.text((x + 2, y + thumb + 1), l1, fill="black", font=f_small)
                d.text((x + 2, y + thumb + 16), l2, fill="black", font=f_small)
            else:
                d.rectangle([x, y, x + thumb, y + thumb], outline="red", width=3)
                d.text((x + 20, y + thumb // 2), "missing", fill="red", font=f_small)
    sheet.save(out_png)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_tracks(cells: list[dict], proj_ref: dict, search_steps: list[int], out_png: Path, pair: str) -> None:
    """Running estimates of every seed as a track in the landing finding's DINOv2 plane, one panel
    per condition. x: cat centroid (left) to dog centroid (right); y: solo midpoint toward the
    joint-prompt centroid (both-ness). Reference clouds from the landing finding are drawn faint."""
    plt = _plt()
    conds = ["poe_eta0", "adapter_eta0", "control_eta1", "search_eta1"]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.6 * len(conds), 4.8), sharex=True, sharey=True)
    for ax, cond in zip(axes, conds):
        for name, col, lab in (("solo_a", "#d95f02", "a cat alone"), ("solo_b", "#1b9e77", "a dog alone"),
                               ("joint", "#7570b3", "\"a cat and a dog\"")):
            pts = np.array(proj_ref[name])
            ax.scatter(pts[:, 0], pts[:, 1], s=60, c=col, alpha=0.18, edgecolors="none", label=lab)
            ax.scatter(pts[:, 0].mean(), pts[:, 1].mean(), s=160, c=col, marker="*", edgecolors="black", linewidths=0.5)
        for cell in cells:
            tr = cell["frames"].get(cond)
            if not tr:
                continue
            steps = sorted(int(s) for s in tr)
            xy = np.array([tr[str(s)]["xy"] for s in steps])
            ax.plot(xy[:, 0], xy[:, 1], "-", color=COLORS[cond], lw=0.9, alpha=0.6)
            inwin = [i for i, s in enumerate(steps) if s in search_steps]
            if inwin and cond == "search_eta1":
                ax.plot(xy[inwin, 0], xy[inwin, 1], "-", color=COLORS[cond], lw=2.4, alpha=0.9)
            ax.scatter(xy[0, 0], xy[0, 1], s=14, c="grey", zorder=3)
            ax.scatter(xy[-1, 0], xy[-1, 1], s=42, c=COLORS[cond], edgecolors="black", linewidths=0.6, zorder=4)
            ax.annotate(str(cell["seed"]), xy[-1], fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(LABELS[cond].replace("\n", ", "), fontsize=10)
        ax.set_xlabel("cat centroid  ←  which animal  →  dog centroid")
        ax.axhline(0, color="#bbb", lw=0.6); ax.axvline(0, color="#bbb", lw=0.6)
    axes[0].set_ylabel("both-ness: toward the joint-prompt centroid")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{pair.replace('__x__', ' x ').replace('_', ' ')}: running estimates over the run in the DINOv2 plane "
                 f"(grey dot: step 0, ring: final; thick: the searched steps {search_steps[0]} to {search_steps[-1]})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140); plt.close(fig)


def fig_over_steps(cells: list[dict], search_steps: list[int], out_png: Path, pair: str, judged: bool) -> None:
    """Both-ness (cat x dog only) and sharpness of the running estimate against the denoising step,
    thin per seed, thick mean per condition, the searched steps shaded."""
    plt = _plt()
    conds = ["poe_eta0", "adapter_eta0", "control_eta1", "search_eta1"]
    panels = (["both_ness", "sharpness"] if judged else ["sharpness"])
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.4), squeeze=False)
    for ax, key in zip(axes[0], panels):
        ax.axvspan(search_steps[0] - 0.5, search_steps[-1] + 0.5, color="#f2e6d9", zorder=0, label="searched steps")
        for cond in conds:
            per_seed = []
            for cell in cells:
                tr = cell["frames"].get(cond)
                if not tr:
                    continue
                steps = sorted(int(s) for s in tr)
                vals = [tr[str(s)][key] for s in steps]
                per_seed.append((steps, vals))
                ax.plot(steps, vals, "-", color=COLORS[cond], lw=0.6, alpha=0.35)
            if per_seed:
                steps = per_seed[0][0]
                mean = np.mean([v for _, v in per_seed], axis=0)
                ax.plot(steps, mean, "-", color=COLORS[cond], lw=2.4, label=LABELS[cond].replace("\n", ", "))
        ax.set_xlabel("denoising step (0 noise, 50 image)")
        ax.set_ylabel("both-ness of the running estimate (DINOv2 plane, y)" if key == "both_ness"
                      else "sharpness of the 256 px running estimate (Laplacian variance)")
        if key == "sharpness":
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{pair.replace('__x__', ' x ').replace('_', ' ')}: thin lines are seeds, thick lines the mean over seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140); plt.close(fig)


def fig_reward(cells: list[dict], search_steps: list[int], out_png: Path, pair: str) -> None:
    """The reward of the control's own draw (the pivot) against the chosen draw at every searched
    step, and which kind of candidate won."""
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    piv, cho, kinds = [], [], {"pivot": np.zeros(len(search_steps)), "local": np.zeros(len(search_steps)), "fresh": np.zeros(len(search_steps))}
    for cell in cells:
        s = cell["search"]
        pr = [s["pivot_reward_by_step"].get(str(k), np.nan) for k in search_steps]
        cr = [s["chosen_reward_by_step"].get(str(k), np.nan) for k in search_steps]
        piv.append(pr); cho.append(cr)
        a1.plot(search_steps, pr, "-", color=COLORS["control_eta1"], lw=0.6, alpha=0.35)
        a1.plot(search_steps, cr, "-", color=COLORS["search_eta1"], lw=0.6, alpha=0.35)
        for i, k in enumerate(search_steps):
            kd = s["chosen_kind_by_step"].get(str(k))
            if kd in kinds:
                kinds[kd][i] += 1
    if piv:
        a1.plot(search_steps, np.nanmean(piv, axis=0), "-", color=COLORS["control_eta1"], lw=2.4, label="the control's draw (pivot)")
        a1.plot(search_steps, np.nanmean(cho, axis=0), "-", color=COLORS["search_eta1"], lw=2.4, label="the chosen draw")
    a1.set_xlabel("denoising step"); a1.set_ylabel("reward = min(count, 2) + 0.5 sigmoid(ImageReward)")
    a1.set_title("reward of the running estimate one step ahead", fontsize=10); a1.legend(fontsize=8); a1.grid(alpha=0.25)
    bottom = np.zeros(len(search_steps))
    for kd, col in (("pivot", COLORS["control_eta1"]), ("local", COLORS["search_eta1"]), ("fresh", "#1b9e77")):
        a2.bar(search_steps, kinds[kd], bottom=bottom, color=col, label=kd); bottom += kinds[kd]
    a2.set_xlabel("denoising step"); a2.set_ylabel("seeds"); a2.set_title("which candidate won: the control's own draw, a local nudge, or a fresh draw", fontsize=10)
    a2.legend(fontsize=8)
    fig.suptitle(pair.replace("__x__", " x ").replace("_", " "), fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def fig_paired(cells: list[dict], out_png: Path, pair: str, judged: bool) -> None:
    """Per seed, the control at eta 1 against the searched run: sharpness, both-ness (or butterfly
    confidence), ImageReward, distance to the joint-prompt render. One line per seed."""
    plt = _plt()
    keys = [("sharpness", "sharpness at 1024 px (Laplacian variance, log)", True),
            ("both_ness" if judged else "conf_a", "both-ness (DINOv2 plane, y)" if judged else "butterfly confidence", False),
            ("imagereward", "ImageReward against the joint prompt", False)]
    if judged:
        keys.append(("dist_to_mono", "DINOv2 cosine distance to the seed's joint-prompt render (lower is nearer)", False))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 4.4))
    for ax, (key, lab, logy) in zip(axes, keys):
        for cell in cells:
            c, s = cell["renders"]["control_eta1"][key], cell["renders"]["search_eta1"][key]
            better = (s > c) if key != "dist_to_mono" else (s < c)
            ax.plot([0, 1], [c, s], "-o", color="#2ca02c" if better else "#d62728", lw=1.2, ms=4, alpha=0.8)
            ax.annotate(str(cell["seed"]), (1, s), fontsize=7, xytext=(4, 0), textcoords="offset points")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["control\neta 1", "search\neta 1"])
        import textwrap
        ax.set_title("\n".join(textwrap.wrap(lab, 34)), fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{pair.replace('__x__', ' x ').replace('_', ' ')}: one line per seed, green where the searched run is better", fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


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
    run_id = a.run_id or (("smoke_" if a.smoke else "") + f"nts_lam{a.adapter_lambda:g}_s{a.steps}_{stamp}")
    run_dir = a.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    constants = {k: getattr(search, k) for k in ("SUPPORT_MAX_COMPOSE_LOSS", "NULL_MIN_COMPOSE_LOSS", "SUPPORT_MIN_SHARPER_SEEDS",
                                                  "NULL_MAX_SHARPER_SEEDS", "CONTROL_MAX_PRESENCE_LOSS", "BUTTERFLY_PRESENT_CONF",
                                                  "ETA", "LAMBDA_ADAPTER", "SEARCH_STEPS", "NUM_CANDIDATES", "NUM_ROUNDS",
                                                  "EPSILON", "LOCAL_SIGMA", "REWARD_CLIP", "FIDELITY_WEIGHT", "FRAME_STEPS")}
    config = {**vars(a), "out_root": str(a.out_root), "cache_root": str(a.cache_root), "adapter_checkpoint": str(a.adapter_checkpoint),
              "run_id": run_id, "node": os.uname().nodename, "pid": os.getpid(), "constants": constants}
    (run_dir / "config.json").write_text(json.dumps(config, indent=1, default=str))
    log.info("run dir %s (node %s, pid %d)", run_dir, os.uname().nodename, os.getpid())

    models = load_sdxl_models(model_id=a.model_id, device=device, dtype=dtype)
    scheduler = load_ddim_scheduler(a.model_id)
    scorer = Scorer(device)
    wb = WandB(mode=a.wandb_mode, project=a.wandb_project, entity=a.wandb_entity, name=run_id, run_dir=run_dir,
               config=json.loads((run_dir / "config.json").read_text()), tags=["noise-trajectory-search", "scope-01", "plan-16"],
               group="noise-trajectory-search")
    if wb.run is not None:
        log.info("wandb run id %s url %s", wb.run.id, wb.run.url)
    h = w = a.size // 8
    enc = lambda t: encode_prompt_sdxl(t, models=models, device=device, dtype=dtype)  # noqa: E731
    seq_e, pool_e = enc("")
    common = dict(models=models, scheduler=scheduler, seq_e=seq_e, pool_e=pool_e, guidance_scale=a.guidance_scale,
                  num_inference_steps=a.steps, height=a.size, width=a.size, device=device, dtype=dtype)
    origin, u1, u2, proj_ref = cloud_axes()

    def project(f: np.ndarray) -> tuple[float, float]:
        return float((f - origin) @ u1), float((f - origin) @ u2)

    summary: dict = {"run_id": run_id, "constants": constants, "settings": {"steps": a.steps, "eta": a.eta, "adapter_lambda": a.adapter_lambda,
                     "search_steps": a.search_steps, "num_candidates": a.num_candidates, "num_rounds": a.num_rounds, "epsilon": a.epsilon,
                     "local_sigma": a.local_sigma, "size": a.size, "guidance": a.guidance_scale, "frame_steps": a.frame_steps}, "pairs": {}}
    counter = 0
    adapter_info = None
    # Mono and plain PoE are rendered for every seed before the adapter is attached (memory note 2026-09-05:
    # a windowed sampler once left the adapter enabled and contaminated the references).
    pre = {}
    for pair in a.pairs:
        pa, pb, pj, qa, qb = PROMPTS[pair]
        seq_a, pool_a = enc(pa); seq_b, pool_b = enc(pb); seq_j, pool_j = enc(pj)
        for seed in a.seeds:
            sd = run_dir / pair / f"seed_{seed}"; sd.mkdir(parents=True, exist_ok=True)
            base, origin_noise = pinned_init(a.cache_root, pair, seed, h, w)
            pre[(pair, seed)] = origin_noise
            png = sd / "mono_eta0.png"
            if not png.exists():
                out = run_particles(init_latents=base[None], composition="mono", seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                                    seq_j=seq_j, pool_j=pool_j, eta=0.0, generator=None, twist=None, unet_chunk=1, **common)
                write_decoded_image(decode_one(models, out.latents[0]), png)
            png = sd / "poe_eta0.png"
            if not png.exists():
                g = torch.Generator(device="cpu").manual_seed(seed)
                out = search.run_search(init_latents=base[None], seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                                        adapter_lambda=0.0, eta=0.0, noise_generator=g, reward_fn=None,
                                        work_dir=sd / "poe_eta0", frame_steps=tuple(a.frame_steps), **common)
                write_decoded_image(decode_one(models, out.latents[0]), png)
                search.dump(out, sd / "poe_eta0" / "run.json")
            log.info("[%s] seed %d references done (init noise from %s)", pair, seed, origin_noise)
    adapter_info = attach_adapter(models["unet"], a.adapter_checkpoint, a.adapter_rank, a.adapter_rank)
    (run_dir / "adapter.json").write_text(json.dumps(adapter_info, indent=1, default=str))
    log.info("adapter attached: %s", adapter_info)

    for pair in a.pairs:
        pa, pb, pj, qa, qb = PROMPTS[pair]
        judged = pair == JUDGED_PAIR
        scorer.prompt = pj
        seq_a, pool_a = enc(pa); seq_b, pool_b = enc(pb)
        pair_dir = run_dir / pair
        cells = []
        for seed in a.seeds:
            t0 = time.time()
            sd = pair_dir / f"seed_{seed}"
            cell_path = sd / "cell.json"
            if cell_path.exists():
                cells.append(json.loads(cell_path.read_text()))
                log.info("[%s] seed %d already scored, reusing cell.json", pair, seed)
                continue
            base, _ = pinned_init(a.cache_root, pair, seed, h, w)
            cell: dict = {"pair": pair, "seed": seed, "init_noise_from": pre.get((pair, seed)), "renders": {}, "frames": {}, "search": {}, "timing": {}}
            runs = {}
            for cond in ("adapter_eta0", "control_eta1", "search_eta1"):
                png = sd / f"{cond}.png"
                t1 = time.time()
                if not png.exists() or not (sd / cond / "run.json").exists():
                    g = torch.Generator(device="cpu").manual_seed(seed)
                    gs = torch.Generator(device="cpu").manual_seed(seed + 7_919)
                    out = search.run_search(init_latents=base[None], seq_a=seq_a, pool_a=pool_a, seq_b=seq_b, pool_b=pool_b,
                                            adapter_lambda=a.adapter_lambda, eta=(0.0 if cond == "adapter_eta0" else a.eta),
                                            noise_generator=g, search_generator=gs,
                                            reward_fn=(scorer.reward if cond == "search_eta1" else None),
                                            search_steps=tuple(a.search_steps), num_candidates=a.num_candidates, num_rounds=a.num_rounds,
                                            epsilon=a.epsilon, local_sigma=a.local_sigma,
                                            work_dir=sd / cond, frame_steps=tuple(a.frame_steps), **common)
                    write_decoded_image(decode_one(models, out.latents[0]), png)
                    search.dump(out, sd / cond / "run.json", extra={"seconds": time.time() - t1})
                    del out; torch.cuda.empty_cache()
                runs[cond] = json.loads((sd / cond / "run.json").read_text())
                cell["timing"][cond] = runs[cond].get("seconds")
            runs["poe_eta0"] = json.loads((sd / "poe_eta0" / "run.json").read_text())
            # score the five finals
            mono_feat = scorer.embed(sd / "mono_eta0.png")
            for cond in CONDITIONS:
                png = sd / f"{cond}.png"
                rec = scorer.final(png, qa, qb)
                f = scorer.embed(png)
                rec["png"] = str(png)
                rec["dist_to_mono"] = float(1.0 - f @ mono_feat)
                if judged:
                    rec["which_animal"], rec["both_ness"] = project(f)
                cell["renders"][cond] = rec
            # embed the running estimates
            for cond in ("poe_eta0", "adapter_eta0", "control_eta1", "search_eta1"):
                fr = runs[cond]["frames"]
                steps = sorted(int(s) for s in fr)
                feats = scorer.embed_many([Path(fr[str(s)]) for s in steps])
                cell["frames"][cond] = {}
                for s, f in zip(steps, feats):
                    xy = project(f)
                    cell["frames"][cond][str(s)] = {"png": fr[str(s)], "xy": list(xy), "both_ness": xy[1], "which_animal": xy[0],
                                                    "sharpness": laplacian_var(Path(fr[str(s)])), "dist_to_mono": float(1.0 - f @ mono_feat)}
            cell["search"] = {k: runs["search_eta1"][k] for k in ("pivot_reward_by_step", "chosen_reward_by_step", "chosen_kind_by_step",
                                                                   "chosen_count_by_step", "chosen_fidelity_by_step", "evaluations")}
            cell["seconds"] = time.time() - t0
            cell_path.write_text(json.dumps(cell, indent=1))
            cells.append(cell)
            counter += 1
            payload = {f"progress/{pair}/seed": seed, f"progress/{pair}/seconds": cell["seconds"]}
            for cond in CONDITIONS:
                r = cell["renders"][cond]
                for k in ("count", "compose", "sharpness", "imagereward", "dist_to_mono", "conf_a"):
                    payload[f"final/{pair}/{cond}/{k}"] = r[k]
                if judged:
                    payload[f"final/{pair}/{cond}/both_ness"] = r["both_ness"]
            for s in a.search_steps:
                pr, cr = cell["search"]["pivot_reward_by_step"].get(str(s)), cell["search"]["chosen_reward_by_step"].get(str(s))
                if pr is not None:
                    payload[f"reward/{pair}/seed_{seed}/gain_step_{s:02d}"] = cr - pr
            wb.log(payload, step=counter)
            log.info("[%s] seed %d done in %.0f s: counts %s, sharpness %s, evaluations %d", pair, seed, cell["seconds"],
                     {c: cell["renders"][c]["count"] for c in CONDITIONS},
                     {c: round(cell["renders"][c]["sharpness"]) for c in CONDITIONS}, cell["search"]["evaluations"])

        # -- pair-level outputs ------------------------------------------------------------
        n = len(cells)
        columns = [(c, LABELS[c]) for c in CONDITIONS]
        sheet_png = run_dir / f"sheet_{pair}.png"
        draw_sheet(cells, columns, f"{pair.replace('__x__', ' x ').replace('_', ' ')}: {a.steps} DDIM steps, guidance {a.guidance_scale:g}, "
                   f"{a.size}², correction lambda {a.adapter_lambda:g}; searched steps {a.search_steps[0]} to {a.search_steps[-1]}, "
                   f"{a.num_rounds} rounds of {a.num_candidates}; the number on a tile is the detector's animal count (green = 2 or more)",
                   sheet_png, a.thumb, judged)
        strip_png = run_dir / f"strip_{pair}.png"
        draw_sheet(cells, [("mono_eta0", "Mono, joint prompt"), ("poe_eta0", "plain PoE"), ("search_eta1", "PoE + correction\n+ noise search")],
                   f"{pair.replace('__x__', ' x ').replace('_', ' ')}: Mono, plain PoE, and the corrected sampler with the noise searched in steps "
                   f"{a.search_steps[0]} to {a.search_steps[-1]}", strip_png, a.thumb, judged)
        figs = {}
        if judged:
            figs["tracks"] = run_dir / f"tracks_in_dino_plane_{pair}.png"
            fig_tracks(cells, proj_ref, a.search_steps, figs["tracks"], pair)
        figs["over_steps"] = run_dir / f"over_steps_{pair}.png"
        fig_over_steps(cells, a.search_steps, figs["over_steps"], pair, judged)
        figs["reward"] = run_dir / f"reward_over_steps_{pair}.png"
        fig_reward(cells, a.search_steps, figs["reward"], pair)
        figs["paired"] = run_dir / f"paired_per_seed_{pair}.png"
        fig_paired(cells, figs["paired"], pair, judged)

        per_cond = {}
        for cond in CONDITIONS:
            rs = [c["renders"][cond] for c in cells]
            per_cond[cond] = {"compose_n": sum(r["compose"] for r in rs), "compose_rate": sum(r["compose"] for r in rs) / n,
                              "presence_n": sum(r["presence_a"] for r in rs),
                              "sharpness_mean": float(np.mean([r["sharpness"] for r in rs])),
                              "sharpness_min": float(min(r["sharpness"] for r in rs)), "sharpness_max": float(max(r["sharpness"] for r in rs)),
                              "imagereward_mean": float(np.mean([r["imagereward"] for r in rs])),
                              "dist_to_mono_mean": float(np.mean([r["dist_to_mono"] for r in rs])),
                              "per_seed": {str(c["seed"]): {k: c["renders"][cond][k] for k in
                                                            ["count", "compose", "sharpness", "imagereward", "dist_to_mono", "conf_a"] + (["both_ness"] if judged else [])}
                                           for c in cells}}
            if judged:
                per_cond[cond]["both_ness_mean"] = float(np.mean([r["both_ness"] for r in rs]))
        paired = {"sharper_seeds": sum(c["renders"]["search_eta1"]["sharpness"] > c["renders"]["control_eta1"]["sharpness"] for c in cells),
                  "higher_imagereward_seeds": sum(c["renders"]["search_eta1"]["imagereward"] > c["renders"]["control_eta1"]["imagereward"] for c in cells),
                  "nearer_mono_seeds": sum(c["renders"]["search_eta1"]["dist_to_mono"] < c["renders"]["control_eta1"]["dist_to_mono"] for c in cells),
                  "sharper_than_shipped_seeds": sum(c["renders"]["search_eta1"]["sharpness"] > c["renders"]["adapter_eta0"]["sharpness"] for c in cells)}
        if judged:
            paired["higher_both_ness_seeds"] = sum(c["renders"]["search_eta1"]["both_ness"] > c["renders"]["control_eta1"]["both_ness"] for c in cells)
        gains = [c["search"]["chosen_reward_by_step"][s] - c["search"]["pivot_reward_by_step"][s]
                 for c in cells for s in c["search"]["chosen_reward_by_step"]]
        kinds = [k for c in cells for k in c["search"]["chosen_kind_by_step"].values()]
        secondary = {"mean_reward_gain_per_searched_step": float(np.mean(gains)) if gains else None,
                     "chosen_kind_counts": {k: kinds.count(k) for k in ("pivot", "local", "fresh")},
                     "evaluations_per_seed": float(np.mean([c["search"]["evaluations"] for c in cells])),
                     "seconds_per_seed": float(np.mean([c["seconds"] for c in cells]))}
        if judged:
            v = search.verdict(compose_search=per_cond["search_eta1"]["compose_n"], compose_control=per_cond["control_eta1"]["compose_n"],
                               sharper_seeds=paired["sharper_seeds"], n_seeds=n)
        else:
            v = search.control_verdict(presence_search=per_cond["search_eta1"]["presence_n"], presence_control=per_cond["control_eta1"]["presence_n"])
        summary["pairs"][pair] = {"n_seeds": n, "conditions": per_cond, "paired_search_vs_control": paired, "secondary": secondary, "verdict": v,
                                  "sheet": str(sheet_png), "strip": str(strip_png), "figures": {k: str(p) for k, p in figs.items()}}
        (run_dir / f"summary_{pair}.json").write_text(json.dumps(summary["pairs"][pair], indent=1))
        counter += 1
        wb.log_image(f"sheets/{pair}", sheet_png, step=counter, caption=v)
        wb.log_image(f"strips/{pair}", strip_png, step=counter, caption="Mono | plain PoE | corrected + noise search")
        for k, p in figs.items():
            wb.log_image(f"figures/{pair}/{k}", p, step=counter)
        payload = {f"summary/{pair}/{cond}/{k}": val for cond in CONDITIONS for k, val in per_cond[cond].items() if not isinstance(val, dict)}
        payload.update({f"summary/{pair}/paired/{k}": val for k, val in paired.items()})
        wb.log(payload, step=counter)
        if wb.run is not None:
            try:
                import wandb
                cols = ["pair", "seed", "step", "pivot_reward", "chosen_reward", "chosen_kind", "chosen_count", "chosen_imagereward"]
                tab = wandb.Table(columns=cols)
                for c in cells:
                    for s in sorted(c["search"]["chosen_reward_by_step"], key=int):
                        tab.add_data(pair, c["seed"], int(s), c["search"]["pivot_reward_by_step"][s], c["search"]["chosen_reward_by_step"][s],
                                     c["search"]["chosen_kind_by_step"][s], c["search"]["chosen_count_by_step"].get(s),
                                     c["search"]["chosen_fidelity_by_step"].get(s))
                wb.run.log({f"search_steps/{pair}": tab}, step=counter)
            except Exception as exc:  # noqa: BLE001
                log.warning("wandb table failed: %s", exc)
        log.info("[%s] verdict: %s | per condition compose %s | paired %s", pair, v,
                 {c: per_cond[c]["compose_n"] for c in CONDITIONS}, paired)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    verdict = {"judged_pair": JUDGED_PAIR, "constants": constants,
               "verdict": summary["pairs"].get(JUDGED_PAIR, {}).get("verdict"),
               "control_pair": summary["pairs"].get(CONTROL_PAIR, {}).get("verdict"), "adapter": adapter_info}
    (run_dir / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    art_dir = run_dir / "_artifact"; art_dir.mkdir(exist_ok=True)
    for f in list(run_dir.glob("*.png")) + list(run_dir.glob("*.json")):
        (art_dir / f.name).write_bytes(f.read_bytes())
    wb.log_artifact_dir(art_dir, name=f"noise-trajectory-search-{run_id}", kind="results", aliases=["latest"])
    wb.finish()
    log.info("verdict: %s", verdict["verdict"])
    log.info("[done] %s", run_dir)


if __name__ == "__main__":
    main()
