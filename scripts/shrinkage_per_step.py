#!/usr/bin/env python
"""How much of the correction the adapter actually applies, per denoising step.

Task 1.1 of scope 09's plan 06. The correction the adapter is trying to supply is
``r = eps_J - eps_PoE``, which the training cache already holds as its delta_t. What the
adapter supplies is ``r_hat = eps_poe_lora - eps_poe_frozen``, the quantity the trainer
forms at one_pair_one_seed/trainer.py:588. This script reports ``||r_hat|| / ||r||`` per
denoising step, per seed, per checkpoint, and nothing else: no renders, no scoring, no
training, and no writes to either checkpoint tree.

Two checkpoints, because one of them never trained on the late steps. v1_freeze_null_r16_s0_25
trained on cached steps 0 to 25, so a fall-off after step 25 there is what an untrained range
looks like as much as what shrinkage looks like. pool43-all50 trained on all fifty.

One checkpoint per process: attaching a second adapter to the same UNet would stack on the
first. Run --checkpoint once per name, then --merge.

    python scripts/shrinkage_per_step.py --checkpoint v1_freeze_null_r16_s0_25_30k
    python scripts/shrinkage_per_step.py --checkpoint pool43_all50_40k
    python scripts/shrinkage_per_step.py --merge
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "showcase"))

from poe_repair._sdxl.metrics import guided_eps, poe_eps          # noqa: E402
from poe_repair.methods._sampling import add_time_ids             # noqa: E402
from poe_repair.run import make_ctx                               # noqa: E402
from poe_repair.training_cache import CellPath, load_step_raw     # noqa: E402

# (checkpoint path, LoRA rank). The same three the hand-off sweep reads, at
# scripts/clean_tail_k_sweep.py:123.
CHECKPOINTS = {
    "v1_freeze_null_r16_s0_25_30k": (
        "/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants/"
        "v1_freeze_null_r16_s0_25/checkpoints/lora_step_030000.pt", 16),
    "pool43_all50_40k": (
        "/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32/"
        "pool43-all50/checkpoints/lora_step_040000.pt", 32),
}

PAIR = "a_cat__x__a_dog"
SEEDS = (9, 10, 11, 12, 13, 14, 15, 16)
GUIDANCE = 7.5
HEIGHT = WIDTH = 1024

OUT_DIR = REPO / "artifacts/results/designing-the-correction-loss/why-the-renders-are-not-crisp"
MERGED_JSON = OUT_DIR / "shrinkage-per-step.json"
PLOT_PNG = OUT_DIR / "shrinkage-per-step.png"


def _shard(name: str) -> Path:
    return OUT_DIR / f"shrinkage-per-step.{name}.json"


def _host() -> dict:
    return {"node": socket.gethostname(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def _git_stamp() -> dict:
    def sh(*a):
        return subprocess.run(a, cwd=str(REPO), capture_output=True, text=True).stdout.strip()
    return {"commit": sh("git", "rev-parse", "--short", "HEAD"),
            "dirty": bool(sh("git", "status", "--porcelain"))}


def _disk_guard(root: Path) -> None:
    """Guard the filesystem this script actually writes to, not some other mount."""
    p = root
    while not p.exists():
        p = p.parent
    st = os.statvfs(str(p))
    used = 1.0 - st.f_bavail / max(st.f_blocks, 1)
    if used >= 0.90:
        raise SystemExit(f"disk guard: {root} filesystem at {used:.0%}, refusing to write")
    print(f"disk guard: {p} at {used:.0%}", flush=True)


def _device_guard() -> None:
    """Abort rather than share a device someone else is already using."""
    if not torch.cuda.is_available():
        raise SystemExit("device guard: no CUDA device visible")
    free, total = torch.cuda.mem_get_info(0)
    used_gb = (total - free) / 1e9
    if used_gb > 1.0:
        raise SystemExit(f"device guard: {used_gb:.1f} GB already in use, refusing to share")
    print(f"device guard: device clear ({used_gb:.2f} GB used of {total / 1e9:.0f} GB)", flush=True)


def _freeze_null_of(ckpt_path: str) -> bool:
    """Read the null-branch rule off the run's own config, never off the checkpoint name."""
    cfg = Path(ckpt_path).parent.parent / "config.json"
    if not cfg.exists():
        raise SystemExit(f"no config.json beside {ckpt_path}; cannot tell how to compose the null branch")
    return bool(json.loads(cfg.read_text()).get("freeze_null") or False)


@torch.no_grad()
def measure(name: str, *, batch_steps: int, max_steps: int | None, seeds) -> dict:
    ckpt, rank = CHECKPOINTS[name]
    freeze_null = _freeze_null_of(ckpt)

    _disk_guard(OUT_DIR)
    _device_guard()

    import lambda_boundary_probe as lbp
    ctx = make_ctx()
    unet = ctx.models["unet"]
    lbp.LORA_RANK = lbp.LORA_ALPHA = rank
    info = lbp._attach_and_load_lora(unet, Path(ckpt))
    unet.eval()
    device, dtype = ctx.device, ctx.dtype
    print(f"{name}: rank {rank}, n_matched={info['n_matched']}, n_loaded={info['n_loaded']}, "
          f"checkpoint_step={info['checkpoint_step']}, freeze_null={freeze_null}", flush=True)
    print(f"host={_host()} git={_git_stamp()}", flush=True)

    rows = []
    for seed in seeds:
        cell = CellPath.from_root(PAIR, seed, split="heldout")
        emb = torch.load(cell.root / "embeddings.pt", map_location="cpu", weights_only=False)
        seq_3 = torch.cat([emb["seq_a"], emb["seq_b"], emb["seq_uncond"]], dim=0).to(device=device, dtype=dtype)
        pool_3 = torch.cat([emb["pool_a"], emb["pool_b"], emb["pool_uncond"]], dim=0).to(device=device, dtype=dtype)

        files = cell.step_files()
        if max_steps is not None:
            files = files[:max_steps]
        t0 = time.time()
        for start in range(0, len(files), batch_steps):
            chunk = [load_step_raw(f) for f in files[start:start + batch_steps]]
            K = len(chunk)
            # (3K, 4, H, W): each step's x_t repeated for the A / B / null branches, in that order.
            x_t = torch.cat([c["x_t"].to(device=device, dtype=dtype).repeat(3, 1, 1, 1) for c in chunk], dim=0)
            ts = torch.tensor([c["timestep"] for c in chunk], device=device,
                              dtype=torch.long).repeat_interleave(3)
            cond = {"text_embeds": pool_3.repeat(K, 1),
                    "time_ids": add_time_ids(height=HEIGHT, width=WIDTH, batch_size=3 * K,
                                             device=device, dtype=dtype)}
            x_in = ctx.scheduler.scale_model_input(x_t, ts[0])
            out = unet(x_in, ts, encoder_hidden_states=seq_3.repeat(K, 1, 1),
                       added_cond_kwargs=cond, timestep_cond=None).sample.float()
            out = out.view(K, 3, *out.shape[1:])
            ea_l, eb_l, eu_l = out[:, 0], out[:, 1], out[:, 2]

            for i, c in enumerate(chunk):
                ea = c["eps_a_raw"].to(device)
                eb = c["eps_b_raw"].to(device)
                ej = c["eps_j_raw"].to(device)
                eu = c["eps_uncond"].to(device)
                # The frozen composition, and the correction it is missing. r is the cache's own
                # delta_t: w * (eps_j_raw - eps_a_raw - eps_b_raw + eps_uncond).
                eps_poe_frozen = poe_eps(guided_eps(ea, eu, GUIDANCE), guided_eps(eb, eu, GUIDANCE), eu)
                r = guided_eps(ej, eu, GUIDANCE) - eps_poe_frozen
                # trainer.py:543-544: the empty branch is the cached one under --freeze-null and
                # the adapted one otherwise.
                u = eu if freeze_null else eu_l[i:i + 1]
                eps_poe_lora = poe_eps(guided_eps(ea_l[i:i + 1], u, GUIDANCE),
                                       guided_eps(eb_l[i:i + 1], u, GUIDANCE), u)
                r_hat = eps_poe_lora - eps_poe_frozen
                rn = float(r.norm().item())
                rhn = float(r_hat.norm().item())
                cos = float(torch.nn.functional.cosine_similarity(
                    r_hat.reshape(1, -1), r.reshape(1, -1), dim=1).item())
                rows.append({"checkpoint_name": name, "checkpoint": ckpt, "rank": rank,
                             "freeze_null": freeze_null, "pair": PAIR, "seed": seed,
                             "step_index": c["step_index"], "timestep": c["timestep"],
                             "r_norm": rn, "r_hat_norm": rhn,
                             "ratio": rhn / max(rn, 1e-12), "cosine": cos})
        print(f"[{time.strftime('%H:%M:%S')}] {name} seed {seed}: {len(files)} steps "
              f"({time.time() - t0:.0f}s)", flush=True)

    payload = {"checkpoint_name": name, "checkpoint": ckpt, "rank": rank,
               "freeze_null": freeze_null, "pair": PAIR, "seeds": list(seeds),
               "guidance_scale": GUIDANCE,
               "what_r_is": "eps_J - eps_PoE, the cache's delta_t = w*(eps_j_raw - eps_a_raw - eps_b_raw + eps_uncond)",
               "what_r_hat_is": "eps_poe_lora - eps_poe_frozen, as trainer.py:588 forms it",
               "host": _host(), "git": _git_stamp(), "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _shard(name).write_text(json.dumps(payload, indent=2))
    print(f"wrote {_shard(name)}  ({len(rows)} rows)", flush=True)
    return payload


def merge_and_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shards = [json.loads(_shard(n).read_text()) for n in CHECKPOINTS if _shard(n).exists()]
    if not shards:
        raise SystemExit("no shards to merge; run --checkpoint first")
    rows = [r for s in shards for r in s["rows"]]
    MERGED_JSON.write_text(json.dumps(
        {"produced_by": "scripts/shrinkage_per_step.py",
         "task": "plans/09-designing-the-correction-loss/plans/hypothesis/"
                 "06-fitting-the-whole-path-and-handing-back-the-tail.md task 1.1",
         "shards": [{k: s[k] for k in ("checkpoint_name", "checkpoint", "rank", "freeze_null",
                                       "host", "git")} for s in shards],
         "what_r_is": shards[0]["what_r_is"], "what_r_hat_is": shards[0]["what_r_hat_is"],
         "rows": rows}, indent=2))

    fig, axes = plt.subplots(1, len(shards), figsize=(6 * len(shards), 4.2), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], shards):
        for seed in s["seeds"]:
            pts = sorted([r for r in s["rows"] if r["seed"] == seed], key=lambda r: r["step_index"])
            if pts:
                ax.plot([p["step_index"] for p in pts], [p["ratio"] for p in pts],
                        lw=1.2, label=f"seed {seed}")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("denoising step")
        ax.set_title(f"{s['checkpoint_name']}  (rank {s['rank']})", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(r"$\|\hat{r}\| \; / \; \|r\|$")
    axes[0][-1].legend(fontsize=7, ncol=2)
    fig.suptitle("How much of the correction the adapter applies, per denoising step "
                 f"({PAIR.replace('__x__', ' x ')}, held-out seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150)
    print(f"wrote {MERGED_JSON}  ({len(rows)} rows)\nwrote {PLOT_PNG}", flush=True)

    for s in shards:
        early = [r["ratio"] for r in s["rows"] if r["step_index"] <= 9]
        late = [r["ratio"] for r in s["rows"] if r["step_index"] >= 40]
        if early and late:
            print(f"{s['checkpoint_name']}: steps 0-9 mean ratio {sum(early)/len(early):.3f}, "
                  f"steps 40-49 mean ratio {sum(late)/len(late):.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=sorted(CHECKPOINTS))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--batch-steps", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke: first N denoising steps only")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    a = ap.parse_args()
    if a.merge:
        merge_and_plot()
        return
    if not a.checkpoint:
        raise SystemExit("pass --checkpoint <name> or --merge")
    measure(a.checkpoint, batch_steps=a.batch_steps, max_steps=a.max_steps, seeds=a.seeds)


if __name__ == "__main__":
    main()
