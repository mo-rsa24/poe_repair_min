# Group A — external score-space correctors: design, experiment, code plan

## What this plan is

Stage 3 of [poe-correction-mvp.md](poe-correction-mvp.md) groups remedies by *where* the correction lives. Group A puts the correction outside the UNet — a small standalone network reads `(z_t, t, e_J)` and outputs an additive ε-correction `r̂_t`. SDXL is frozen end-to-end.

This document covers Group A's three techniques:

- **A1. Latent-CNN corrector** — a flat CNN with FiLM on `t` and `e_J`.
- **A2. Latent-UNet corrector** — a small UNet with skip connections, t-conditioned.
- **A3. Frozen-feature MLP head** — a dense head on top of frozen SDXL mid-block features.

The plan mirrors the existing M5 (cross-attention LoRA) experiment in artifact layout, probe cadence, and CLI ergonomics. The user already knows how M5 saves probes at epoch intervals across a λ-grid; Group A reproduces that machinery for a different parameterisation.

**Scope.** `a_cat__x__a_dog`, seed 42. Mono never invoked at inference. The qualitative success criterion is the *progressive morph* defined in [poe-correction-mvp.md](poe-correction-mvp.md) — at λ=1 the corrected rollout walks the chimera into two distinct animals over the trajectory.

**Hardware.** `mscluster106` and `mscluster107`, each with two visible CUDA devices (49 GB VRAM each). Runs go directly in the terminal with logs to stdout; no slurm submission, no batch system. Each technique fits on one device; three techniques can run in parallel across the two nodes' four GPUs.

---

## 1. Design

### 1.1 What every Group A corrector does, end to end

At every DDIM denoising step `t`:

1. PoE is computed exactly as today via frozen SDXL: `ε̃_PoE = poe_eps(ε̃_a, ε̃_b, ε_∅)` from a 3-branch (A, B, ∅) forward.
2. The corrector runs once: `r̂_t = f_θ(z_t, t, e_J)` (or, for A3, on frozen UNet features instead of `z_t`).
3. The corrected prediction is `ε_final = ε̃_PoE + λ · r̂_t` where `λ ∈ [0, 1]` is the probe sweep knob (training uses `λ ≡ 1`; the probe scans the grid).
4. DDIM consumes `ε_final` and produces `z_{t-1}`.

The corrector never sees the joint-prompt ε directly at inference. It must predict the residual from `(z_t, t, e_J)` alone.

### 1.2 Common interface

Every corrector exposes the same callable shape:

```python
class Corrector(nn.Module):
    def forward(
        self,
        z_t: torch.Tensor,      # (1, 4, H, W)
        t: torch.Tensor,        # (1,) long
        e_J_seq: torch.Tensor,  # (1, 77, 2048)
        e_J_pool: torch.Tensor, # (1, 1280)
    ) -> torch.Tensor:          # (1, 4, H, W) — r̂_t
```

Same input and output for A1/A2/A3. Differences live only inside the module body. The training loop, the probe, and the sampler do not branch on the corrector type.

### 1.3 A1 — Latent-CNN corrector

**What it is.** A 4–6 block convolutional network on the raw latent with FiLM conditioning at every block.

**Architecture.**

```
z_t (1,4,128,128) ──► Conv 4→C  ──► [FiLM-Block]×N ──► Conv C→4 ──► r̂_t (1,4,128,128)

FiLM-Block:
    h ← GroupNorm(h)
    γ, β ← FiLM_MLP(time_embed ⊕ pool_proj_e_J)
    h ← γ · h + β
    h ← SiLU(Conv 3×3 C→C)
    h ← Residual add
```

- `C` = 128 channels; `N` = 5 blocks. ~1.2 M params.
- `time_embed`: sinusoidal embedding of `t` (length 256) → MLP → 256-dim conditioning vector.
- `pool_proj_e_J`: linear projection of `e_J_pool` (1280 → 256).
- FiLM MLP per block: `Linear(512 → 2C)` split into `(γ, β)`.

**Why this shape.** Cheapest plausible learner that respects spatial structure at the latent's native resolution. Single-resolution, no skip connections. Matches what was previously called "M2b."

**File.** `poe_repair/students/latent_cnn.py`.

### 1.4 A2 — Latent-UNet corrector

**What it is.** A 2-scale encoder-decoder with skip connections. Same input/output as A1.

**Architecture.**

```
              z_t (1,4,128,128)
                    │
        Conv 4→C ─► Enc1 (C, 128²)  ────────────────┐
                    │                                │  skip
                    ▼                                │
        Down 2×    Enc2 (2C, 64²) ────┐              │
                    │                  │  skip        │
                    ▼                  │              │
        Down 2×    Bottleneck (4C, 32²)             │
                    │                                │
                    ▼                                │
        Up 2×      Dec2 (2C, 64²) ◄────┘             │
                    │                                │
                    ▼                                │
        Up 2×      Dec1 (C, 128²) ◄──────────────────┘
                    │
                    ▼
        Conv C→4 ─► r̂_t (1,4,128,128)

Every Enc/Dec block: GroupNorm → SiLU → Conv → t/e_J FiLM → Residual.
```

- `C` = 96 channels; ~6 M params at this width.
- Conditioning entry: same FiLM mechanism as A1, applied at each block at each scale.
- `e_J_seq` is mean-pooled to a single vector; the sequence dimension is not exposed at this stage (no cross-attention in A2).

**Why this shape.** Multi-scale residual structure (coarse "split the animals" plus finer texture corrections) gets a structured place to live.

**File.** `poe_repair/students/latent_unet.py`.

### 1.5 A3 — Frozen-feature MLP head

**What it is.** SDXL's UNet is run frozen on `(z_t, t, e_J)`. A forward hook captures the mid-block output. The MLP head reads those features and produces the correction.

**Architecture.**

```
(z_t, t, e_J) ──► SDXL UNet (frozen) ──[hook]──► h_mid  (1, 1280, 32, 32)
                                                  │
                                Conv 1×1: 1280 → 256 ──► (1, 256, 32, 32)
                                                  │
                                  Spatial upsample 32 → 128 (nearest)
                                                  │
                                  3×3 Conv stack with FiLM(t, e_J_pool)
                                                  │
                                  Conv 256 → 4 ──► r̂_t (1, 4, 128, 128)
```

- Hook target: `unet.mid_block` (one of the two mid-block transformer attentions; pick the output of `mid_block` itself for the 1280-channel, 32×32 feature map).
- Head: ~3 M params. The SDXL forward is what dominates per-step cost.
- A 256-dim sinusoidal-t bias is added at the input of the head for explicit t conditioning, even though SDXL's internal forward already consumed t.

**Why this shape.** Reuses SDXL's pretrained representation of `(z_t, t, e_J)` without modifying SDXL. If the residual is a function of what SDXL already knows about the joint conditioning, this has the right inductive bias.

**File.** `poe_repair/students/frozen_feature_mlp.py`.

**Implementation note.** A3 needs to call SDXL with `e_J_seq` once per step on top of the existing 3-branch (A, B, ∅) forward. Wallclock per step is roughly +33% versus A1/A2. We can amortise this by piggy-backing on the joint forward if a future variant ever needs it — for the MVP we run a separate joint forward and pay the cost.

---

## 2. Sampler abstraction

The existing samplers (`run_vanilla_poe`, `run_lora_residual_inject`, …) each bake in a fixed correction mechanism. Group A needs one new sampler that takes the corrector as a callback.

### 2.1 New sampler: `run_external_corrector_inject`

**File.** `poe_repair/methods/_sampling.py` — append a new function (do not refactor existing ones).

**Signature.**

```python
@torch.no_grad()
def run_external_corrector_inject(
    *,
    init_latents: torch.Tensor,
    models: dict,
    scheduler,
    seq_a, pool_a,
    seq_b, pool_b,
    seq_j, pool_j,            # consumed by the corrector only
    seq_e, pool_e,
    guidance_scale: float,
    num_inference_steps: int,
    height: int, width: int,
    euler_init_noise_sigma: float,
    device, dtype,
    lambda_value: float,
    corrector: nn.Module,     # the Group A model in eval mode
    record_delta_at_steps: list[int] | None = None,
    correction_max_rel_norm: float | None = None,
) -> SamplerOutputs:
```

**Per-step body.**

1. 3-branch SDXL forward on `(A, B, ∅)` → ε_a_raw, ε_b_raw, ε_uncond. Compose `ε̃_PoE` exactly as `run_vanilla_poe` does.
2. Call the corrector: `r̂_t = corrector(z_t, t, seq_j, pool_j)`. Cast to fp32 for the addition.
3. Optional norm cap: `r̂_t ← _maybe_cap_correction(r̂_t, ε̃_PoE, correction_max_rel_norm)`.
4. `ε_final = ε̃_PoE + λ · r̂_t`. At `λ = 0` we return vanilla PoE byte-identical (canary).
5. Standard DDIM update from `(x0, eps_final)`.
6. If `step_index ∈ record_delta_at_steps`: cache `{r̂_t, ε̃_PoE, z_t, tweedie_x0, timestep}` for the where-applied overlay.

**Extras returned.**
- `lambda_per_step`, `r_hat_norm_per_step`, `eps_poe_norm_per_step`, `cap_scale_per_step`, `where_applied_cache`.

**Why one sampler for all three.** A1/A2/A3 differ only inside `corrector.forward(...)`. The sampler loop is identical, so the abstraction lives at the callback boundary.

### 2.2 Replay sampler (a sanity baseline)

The Stage 3 prerequisite per [poe-correction-mvp.md](poe-correction-mvp.md): inject *cached* `r_t` at every step with no learner. Already covered by the existing `run_teacher_residual` sampler (used by the Stage 1 λ-walk grids). No new code needed — reuse it as the replay baseline.

---

## 3. Training data

### 3.1 Cache reuse

The exact cache M5 trains on:

```
outputs/training_cache/heldout/a_cat__x__a_dog/seed_42/
    residuals/step_NNN.pt   # each contains x_t, eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond, timestep, step_index
    meta.json
    embeddings.pt
```

Loaded via `poe_repair.experiments.thread_c_structure.loader.CellPath` and `load_step_raw`. The cache holds 50 steps of (z_t, raw-eps) per cell.

### 3.2 Target

```python
r_t = guidance_scale * (eps_j_raw - eps_a_raw - eps_b_raw + eps_uncond)
```

— the guided residual `r_t = ε̃_J − ε̃_PoE`, identical to the target M5 uses. Compute once at load time via the existing `delta_t_from_raw` helper.

### 3.3 Conditioning

`e_J_seq` and `e_J_pool` for "a cat and a dog" are encoded once at run start via `encode_prompt_sdxl` (same path M5 already uses) and reused across all training steps.

### 3.4 σ-windowed loss (default schedule)

Per [poe-correction-mvp.md](poe-correction-mvp.md), the default training schedule weights the per-step MSE by `‖r_t‖`. Implementation:

```python
weights = torch.tensor([entry.delta_t.norm().item() for entry in dataset])
weights = weights / weights.sum()
# Sample step_index ~ Categorical(weights) instead of Uniform.
```

A CLI flag `--t-sampler {uniform,sigma_weighted,commit_window}` lets us switch back to uniform for ablation, or to a hard mask `1[t ∈ commit_window]`.

---

## 4. Probe — what success looks like, captured every K epochs

Mirrors M5's probe exactly, with two changes: (a) it calls `run_external_corrector_inject` instead of `run_lora_residual_inject`; (b) `where_applied` overlay payload uses `r̂_t` from the corrector instead of LoRA's `Δ̂`.

### 4.1 Probe inputs (pinned across the entire run)

- `init_latents` recovered from `cell.step_files()[0]['x_t']` × `euler_init_noise_sigma` (the existing M5 trick — the λ=0 column then is byte-identical across probes within a run, and identical to the cache's frozen PoE).
- `seq_{a,b,j,e}`, `pool_{a,b,j,e}` — encoded once.
- DDIM scheduler with 50 timesteps.

Only the corrector weights change across probes. Only λ changes within a probe.

### 4.2 Probe loop

For each `λ ∈ cfg.probe.lambda_grid` (default `(0.0, 0.25, 0.5, 0.75, 1.0)`):

1. Call `run_external_corrector_inject(..., lambda_value=λ, corrector=net, record_delta_at_steps=cfg.probe.where_applied_steps)`.
2. Write `decoded.png` via the existing `write_decoded_image`.
3. Dump the where-applied payload to `delta_overlays/step_NN.pt`.
4. Optionally score with GroundingDINO + VQA (skipped via `--skip-scoring` for the MVP — the headline is qualitative).
5. Write `metrics.json` per λ; aggregate to `summary.json` per epoch.

### 4.3 Qualitative artifact set per epoch

The headline output the user reads at each epoch:

- `thumbnails_epoch_NNNN.png` — one strip of 5 decoded images, one per λ. This is the *morph indicator*: at trained epochs, this strip should walk chimera → split → co-occurrence as λ increases.
- `cumulative_grid.png` — rows are epochs (every probed checkpoint), columns are λ. Lets the user see *both* axes (training progress and λ sweep) in one place.
- `where_applied_epoch_NNNN.png` — for the top λ, overlay `‖r̂_t‖` on the decoded image at the 3 chosen step indices (default `(7, 15, 22)`). Diagnoses where in space the correction is being applied.

### 4.4 Quantitative artifact (supporting only)

- `curve_r_hat_norm_vs_epoch.png` — `‖r̂_t‖` summed across steps at λ=1, plotted over epochs. Replaces M5's VQA curve since we are not gating on quantitative scores.
- If `--skip-scoring=false` is passed, the same VQA/regime curves M5 produces are written.

---

## 5. Run-directory layout (mirrors M5)

```
outputs/group_a/<technique>/<pair_slug>/seed_<N>/<run_id>/
    config.json                       # serialised RunConfig
    attach.json                       # corrector arch summary (params, channels, scales)
    history.json                      # offline mirror of every logger.log() call
    probes/
        epoch_0000/
            lambda_0.00/
                decoded.png
                metrics.json
                delta_overlays/
                    step_07.pt
                    step_15.pt
                    step_22.pt
            lambda_0.25/...
            lambda_0.50/...
            lambda_0.75/...
            lambda_1.00/...
            summary.json
        epoch_0050/...
        epoch_0100/...
        ...
    figures/
        thumbnails_epoch_0000.png
        thumbnails_epoch_0050.png
        ...
        cumulative_grid.png           # updated every probe
        where_applied_epoch_NNNN.png
        curve_r_hat_norm_vs_epoch.png
    checkpoints/
        student_step_000050.pt
        student_step_000100.pt
        ...
        student_step_latest.pt        # symlink
```

Each `student_step_*.pt` contains:

```python
{
    "model_state": net.state_dict(),
    "optimizer_state": opt.state_dict(),
    "scheduler_state": ...,
    "step": int,
    "epoch": int,
    "config": cfg.to_dict(),
    "probe_summary": [...],   # mirror of the epoch's summary.json
}
```

`<technique>` is one of `latent_cnn`, `latent_unet`, `frozen_feature_mlp`. Run IDs follow the M5 pattern: `<technique>__<pair>__seed<N>__<hparams>__<timestamp>`.

---

## 6. Code structure

Mirror M5 module-for-module. One new experiment package per technique, two shared modules.

### 6.1 New files

```
poe_repair/
    students/
        __init__.py
        common.py                    # FiLM block, sinusoidal-t embedding, shared probe-data structs
        latent_cnn.py                # A1 nn.Module
        latent_unet.py               # A2 nn.Module
        frozen_feature_mlp.py        # A3 nn.Module + SDXL hook helper
    methods/
        _sampling.py                 # APPEND run_external_corrector_inject (no refactor)
    experiments/
        group_a_corrector/
            __init__.py
            __main__.py              # raise SystemExit(main())
            config.py                # RunConfig + technique-specific sub-configs
            main.py                  # CLI parser, orchestration, W&B shim
            trainer.py               # dataset loader, train_epoch, kill criteria
            probe.py                 # run_probe (calls run_external_corrector_inject)
            figures.py               # thumbnail strip, cumulative grid, where-applied overlay
```

`group_a_corrector` is **one** experiment package that handles all three techniques. Selection is by `--technique {latent_cnn,latent_unet,frozen_feature_mlp}`. This keeps the probe, figures, and run-dir layout shared across A1/A2/A3 — only the corrector module imported in `trainer.py` differs.

### 6.2 What we copy from M5

- `WandBLogger` class — copy verbatim (rename project to `poe-repair-group-a`).
- `_do_probe` orchestration — same shape; pass `corrector` instead of LoRA-on-unet flag.
- Checkpoint cadence: every `cfg.probe.every_epochs` epochs, after the probe completes, write a checkpoint and log as W&B artifact.
- Resume path: `--resume-from path/to/student_step_NNNNNN.pt`.
- Dry-run mode: `--dry-run` attaches the corrector, runs one probe at epoch 0, exits.

### 6.3 What changes vs M5

- `attach_lora` → replaced by `build_corrector(technique, cfg)`. Returns the `nn.Module` (already on `device/dtype`).
- `lora_state_dict` → replaced by `net.state_dict()`. No PEFT involvement.
- Trainer's `_train_one_step`: instead of running SDXL twice (LoRA-on + LoRA-off), it runs the corrector once and MSEs against cached `r_t`. The frozen PoE marginals are computed *from cache* (`eps_*_raw` are already on disk), so no SDXL forward is needed during training for A1/A2. For A3, one frozen SDXL forward is needed per training step (the hook captures features).
- `run_lora_residual_inject` call inside `probe.py` → `run_external_corrector_inject` with the loaded corrector.

### 6.4 Training-step pseudocode

```python
def _train_one_step(net, dataset, e_J_seq, e_J_pool, cfg, sampler):
    idx = sampler.sample_index()           # σ-weighted or uniform
    entry = dataset[idx]
    z_t   = entry.x_t.to(device, dtype)
    t     = torch.tensor([entry.timestep], device=device)
    r_t   = entry.delta_t.to(device, torch.float32)

    r_hat = net(z_t, t, e_J_seq, e_J_pool).float()
    loss  = F.mse_loss(r_hat, r_t)
    return loss, {"r_hat_norm": r_hat.norm().item(),
                  "r_t_norm":   r_t.norm().item(),
                  "step_index": entry.step_index}
```

For A3, `net.forward` internally calls frozen SDXL with the hook installed; the same signature holds at the trainer level.

---

## 7. CLI and config

### 7.1 RunConfig (one dataclass tree, shared across techniques)

```python
@dataclass
class TechniqueConfig:
    name: str = "latent_cnn"     # "latent_cnn" | "latent_unet" | "frozen_feature_mlp"
    channels: int = 128          # A1/A2 width; A3 head width
    n_blocks: int = 5            # A1 only
    bottleneck_resolution: int = 32   # A2 only
    hook_module: str = "mid_block"    # A3 only
```

`CellConfig`, `OptimConfig`, `ScheduleConfig`, `ProbeConfig`, `KillConfig`, `SamplerConfig`, `WandBConfig` — copied verbatim from M5's `config.py`.

`RunConfig` field changes vs M5:
- Drop `lora: LoRAConfig`. Add `technique: TechniqueConfig`.
- WandB project default: `"poe-repair-group-a"`.
- Output root: `outputs/group_a/<technique.name>/`.

### 7.2 Default hyperparameters

| Field | Default | Notes |
|---|---|---|
| `optim.lr` | `1e-4` | Same as M5 starting point. |
| `optim.weight_decay` | `0.0` | Same. |
| `optim.grad_clip` | `1.0` | Same. |
| `schedule.total_epochs` | `600` | Match M5's known-working budget. |
| `schedule.epoch_size` | `50` | One pass over the 50 cached steps per epoch (when uniform). |
| `schedule.train_batch_size` | `4` | A1/A2 fit comfortably; A3 may need `2`. |
| `probe.every_epochs` | `50` | Same cadence as M5. |
| `probe.lambda_grid` | `(0.0, 0.25, 0.5, 0.75, 1.0)` | Same. |
| `probe.where_applied_steps` | `(7, 15, 22)` | Same — anchors inside the commit window. |
| `probe.commit_window` | `(5, 25)` | Same. |
| `kill.loss_threshold` | n/a | Drop M5's `commit-bucket loss > 0.1 after 3k steps` kill — for Group A the loss scales are different. Replace with: `kill if r_hat_norm_at_probe collapses to < 5% of r_t_norm for 3 consecutive probes`. |
| `t_sampler` | `sigma_weighted` | σ-window is the default per the rewritten MVP. |

### 7.3 Example invocations

```bash
# A1 — latent CNN, default everything
CUDA_VISIBLE_DEVICES=0 python -m poe_repair.experiments.group_a_corrector \
    --technique latent_cnn \
    --pair a_cat__x__a_dog --seed 42 --split heldout \
    --total-epochs 600 --probe-every-epochs 50 \
    --lr 1e-4

# A2 — small UNet
CUDA_VISIBLE_DEVICES=1 python -m poe_repair.experiments.group_a_corrector \
    --technique latent_unet \
    --pair a_cat__x__a_dog --seed 42 --split heldout \
    --total-epochs 600 --probe-every-epochs 50 \
    --lr 1e-4

# A3 — frozen-feature MLP
CUDA_VISIBLE_DEVICES=0 python -m poe_repair.experiments.group_a_corrector \
    --technique frozen_feature_mlp \
    --pair a_cat__x__a_dog --seed 42 --split heldout \
    --total-epochs 600 --probe-every-epochs 50 \
    --lr 1e-4 --train-batch-size 2

# Dry-run sanity (any technique)
python -m poe_repair.experiments.group_a_corrector --technique latent_cnn --dry-run

# Resume from checkpoint
python -m poe_repair.experiments.group_a_corrector \
    --technique latent_cnn \
    --resume-from outputs/group_a/latent_cnn/.../checkpoints/student_step_001500.pt
```

All three run directly in the terminal. Logs print to stdout via the `logging.basicConfig` already configured in M5's `main.py`. No slurm wrappers.

### 7.4 W&B mode

Default `--wandb-mode online`. Falls back to `offline` if the node has no internet (history.json is always written regardless). Project: `poe-repair-group-a`. Run group: pair slug. Tags: `group_a`, technique name, seed.

---

## 8. Hardware mapping and parallel run plan

Two nodes × two devices × ~49 GB VRAM each = four independent training slots.

| Slot | Node | CUDA device | Technique | Why |
|---|---|---|---|---|
| 1 | mscluster106 | 0 | A1 latent CNN | Cheapest; first to reach a useful probe. |
| 2 | mscluster106 | 1 | B3 cross-attention LoRA (reference rerun) | Re-baselines M5 with the σ-windowed loss for fair comparison. Listed for context — not new code, just `--t-sampler sigma_weighted` on the existing M5 entrypoint. |
| 3 | mscluster107 | 0 | A2 latent UNet | Runs in parallel with A1; comparable wallclock. |
| 4 | mscluster107 | 1 | A3 frozen-feature MLP | Heavier per-step (extra SDXL forward); reserve a full GPU. |

If only Group A is being run (no M5 rebaseline), use slots 1, 3, 4 and leave slot 2 idle.

**Per-technique VRAM budget (estimates).**

- A1: SDXL fp16 (~7 GB) + cache batch (~3 GB) + corrector + optimizer state (<1 GB) ≈ 11 GB. Plenty of headroom.
- A2: same shape + larger corrector ≈ 14 GB.
- A3: SDXL fp16 + frozen forward activations retained for the hook + head ≈ 18 GB.

All three fit on one 49 GB device with batch size 4. Headroom lets us raise `train_batch_size` if wallclock matters.

**Terminal run pattern.**

```bash
# On mscluster106
ssh mscluster106
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
source /home-mscluster/mmolefe/miniforge3/etc/profile.d/conda.sh && conda activate co3

# Slot 1 — foreground in tmux pane 1
CUDA_VISIBLE_DEVICES=0 python -m poe_repair.experiments.group_a_corrector \
    --technique latent_cnn --pair a_cat__x__a_dog --seed 42 \
    --total-epochs 600 --probe-every-epochs 50 \
    2>&1 | tee outputs/group_a/latent_cnn/run_$(date +%Y%m%d-%H%M%S).log

# Slot 2 — pane 2 (or skip)
CUDA_VISIBLE_DEVICES=1 python -m poe_repair.experiments.m5_lora_sdxl \
    --pair a_cat__x__a_dog --seed 42 --t-sampler sigma_weighted \
    --total-epochs 600 --probe-every-epochs 50 \
    2>&1 | tee outputs/m5_lora_sdxl/.../run_$(date +%Y%m%d-%H%M%S).log
```

Same pattern on `mscluster107` for slots 3 and 4. `tee` mirrors stdout to a log file inside the run dir so we can replay output without losing the terminal stream.

---

## 9. Phased implementation order

Each phase produces one concrete deliverable. Don't start the next phase until the previous one's deliverable is on disk.

### Phase 0 — Sampler abstraction (½ day)

- [ ] Append `run_external_corrector_inject` to `poe_repair/methods/_sampling.py`. Body modelled on `run_lora_residual_inject` minus adapter management, plus a single corrector call per step.
- [ ] Unit test: instantiate a stub corrector that returns zeros; verify the sampler at `λ=0` and `λ=1` (with zero corrector) produces a decoded image byte-identical to `run_vanilla_poe` on the same `init_latents`.

**Deliverable.** `run_external_corrector_inject` lands; canary test passes.

### Phase 1 — Common scaffolding (½ day)

- [ ] Create `poe_repair/students/common.py` with `FiLMBlock`, `sinusoidal_time_embedding`, `pool_proj`.
- [ ] Create `poe_repair/experiments/group_a_corrector/` skeleton: `config.py`, `main.py`, `trainer.py`, `probe.py`, `figures.py`. Copy M5's structure file-for-file, swapping the LoRA-specific names.
- [ ] Implement `build_corrector(technique, cfg)` that returns a `nn.Module`. Stub all three to a zero-output module.

**Deliverable.** `python -m poe_repair.experiments.group_a_corrector --technique latent_cnn --dry-run` runs to completion against the stub and writes `probes/epoch_0000/` with five λ thumbnails. λ=0 = vanilla PoE; λ=1 = identical to λ=0 since the stub outputs zero.

### Phase 2 — A1 latent CNN (1 day)

- [ ] Implement `poe_repair/students/latent_cnn.py` per §1.3.
- [ ] Wire `build_corrector("latent_cnn", cfg)` to it.
- [ ] Implement σ-weighted sampler in `trainer.py`.
- [ ] Implement `_do_probe` and figures (`thumbnails_epoch_NNNN.png`, `cumulative_grid.png`).
- [ ] Start the first 600-epoch run on `mscluster106:0`.

**Deliverable.** A run directory with probes every 50 epochs. The cumulative grid shows whether the morph is happening qualitatively.

### Phase 3 — A2 latent UNet (½ day after Phase 2)

- [ ] Implement `poe_repair/students/latent_unet.py`. Reuse Phase 2's training loop unchanged.
- [ ] Start the run on `mscluster107:0`.

**Deliverable.** Comparable run directory to A1.

### Phase 4 — A3 frozen-feature MLP (1 day)

- [ ] Implement `poe_repair/students/frozen_feature_mlp.py` including the forward hook.
- [ ] Verify the hook captures the expected tensor shape (likely `(1, 1280, 32, 32)`; confirm by running a single SDXL forward and printing shapes).
- [ ] Start the run on `mscluster107:1` with `--train-batch-size 2`.

**Deliverable.** Run directory with probes; cumulative grid renders without OOMs.

### Phase 5 — Qualitative comparison (½ day after all three finish or hit 600 epochs)

- [ ] Render a single combined figure: three rows (A1, A2, A3), columns are five λ values, evaluated at the final epoch of each run.
- [ ] Render a per-technique cumulative grid (epochs × λ) for the writeup.
- [ ] Write a short verdict file per technique: pass / fail by the morph criterion, with one sentence of qualitative justification.

**Deliverable.** Three verdict files, one combined figure, ready for the writeup.

---

## 10. Success and failure criteria

### 10.1 Per-technique success

The corrected rollout at the final probe (or any probe after the loss plateau) shows the chimera-to-co-occurrence morph qualitatively across the λ-grid: at λ=0 the image is the PoE chimera; at λ=1 the image is two distinct animals; the intermediate λ values show a gradual split rather than a jump.

### 10.2 Per-technique failure modes (and what each implies)

| Symptom at λ=1 | Implied failure | Next step |
|---|---|---|
| Image identical to PoE | Corrector outputs zero in expectation | Optimisation issue — check gradients, σ-window, sign convention. Not architecture. |
| Image is noise or pure-garbage | Corrector overshoots; correction has wrong scale | Add `--correction-max-rel-norm 1.0` cap; reduce LR. |
| Image is plausibly cat-like or dog-like but not both | Partial morph plateau | Architecture-level — escalate per the MVP's bottom-up ladder. |
| Image flickers across epochs / non-monotone with λ | Rollout drift between cached and inference trajectories | Real but expected — note it; escalate to outcome supervision is out of scope for the MVP. |

### 10.3 Cross-technique comparison

If A1 passes, A2 and A3 are ablation rather than necessity. If A1 fails and the failure looks scale-specific (e.g. fine textures wrong), A2 is justified. If A1 fails and the failure looks input-bottlenecked (texture-blind, missing semantic structure), A3 is justified.

---

## 11. Risks and mitigations

- **Risk: A3 SDXL hook captures the wrong tensor.** Pre-step: write a 10-line script that runs one SDXL forward with the hook installed and prints `feature.shape` and `feature.dtype`. Confirm `(1, 1280, 32, 32)` before integrating into the trainer.
- **Risk: σ-windowed sampler underweights tail timesteps to the point that they regress.** Mitigation: add a "min weight floor" (e.g. each step contributes at least 1% of the average weight). Switchable via `--t-sampler-floor 0.01`.
- **Risk: probe wallclock dominates training wallclock.** With 5 λ × 50 DDIM steps × 1024² decode at probe time, each probe takes ~3 minutes on SDXL. At 50-epoch cadence over 600 epochs that's ~36 minutes total — acceptable.
- **Risk: replaying the cached `r_t` doesn't reproduce mono.** This is the Stage 3 prerequisite (per the MVP). Verify before training: `run_teacher_residual` at λ=1 on this cell should already be on disk under `outputs/veracity/.../teacher_residual_const_lam100/decoded.png`. If it's a clean co-occurrence, the cache is good. If not, fix the cache before starting any Group A training.
- **Risk: history.json grows unbounded.** Match M5's pattern: it's only an offline mirror, not used for queries; rotation isn't needed at this scale (600 epochs × ~30 log payloads per epoch ≈ 18k records, well under any size limit).
- **Risk: simultaneous W&B logging from 3–4 runs hits rate limits.** Use `--wandb-mode offline` on at least one of the parallel runs; sync after completion.

---

## 12. What this plan does NOT include

- Cross-seed or cross-pair runs.
- Outcome-supervised fine-tuning (DRaFT / DDPO).
- Quantitative VQA/GroundingDINO scoring as a gating criterion. The headline is qualitative; quantitative scoring is available via removing `--skip-scoring` but is not what success is measured against.
- Hyperparameter sweeps. One default config per technique on the seed-42 beachhead. If results are mixed, *then* sweep.
- Hypernetwork or multi-cell variants. Group A is single-cell.

---

## 13. Quick links

- Active MVP plan: [poe-correction-mvp.md](poe-correction-mvp.md).
- Full design space (where Group A sits): [poe-correction-design-space.md](poe-correction-design-space.md).
- Reference experiment (M5 LoRA, the layout we mirror): [poe_repair/experiments/m5_lora_sdxl/](poe_repair/experiments/m5_lora_sdxl/).
- Training cache root: [outputs/training_cache/heldout/a_cat__x__a_dog/seed_42/](outputs/training_cache/heldout/a_cat__x__a_dog/seed_42/).
- Stage 1 λ-walk grids on disk: [outputs/veracity/pairs/a_cat__x__a_dog/seed_42/](outputs/veracity/pairs/a_cat__x__a_dog/seed_42/).
