# Scope call: where the eight unfiled dirs go

Date: 2026-08-04 · Decision owner: human (recorded by the reconciliation session)
Follows: `inventory/sweeps/2026-08-04-resweep-detection.md` (detection, read-only)

This answers task 1 of the parked list: do the diagnostics / mechanism-study dirs
belong in the rung1-4 scheme, or somewhere else? Tasks 2-6 were gated on it.

## The call

**Rung1-4 is for the ladder experiments only. Diagnostics and study output get
their own top-level buckets.**

The rung scheme encodes one progression: overfit a pair, survive noise, pool a
group, scale to all groups. A directory earns a rung slot when it is a training
run at one of those stages. Filing a residual-tensor dump or a failed-architecture
sweep under `rung1-overfit/` would say something false about what it is, and it
would make the rung tree stop meaning anything.

So three buckets instead of one:

| Bucket | What goes in it | Why |
|---|---|---|
| `artifacts/rung{1,2,3,4}-*` | ladder training runs | unchanged, as today |
| `artifacts/diagnostics/` | measurement and study output that no rung produced | `residual_diagnostics`, `conditioning_window{,_lora}`, `group_a_failure` |
| `artifacts/scopes/<scope>/` | output owned by a named plan scope | `animals_compose_transfer`, `compose_scorer`, `poe` |
| `artifacts/_shared/presentation/` | cross-experiment figures | `presentation` reads from several rungs, belongs to none |

`artifacts/scopes/` is the one new idea. Three of the eight dirs are outputs of
plan scopes that already exist in `plans/`, so the scope name is the honest
owner. It also means the next scope that lands output has an obvious destination
and does not force another scope call.

## Per-directory destinations

All eight are repo-side, so every move stays within one filesystem and the
existing `sym()` compat-symlink policy in `04_apply_layout.sh` applies unchanged.

| top-dir | disk | destination | disposition |
|---|---|---|---|
| `group_a_failure` | 7G | `artifacts/diagnostics/group_a_failure/` | **keep (reference)** |
| `residual_diagnostics` | 855M | `artifacts/diagnostics/residual_diagnostics/` | **keep** |
| `conditioning_window` | 697M | `artifacts/diagnostics/conditioning_window/` | **keep** |
| `conditioning_window_lora` | 5G | `artifacts/diagnostics/conditioning_window_lora/` | **keep** |
| `animals_compose_transfer` | 6G | `artifacts/scopes/does-the-fix-reach-unseen-pairs/` | **keep** |
| `compose_scorer` | 15M | `artifacts/scopes/compose-scorer/` | **keep** |
| `poe` | 184M | `artifacts/scopes/poe-baselines/` | **keep** |
| `presentation` | 6M | `artifacts/_shared/presentation/` | **keep** |

`group_a_failure` is kept as a reference negative, not as a live result: it is the
recorded failure of the direct-eps / latent-CNN / latent-UNet architectures, and
the paper's negative claims lean on it. Keeping 7G for that is a judgment call
worth revisiting once those claims are written and cited; flagged, not actioned.

## Load test (this sweep)

Read-only, `torch.load(weights_only=True)` on CPU.

| Artifact | Load | Structure | Decision |
|---|---|---|---|
| `animals_compose_transfer/pooled_lora/phase1_r8_100k/checkpoints/lora_step_100000.pt` | PASS | `lora_state` 420 keys, rank-8 shapes `(8,640)/(8,1280)/(8,2048)/(640,8)`; `step=100000`, `epoch=2000`; optimizer + scaler state present | **keep** |
| same dir, `lora_step_005000.pt` and `lora_step_055000.pt` | PASS | identical envelope | **keep** |
| `group_a_failure/.../direct_eps_overfit_catdog/best.pt` | PASS | `student`, `step`, `val_metrics`, `guidance_scale`, `target_kind` | **keep (reference)** |
| `group_a_failure/latent_unet/a_cat__x__a_dog/seed_42/...` | PASS | `delta_hat`, `eps_poe`, `x_t`, `tweedie_x0`, `timestep` | **keep (reference)** |
| `residual_diagnostics/delta_structure/.../teacher_residual_*.pt` | PASS | 13 keys incl. `x_t`, `timestep`, `seq_a`, `pool_a` | **keep** |
| `residual_diagnostics/delta_structure_unguided/tensors.pt` | PASS | `delta`, `eps_poe`, `eps_mono`, `timesteps`, `seeds` | **keep** |

Sampling: first and last `.pt` per directory group, plus three points across the
headline run (20 checkpoints on disk). 1031 files in `group_a_failure` and 402 in
`residual_diagnostics` were not each opened.

### One thing worth knowing about the headline checkpoint

`lora_step_100000.pt` does **not** match the flat 420-key layout recorded in
`inventory/03`. The 420 LoRA tensors sit under a `lora_state` sub-dict, alongside
optimizer and scaler state. Nothing is wrong with the file, but a loader written
against the older flat contract will fail on it. Anything reading this checkpoint
needs `sd["lora_state"]`, not `sd`.

`conditioning_window` and `conditioning_window_lora` hold no `.pt` files at all
(figures and sample grids only), so there is nothing to load-test there.

## Note for the does-the-correction-cause-composition scope

`residual_diagnostics/delta_structure_unguided/tensors.pt` holds `delta`,
`eps_poe`, and `eps_mono` keyed by timestep and seed. That is the same quantity
the does-the-correction-cause-composition scope calls r_t. Filing it under `artifacts/diagnostics/`
puts it on a stable path before that scope starts writing its own caches.
