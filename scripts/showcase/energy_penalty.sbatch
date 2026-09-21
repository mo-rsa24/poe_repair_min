#!/bin/bash
#SBATCH --job-name=energy_penalty
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83,mscluster85
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=16:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/showcase/logs/energy_penalty_%x_%j.log
# Plan 19 of scope 01 (01-showcase-the-trained-lora/19-charge-the-adapter-for-its-energy).
#
# One arm of the energy-penalty experiment, resumed from the shipped rank-32 checkpoint at step
# 30,050 and trained for 10,000 more steps with every flag of the run that made phase1_r32_100k
# (the argv W&B recorded for run 6xc2l8ix), plus:
#   --energy-penalty $BETA        the one axis under test (0 is the control arm)
#   --gradient-checkpointing      the same numbers with less memory; rank 32 went out of memory on
#                                 a 24 GB bigbatch card without it (Slurm job 50339, 2026-09-06)
#   --kill-halve-after-steps 1e9  a resume never latches the kill criterion anyway; explicit
#   --resume-from <30050>         the resume, so the optimizer moments continue too
# Then, on the same card: the 8-seed held-out grid of the arm's 40,050 checkpoint at lambda 1.0,
# its per-step frames at lambda 1.0, and the on-policy energy read.
#
#   STAGE=smoke   sbatch --time=00:40:00 scripts/showcase/energy_penalty.sbatch        # one-epoch dry run, W&B off
#   BETA=0.05     sbatch scripts/showcase/energy_penalty.sbatch                          # one arm
#   STAGE=probe BETA=0.05 sbatch scripts/showcase/energy_penalty.sbatch                 # rerun the post-training stages
set -uo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
SHOWCASE=/datasets/mmolefe/poe_repair_min/outputs/showcase
SCOPE="$REPO/artifacts/results/does-the-fix-reach-unseen-pairs"
RESUME="$SHOWCASE/phase1_r32_100k/checkpoints/lora_step_030050.pt"
export POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE="${STAGE:-arm}"
BETA="${BETA:-0.05}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-800}"          # 30,050 is epoch 600; 800 epochs x 50 steps = 40,050
RUN_ID="${RUN_ID:-phase1_r32_energy${BETA}_from30050_40k}"
RUN_DIR="$SHOWCASE/$RUN_ID"
PROBE_DIR="$SHOWCASE/figure_r32_energy${BETA}_040050"
FRAMES_DIR="$SHOWCASE/where_each_condition_lands/frames_energy${BETA}"
ONPOLICY_DIR="$SHOWCASE/energy_penalty/on_policy/energy${BETA}_040050"
mkdir -p "$SHOWCASE/logs" "$SHOWCASE/energy_penalty"

cd "$REPO" || exit 2

# --- Guards (environment/hpc/execution-protocol.md) ---------------------------------------
USE_PCT=$(df --output=pcent "$SHOWCASE" | tail -1 | tr -dc '0-9')
if [ "$USE_PCT" -ge 90 ]; then echo "ABORT: /datasets at ${USE_PCT}% (>=90% guard)" >&2; exit 3; fi
[ -x "$PY" ] || { echo "ABORT: co3 python not found at $PY" >&2; exit 2; }
[ -f "$RESUME" ] || { echo "ABORT: resume checkpoint missing: $RESUME" >&2; exit 2; }
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || { echo "ABORT: no GPU visible" >&2; exit 4; }
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i 0)
if [[ "$UTIL" == *"N/A"* || "$UTIL" == *"reset"* ]]; then echo "ABORT: device 0 faulted (util '$UTIL', poe-launch-002)" >&2; exit 6; fi
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 7)" || { echo "ABORT: torch.cuda.is_available() is False (poe-launch-002)" >&2; exit 7; }
echo "[$(date -u '+%F %T')] node=$(hostname -s) job=${SLURM_JOB_ID:-none} commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) disk_use=${USE_PCT}% stage=$STAGE beta=$BETA run_id=$RUN_ID"
$PY -c "import torch; print('[gpu]', torch.cuda.get_device_name(0))"

train_args=(
    --pair-pool      "$SCOPE/pair_pool.yaml"
    --pair-prompts   "$SCOPE/pair_prompts.yaml"
    --seed-pool-path "$SCOPE/seed_pool.yaml"
    --cache-root     "$POE_REPAIR_TRAINING_CACHE"
    --lora-rank 32 --lora-alpha 32 --lr 1e-4
    --resume-from "$RESUME"
    --energy-penalty "$BETA" --gradient-checkpointing
    --kill-halve-after-steps 1000000000
    --total-epochs "$TOTAL_EPOCHS" --epoch-size 50
    --ckpt-every-epochs 100 --log-every-epochs 20
    --sample-every-epochs 200 --sample-cells-per-train-pair 2 --sample-cells-per-heldout-pair 2
    --sample-train-pairs a_wolf__x__a_husky --sample-heldout-pairs a_cat__x__a_dog
    --sample-num-inference-steps 50 --sample-thumb 256
    --wandb-project poe-repair-animals-compose
    --output-root "$SHOWCASE"
)

if [ "$STAGE" = "smoke" ]; then
  SMOKE_ID="smoke_energy${BETA}_$(date -u +%H%M%S)"
  echo "[$(date -u '+%F %T')] smoke: one epoch (50 steps) from the 30,050 checkpoint, W&B off, run_id=$SMOKE_ID"
  $PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled "${train_args[@]}" \
      --dry-run --wandb-mode disabled --run-id "$SMOKE_ID" --sample-every-epochs 0
  rc=$?
  echo "[$(date -u '+%F %T')] smoke exit=$rc; peak memory and the loss_fit / loss_energy lines are in this log"
  $PY - "$SHOWCASE/$SMOKE_ID/history.json" "$SHOWCASE/energy_penalty/smoke.json" "$BETA" "$rc" <<'PYEOF'
import json, sys
h = json.load(open(sys.argv[1]))
rows = [r for r in h if "train/loss_fit" in r]
out = {"beta": float(sys.argv[3]), "train_exit": int(sys.argv[4]), "steps": len(rows)}
if rows:
    fit = sum(r["train/loss_fit"] for r in rows) / len(rows)
    en = sum(r["train/loss_energy"] for r in rows) / len(rows)
    beta = rows[-1].get("train/energy_penalty_beta", 0.0)
    out.update(mean_loss_fit=fit, mean_loss_energy=en, price_to_fit_ratio=beta * en / max(fit, 1e-12))
    print(f"[smoke] {len(rows)} steps: mean loss_fit {fit:.5f}, mean loss_energy {en:.5f}, "
          f"beta*energy / fit = {out['price_to_fit_ratio']:.3f}")
vram = [r["train/peak_vram_gb"] for r in h if "train/peak_vram_gb" in r]
out["peak_vram_gb"] = max(vram) if vram else None
print(f"[smoke] peak VRAM GB: {out['peak_vram_gb']}")
json.dump(out, open(sys.argv[2], "w"), indent=1)
PYEOF
  [ -n "${SMOKE_ID:-}" ] && rm -rf "$SHOWCASE/$SMOKE_ID"
  exit $rc
fi

if [ "$STAGE" = "arm" ]; then
  if [ -e "$RUN_DIR" ]; then echo "ABORT: $RUN_DIR already exists; set RUN_ID or move it aside" >&2; exit 1; fi
  # The plan's calibration gate (What has to pass): the smoke's price-to-fit ratio at beta 0.05 must
  # sit in [0.2, 5]; otherwise no arm trains until beta is rescaled and written into the plan.
  if [ "${SKIP_SMOKE_GATE:-0}" != "1" ]; then
    $PY - "$SHOWCASE/energy_penalty/smoke.json" <<'PYEOF' || exit 8
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except FileNotFoundError:
    print("ABORT: no smoke.json; run STAGE=smoke first or set SKIP_SMOKE_GATE=1", file=sys.stderr); sys.exit(1)
r = s.get("price_to_fit_ratio")
if s.get("train_exit", 1) != 0 or r is None:
    print(f"ABORT: the smoke did not finish cleanly: {s}", file=sys.stderr); sys.exit(1)
if not (0.2 <= r <= 5.0):
    print(f"ABORT: price-to-fit ratio {r:.3f} at beta {s['beta']} is outside [0.2, 5]; rescale beta per the plan", file=sys.stderr); sys.exit(1)
print(f"[gate] smoke ratio {r:.3f} at beta {s['beta']}, peak VRAM {s.get('peak_vram_gb')} GB: arms may train")
PYEOF
  fi
  echo "[$(date -u '+%F %T')] training arm beta=$BETA -> $RUN_DIR"
  $PY -m poe_repair.experiments.cross_pair_lora_pooling.train_pooled "${train_args[@]}" \
      --wandb-mode online --run-id "$RUN_ID"
  rc=$?
  echo "[$(date -u '+%F %T')] training exit=$rc"
  [ $rc -eq 0 ] || exit $rc
fi

# --- Post-training stages (STAGE=arm continues here; STAGE=probe starts here) ----------------
CKPT="$RUN_DIR/checkpoints/lora_step_040050.pt"
[ -f "$CKPT" ] || { echo "ABORT: expected checkpoint missing: $CKPT" >&2; ls "$RUN_DIR/checkpoints" >&2; exit 5; }

echo "[$(date -u '+%F %T')] probe: 8-seed held-out grid of $CKPT at lambda 0 and 1.0 -> $PROBE_DIR"
$PY scripts/showcase/lambda_boundary_probe.py --rank 32 --checkpoint "$CKPT" --out-root "$PROBE_DIR" \
    --windows full --lambdas 1.0 || echo "WARN: probe exit=$?"

echo "[$(date -u '+%F %T')] frames: per-step running estimates at lambda 1.0 -> $FRAMES_DIR"
$PY - "$CKPT" "$FRAMES_DIR" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/showcase")
import where_each_condition_lands_trajectories as W
W.CHECKPOINT = Path(sys.argv[1])
raise SystemExit(W.main(["--out-root", sys.argv[2], "--conditions", "lora_1.0"]))
EOF
echo "[$(date -u '+%F %T')] frames exit=$?"

echo "[$(date -u '+%F %T')] on-policy energy read of $CKPT -> $ONPOLICY_DIR"
$PY scripts/showcase/on_policy_energy.py --checkpoint "$CKPT" --rank 32 --out "$ONPOLICY_DIR" \
    --tag "energy penalty beta $BETA, step 40,050" || echo "WARN: on-policy exit=$?"

echo "[$(date -u '+%F %T')] DONE arm beta=$BETA"
