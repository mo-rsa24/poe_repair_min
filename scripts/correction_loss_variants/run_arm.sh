#!/usr/bin/env bash
# One arm of the correction-loss variant comparison.
#
#   GPU=<idx> ARM=<plain|anchor|plurality|connective> [SMOKE=1] [EPOCHS=n] [MU=0.1] \
#     bash scripts/correction_loss_variants/run_arm.sh
#
# Four arms differ in one thing: what the three branches are conditioned on, and whether the
# null branch is pinned. Everything else is set here, once, so it cannot drift between arms.
#
#   plain        the current recipe. The control every other arm is read against.
#   anchor       plain plus mu * ||eps_theta(null) - eps_frozen(null)||^2. The fit loss pins
#                only eps_1 + eps_2 - eps_null, while the sampler reads the null branch again
#                at -(w-1) = -6.5, so a perturbation the loss cannot see moves the picture.
#   plurality    "a cat, two animals" / "a dog, two animals" against a null of "two animals",
#                so the composition divides by the plurality-conditioned base.
#   connective   "a cat and" / "a dog and" against a null of "and".
#
# plain, plurality and connective reach the same composed predictions at a fixed state, so a
# difference between them is optimisation and inductive bias and never capacity. Write it up
# that way.
#
# Why orth-weight stays at its default 1.0: any other value activates a projection onto the
# plane spanned by the CACHED eps_a/eps_b, which were built for the plain prompts. Under
# plurality and connective that plane is the wrong one, and at 1.0 the trainer short-circuits
# the projection entirely, so the arms stay comparable.
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
ARM="${ARM:?set ARM}"; GPU="${GPU:?set GPU}"
MU="${MU:-0.1}"

case "$ARM" in
  plain)       ARM_EXTRA="" ;;
  anchor)      ARM_EXTRA="--null-anchor $MU" ;;
  plurality)   ARM_EXTRA="--branch-prompt-style plurality" ;;
  connective)  ARM_EXTRA="--branch-prompt-style connective" ;;
  *) echo "ABORT: unknown ARM $ARM" >&2; exit 2 ;;
esac

# --- pinned across every arm ------------------------------------------------
export RANK=16 ALPHA=16
export STEPRANGE="0 25"          # the first half of the schedule, where the correction lives
export POOLGEN=v54               # cat x dog and elephant x penguin are heldout in this pool
export CELLS="$REPO/artifacts/_shared/cross_pair_pool_configs/cells_v54.json"
export WD=0                      # the reference run that composed used weight decay 0
export BATCH=1
export EPOCHSIZE="${EPOCHSIZE:-50}"
# Rank 16 with checkpointing peaks at 11.2 GB of a 24 GB card (smoke 53310), so there is room to
# turn it off and buy back the third of the speed it costs. Overridable per launch; whatever is
# chosen must be identical across all four arms, because it changes the optimisation timing.
export CKPT="${CKPT-1}"
export MAXUSED="${MAXUSED:-1024}"

if [ "${SMOKE:-0}" = "1" ]; then
  export EPOCHS="${EPOCHS:-2}" EPOCHSIZE=10 SAMPLEEVERY=0
  export RUN="smoke_r16_s0_25_${ARM}"
  export OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants/smoke
  EXTRA="$ARM_EXTRA --wandb-mode disabled"
else
  export EPOCHS="${EPOCHS:-600}" SAMPLEEVERY="${SAMPLEEVERY:-50}"
  export RUN="r16_s0_25_${ARM}"
  export OUTROOT=/datasets/mmolefe/poe_repair_min/outputs/correction_loss_variants
  # W&B groups the four arms so they plot against each other without hand-picking run ids.
  export WANDB_NAME="corrloss_r16_s0_25_${ARM}"
  export WANDB_RUN_GROUP="correction_loss_variants_r16_s0_25"
  EXTRA="$ARM_EXTRA"
fi
export EXTRA

echo "=== arm $ARM  extra '${EXTRA}'  run $RUN"
exec bash "$REPO/scripts/showcase/train_pool_run.sh"
