#!/usr/bin/env bash
# Canonical-layout mover for the surviving poe_repair artifacts.
# DRY-RUN by default: prints every planned action, changes nothing.
# Execute with: APPLY=1 bash inventory/scripts/04_apply_layout.sh
#
# Policy: within-root moves only (no cross-filesystem copies); every move
# leaves a backward-compat symlink at the old path; discards go to
# _quarantine/ (moved, not deleted); only verified-empty 0-byte dirs are
# removed. Refuses to overwrite an existing destination.
set -euo pipefail

REPO="/home-mscluster/mmolefe/Playground/PhD/poe_repair_min"
DAT="/datasets/mmolefe/poe_repair_min"
APPLY="${APPLY:-0}"

say()  { printf '%s\n' "$*"; }
run()  { if [ "$APPLY" = "1" ]; then eval "$@"; else say "  [DRY] $*"; fi; }

# move SRC -> DST, leave a compat symlink at SRC (absolute target)
sym() {
  local src="$1" dst="$2"
  if [ ! -e "$src" ] && [ ! -L "$src" ]; then say "SKIP (missing): $src"; return; fi
  if [ -e "$dst" ]; then say "REFUSE (dest exists): $dst"; return; fi
  say "MOVE+LINK: $src"
  say "        -> $dst"
  run "mkdir -p \"$(dirname "$dst")\""
  run "mv \"$src\" \"$dst\""
  run "ln -s \"$dst\" \"$src\""
}

# move SRC -> DST (quarantine / no symlink)
quar() {
  local src="$1" dst="$2"
  if [ ! -e "$src" ] && [ ! -L "$src" ]; then say "SKIP (missing): $src"; return; fi
  if [ -e "$dst" ]; then say "REFUSE (dest exists): $dst"; return; fi
  say "QUARANTINE: $src -> $dst"
  run "mkdir -p \"$(dirname "$dst")\""
  run "mv \"$src\" \"$dst\""
}

# remove SRC only if it exists and contains zero files
rmempty() {
  local src="$1"
  if [ ! -d "$src" ]; then say "SKIP (missing): $src"; return; fi
  local n; n="$(find "$src" -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$n" != "0" ]; then say "REFUSE (not empty, $n files): $src"; return; fi
  say "RMDIR (empty): $src"
  run "rmdir \"$src\" 2>/dev/null || find \"$src\" -type d -empty -delete"
}

say "=== APPLY=$APPLY  (set APPLY=1 to execute) ==="

say ""; say "--- Rung 1: overfit (lora, single seed) ---"
sym  "$REPO/outputs/lora/cat_dog/seed_42/results" \
     "$REPO/artifacts/rung1-overfit/lora/a_cat__x__a_dog/seed_42/run__local"
sym  "$REPO/outputs/lora/a_typewriter__x__a_cactus/seed_42/lora_sdxl__a_typewriter__x__a_cactus__seed42__r8__lr1e-04__20260520-133632" \
     "$REPO/artifacts/rung1-overfit/lora/a_typewriter__x__a_cactus/seed_42/run__wandb-wag4z592"
sym  "$REPO/outputs/lora/a_camel__x__a_desert_landscape/seed_42/lora_sdxl__a_camel__x__a_desert_landscape__seed42__r8__lr1e-04__20260520-131542" \
     "$REPO/artifacts/rung1-overfit/lora/a_camel__x__a_desert_landscape/seed_42/run__wandb-8p1spi5b"
quar "$REPO/outputs/lora/a_typewriter__x__a_cactus/seed_42/lora_sdxl__a_typewriter__x__a_cactus__seed42__r8__lr1e-04__20260520-131542" \
     "$REPO/artifacts/_quarantine/false-start/lora_typewriter_131542__wandb-lu7g7svh"
# NOTE: outputs/lora/{a_dog__x__oil_painting_style,a_dolphin__x__an_ocean_wave,a_mailbox__x__a_snowfield}
# were mislabelled "0-byte stubs" in the inventory. They are LIVE alias views —
# seed_42/results symlinks into the pair's cross_seed seed bank (resolves post-reorg).
# Kept in place, not deleted. See inventory/03 and 04 corrections.

say ""; say "--- Rung 2: survive-noise (cross_seed) — datasets seed banks ---"
sym  "$DAT/outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep2000_resumed" \
     "$DAT/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl"
for pair in a_dog__x__oil_painting_style a_dolphin__x__an_ocean_wave a_mailbox__x__a_snowfield a_typewriter__x__a_cactus; do
  sym "$DAT/outputs/cross_seed_lora_pooling/$pair/task_b_learning_curve/k04__ep2000" \
      "$DAT/artifacts/rung2-survive-noise/cross_seed/$pair/taskB__k04_ep2000"
done
quar "$DAT/outputs/cross_seed_lora_pooling/task_b_learning_curve/k04__ep200" \
     "$DAT/artifacts/_quarantine/superseded/cross_seed_catdog_k04_ep200"

say ""; say "--- Rung 2: survive-noise (cross_seed) — repo eval/results ---"
# pair root first so cat_dog siblings land inside it
sym  "$REPO/outputs/cross_seed_lora_pooling/a_cat__x__a_dog" \
     "$REPO/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog"
sym  "$REPO/outputs/cross_seed_lora_pooling/task_b_learning_curve/k01_pick1__ep1600" \
     "$REPO/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k01_pick1_ep1600__wandb-hbpotmnk"
sym  "$REPO/outputs/cross_seed_lora_pooling/task_c_per_seed_ceiling/per_seed_s9__ep1600" \
     "$REPO/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskC__s9_ep1600__wandb-d5b2706v"
sym  "$REPO/outputs/cross_seed_lora_pooling/step0_prescreen" \
     "$REPO/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/step0_prescreen/heldout"
sym  "$REPO/outputs/cross_seed_lora_pooling/step0_prescreen_seed42" \
     "$REPO/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/step0_prescreen/seed42"
sym  "$REPO/outputs/cross_seed_lora_pooling/trajectory_diagram/seed_42" \
     "$REPO/artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/trajectory_diagram/seed_42"
for pair in a_dog__x__oil_painting_style a_dolphin__x__an_ocean_wave a_mailbox__x__a_snowfield a_typewriter__x__a_cactus; do
  sym "$REPO/outputs/cross_seed_lora_pooling/$pair" \
      "$REPO/artifacts/rung2-survive-noise/cross_seed/$pair"
done

say ""; say "--- Rung 3: group-wise (cross_pair within_group) ---"
sym  "$REPO/outputs/cross_pair_lora_pooling/within_group/g6/main" \
     "$REPO/artifacts/rung3-group-wise/cross_pair/within_group/g6/main__wandb-ow1jo0xq"
for f in main.log pair_pool.yaml; do
  sym "$REPO/outputs/cross_pair_lora_pooling/within_group/g6/$f" \
      "$REPO/artifacts/rung3-group-wise/cross_pair/within_group/g6/$f"
done
for g in g1 g2 g3 g4; do
  sym "$REPO/outputs/cross_pair_lora_pooling/within_group/$g" \
      "$REPO/artifacts/rung3-group-wise/cross_pair/within_group/$g"
done
sym  "$REPO/outputs/cross_pair_lora_pooling/within_group/seed_pool.yaml" \
     "$REPO/artifacts/rung3-group-wise/cross_pair/within_group/seed_pool.yaml"

say ""; say "--- Rung 4: scale (cross_pair all_groups) ---"
sym  "$REPO/outputs/cross_pair_lora_pooling/all_groups/main" \
     "$REPO/artifacts/rung4-scale/cross_pair/all_groups/main__wandb-2em6frqv"
quar "$REPO/outputs/cross_pair_lora_pooling/all_groups/dryrun" \
     "$REPO/artifacts/_quarantine/smoke/cross_pair_all_groups_dryrun"
for y in pair_pool.yaml seed_pool.yaml pair_prompts.yaml; do
  sym "$REPO/outputs/cross_pair_lora_pooling/$y" \
      "$REPO/artifacts/_shared/cross_pair_pool_configs/$y"
done

say ""; say "--- Diagnostics and study output (scope call 2026-08-04) ---"
# Not rung experiments: measurement and study output that no rung produced.
# See inventory/sweeps/2026-08-04-scope-call.md for the decision and load tests.
for d in group_a_failure residual_diagnostics conditioning_window conditioning_window_lora; do
  sym "$REPO/outputs/$d" "$REPO/artifacts/diagnostics/$d"
done

say ""; say "--- Scope-owned output (scope call 2026-08-04) ---"
sym  "$REPO/outputs/animals_compose_transfer" \
     "$REPO/artifacts/scopes/does-the-fix-reach-unseen-pairs"
sym  "$REPO/outputs/compose_scorer" \
     "$REPO/artifacts/scopes/compose-scorer"
sym  "$REPO/outputs/poe" \
     "$REPO/artifacts/scopes/poe-baselines"

say ""; say "--- Cross-experiment presentation figures ---"
sym  "$REPO/outputs/presentation" \
     "$REPO/artifacts/_shared/presentation"

say ""; say "--- Caches ---"
sym  "$DAT/outputs/training_cache" \
     "$DAT/artifacts/caches/training_cache"
# training_cache_overfit_catdog is a symlink view (not empty) whose links were
# ALREADY broken pre-reorg (target a nonexistent repo outputs/training_cache).
# Quarantined by hand to artifacts/_quarantine/broken-symlink-view/. Not re-actioned here.
sym  "$REPO/outputs/manifold_cache" \
     "$REPO/artifacts/caches/manifold_cache"

say ""; say "=== done (APPLY=$APPLY) ==="
