#!/usr/bin/env bash
# Drive the mscluster checkout from this local one. Code is written here, committed,
# and pushed through GitHub; the cluster checkout is fast-forwarded to the same commit
# and every launch runs there, from the repo root, exactly as it would in a shell on
# the login node.
#
#   scripts/cluster.sh sync                push this branch, fast-forward the cluster to it
#   scripts/cluster.sh run "<command>"     sync, then run <command> in the cluster checkout
#                                          e.g. scripts/cluster.sh run "sbatch scripts/foo.sbatch"
#   scripts/cluster.sh sh "<command>"      run <command> in the cluster checkout, no sync
#   scripts/cluster.sh jobs                squeue for this user
#   scripts/cluster.sh pull <path>...      rsync git-ignored outputs back through the compute node
#                                          (skips model weights: *.pt *.safetensors *.npy)
#
# Env overrides: CLUSTER_LOGIN (default mscluster), CLUSTER_NODE (default mscluster84),
# CLUSTER_DIR (default the path below).
set -euo pipefail

LOGIN="${CLUSTER_LOGIN:-mscluster}"
NODE="${CLUSTER_NODE:-mscluster84}"
RDIR="${CLUSTER_DIR:-/home-mscluster/mmolefe/Playground/PhD/poe_repair_min}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"

die() { echo "cluster.sh: $*" >&2; exit 1; }

remote() { ssh "$LOGIN" "cd '$RDIR' && bash -lc $(printf '%q' "$1")"; }

sync() {
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD)"
  [ "$branch" = HEAD ] && die "detached HEAD; check out a branch first"
  if ! git diff --quiet || ! git diff --cached --quiet; then
    die "uncommitted changes to tracked files; commit them first so the cluster runs what you see"
  fi
  git push -q origin "$branch"
  # --ff-only: if the cluster checkout has commits of its own, stop and say so
  # rather than merging on the login node.
  remote "git fetch -q origin && git checkout -q '$branch' && git merge -q --ff-only 'origin/$branch'" \
    || die "cluster could not fast-forward to origin/$branch (diverged or dirty); reconcile with /reconcile-machines"
  echo "cluster at $(remote 'git log --oneline -1')"
}

case "${1:-}" in
  sync) sync ;;
  run)  [ $# -ge 2 ] || die 'usage: run "<command>"'; sync; remote "$2" ;;
  sh)   [ $# -ge 2 ] || die 'usage: sh "<command>"'; remote "$2" ;;
  jobs) remote 'squeue -u "$USER"' ;;
  pull)
    shift; [ $# -ge 1 ] || die "usage: pull <path>..."
    for p in "$@"; do
      p="${p%/}"
      mkdir -p "$(dirname "$p")"
      rsync -az --info=progress2 --exclude='*.pt' --exclude='*.safetensors' --exclude='*.npy' \
        "$NODE:$RDIR/$p" "$(dirname "$p")/"
    done ;;
  *) sed -n '2,17p' "$0"; exit 1 ;;
esac
