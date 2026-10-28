#!/bin/bash
# Train the ten notebook datasets, spread over the free cards.
# GPU guard: a card is used only if it has under 1 GB in use when this script starts.
set -u
REPO=/home-mscluster/mmolefe/PhD/reduce-reuse-recycle/.claude/worktrees/unbuild-walk
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=$REPO/rebuild/out/phase1/ckpt
mkdir -p "$OUT"
[ -x "$PY" ] || { echo "co3 python missing"; exit 1; }
nvidia-smi > /dev/null || { echo "no GPU visible"; exit 1; }
used=$(df --output=pcent "$REPO" | tail -1 | tr -dc 0-9)
[ "$used" -lt 90 ] || { echo "disk ${used}% full"; exit 1; }
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1024{print $1}')
[ ${#FREE[@]} -gt 0 ] || { echo "no card under 1 GB"; exit 1; }
echo "node $(hostname) free cards ${FREE[*]}"
PANELS=("toy_gmm(std=.03)" "toy_gauss()" "toy_box(0.7)" "toy_box(0.3)" "bar(scale=0.2)" "bar_horizontal()" "toy_gmm_left()" "toy_gmm_right()" "right_bar()" "left_bar()")
i=0
cd "$REPO"
for p in "${PANELS[@]}"; do
  card=${FREE[$((i % ${#FREE[@]}))]}
  slug=$(echo "$p" | sed 's/[^[:alnum:]]/_/g; s/^_*//; s/_*$//')
  REBUILD_DEVICE=cuda:$card "$PY" -m rebuild.train --panel "$p" --out "$OUT" > "$OUT/$slug.log" 2>&1 &
  echo "launched $p on card $card pid $!"
  i=$((i+1))
done
wait
echo "all done"
