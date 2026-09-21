#!/usr/bin/env bash
# Fully qualified paths for every sheet the conveyor has finished, and what is still cooking.
G=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from
L=/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32
echo "=== SHEETS READY TO JUDGE          $(date '+%H:%M')"
if [ -s "$G/conveyor-sheets.txt" ]; then
  nl -w2 -s'. ' "$G/conveyor-sheets.txt" | sed 's/^/  /'
else
  echo "  none yet from the conveyor"
fi
echo
echo "=== EVERY SHEET IN THE FOLDER, NEWEST FIRST"
ls -t "$G/inspection-grids"/*.png 2>/dev/null | head -20 | sed 's/^/  /'
echo
echo "=== RUNNING NOW"
for n in 112 109; do
  printf "  mscluster%s  " $n
  timeout 12 ssh -o BatchMode=yes mscluster$n 'pgrep -af build_training_cache | grep -v pgrep | sed "s/.*--prompt-a /rendering: /; s/--joint-prompt.*--seed /seed /; s/--split.*//" | head -1' 2>/dev/null || true
  echo
done
echo "  conveyor rounds logged:"; grep -c '^=== round' "$L/logs/conveyor_112.log" 2>/dev/null || echo "    0"
