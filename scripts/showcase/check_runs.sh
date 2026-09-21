#!/usr/bin/env bash
# Did the pooled runs finish, or stop early? Exit code alone cannot tell you: an aborted run exits 0
# and its log reads like a clean completion. The two things that separate them are verdict.json and
# the last checkpoint's step against the intended total.
L=/datasets/mmolefe/poe_repair_min/outputs/showcase/improve_r32
TARGET="${TARGET:-30000}"
printf "%-20s %-10s %8s %10s  %s\n" run verdict step alive reason
for d in "$@"; do
  r=$(basename "$d")
  v=$(python3 -c "import json;print(json.load(open('$L/$r/verdict.json'))['verdict'])" 2>/dev/null || echo "-")
  why=$(python3 -c "import json;print(json.load(open('$L/$r/verdict.json')).get('reason','')[:58])" 2>/dev/null)
  step=$(ls "$L/$r/checkpoints"/lora_step_*.pt 2>/dev/null | sed 's/.*lora_step_0*//; s/\.pt//' | sort -n | tail -1)
  # The runs live on other nodes, so a local pgrep always returns zero and would report a healthy
  # run as dead. Ask each node instead.
  alive=0
  for n in 106 108 109 110 112; do
    c=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 mscluster$n "pgrep -fc 'run-id $r'" 2>/dev/null)
    [ -n "$c" ] && alive=$((alive + c))
  done
  flag=""
  [ "$v" = "aborted" ] && flag="  <-- ABORTED, not finished"
  [ -n "$step" ] && [ "$step" -lt "$TARGET" ] && [ "$v" = "completed" ] && flag="  <-- short of $TARGET"
  printf "%-20s %-10s %8s %10s  %s%s\n" "$r" "$v" "${step:--}" "$alive" "$why" "$flag"
done
