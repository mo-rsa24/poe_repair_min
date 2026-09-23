#!/bin/bash
# Keep only the CO3 renders chosen by eye; delete everything else under the CO3 tree.
ROOT="$1"
KEEP="a_bison_and_a_horse:42
a_bobcat_and_a_quail:42
a_camel_and_a_llama:2,9
a_cow_and_a_chicken:2,9,42
a_cow_and_a_milk_churn:2,9,42
a_goat_and_a_sheep:2
a_leopard_and_a_cheetah:42
a_rabbit_and_a_hare:42
a_soccer_ball_and_a_park_bench:2,9,11,42
a_suitcase_and_a_globe:2,11,42
a_tiger_and_a_parrot:2,9
a_wolf_and_a_coyote:9,42
a_wolf_and_a_raven:2,9,42
a_zebra_and_a_horse:2,9"
cd "$ROOT" 2>/dev/null || exit 0
for d in */; do
  p=${d%/}
  line=$(echo "$KEEP" | grep "^$p:")
  if [ -z "$line" ]; then rm -rf "$p" && echo "deleted $p"; continue; fi
  seeds=${line#*:}
  for s in "$p"/seed-*; do
    n=${s##*seed-}
    echo ",$seeds," | grep -q ",$n," || { rm -rf "$s" && echo "deleted $s"; }
  done
done
