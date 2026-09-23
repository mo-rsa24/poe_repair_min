#!/bin/bash
# Delete every render whose pair names a subject on either list in
# scripts/showcase/disallowed_subjects.py. The audition and CO3 render paths do not import that
# check, so these have to be swept out by name.
ROOT="$1"
WORDS="iguana lizard gecko reptile crocodile alligator gator frog toad snake serpent rat rats mouse mice shark eel spider scorpion worm slug moth beetle bat wasp hornet mosquito tick leech"
cd "$ROOT" 2>/dev/null || exit 0
for d in */; do
  p=${d%/}
  for w in $WORDS; do
    case "_${p}_" in
      *_${w}_*|*_${w}s_*) rm -rf "$p" && echo "deleted $p"; break;;
    esac
  done
done
