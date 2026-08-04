#!/usr/bin/env bash
# Run /execute-plan-tree over a plan scope, unattended.
#
# Only this launcher is permissive. It loads .claude/execute-plan-tree.settings.json
# and runs in bypassPermissions mode, so no command prompts appear. A normal
# `claude` session in this repo is unaffected and still asks as before.
#
# Before launching it does three things:
#   1. commits any uncommitted work, so there is always a way back
#   2. runs /verify-plan over the scope and stops if any plan has thin
#      Engagement Instructions (those tasks would only be parked anyway)
#   3. prints the SHA to revert to if the run goes wrong
#
# Usage:
#   scripts/run-plan-tree.sh [options] <scope-path>
#
# Options:
#   --verify-only   run the admissions check and stop; never launches
#   --no-verify     skip the admissions check (you already ran /verify-plan)
#   -h, --help      this text
#
# Examples:
#   scripts/run-plan-tree.sh plans/interaction-term
#   scripts/run-plan-tree.sh --verify-only plans/compose-scorer

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTINGS="$REPO_ROOT/.claude/execute-plan-tree.settings.json"

usage() { sed -n '2,${/^[^#]/q;p;}' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

VERIFY=1
LAUNCH=1
SCOPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify-only) LAUNCH=0; shift ;;
    --no-verify)   VERIFY=0; shift ;;
    -h|--help)     usage; exit 0 ;;
    -*)            echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)             SCOPE="$1"; shift ;;
  esac
done

[[ -n "$SCOPE" ]] || { echo "error: no scope given" >&2; usage >&2; exit 2; }

cd "$REPO_ROOT"

# --- scope must exist and be a real plan scope -------------------------------
SCOPE="${SCOPE%/}"
if [[ ! -d "$SCOPE" ]]; then
  echo "error: no such scope directory: $SCOPE" >&2
  exit 2
fi
if [[ ! -f "$SCOPE/MASTER_PLAN.md" ]]; then
  echo "error: $SCOPE has no MASTER_PLAN.md" >&2
  echo "       execute-plan-tree refuses a scope without one; run /init-master-plan first." >&2
  exit 2
fi
if ! compgen -G "$SCOPE/plans/*.md" >/dev/null; then
  echo "error: $SCOPE/plans/ has no plan files" >&2
  echo "       run /populate-plans on this scope first." >&2
  exit 2
fi

[[ -f "$SETTINGS" ]] || { echo "error: missing settings profile: $SETTINGS" >&2; exit 1; }

echo "scope:  $SCOPE"
echo "plans:  $(compgen -G "$SCOPE/plans/*.md" | wc -l) file(s)"
echo

# --- 1. never run with uncommitted work ---------------------------------------
DIRTY=$(git status --porcelain --untracked-files=all | wc -l)
if (( DIRTY > 0 )); then
  echo "working tree dirty: $DIRTY path(s); committing a safety point first"
  git add -A
  git commit --quiet -m "$(cat <<EOF
pre-run safety commit before executing $SCOPE

Automatic commit made by scripts/run-plan-tree.sh so the unattended run
has a clean revert point. No content changes.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
  echo "committed $(git rev-parse --short HEAD)"
  echo
fi

REVERT_TO="$(git rev-parse --short HEAD)"

# --- 2. admissions check: are the plans actually runnable? ---------------------
if (( VERIFY )); then
  echo "checking Engagement Instructions across $SCOPE ..."
  VERIFY_PROMPT="/verify-plan $SCOPE

Report only. Do not edit any file.
List every plan file whose Engagement Instructions are missing, thin, or
placeholder rather than a concrete command, query, or gate with a pass
condition. Then print a final line in exactly this form and nothing after it:
THIN_COUNT=<n>"

  VERIFY_OUT="$(claude -p "$VERIFY_PROMPT" --permission-mode plan 2>&1 || true)"
  echo "$VERIFY_OUT"
  echo

  THIN="$(printf '%s' "$VERIFY_OUT" | grep -oE 'THIN_COUNT=[0-9]+' | tail -1 | cut -d= -f2 || true)"
  if [[ -z "$THIN" ]]; then
    echo "warning: could not read a THIN_COUNT from the check; treating as inconclusive." >&2
    echo "         re-run with --no-verify to proceed anyway." >&2
    exit 3
  fi
  if (( THIN > 0 )); then
    echo "$THIN plan(s) have thin Engagement Instructions."
    echo "Their tasks would only be parked, so this run would accomplish little."
    echo "Fix them first:  /verify-plan $SCOPE   (it proposes repairs)"
    exit 3
  fi
  echo "all plans have concrete Engagement Instructions."
  echo
fi

if (( ! LAUNCH )); then
  echo "--verify-only: stopping here."
  exit 0
fi

# --- 3. launch ----------------------------------------------------------------
echo "revert point if this goes wrong:  git reset --hard $REVERT_TO"
echo

# The manifest pre-approval lives here, not in SKILL.md, so it applies only to
# sessions started through this launcher.
read -r -d '' PROMPT <<EOF || true
/execute-plan-tree $SCOPE

This session is running unattended through scripts/run-plan-tree.sh.
The dry-run manifest is pre-approved for this run: print it for the record,
then proceed straight into execution without waiting for my approval.
Everything else in the skill still applies. Keep parking any task whose
verification needs a human to read the output, and keep committing one task
at a time.
EOF

exec claude \
  --settings "$SETTINGS" \
  --permission-mode bypassPermissions \
  "$PROMPT"
