#!/usr/bin/env bash
# Start a Claude Code session that runs /execute-plan-tree unattended.
#
# Only this launcher is permissive. It loads .claude/execute-plan-tree.settings.json
# and runs in bypassPermissions mode, so no command prompts appear. A normal
# `claude` session in this repo is unaffected and still asks as before.
#
# Usage:
#   scripts/run-plan-tree.sh <scope-path>
#
# Example:
#   scripts/run-plan-tree.sh plans/artifact-reconciliation

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTINGS="$REPO_ROOT/.claude/execute-plan-tree.settings.json"

if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <scope-path>" >&2
  echo "example: $(basename "$0") plans/artifact-reconciliation" >&2
  exit 2
fi

SCOPE="$1"

if [[ ! -f "$SETTINGS" ]]; then
  echo "missing settings profile: $SETTINGS" >&2
  exit 1
fi

# The pre-approval line lives here, not in SKILL.md, so it applies only to
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
