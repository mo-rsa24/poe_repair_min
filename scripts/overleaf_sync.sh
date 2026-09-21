#!/usr/bin/env bash
# Move the manuscript between paper/iclr (source of truth) and a local clone of
# the Overleaf project's git bridge.
#
#   scripts/overleaf_sync.sh push            # copy paper/iclr -> clone, show the diff
#   scripts/overleaf_sync.sh push -m "msg"   # ... and commit + push it
#   scripts/overleaf_sync.sh pull            # git pull in the clone, copy clone -> paper/iclr
#   scripts/overleaf_sync.sh status          # what differs, in both directions, no writes
#
# The clone lives at paper/overleaf-iclr, gitignored by this repo because it
# carries the Overleaf project's own history. Override with OVERLEAF_CLONE.
set -euo pipefail

root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
paper_dir="$root/paper/iclr"
clone="${OVERLEAF_CLONE:-$root/paper/overleaf-iclr}"
all_figures=0
prune=0
message=""

mode="${1:-status}"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message) message="$2"; shift 2 ;;
    --clone) clone="$2"; shift 2 ;;
    --all-figures) all_figures=1; shift ;;
    --prune) prune=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$clone/.git" ]]; then
  cat >&2 <<MSG
No Overleaf clone at: $clone

Clone it first (Overleaf: Menu -> Sync -> Git for the project id; Account
Settings -> Git integration for the token, used as the password):

  git clone https://git.overleaf.com/<project-id> "$clone"

Then re-run this. Set OVERLEAF_CLONE to use a different path.
MSG
  exit 1
fi

# The file set Overleaf gets: what the manuscript needs to compile, and nothing
# else. No writing notes, no build products, no reference PDFs, no standalone
# abstract drafts. Overleaf is where the paper is read, not where it is worked on.
includes=(
  --include='/figures/' --include='/figures/***'
  --exclude='/abstract_candidate_*'
  --include='/*.tex'
  --include='/*.sty' --include='/*.bst' --include='/*.bib'
  # The compiled manuscript ships too, so the project has a readable PDF the
  # moment it opens, before anyone waits on Overleaf's own compile.
  --include='/iclr2027_conference.pdf'
  --exclude='*'
)

# Keep only the figures the manuscript includes, and every variant of each one
# (a stem named as .pdf in the tex keeps its .png sibling too, so a co-author can
# swap the format in Overleaf without the file being missing). Everything else
# under figures/ is drafts for sections not yet written.
figure_filter() {
  local target="$1"
  [[ "$all_figures" == "1" ]] && return 0
  [[ -d "$target/figures" ]] || return 0
  local stems
  stems="$(grep -ohE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$paper_dir"/*.tex \
           | sed -E 's/.*\{([^}]+)\}/\1/' | sed 's#^figures/##' | sed -E 's/\.[A-Za-z0-9]+$//' \
           | sort -u)"
  find "$target/figures" -type f -print0 | while IFS= read -r -d '' f; do
    base="${f#$target/figures/}"
    case "${base##*.}" in
      pdf|png|jpg|jpeg|eps) ;;
      # .svg is an editing source LaTeX cannot include, and the two in this paper
      # are 21 MB between them. .json sidecars are data. Neither is a figure variant.
      *) rm -f "$f"; continue ;;
    esac
    grep -qxF "${base%.*}" <<< "$stems" || rm -f "$f"
  done
  find "$target/figures" -type d -empty -delete
  echo "figure stems kept:" >&2
  sed 's/^/  /' <<< "$stems" >&2
}

case "$mode" in
  push)
    stage="$(mktemp -d)"; trap 'rm -rf "$stage"' EXIT
    rsync -a "${includes[@]}" "$paper_dir/" "$stage/"
    figure_filter "$stage"
    rsync -a $([[ "$prune" == "1" ]] && echo --delete) \
      --exclude='.git' "$stage/" "$clone/"
    echo
    git -C "$clone" status --short
    if [[ -n "$message" ]]; then
      git -C "$clone" add -A
      if git -C "$clone" diff --cached --quiet; then
        echo "nothing to commit; Overleaf already matches paper/iclr"
      else
        git -C "$clone" commit -m "$message"
        git -C "$clone" push
      fi
    else
      echo
      echo "Staged in the clone but not committed. Review, then:"
      echo "  git -C \"$clone\" add -A && git -C \"$clone\" commit -m '...' && git -C \"$clone\" push"
      echo "Or re-run with -m 'message' to do all three."
    fi
    ;;
  pull)
    git -C "$clone" pull --ff-only
    # Never deletes locally: a file that only exists in paper/iclr is work in
    # progress that was deliberately kept off Overleaf.
    rsync -a "${includes[@]}" --exclude='.git' "$clone/" "$paper_dir/"
    echo
    git -C "$root" status --short -- paper/iclr
    ;;
  status)
    git -C "$clone" fetch --quiet || true
    echo "== clone: $clone"
    git -C "$clone" status --short --branch
    echo
    echo "== files differing between paper/iclr and the clone"
    stage="$(mktemp -d)"; trap 'rm -rf "$stage"' EXIT
    rsync -a "${includes[@]}" "$paper_dir/" "$stage/"
    figure_filter "$stage" >/dev/null 2>&1
    diff -rq "$stage" "$clone" -x .git || true
    ;;
  *)
    echo "usage: $(basename "$0") {push|pull|status} [-m msg] [--clone DIR] [--all-figures] [--prune]" >&2
    exit 2 ;;
esac
