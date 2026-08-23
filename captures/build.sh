#!/usr/bin/env bash
# Render each captured markdown file as its own self-contained web page with real maths.
# These are independent documents sharing a directory, so one page per file, not one per folder.
set -euo pipefail
cd "$(dirname "$0")"
ASSETS="$HOME/.claude/skills/polish/assets"

command -v pandoc >/dev/null || { echo "pandoc not installed: the markdown is the artifact, nothing to do"; exit 0; }

OPTS=()
command -v pandoc-crossref >/dev/null && OPTS+=(--filter pandoc-crossref)
[ -f "$ASSETS/polish.lua" ] && OPTS+=(--lua-filter "$ASSETS/polish.lua")

# Local KaTeX so the page renders with no network. Falls back to the CDN.
if [ -d "$ASSETS/katex" ]; then
  [ -d katex ] || cp -r "$ASSETS/katex" katex
  OPTS+=(--katex=katex/)
else
  OPTS+=(--katex)
fi

for src in *.md; do
  [ -e "$src" ] || continue
  out="${src%.md}.html"
  pandoc "$src" "${OPTS[@]}" \
    --toc --standalone --css tufte.css \
    -o "$out"
  echo "wrote $out"
done

# crossref's inline label syntax only renders under pandoc-with-crossref. In any
# other previewer it reaches KaTeX as literal text and throws a parse error, so
# the markdown stops being readable on its own. Check the SOURCE, not the output.
STATUS=0
if grep -l '{#eq:\|\[@eq:' *.md 2>/dev/null | grep -q .; then
  echo "ERROR: pandoc-crossref equation syntax found in:" >&2
  grep -n '{#eq:\|\[@eq:' *.md >&2
  echo "This renders only under pandoc+crossref and throws a KaTeX parse error" >&2
  echo "in VS Code preview and on GitHub. Use an anchor instead:" >&2
  echo '  <a id="eq-name"></a>**(1) Caption**  ... cite as [(1)](#eq-name)' >&2
  STATUS=1
fi
exit "$STATUS"
