# Paper: where the LaTeX lives and how it is built

Navigation: 📋 [Index](00-INDEX.md) | [Overview](overview.md#paper-latex-build)

The manuscript is `paper/iclr/`, owned by the
`plans/closing-the-compositional-gap/plans/writing-the-paper/` scope. Editing and building it
is a different loop from running experiments: no GPU, no queue, no job.

## Where to make changes

All prose goes in `paper/iclr/iclr2027_conference.tex`, the single manuscript file. The other
files in that folder are the ICLR distribution and are not edited: `iclr2027_conference.sty`
(layout), `.bst` (bibliography style), `natbib.sty`, `fancyhdr.sty`, and `math_commands.tex`
(macro definitions, worth reading before defining a new macro since it likely already exists).
References go in `iclr2027_conference.bib`. Figures are referenced, never copied in: they are
produced by `plans/closing-the-compositional-gap/plans/does-the-correction-cause-composition/`
and `plans/closing-the-compositional-gap/plans/does-the-fix-reach-unseen-pairs/`, and
`paper/iclr/README.md` holds the path rule.

## No system LaTeX exists on this cluster

Confirmed live on 2026-08-24: `pdflatex`, `xelatex`, `latexmk`, and `bibtex` are all absent from
PATH (`which` returns nothing for each). Every LaTeX recipe except `tectonic` fails here,
including the extension's own default recipe (`latexmk`, see below). Any instruction, script, or
CI step that assumes a normal TeX install is wrong on this cluster.

## Building it, the terminal route (what Claude and any script uses)

```bash
cd paper/iclr
/home-mscluster/mmolefe/.local/bin/tectonic iclr2027_conference.tex
```
Confirmed present and executable at that path live on 2026-08-24. `tectonic` is a
self-contained engine: it needs no TeX installation and fetches packages and fonts on demand
into `~/.cache/Tectonic`. The FIRST build on a node therefore needs network access. Later builds
are offline and fast.

**Reading the output.** The build writes `iclr2027_conference.pdf` next to the `.tex`. `*.pdf`
is gitignored (`.gitignore:31`), so the PDF is never committed: the paper is read by building
it, either through the VS Code PDF viewer or from a laptop over the same SSH tunnel the LoRA
Inspector is reached through.

**Build noise that is not failure.** A successful build still prints underfull `\vbox` warnings
and `TeX rerun seems needed, but stopping at 6 passes`. Both are normal here. Judge a build by
whether the PDF was written, not by whether the output was silent.

## Building it, the VS Code route

LaTeX Workshop (`james-yu.latex-workshop`, version `10.15.2`, installed server-side, confirmed
present live via `ls ~/.vscode-server/extensions/` on 2026-08-24) is the extension.
`Ctrl+Shift+P` -> `LaTeX Workshop: Build with recipe` -> `tectonic`, then `LaTeX Workshop: View
LaTeX PDF` to see it.

**The extension's default recipe is `latexmk`, which does not exist on this cluster.** Plain
"Build LaTeX project" (no recipe chosen) therefore fails, not because anything is wrong with the
document. The recipe has to be picked by hand as above. Nine recipes ship by default;
`latexmk` is first, `tectonic` is last (confirmed by reading
`~/.vscode-server/extensions/james-yu.latex-workshop-10.15.2/package.json` on 2026-08-05).

**This may be partly fixed and needs a re-check, not a rebuild, to confirm.** A tracked
`.vscode/settings.json` now exists in the repo (added in commit `2c4d0b6`, "Working-tree state
before the retrofit rename"; confirmed live by reading the file on 2026-08-24) and defines a
custom `tectonic` recipe and tool pointing at the absolute `tectonic` path above. It does
**not** set `latex-workshop.latex.recipe.default`, so whether plain "Build LaTeX project" now
uses `tectonic` automatically, or still tries `latexmk` first and fails, is unverified: this
sitting deliberately did not run a build (that would write a PDF outside the environment folder's
scope). This contradicts the earlier-verified claim below that "there is no project or machine
settings file," and is logged as an open item rather than resolved silently. See "Still open" in
`00-INDEX.md`.

Earlier verification (2026-08-05): "Absence of project or machine settings confirmed by `ls`."
That was true on 2026-08-05 and is no longer true as of the `.vscode/settings.json` addition; the
2026-08-05 mark is kept here as a dated fact, not deleted, per this file's own currency rule.

The VS Code keystroke route above is user-stated (2026-08-05) and consistent with the recipe
list read from the extension's `package.json`, but the command-palette action itself cannot be
driven from a shell, so it has never been machine-verified.
