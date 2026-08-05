# 🔨 Make the template build, and keep it building

## Description
Confirm the ICLR template compiles to a PDF from this repo, replace the stock
placeholders with real content, and write down the one figure-path rule that
every later section follows.

## Purpose
Nothing else in this scope is checkable until a PDF comes out. This plan also
settles the two things that get expensive to change later: how figures are
included (before any figure exists), and what the bib is called. Serves DoD 1-3.

## Goal
`tectonic paper/iclr/iclr2027_conference.tex` produces a PDF; the sample title
and the Cranberry-Lemon authors are gone; the figure-path rule is written into
`paper/iclr/README.md`.

## Environment Facts This Plan Depends On
- No system LaTeX on the cluster nodes: `pdflatex`, `xelatex`, `latexmk`, and
  `bibtex` are all absent from PATH.
- The build path is `tectonic` at `/home-mscluster/mmolefe/.local/bin/tectonic`.
  It fetches packages and fonts on demand, so the FIRST build on a given node
  needs network access. Later builds use its cache.
- Verified working 2026-08-05: a clean build of the stock template produced a
  73 KB PDF.
- `*.pdf` is gitignored (`.gitignore:31`). The built PDF is local-only, never
  committed. Look at it by building it.
- Known harmless noise: the build reports "TeX rerun seems needed, but stopping
  at 6 passes" and underfull-vbox warnings on the stock template. Neither is a
  failure. Judge the build by whether the PDF is written.

## Tasks
- [x] ✅ confirm the stock template builds with tectonic (done 2026-08-05, 73 KB
      PDF written)
- [ ] ⚠️ replace the stock title and the Cranberry-Lemon author block with the
      real author block; keep `\iclrfinalcopy` commented out so the submission
      stays anonymous  [inferred]
- [ ] ⚠️ decide the bib: keep the 23-line stub filename or rename, and add the
      references this paper actually cites  [inferred]
- [ ] ⚠️ write the figure-path rule into `paper/iclr/README.md`: one
      `\graphicspath` root plus a naming rule, chosen so a figure that does not
      exist yet can be referenced without breaking the build  [inferred]
- [ ] ⚠️ add a placeholder-figure macro so an owed figure renders as a visible
      grey box with its slot name, instead of failing the build  [inferred]

## Success/Failure Outcomes
- **the build**
  - Success: a PDF is written. Warnings about vboxes and rerun passes are noise.
  - Failure: no PDF, or an "I can't find file" error. If the first build on a
    node fails on a download, the node has no network: build on a login node.
- **the placeholder macro**
  - Success: a section referencing a figure that does not exist still compiles,
    and the reader sees a labelled grey box.
  - Failure: a missing figure breaks the build. That makes the whole draft
    hostage to unfinished runs, which is the thing this plan exists to prevent.

## Engagement Instructions
```bash
cd paper/iclr
/home-mscluster/mmolefe/.local/bin/tectonic iclr2027_conference.tex
ls -la iclr2027_conference.pdf          # expect a PDF, ~70 KB or larger

grep -c "Cranberry-Lemon" iclr2027_conference.tex   # expect 0
grep -c "Formatting Instructions" iclr2027_conference.tex  # expect 0
cat README.md                            # expect the figure-path rule
```
