# 🔨 Make the template build, and keep it building

**Step 16 of 22.** Waits on nothing. The one order is the `## Running order` table in the [repo root MASTER_PLAN.md](../../../../../MASTER_PLAN.md).

| Step | Plan | Status |
|---|---|---|
| 15 | [gate-01-two-literature-checks-before-print](../../does-the-correction-cause-composition/plans/gate-01-two-literature-checks-before-print.md) | ⚠️ |
| **16** | **this plan** | **◑ title still a stub** |
| 17 | [writing-02-the-title-and-the-section-spine](writing-02-the-title-and-the-section-spine.md) | ⚠️ |

## What this asks, in one line
Prove the paper builds to a PDF on this cluster, remove the template's stock content, and fix the two conventions that get expensive to change later: how figures are pathed and what the bibliography is called.

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
See `environment/paper.md`, "Paper: where the LaTeX lives and how it is built",
for the full picture. What this plan depends on specifically:
- No system LaTeX: every recipe except `tectonic` fails, including LaTeX
  Workshop's default (`latexmk`). This plan's tasks assume the tectonic path.
- The FIRST build on a node needs network access, since tectonic downloads
  packages. On a walled-off node, the build task moves to a node with network.
- `*.pdf` is gitignored, so the de-stubbing tasks are checked by grepping the
  `.tex` and by looking at a locally built PDF, never by a committed artifact.
- Verified working 2026-08-05: a clean build of the stock template produced a
  73 KB PDF, so the toolchain question is settled before this plan starts.

## Tasks
- [x] confirm the stock template builds with tectonic (done 2026-08-05, 73 KB
      PDF written)
- [ ] replace the stock title `Lorem Ipsum for a Future ICLR 2027 Submission`
      with the real one (author block done: now `Anonymous Authors`, and
      `\iclrfinalcopy` stays commented out so the submission is anonymous)
      [inferred]
- [ ] finish the bib: it is past the stub at 67 lines, so what remains is the
      references this paper actually cites  [inferred]
- [x] write the figure-path rule into `paper/iclr/README.md`: one
      `\graphicspath` root plus a naming rule, chosen so a figure that does not
      exist yet can be referenced without breaking the build  [inferred]
      ✓ verified (`paper/iclr/README.md`: root, names, formats; untracked in git)
- [ ] add a placeholder-figure macro so an owed figure renders as a visible
      grey box with its slot name, instead of failing the build  [inferred]
- [ ] set `latex-workshop.latex.recipe.default` to `tectonic` in
      `.vscode/settings.json`, so plain Build works instead of failing on the
      missing `latexmk`, and the recipe stops needing to be picked by hand
      [inferred]

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
