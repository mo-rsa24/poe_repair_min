# Overleaf Handoff

Upload `paper/iclr-overleaf.zip` to Overleaf as a new project.

Project settings:

- Main file: `iclr2027_conference.tex`
- Compiler: start with Overleaf's default pdfLaTeX for the ICLR template; if it fails, switch to XeLaTeX. This bundle was verified locally with Tectonic's XeTeX backend.
- Bibliography: currently disabled in `iclr2027_conference.tex`; uncomment the `\bibliography{iclr2027_conference}` and `\bibliographystyle{iclr2027_conference}` lines once citation keys are finalized.

Bundle contents:

- Manuscript source: `iclr2027_conference.tex`
- Local template/support files: `iclr2027_conference.sty`, `iclr2027_conference.bst`, `natbib.sty`, `fancyhdr.sty`, `math_commands.tex`
- Bibliography source: `iclr2027_conference.bib`
- Figure assets under `figures/`
- Lightweight writing notes: `README.md`, `figures.md`, `SPINE.md`, `DRAFT_MAP.md`, and this file

The bundle deliberately excludes generated build files, the compiled manuscript PDF, and root-level reference PDFs.

Regenerate the bundle from the repository root with:

```bash
scripts/make_overleaf_bundle.sh
```
