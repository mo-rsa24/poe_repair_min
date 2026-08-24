# Prompt: Scan, organize, and catalog all figures related to this plan

## Goal

Scan the entire repository for all image files (PNG, JPG, etc.) that are relevant to this plan. Rename them consistently, move them to a central folder, and generate a catalog table that maps each figure to:
- Its axes and labels
- What it shows (in one sentence)
- Which step(s) of the plan it supports
- Where it originally lived

## Search scope

Search recursively in:
- `artifacts/results/ (per-question) and report/paper-evidence-index.md`
- `outputs/interaction_term/`
- `paper/iclr/figures/`
- Any ``artifacts/notes/` items (formerly `/show-me`) or results directories
- Any previously generated figures from earlier runs of this plan

**Filter:** Only images created or modified within the last 3 months (or since the last plan run).

## Naming convention

Rename all figures to follow this pattern:
```
step-09_metric-XXXX_description.png
```

Where:
- `step-09`: The step number this plan file is (always 9 for this plan)
- `metric-XXXX`: The metric or concept the figure illustrates (e.g., `metric-compose-rate`, `metric-direction-cosine`, `method-live-logging`)
- `description`: A short kebab-case description (e.g., `compose-rate-climb`, `before-after-comparison`)

Examples:
- `step-09_metric-compose-rate_climb-over-epoch.png`
- `step-09_method-live-logging_before-after.png`
- `step-09_metric-direction-cosine_pool-alignment.png`

## Organization

**Destination folder:** `outputs/interaction_term/live_curves_smoke_run/figures/`

Move or copy (do not delete originals) all identified figures to this folder, applying the naming convention above.

## Catalog generation

After organizing, create a file: `outputs/interaction_term/live_curves_smoke_run/FIGURE_CATALOG.md`

The catalog must be a table with columns:

| Step | Figure (file path + title) | What it shows | Axes | Original location |
|------|----------------------------|---------------|------|-------------------|
| 9 | `step-09_metric-compose-rate_climb.png` <br> "Compose-rate climb over epoch 1" | Shows whether the PoE-blend scorer outputs increase over training. Flat line = fix didn't arrive. Climb = fix is working. | X: epoch step or training iteration. Y: compose-rate (0–1, fraction classified as PoE-blend). | `outputs/interaction_term/dose/pairs/` |
| 9 | `step-09_metric-direction-cosine_alignment.png` <br> "Direction alignment to pool-mean" | Shows whether the correction moves in the ensemble direction. High value = aligned and transferable. Low/zero = diverged. | X: epoch step. Y: cosine similarity (-1 to +1). | `artifacts/results/ (per-question) and report/paper-evidence-index.md` |
| ... | ... | ... | ... | ... |

## Instructions for you

1. **Run the scan:** Open a Python shell and execute a script (or manually browse) to find all image files matching the scope above.
2. **Examine each figure:** Open each image. Note its title, axes, legend, and what it measures. Add a row to the table.
3. **Rename and move:** Rename each figure to the naming convention. Copy to `outputs/interaction_term/live_curves_smoke_run/figures/`.
4. **Generate the table:** Build the FIGURE_CATALOG.md table with all rows filled in.
5. **Link it:** Once complete, update the plan file's [Figure Catalog](#figure-catalog) section to embed or link to this catalog.

## Expected figures for this plan

Based on the plan design, expect to find or generate:
- A compose-rate curve (climb or flat)
- A direction-cosine curve (high or low)
- A fraction-of-distance-reached curve (plateau around 40%)
- A before/after comparison (showing the value of live logging)
- A diagram of the three metrics at a glance
- A context diagram (training loop → eval hook → three metrics → W&B)

If fewer than 5 figures exist, generate them using the diagram prompts in `diagrams/*.prompt.md`.

## Output

Place the completed `FIGURE_CATALOG.md` in `outputs/interaction_term/live_curves_smoke_run/` and link it from the plan file.
