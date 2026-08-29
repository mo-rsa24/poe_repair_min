# Section 4 teaching figure: make the λ dial unmistakable

The figure this note specifies belongs to section 4, "Restoring the Plurality Term Restores Composition". Its one job: a reader must not confuse the dial (λ, the fraction of the measured correction r_t added back at each step, 0 none to 1 all) with the measure (the per-step size ratio ‖r_t‖/‖ε_PoE‖, 5 to 14% per step from `figures/correction-size-over-the-denoising-run.json`, which never enters the sampler).

## The chosen layout

Rebuild `figures/compose-rate-as-correction-rises.pdf` (rows: real correction, wrong pair, wrong seed, wrong order; columns: λ 0, 0.25, 0.5, 0.75, 1; compose-rate curve below) with three changes:

**The formula in the column header.** Each column is labelled with the injection rule with λ substituted, ε_PoE + 0.25·r_t and so on, the numeral bold, replacing the bare λ labels. Under the grid, one fixed muted line: the size of r_t is 5 to 14% of ε_PoE per step and does not change with λ.

**The λ=1 footnote.** λ=1 reproduces the joint prediction by construction (endpoint drift 1.9 grey levels of 255, from `figures/F5-one-dial-three-instruments.json`), so that column is a consistency check, not evidence; the persuasive columns are 0.25 to 0.75. The footnote marker sits on the column header, the sentence in the caption.

**The clipping fix.** Both existing dose grids clip their top-right annotation ("filled = composed", "compose needs two"). Every annotation is placed inside the axes bounding box with explicit right-margin padding, and the build is checked at final column width before export.

Numbers stay sourced from `outputs/interaction_term/dose/dose_curves.json`: real-correction compose rate 3.1% at λ=0 rising to 93.8% at λ=1, controls at or below 9.4% everywhere, AUC 0.387 against 0.023 to 0.047, 32 cells per row.

## The two alternatives, kept in case the page budget bites

**The dial strip.** One row (cat×dog seed 9), five columns, formula header, ratio sentence beneath, detector-count chips (1, 1, 1, 2, 2). Teaches the dial in a third of the height but carries no controls and no population; it must sit beside the curve figure, not replace it.

**The window as a second dial.** Two rows (correction injected at steps 0 to 10 versus 40 to 50), columns λ within the window, same formula header. Shows the effect is about when, not only how much. Blocked on renders: `outputs/interaction_term/dose_matched/pairs/` holds four matched-total cells, not a λ sweep per window, so building it needs 2 windows × 5 λ values × 1 seed of new renders. It also cannot make the matched-total comparison (that forces λ up to 2.96, outside the 0-to-1 dial), so the existing 2×2 matched-total figure keeps that job.
