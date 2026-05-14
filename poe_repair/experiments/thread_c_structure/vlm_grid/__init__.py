"""§7c VLM-projection grid (Thread C).

For each chosen injected timestep t and each seed, finish a vanilla PoE
trajectory in which step t alone is replaced by ``ε̃_PoE + α · Δ_t`` for
α ∈ {0, α_partial, 1}. Grade the decoded image with the two-axis VLM
protocol (co-occurrence score + separation confidence) and plot each
seed's three points connected by arrows.

This is the §7c VLM grid — see the consolidated plan §7c "Method 2".

Modules:
    runner   — orchestrate the sampler + grader sweep.
    figures  — render the 6-panel arrow grid with ellipses and route tags.
"""

from poe_repair.experiments.thread_c_structure.vlm_grid.runner import (  # noqa: F401
    VlmGridResult, VlmGridSample, run_vlm_grid,
)
from poe_repair.experiments.thread_c_structure.vlm_grid.figures import (  # noqa: F401
    render_vlm_grid, render_vlm_calibration,
)
