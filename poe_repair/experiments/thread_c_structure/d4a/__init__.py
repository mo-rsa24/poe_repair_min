"""D4-A / D4-A-t substitution test (Thread C).

Per the phase-0 plan §7b, this submodule plumbs a per-step Δ_t override
through the inner sampling loop and runs the four substitution conditions
(oracle / shared-mean / shuffle / zero) over the seeds in the cache, then
grades each output with the §4 detection + VQA protocol.

Modules:
    overrides — build the four per-condition Δ tensors from cached eps.
    runner    — drive ``run_delta_override`` and grade the outputs.
    figures   — render the D4-A bar chart and the D4-A-t small-multiples.
"""

from poe_repair.experiments.thread_c_structure.d4a.overrides import (  # noqa: F401
    Condition, OverrideBuilder, build_overrides_for_seed,
)
from poe_repair.experiments.thread_c_structure.d4a.runner import (  # noqa: F401
    D4aResult, D4aSeedRow, D4aGradeRecord, run_d4a,
)
from poe_repair.experiments.thread_c_structure.d4a.figures import (  # noqa: F401
    render_d4a, render_d4a_t,
)
