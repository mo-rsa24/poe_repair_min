"""Run both residual_diagnostics sub-experiments back-to-back.

Equivalent to::

    python -m poe_repair.experiments.residual_diagnostics.existence
    python -m poe_repair.experiments.residual_diagnostics.clip_window

The clip_window stage reads cached artifacts produced by existence, so
running existence first is required. Both sub-mains parse the same
``sys.argv[1:]``; pass shared flags like ``--pair`` / ``--seed`` once.
"""

from __future__ import annotations

from poe_repair.experiments.residual_diagnostics.existence.main import (
    main as existence_main,
)
from poe_repair.experiments.residual_diagnostics.clip_window.main import (
    main as clip_window_main,
)


def main() -> None:
    existence_main()
    clip_window_main()


if __name__ == "__main__":
    main()
