"""Experiments — one file per scientific question.

Each experiment file exposes a ``main()`` that takes CLI args and writes
results under ``outputs/<exp_name>/``. The experiments are:

  - ``e1_held_out``       : main result. Held-out collision pairs × seeds.
                            Compare Mono / PoE / sched-M2 (literal e_J) /
                            sched-M2 (ê_J) / each with FOCUS/AAE/CO3.
  - ``e2_synth_audit``    : synthesizer reconstruction error of ê_J vs e_J,
                            both at the embedding level (cosine, MSE) and
                            at the UNet output level (‖ε(x_t,t,ê_J) −
                            ε(x_t,t,e_J)‖ averaged over trajectories).
  - ``e3_mono_methods``   : sanity. Mono + {FOCUS, AAE, P2P, CO3} on the
                            same seeds, to confirm each method actually
                            fixes Mono in our SDXL setup.
  - ``e4_appendix_xstack``: optional. Run AAE (SD2.1 native) on one prompt
                            via its official pipeline, compare to our SDXL
                            port qualitatively. Faithfulness sanity check.

ENVIRONMENT
-----------
All experiments require the ``co3`` conda env (or any env with diffusers,
xformers, sentence_transformers, and the project's deps). The base /
jaxstack envs do NOT work. Use::

    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m poe_repair.experiments.<exp> ...

or activate first via ``conda activate co3``. The ``_assert_env_ok`` helper
below performs a fast sanity check; each experiment ``main()`` calls it
before doing any work so failures surface with a clear message.
"""

from __future__ import annotations

import sys


def _assert_env_ok() -> None:
    """Fail fast if the running interpreter doesn't have the project deps.

    This catches the common error of running ``python -m poe_repair.experiments.e3_mono_methods``
    with the system / jaxstack interpreter instead of the ``co3`` env.
    """
    missing: list[str] = []
    for mod in ("torch", "diffusers", "transformers", "xformers"):
        try:
            __import__(mod)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{mod}  ({exc.__class__.__name__}: {exc})")
    if missing:
        msg = (
            "[poe_repair] required modules failed to import in this Python:\n"
            "  " + "\n  ".join(missing) + "\n"
            f"current interpreter: {sys.executable}\n"
            "Run with the co3 env, e.g.:\n"
            "  /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m "
            "poe_repair.experiments.<exp> ...\n"
            "or  conda activate co3  before invoking python."
        )
        print(msg, file=sys.stderr)
        sys.exit(2)
