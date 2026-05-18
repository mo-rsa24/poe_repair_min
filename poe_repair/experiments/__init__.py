"""Experiments — one package per scientific question.

Active experiments:

  - ``m5_lora_sdxl``      : LoRA per-arm composition on SDXL (success thread).
  - ``group_a_corrector`` : latent UNet / latent CNN / frozen-feature-MLP
                            students as demonstrable failure cases.
  - ``idea1``             : mono-during-inference residual diagnostic.
  - ``idea5a``            : mono-during-inference residual diagnostic.
  - ``veracity``          : mono-during-inference residual diagnostic.

ENVIRONMENT
-----------
All experiments require the ``co3`` conda env (or any env with diffusers,
xformers, sentence_transformers, and the project's deps). Use::

    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python -m poe_repair.experiments.<exp> ...

or activate first via ``conda activate co3``. The ``_assert_env_ok`` helper
below performs a fast sanity check; each experiment ``main()`` calls it
before doing any work so failures surface with a clear message.
"""

from __future__ import annotations

import sys


def _assert_env_ok() -> None:
    """Fail fast if the running interpreter doesn't have the project deps."""
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
