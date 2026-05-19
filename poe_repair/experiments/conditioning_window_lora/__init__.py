"""CFG conditioning-window ablation with the trained LoRA in the loop.

Mirror of ``conditioning_window`` but with the per-arm LoRA active during
sampling. Renders the same schedule grammar twice per mode:

  - ``with_prompt``: LoRA only fires on prompt-on steps (gated by mask).
  - ``always``:      LoRA fires every step, including off-steps where
                     it acts on the unconditional branch alone.

Outputs land under
``outputs/conditioning_window_lora/<pair_slug>/seed_<n>/<mode>/``. The
existing no-LoRA outputs at ``outputs/conditioning_window/...`` are the
side-by-side baseline.
"""

from poe_repair.experiments.conditioning_window_lora.main import main

__all__ = ["main"]
