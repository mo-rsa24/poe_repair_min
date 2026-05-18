"""Contact-sheet rendering for the CFG conditioning-window ablation.

Static fallback for papers / share-by-link. The interactive inspector
slider is the primary readout (see ``scripts/lora_inspector.py`` route
``/conditioning_window``).

One PNG: rows = schedule family (prefix → suffix → window → punctate →
sanity), columns = schedules within family in suite order. Each cell is
the decoded image thumbnail with the schedule id and ``num_on / N``
written beneath it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from poe_repair.experiments.conditioning_window.schedules import FAMILY_ORDER
from poe_repair.figures._common import save_fig
from poe_repair.runtime import ensure_dir


def _group_by_family(schedules: list[dict]) -> dict[str, list[dict]]:
    by_family: dict[str, list[dict]] = {fam: [] for fam in FAMILY_ORDER}
    for s in schedules:
        fam = s.get("family", "punctate")
        by_family.setdefault(fam, []).append(s)
    return {fam: by_family[fam] for fam in FAMILY_ORDER if by_family.get(fam)}


def render_contact_sheet(manifest_path: Path, out_path: Path) -> Path:
    """Read the manifest, lay out one thumbnail per schedule, save to ``out_path``."""
    manifest = json.loads(Path(manifest_path).read_text())
    by_family = _group_by_family(manifest["schedules"])
    repo_root = Path(__file__).resolve().parents[3]

    n_rows = len(by_family)
    n_cols = max(len(rows) for rows in by_family.values()) if by_family else 1
    thumb = 1.6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(thumb * n_cols + 0.6, thumb * n_rows + 0.8),
        squeeze=False,
    )

    for r, (family, rows) in enumerate(by_family.items()):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if c >= len(rows):
                ax.set_visible(False)
                continue
            rec = rows[c]
            img_path = repo_root / rec["image_path"]
            try:
                ax.imshow(Image.open(img_path))
            except FileNotFoundError:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            n_total = manifest["num_inference_steps"]
            ax.set_title(
                f"{rec['id']}\n{rec['num_on']}/{n_total} on",
                fontsize=7,
            )
            if c == 0:
                ax.set_ylabel(family, fontsize=9, rotation=0,
                              labelpad=28, ha="right", va="center")

    fig.suptitle(
        f"conditioning_window — {manifest['prompt']!r} · seed {manifest['seed']}",
        fontsize=10,
    )
    ensure_dir(Path(out_path).parent)
    return save_fig(fig, Path(out_path))
