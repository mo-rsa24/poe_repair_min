"""Answer key for abstract sentence 4: the baseline model against the fine-tuned one,
on the same 100 patients. Only the model's call for two diseased patients changes.
"""
from step1_patients import patients as finetuned  # clinic A: 16 caught, 4 missed, 8/72

# Same 100 patients, same expert grades. The baseline model misses two more of the
# diseased patients than the fine-tuned one; every healthy call stays identical.
baseline = (
    [(1, 1)] * 14   # has disease, baseline also catches these
    + [(1, 0)] * 6  # has disease, baseline misses these (2 of them the fine-tuned model catches)
    + [(0, 1)] * 8
    + [(0, 0)] * 72
)


def paired_change(before, after):
    """Count the patients whose model call differs, expert grade held fixed and paired by position."""
    return sum(1 for (e1, m1), (e2, m2) in zip(before, after) if e1 == e2 and m1 != m2)


if __name__ == "__main__":
    changed = paired_change(baseline, finetuned)
    print(changed)
