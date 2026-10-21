"""Pair table for the NIH ChestX-ray14 labels.

For each finding: images carrying it. For each pair of findings: images carrying
both, the count expected if the two were independent, and observed / expected.
Output: artifacts/results/does-roentgen-draw-two-findings/pair-table.csv
"""
import csv
import itertools
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS = ROOT / "data/nih_cxr14/Data_Entry_2017_v2020.csv"
OUT = ROOT / "artifacts/results/does-roentgen-draw-two-findings/pair-table.csv"

rows = [r["Finding Labels"].split("|") for r in csv.DictReader(open(LABELS))]
n_images = len(rows)
singles = Counter(f for labels in rows for f in labels if f != "No Finding")
pairs = Counter()
for labels in rows:
    for a, b in itertools.combinations(sorted(set(labels) - {"No Finding"}), 2):
        pairs[(a, b)] += 1

print(f"{n_images} images, {len(singles)} findings")
for f, n in singles.most_common():
    print(f"  {f:20s} {n}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["finding_a", "finding_b", "n_a", "n_b", "n_both", "expected_both", "observed_over_expected"])
    for a, b in itertools.combinations(sorted(singles), 2):
        expected = singles[a] * singles[b] / n_images
        both = pairs[(a, b)]
        w.writerow([a, b, singles[a], singles[b], both, f"{expected:.1f}", f"{both / expected:.3f}"])
print(f"wrote {OUT}")
