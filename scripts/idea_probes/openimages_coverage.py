"""Does an annotated corpus cover the pool's concepts and pairs?

Stage 1: match the pool's concept vocabulary against Open Images' 601 boxable classes.
Stage 2: count, per cached pair, how many validation images carry boxes for both classes.

Prints matched concepts and per-pair counts, never only totals.
"""
from __future__ import annotations
import collections, csv, io, json, os, re, urllib.request

CACHE = "/datasets/mmolefe/poe_repair_min/outputs/training_cache/train"
OUT = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/openimages_coverage.json"
CLASSES = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
VAL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read().decode("utf-8", "replace")


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"^(a|an|the)\s+", "", s)
    return re.sub(r"[^a-z ]", "", s)


def singular(s: str) -> str:
    if s.endswith("ies"):
        return s[:-3] + "y"
    if s.endswith("es") and not s.endswith("ses"):
        return s[:-2]
    return s[:-1] if s.endswith("s") else s


def concept_of(slug_part: str) -> str:
    return norm(slug_part.replace("_", " "))


pairs = []
for p in sorted(os.listdir(CACHE)):
    if "__x__" in p:
        a, b = p.split("__x__")
        pairs.append((p, concept_of(a), concept_of(b)))
concepts = sorted({c for _, a, b in pairs for c in (a, b)})
print(f"{len(pairs)} pairs, {len(concepts)} concepts", flush=True)

rows = list(csv.reader(io.StringIO(fetch(CLASSES))))
oi = {}
for r in rows:
    if len(r) >= 2 and r[0] != "LabelName":
        oi[norm(r[1])] = r[0]
print(f"Open Images boxable classes: {len(oi)}", flush=True)

exact, variant, miss = {}, {}, []
for c in concepts:
    if c in oi:
        exact[c] = (c, oi[c])
        continue
    cs = singular(c)
    hit = next((k for k in oi if singular(k) == cs), None)
    if hit is None:
        hit = next((k for k in oi if k == c.split()[-1] or singular(k) == singular(c.split()[-1])), None)
        if hit is not None:
            variant[c] = (hit, oi[hit])
            continue
        miss.append(c)
    else:
        variant[c] = (hit, oi[hit])

print(f"\nEXACT {len(exact)}: {', '.join(sorted(exact))}", flush=True)
print(f"\nVARIANT (needs a human) {len(variant)}: "
      f"{', '.join(f'{k}->{v[0]}' for k, v in sorted(variant.items()))}", flush=True)
print(f"\nMISS {len(miss)}: {', '.join(miss[:60])}{' ...' if len(miss) > 60 else ''}", flush=True)

label = {c: v[1] for c, v in {**exact, **variant}.items()}

print("\nfetching validation boxes (25 MB)", flush=True)
per_image = collections.defaultdict(set)
rdr = csv.DictReader(io.StringIO(fetch(VAL)))
for r in rdr:
    per_image[r["ImageID"]].add(r["LabelName"])
print(f"validation images with boxes: {len(per_image)}", flush=True)

co = collections.Counter()
for labels in per_image.values():
    for slug, a, b in pairs:
        if a in label and b in label and label[a] in labels and label[b] in labels:
            co[slug] += 1

covered = [(s, a, b) for s, a, b in pairs if a in label and b in label]
print(f"\npairs with BOTH concepts matched to a class: {len(covered)} of {len(pairs)}", flush=True)
print(f"pairs with >=1 co-occurring validation image: {sum(1 for v in co.values() if v >= 1)}", flush=True)
print(f"pairs with >=10: {sum(1 for v in co.values() if v >= 10)}", flush=True)
print("\ntop co-occurring pairs:", flush=True)
for s, n in co.most_common(15):
    print(f"  {s}: {n}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"pairs": len(pairs), "concepts": len(concepts),
           "exact": sorted(exact), "variant": {k: v[0] for k, v in variant.items()},
           "miss": miss, "pairs_both_matched": len(covered),
           "val_images": len(per_image),
           "co_occurrence": dict(co.most_common())}, open(OUT, "w"), indent=2)
print(f"\nwritten {OUT}", flush=True)
