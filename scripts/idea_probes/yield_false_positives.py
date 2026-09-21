"""Does the two-species filter reject an image holding only one of the two animals?

The yield number is only trustworthy if the filter says no to a single-species photograph.
This takes captions naming exactly one of the pair's two species (and not the other),
downloads them, and runs the same filter. Every pass is a false positive.

Bar, written before the run: the filter is usable if the false-positive rate is below 0.10.
"""
import json, re, glob, io, os, random
import concurrent.futures as cf
import urllib.request
import pyarrow.parquet as pq
from PIL import Image
import torch

SAMPLE_DIR = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/caption_sample"
IMG_DIR = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/fp_images"
OUT = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/yield_false_positives.json"
FP_BAR, CONF, IOU_SEPARATE, N = 0.10, 0.30, 0.5, 120

# pair -> (species A, its words), (species B, its words); we take captions with A and NOT B
PAIRS = {
    "a_turtle__x__a_tortoise": (("turtle", ["turtle", "turtles"]), ("tortoise", ["tortoise", "tortoises"])),
    "a_cat__x__a_dog":         (("cat", ["cat", "cats", "kitten", "kittens"]), ("dog", ["dog", "dogs", "puppy", "puppies"])),
}
def build(ws): return re.compile(r"\b(?:" + "|".join(map(re.escape, ws)) + r")\b")

def fetch(args):
    url, path = args
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research crawl)"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = r.read(12_000_000)
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if min(im.size) < 224: return None
        im.thumbnail((768, 768)); im.save(path, "JPEG", quality=92)
        return path
    except Exception:
        return None

def main():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    os.makedirs(IMG_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev).eval()

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/ua if ua > 0 else 0.0

    def two_species(path, sa, sb):
        im = Image.open(path).convert("RGB")
        inp = proc(images=im, text=f"a {sa}. a {sb}.", return_tensors="pt").to(dev)
        with torch.no_grad(): out = model(**inp)
        res = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=CONF, text_threshold=CONF, target_sizes=[im.size[::-1]])[0]
        boxes, labels, scores = res["boxes"].tolist(), res["labels"], res["scores"].tolist()
        A = [(b, s) for b, l, s in zip(boxes, labels, scores) if sa in l.lower()]
        B = [(b, s) for b, l, s in zip(boxes, labels, scores) if sb in l.lower()]
        if not A or not B: return False
        A.sort(key=lambda t: -t[1]); B.sort(key=lambda t: -t[1])
        return min(iou(a[0], b[0]) for a in A[:3] for b in B[:3]) < IOU_SEPARATE

    regs = {p: ((a[0], build(a[1])), (b[0], build(b[1]))) for p, (a, b) in PAIRS.items()}
    singles = {p: [] for p in PAIRS}
    for s in sorted(glob.glob(f"{SAMPLE_DIR}/*.parquet")):
        tbl = pq.read_table(s, columns=["url", "text"])
        for u, t in zip(tbl.column("url").to_pylist(), tbl.column("text").to_pylist()):
            if not t or not u: continue
            low = t.lower()
            for p, ((sa, ra), (sb, rb)) in regs.items():
                if ra.search(low) and not rb.search(low):
                    singles[p].append(u)

    rng = random.Random(1); report = {"fp_bar": FP_BAR, "pairs": {}}
    for p, ((sa, _), (sb, _)) in regs.items():
        us = singles[p]; rng.shuffle(us); pick = us[:N]
        d = f"{IMG_DIR}/{p}"; os.makedirs(d, exist_ok=True)
        with cf.ThreadPoolExecutor(32) as ex:
            got = [x for x in ex.map(fetch, [(u, f"{d}/{i:04d}.jpg") for i, u in enumerate(pick)]) if x]
        fp = sum(two_species(x, sa, sb) for x in got)
        rate = fp/len(got) if got else None
        report["pairs"][p] = {"single_species_kept": sa, "downloaded": len(got),
                              "false_positives": fp, "false_positive_rate": round(rate, 3) if rate is not None else None,
                              "verdict": "usable" if rate is not None and rate < FP_BAR else "above bar"}
        print(p, report["pairs"][p], flush=True)
    json.dump(report, open(OUT, "w"), indent=1); print("wrote", OUT)

if __name__ == "__main__":
    main()
