"""What fraction of caption-matched web images actually holds both animals as separate objects?

Caption co-occurrence gives candidates. This measures the yield after the filter that V5
would need: GroundingDINO queried per species (not the validated scorer's generic "animal"),
requiring one box of each species whose boxes do not overlap.

The bar is written here, before the run, per EXPERIMENT_CONVENTIONS.md:
  a pair is REACHABLE if the projected usable count over the full corpus is >= 500 images.
  Reason for 500: the additive-transfer result makes the requirement per concept rather than
  per pair, and a LoRA fitted by denoising score matching over noise levels is in the
  hundreds-of-images regime, not the tens (textual inversion) or millions (pretraining).
"""
import json, re, glob, io, random, sys
import concurrent.futures as cf
import urllib.request
import pyarrow.parquet as pq
from PIL import Image
import torch

SAMPLE_DIR = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/caption_sample"
IMG_DIR = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/yield_images"
OUT = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/cooccurrence_yield.json"
REACHABLE_BAR = 500          # projected usable images over the full corpus
N_DOWNLOAD = 400             # images attempted per pair
CONF = 0.30                  # matches the validated scorer's confidence floor
IOU_SEPARATE = 0.5           # matches the validated scorer's NMS threshold

PAIRS = {
    "a_cat__x__a_dog":        (("cat",      ["cat", "cats", "kitten", "kittens"]),
                               ("dog",      ["dog", "dogs", "puppy", "puppies"])),
    "a_turtle__x__a_tortoise":(("turtle",   ["turtle", "turtles"]),
                               ("tortoise", ["tortoise", "tortoises"])),
    "a_wolf__x__a_husky":     (("wolf",     ["wolf", "wolves"]),
                               ("husky",    ["husky", "huskies"])),
}
def build(ws): return re.compile(r"\b(?:" + "|".join(map(re.escape, ws)) + r")\b")

def collect():
    regs = {p: (build(a[1]), build(b[1])) for p, (a, b) in PAIRS.items()}
    urls = {p: [] for p in PAIRS}
    for s in sorted(glob.glob(f"{SAMPLE_DIR}/*.parquet")):
        tbl = pq.read_table(s, columns=["url", "text"])
        for u, t in zip(tbl.column("url").to_pylist(), tbl.column("text").to_pylist()):
            if not t or not u:
                continue
            low = t.lower()
            for p, (ra, rb) in regs.items():
                if ra.search(low) and rb.search(low):
                    urls[p].append((u, t))
    return urls

def fetch(args):
    url, path = args
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research crawl)"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = r.read(12_000_000)
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if min(im.size) < 224:
            return None
        im.thumbnail((768, 768))
        im.save(path, "JPEG", quality=92)
        return path
    except Exception:
        return None

def main():
    import os
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    os.makedirs(IMG_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev).eval()

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def two_species(path, sa, sb):
        """one box of each species, not overlapping. Returns (passes, detail)."""
        im = Image.open(path).convert("RGB")
        text = f"a {sa}. a {sb}."
        inp = proc(images=im, text=text, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=CONF, text_threshold=CONF,
            target_sizes=[im.size[::-1]])[0]
        boxes, labels, scores = res["boxes"].tolist(), res["labels"], res["scores"].tolist()
        A = [(b, s) for b, l, s in zip(boxes, labels, scores) if sa in l.lower()]
        B = [(b, s) for b, l, s in zip(boxes, labels, scores) if sb in l.lower()]
        if not A or not B:
            return False, {"n_a": len(A), "n_b": len(B), "reason": "missing a species"}
        A.sort(key=lambda t: -t[1]); B.sort(key=lambda t: -t[1])
        best = min(iou(a[0], b[0]) for a in A[:3] for b in B[:3])
        ok = best < IOU_SEPARATE
        return ok, {"n_a": len(A), "n_b": len(B), "min_iou": round(best, 3),
                    "reason": "ok" if ok else "boxes overlap (same animal labelled twice)"}

    cooc = json.load(open("/datasets/mmolefe/poe_repair_min/outputs/idea_probes/caption_cooccurrence.json"))
    urls = collect()
    rng = random.Random(0)
    report = {"bar_projected_usable": REACHABLE_BAR, "conf": CONF, "iou_separate": IOU_SEPARATE,
              "n_download_attempted": N_DOWNLOAD, "pairs": {}}

    for pair, ((sa, _), (sb, _)) in PAIRS.items():
        us = urls[pair]; rng.shuffle(us)
        pick = us[:N_DOWNLOAD]
        d = f"{IMG_DIR}/{pair}"; os.makedirs(d, exist_ok=True)
        jobs = [(u, f"{d}/{i:04d}.jpg") for i, (u, t) in enumerate(pick)]
        with cf.ThreadPoolExecutor(32) as ex:
            got = [p for p in ex.map(fetch, jobs) if p]
        usable, details = 0, []
        for p in got:
            ok, det = two_species(p, sa, sb)
            usable += ok
            details.append({"path": p, "ok": ok, **det})
        n_cand_sample = cooc["counts_in_sample"][pair]
        n_cand_full = cooc["projected_full_corpus"][pair]
        yield_rate = usable / len(got) if got else 0.0
        download_rate = len(got) / len(pick) if pick else 0.0
        projected = int(round(n_cand_full * download_rate * yield_rate))
        report["pairs"][pair] = {
            "candidates_in_sample": n_cand_sample,
            "candidates_projected_full_corpus": n_cand_full,
            "attempted": len(pick), "downloaded": len(got),
            "download_rate": round(download_rate, 3),
            "usable": usable, "yield_rate": round(yield_rate, 3),
            "projected_usable_full_corpus": projected,
            "verdict": "reachable" if projected >= REACHABLE_BAR else "below bar",
        }
        print(pair, report["pairs"][pair], flush=True)
        json.dump({"details": details}, open(f"{d}/detections.json", "w"), indent=1)

    json.dump(report, open(OUT, "w"), indent=1)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
