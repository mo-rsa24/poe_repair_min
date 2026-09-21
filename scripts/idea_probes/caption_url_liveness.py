"""Of the captions naming both animals, how many of their image URLs still resolve?

A caption match is a candidate; a usable training image needs the URL to still serve an
image. This walks the same DataComp-1B sample, collects URLs for a few named pairs, and
issues a HEAD request to a random sample of them to measure the live fraction.
"""
import json, re, glob, random, concurrent.futures as cf
import urllib.request
import pyarrow.parquet as pq

SAMPLE_DIR = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/caption_sample"
OUT = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/caption_url_liveness.json"
N_HEAD = 150          # urls probed per pair
TIMEOUT = 8

PAIRS = {
    "a_turtle__x__a_tortoise": (["turtle", "turtles"], ["tortoise", "tortoises"]),
    "a_wolf__x__a_husky": (["wolf", "wolves"], ["husky", "huskies"]),
    "a_cat__x__a_dog": (["cat", "cats", "kitten", "kittens"], ["dog", "dogs", "puppy", "puppies"]),
}
def build(ws): return re.compile(r"\b(?:" + "|".join(map(re.escape, ws)) + r")\b")
REG = {p: (build(a), build(b)) for p, (a, b) in PAIRS.items()}

def alive(url):
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0 (research crawl)"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            ct = r.headers.get("Content-Type", "")
            return r.status == 200 and ct.startswith("image")
    except Exception:
        return False

def main():
    urls = {p: [] for p in PAIRS}
    for s in sorted(glob.glob(f"{SAMPLE_DIR}/*.parquet")):
        tbl = pq.read_table(s, columns=["url", "text"])
        for u, t in zip(tbl.column("url").to_pylist(), tbl.column("text").to_pylist()):
            if not t or not u:
                continue
            low = t.lower()
            for p, (ra, rb) in REG.items():
                if ra.search(low) and rb.search(low):
                    urls[p].append(u)
    out = {"n_head_per_pair": N_HEAD, "pairs": {}}
    rng = random.Random(0)
    for p, us in urls.items():
        rng.shuffle(us)
        probe = us[:N_HEAD]
        with cf.ThreadPoolExecutor(32) as ex:
            res = list(ex.map(alive, probe))
        live = sum(res)
        out["pairs"][p] = {"candidates_in_sample": len(us), "probed": len(probe),
                           "live": live,
                           "live_fraction": round(live / len(probe), 3) if probe else None}
        print(p, out["pairs"][p], flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
