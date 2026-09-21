"""How many web captions name both animals of a pair that fails to compose?

Reads a sample of DataComp-1B parquet shards (url + text), counts captions whose text
contains both animal nouns of each pair under word-boundary matching, and scales the
count to the full 1.4B-row corpus. The counts are candidate images before any detector
filter; they are an upper bound on usable images, never a usable-image count.
"""
import json, re, sys, glob
import pyarrow.parquet as pq

SAMPLE_DIR = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/caption_sample"
OUT = "/datasets/mmolefe/poe_repair_min/outputs/idea_probes/caption_cooccurrence.json"
CORPUS_ROWS = 1_400_000_000
CORPUS_SHARDS = 2663

# pair -> (word forms for concept A, word forms for concept B)
PAIRS = {
    "a_wolf__x__a_husky": (["wolf", "wolves"], ["husky", "huskies"]),
    "a_lion__x__a_tiger": (["lion", "lions"], ["tiger", "tigers"]),
    "a_cheetah__x__a_cougar": (["cheetah", "cheetahs"], ["cougar", "cougars"]),
    "a_horse__x__a_zebra": (["horse", "horses"], ["zebra", "zebras"]),
    "a_rabbit__x__a_hare": (["rabbit", "rabbits"], ["hare", "hares"]),
    "a_dolphin__x__a_porpoise": (["dolphin", "dolphins"], ["porpoise", "porpoises"]),
    "a_crow__x__a_raven": (["crow", "crows"], ["raven", "ravens"]),
    "a_gorilla__x__a_chimpanzee": (["gorilla", "gorillas"], ["chimpanzee", "chimpanzees", "chimp", "chimps"]),
    "a_turtle__x__a_tortoise": (["turtle", "turtles"], ["tortoise", "tortoises"]),
    "a_leopard__x__a_jaguar": (["leopard", "leopards"], ["jaguar", "jaguars"]),
    "a_frog__x__a_toad": (["frog", "frogs"], ["toad", "toads"]),
    "an_eagle__x__a_hawk": (["eagle", "eagles"], ["hawk", "hawks"]),
    "a_seal__x__a_walrus": (["seal", "seals"], ["walrus", "walruses"]),
    "a_donkey__x__a_pony": (["donkey", "donkeys"], ["pony", "ponies"]),
    "a_crocodile__x__an_alligator": (["crocodile", "crocodiles", "croc"], ["alligator", "alligators", "gator"]),
    "a_cat__x__a_dog": (["cat", "cats", "kitten", "kittens"], ["dog", "dogs", "puppy", "puppies"]),
    "an_elephant__x__a_penguin": (["elephant", "elephants"], ["penguin", "penguins"]),
    "a_giraffe__x__a_zebra": (["giraffe", "giraffes"], ["zebra", "zebras"]),
}

def build(words):
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")

REG = {p: (build(a), build(b)) for p, (a, b) in PAIRS.items()}

def main():
    shards = sorted(glob.glob(f"{SAMPLE_DIR}/*.parquet"))
    print(f"{len(shards)} shards", flush=True)
    counts = {p: 0 for p in PAIRS}
    examples = {p: [] for p in PAIRS}
    rows = 0
    for i, s in enumerate(shards):
        try:
            tbl = pq.read_table(s, columns=["text"])
        except Exception as e:
            print(f"skip {s}: {e}", flush=True)
            continue
        texts = tbl.column("text").to_pylist()
        rows += len(texts)
        for t in texts:
            if not t:
                continue
            low = t.lower()
            for p, (ra, rb) in REG.items():
                if ra.search(low) and rb.search(low):
                    counts[p] += 1
                    if len(examples[p]) < 8:
                        examples[p].append(t[:160])
        print(f"  shard {i+1}/{len(shards)} rows so far {rows:,}", flush=True)
    scale = CORPUS_ROWS / rows if rows else 0
    out = {
        "corpus": "DataComp-1B (mlfoundations/datacomp_1b)",
        "shards_read": len(shards),
        "shards_total": CORPUS_SHARDS,
        "rows_read": rows,
        "corpus_rows_assumed": CORPUS_ROWS,
        "scale_factor": scale,
        "counts_in_sample": counts,
        "projected_full_corpus": {p: int(round(c * scale)) for p, c in counts.items()},
        "examples": examples,
        "note": "caption co-occurrence only; no detector filter applied, so these are candidates not usable images",
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["projected_full_corpus"], indent=1))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
