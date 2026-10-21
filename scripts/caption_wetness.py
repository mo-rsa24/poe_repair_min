#!/usr/bin/env python
"""How often does a captioned camel or dog in SDXL's stand-in data look wet?

The stand-in is Re-LAION-2B-en research-safe, mirrored on the cluster (artifacts/results/
does-a-camel-in-the-rain-come-out-wet/caption-counts.md, "Dataset searched"). Captions rarely say
"wet", so the caption only picks the animal and CLIP looks at the picture for the wetness:

    sample  whole-word caption matches over all 128 files, a fixed random draw per animal
    fetch   download each sampled image from its URL; dead links are recorded, not retried
    score   CLIP ViT-bigG-14 (trained on LAION-2B) asks two questions of each picture:
            is the animal there at all (against text, a logo, a product, another animal),
            and is it wet (five wet phrasings against five dry ones, averaged)

    python scripts/caption_wetness.py sample --n 100 --out /datasets/mmolefe/relaion2b-en/wetness/pilot
    python scripts/caption_wetness.py fetch  --out /datasets/mmolefe/relaion2b-en/wetness/pilot
    python scripts/caption_wetness.py score  --out /datasets/mmolefe/relaion2b-en/wetness/pilot
    python scripts/caption_wetness.py explain --out /datasets/mmolefe/relaion2b-en/wetness/pilot

`explain` adds the numbers the explainer scene draws for its 10 pictures (see its docstring).

The by-eye check of the wet calls happens afterwards on a contact sheet. The thresholds that check is
judged against sit below, fixed before any picture was scored.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/datasets/mmolefe/relaion2b-en")
DUCKDB = ROOT / "bin" / "duckdb"
PARQUET = ROOT / "parquet" / "*.parquet"
SEED = 20261021

ANIMALS = {                      # caption pattern (whole words, lower case), CLIP noun, other animal
    "camel": (r"\b(camel|camels|dromedary)\b", "camel", "horse"),
    "dog": (r"\b(dog|dogs|puppy)\b", "dog", "cat"),
}
NOT_THE_ANIMAL = ["a photo of text or a document", "a logo or a graphic design",
                  "a photo of a product or its packaging"]
WET = ["a photo of a wet {a}", "a photo of a {a} soaked by rain", "a photo of a {a} in the rain",
       "a photo of a {a} dripping with water", "a photo of a {a} swimming in water"]
DRY = ["a photo of a dry {a}", "a photo of a {a} on a sunny day", "a photo of a {a} on dry ground",
       "a photo of a {a} in dry weather", "a photo of a {a} indoors"]
CLIP_MODEL = "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

# Fixed before scoring. CLIP's wet count is used only if, for both animals, at least this share of
# the pictures CLIP called wet are wet by eye; otherwise only the eye-confirmed pictures count.
MIN_WET_PRECISION = 0.80
# The two animals differ only if their wet-rate 95% Wilson intervals do not overlap.
WILSON_Z = 1.96


def duckdb(sql: str, db: Path | str = ":memory:") -> list[dict]:
    res = subprocess.run([str(DUCKDB), str(db), "-json", "-c", sql], check=True,
                         capture_output=True, text=True).stdout.strip()
    return json.loads(res) if res else []


def sample(out: Path, n: int) -> None:
    """Count every whole-word match per animal, then keep the n matches with the lowest seeded hash."""
    out.mkdir(parents=True, exist_ok=True)
    # every caption naming either animal (~15M rows), on node-local disk: on /datasets the writes
    # stretched a ~10 minute scan to 47 (pilot job 72598); deleted at the end
    db = Path(os.environ.get("TMPDIR", "/tmp")) / f"caption_wetness_{os.getpid()}.duckdb"
    either = r"\b(camel|camels|dromedary|dog|dogs|puppy)\b"
    duckdb(f"CREATE OR REPLACE TABLE m AS SELECT url, caption, key, similarity, punsafe, width, "
           f"height, filename FROM read_parquet('{PARQUET}', filename=true) "
           f"WHERE regexp_matches(lower(caption), '{either}');", db)
    for name, (pattern, _, _) in ANIMALS.items():
        duckdb(f"COPY (SELECT '{name}' AS animal, * FROM m "
               f"WHERE regexp_matches(lower(caption), '{pattern}') "
               f"ORDER BY hash(url, caption, {SEED}) LIMIT {n}) TO '{out}/sample_{name}.parquet';", db)
    counts = ", ".join(f"count(*) FILTER (WHERE regexp_matches(lower(caption), '{p}')) AS {k}"
                       for k, (p, _, _) in ANIMALS.items())
    totals = duckdb(f"SELECT {counts} FROM m;", db)[0]
    totals["rows_searched"] = duckdb(f"SELECT count(*) AS n FROM read_parquet('{PARQUET}');")[0]["n"]
    totals["files_searched"] = duckdb(f"SELECT count(*) AS n FROM glob('{PARQUET}');")[0]["n"]
    db.unlink()
    totals.update(seed=SEED, n_per_animal=n, patterns={k: v[0] for k, v in ANIMALS.items()})
    (out / "sample_counts.json").write_text(json.dumps(totals, indent=2))
    print(json.dumps(totals, indent=2))


def _fetch_one(row: dict, img_dir: Path) -> dict:
    import requests
    from PIL import Image
    rec = {"animal": row["animal"], "key": row["key"], "url": row["url"], "caption": row["caption"]}
    path = img_dir / f"{row['animal']}_{row['rank']:05d}.jpg"
    try:
        r = requests.get(row["url"], timeout=8, headers={"User-Agent": "Mozilla/5.0 (research)"})
        if r.status_code != 200:
            return rec | {"status": f"http {r.status_code}"}
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        if min(im.size) < 64:
            return rec | {"status": "too small"}
        im.thumbnail((512, 512))
        im.save(path, quality=90)
        return rec | {"status": "ok", "path": path.name}
    except requests.Timeout:
        return rec | {"status": "timeout"}
    except Exception as e:                       # dead host, not an image, truncated bytes
        return rec | {"status": type(e).__name__}


def fetch(out: Path) -> None:
    import pyarrow.parquet as pq
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    rows = []
    for name in ANIMALS:
        for rank, row in enumerate(pq.read_table(out / f"sample_{name}.parquet").to_pylist()):
            rows.append(row | {"rank": rank})
    with ThreadPoolExecutor(32) as pool:
        recs = list(pool.map(lambda r: _fetch_one(r, img_dir), rows))
    for r, row in zip(recs, rows):
        r["rank"] = row["rank"]
    (out / "fetch.json").write_text(json.dumps(recs, indent=1))
    for name in ANIMALS:
        mine = [r for r in recs if r["animal"] == name]
        print(name, sum(r["status"] == "ok" for r in mine), "of", len(mine), "fetched")


def score(out: Path) -> None:
    import open_clip
    import torch
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, device=device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    scale = float(model.logit_scale.exp())

    @torch.no_grad()
    def embed_text(texts: list[str]) -> torch.Tensor:
        e = model.encode_text(tokenizer(texts).to(device)).float()
        return e / e.norm(dim=-1, keepdim=True)

    recs = json.loads((out / "fetch.json").read_text())
    prompts = {}
    for name, (_, noun, other) in ANIMALS.items():
        presence = [f"a photo of a {noun}"] + NOT_THE_ANIMAL + [f"a photo of a {other}"]
        wet = [t.format(a=noun) for t in WET]
        dry = [t.format(a=noun) for t in DRY]
        pe, we, de = embed_text(presence), embed_text(wet), embed_text(dry)
        wet_c, dry_c = we.mean(0), de.mean(0)
        prompts[name] = dict(presence=presence, wet=wet, dry=dry, pe=pe,
                             classes=torch.stack([wet_c / wet_c.norm(), dry_c / dry_c.norm()]),
                             we=we, de=de)
    for r in recs:
        if r["status"] != "ok":
            continue
        p = prompts[r["animal"]]
        with torch.no_grad():
            x = preprocess(Image.open(out / "images" / r["path"])).unsqueeze(0).to(device)
            img = model.encode_image(x).float()
            img = img / img.norm(dim=-1, keepdim=True)
        cos_presence = (img @ p["pe"].T)[0]
        p_presence = (scale * cos_presence).softmax(-1)
        cos_wet = (img @ p["classes"].T)[0]
        p_wet = (scale * cos_wet).softmax(-1)
        r.update(cos_presence=cos_presence.tolist(), p_presence=p_presence.tolist(),
                 present=bool(p_presence.argmax() == 0),
                 cos_wet_templates=(img @ p["we"].T)[0].tolist(),
                 cos_dry_templates=(img @ p["de"].T)[0].tolist(),
                 cos_wet_dry=cos_wet.tolist(), p_wet=float(p_wet[0]),
                 wet=bool(p_wet[0] > 0.5), image_embedding_head=img[0, :16].tolist())
    meta = {"model": CLIP_MODEL, "device": device, "logit_scale": scale,
            "prompts": {k: {"presence": v["presence"], "wet": v["wet"], "dry": v["dry"]}
                        for k, v in prompts.items()},
            "text_embedding_head": {k: {"presence": v["pe"][:, :16].tolist(),
                                        "wet_class": v["classes"][0, :16].tolist(),
                                        "dry_class": v["classes"][1, :16].tolist()}
                                    for k, v in prompts.items()},
            "min_wet_precision": MIN_WET_PRECISION}
    (out / "scores.json").write_text(json.dumps({"meta": meta, "images": recs}, indent=1))
    for name in ANIMALS:
        mine = [r for r in recs if r["animal"] == name and r["status"] == "ok"]
        present = [r for r in mine if r["present"]]
        print(f"{name}: {len(mine)} fetched, {len(present)} show the animal, "
              f"{sum(r['wet'] for r in present)} called wet")


def explain(out: Path, per_animal: int = 5, grid: int = 7) -> None:
    """Extra numbers for the explainer scene, on the first `per_animal` fetched pictures of each animal.

    caption_grid  cosine of every picture against every one of those pictures' own captions
    occlusion     P(wet) with each cell of a grid x grid split of the picture greyed out
    arrows        the picture and its five presence sentences projected onto their top-2 plane
    """
    import open_clip
    import torch
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, device=device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    scale = float(model.logit_scale.exp())
    scores = json.loads((out / "scores.json").read_text())
    walk = []
    for name in ANIMALS:
        ok = sorted((r for r in scores["images"] if r["animal"] == name and r["status"] == "ok"),
                    key=lambda r: r["rank"])
        walk += ok[:per_animal]

    @torch.no_grad()
    def unit(e: torch.Tensor) -> torch.Tensor:
        e = e.float()
        return e / e.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        pix = torch.stack([preprocess(Image.open(out / "images" / r["path"])) for r in walk]).to(device)
        img = unit(model.encode_image(pix))
        cap = unit(model.encode_text(tokenizer([r["caption"] for r in walk]).to(device)))
        caption_grid = (img @ cap.T).tolist()
        occlusion, arrows = [], []
        side = pix.shape[-1]
        edges = [round(i * side / grid) for i in range(grid + 1)]
        for i, r in enumerate(walk):
            _, noun, other = ANIMALS[r["animal"]]
            wet_c = unit(model.encode_text(tokenizer([t.format(a=noun) for t in WET]).to(device))).mean(0)
            dry_c = unit(model.encode_text(tokenizer([t.format(a=noun) for t in DRY]).to(device))).mean(0)
            classes = unit(torch.stack([wet_c, dry_c]))
            batch = []
            for gy in range(grid):
                for gx in range(grid):
                    x = pix[i].clone()
                    x[:, edges[gy]:edges[gy + 1], edges[gx]:edges[gx + 1]] = 0.0   # 0 = dataset mean grey
                    batch.append(x)
            occ = unit(model.encode_image(torch.stack(batch)))
            p_wet = (scale * occ @ classes.T).softmax(-1)[:, 0]
            occlusion.append([p_wet[g * grid:(g + 1) * grid].tolist() for g in range(grid)])
            presence = [f"a photo of a {noun}"] + NOT_THE_ANIMAL + [f"a photo of a {other}"]
            vecs = torch.cat([img[i:i + 1], unit(model.encode_text(tokenizer(presence).to(device)))])
            _, _, v = torch.linalg.svd(vecs, full_matrices=False)
            xy = vecs @ v[:2].T
            arrows.append({"labels": ["the picture"] + presence, "xy": xy.tolist(),
                           "cos_to_picture": (vecs @ img[i]).tolist(),
                           "kept_length": (xy.norm(dim=-1)).tolist()})
    walk_out = [{k: r[k] for k in ("animal", "rank", "key", "url", "caption", "path")} for r in walk]
    (out / "explain.json").write_text(json.dumps(
        {"model": CLIP_MODEL, "grid": grid, "walk": walk_out, "caption_grid": caption_grid,
         "occlusion": occlusion, "arrows": arrows}, indent=1))
    print("explained", len(walk), "pictures;", "diagonal mean",
          round(sum(caption_grid[i][i] for i in range(len(walk))) / len(walk), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["sample", "fetch", "score", "explain"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100, help="rows drawn per animal (sample only)")
    a = ap.parse_args()
    {"sample": lambda: sample(a.out, a.n), "fetch": lambda: fetch(a.out),
     "score": lambda: score(a.out), "explain": lambda: explain(a.out)}[a.stage]()
