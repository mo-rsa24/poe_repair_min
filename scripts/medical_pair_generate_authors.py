"""plans/10-does-roentgen-draw-two-findings: draw RoentGen-v2 images the way its authors do.

Takes real prompts from one parquet shard of the authors' released synthetic dataset (their
`text` column, verbatim), for each single finding, and generates with the settings read from
their inference code (StanfordMIMI/RoentGen-v2 @ 039e34b): PNDM scheduler with skip_prk_steps,
75 steps, bf16, unconditional prompt "". Guidance defaults to 3, the value in their demo config.

The sample is one row per distinct impression, so the repeated impression x demographics rows
that the label-sorted shard holds do not fill the sample with one sentence.
"""
import argparse
import json
import pathlib

import pyarrow.parquet as pq
import torch

from medical_pair_generate import MODEL, contact_sheet, disk_guard

FINDINGS = ["Cardiomegaly", "Pneumothorax", "Pleural Effusion"]


def pick_prompts(shard, per_finding, seed):
    cols = ["text", "impression"] + FINDINGS
    t = pq.read_table(shard, columns=cols).to_pandas()
    picked = {}
    for f in FINDINGS:
        others = [x for x in FINDINGS if x != f]
        rows = t[(t[f] == 1) & (t[others] != 1).all(axis=1)]
        counts = {"rows": len(rows), "distinct_impressions": rows["impression"].nunique()}
        rows = rows.drop_duplicates("impression").sample(n=per_finding, random_state=seed)
        print(f, counts, flush=True)
        picked[f] = [{"row": int(i), "text": r["text"]} for i, r in rows.iterrows()]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--per-finding", type=int, default=16)
    ap.add_argument("--steps", type=int, default=75)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--sample-seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA: a CPU run would crawl without any error"
    out = pathlib.Path(args.out)
    disk_guard(out)
    picked = pick_prompts(args.shard, args.per_finding, args.sample_seed)

    from diffusers import DiffusionPipeline, PNDMScheduler

    pipe = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    shipped = type(pipe.scheduler).__name__
    pipe.scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000, set_alpha_to_one=False, skip_prk_steps=True,
        steps_offset=1, trained_betas=None)
    pipe.set_progress_bar_config(disable=True)
    manifest = {"model": MODEL, "steps": args.steps, "guidance": args.guidance, "dtype": "bf16",
                "scheduler_shipped_with_model": shipped, "scheduler_used": "PNDMScheduler",
                "shard": args.shard, "sample_seed": args.sample_seed, "images": []}
    for f, items in picked.items():
        d = out / f.replace(" ", "_").lower()
        d.mkdir(exist_ok=True)
        files = []
        for k, it in enumerate(items):
            p = d / f"idx_{k:02d}.png"
            if not p.exists():
                g = torch.Generator("cuda").manual_seed(k)
                img = pipe(it["text"], num_inference_steps=args.steps,
                           guidance_scale=args.guidance, generator=g).images[0]
                img.save(p)
            files.append(p)
            manifest["images"].append({"finding": f, "idx": k, "seed": k, "shard_row": it["row"],
                                       "text": it["text"], "file": str(p)})
        contact_sheet(files, out / f"sheet_{d.name}.png", cols=8)
        print(f, len(files), "images", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("done", len(manifest["images"]), "images; scheduler shipped with the model:", shipped, flush=True)


if __name__ == "__main__":
    main()
