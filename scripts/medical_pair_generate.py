"""Generate chest X-rays from RoentGen-v2 for the pair test in
plans/10-does-roentgen-draw-two-findings: five prompt types, fixed seeds.

Prompt form from the RoentGen-v2 paper (arXiv 2508.16783, Methods):
"<AGE> year old <RACE> <SEX>. <IMPRESSION>." Sampling from the same paper:
classifier-free guidance 4.0, 75 steps, default scheduler, 512 x 512.

Run on mscluster107 with the co3 python. Images go under --out (on /datasets).
Existing images are skipped, so a rerun resumes.
"""
import argparse
import json
import os
import pathlib
import shutil

import torch
from PIL import Image

DEMOGRAPHIC = "60 year old White male."
PROMPTS = {
    "cardiomegaly": f"{DEMOGRAPHIC} Cardiomegaly.",
    "pneumothorax": f"{DEMOGRAPHIC} Pneumothorax.",
    "effusion": f"{DEMOGRAPHIC} Pleural effusion.",
    "cardiomegaly_pneumothorax": f"{DEMOGRAPHIC} Cardiomegaly. Pneumothorax.",
    "cardiomegaly_effusion": f"{DEMOGRAPHIC} Cardiomegaly. Pleural effusion.",
}
MODEL = "stanfordmimi/RoentGen-v2"
MAX_DISK_USED = 0.90


def disk_guard(path):
    """Check the filesystem the images are written to, not an assumed one."""
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    if usage.used / usage.total >= MAX_DISK_USED:
        raise SystemExit(f"disk guard: {path} is {usage.used / usage.total:.0%} full")


def contact_sheet(files, out, cols=8, size=192):
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("L", (cols * size, rows * size))
    for i, f in enumerate(files):
        im = Image.open(f).convert("L").resize((size, size))
        sheet.paste(im, ((i % cols) * size, (i // cols) * size))
    sheet.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64, help="images per prompt, seeds 0..n-1")
    ap.add_argument("--only", nargs="*", default=list(PROMPTS), choices=list(PROMPTS))
    ap.add_argument("--steps", type=int, default=75)
    ap.add_argument("--guidance", type=float, default=4.0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA: a CPU run would crawl without any error"
    out = pathlib.Path(args.out)
    disk_guard(out)

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    manifest = {"model": MODEL, "steps": args.steps, "guidance": args.guidance, "prompts": {}}
    for key in args.only:
        d = out / key
        d.mkdir(exist_ok=True)
        files = []
        for seed in range(args.n):
            f = d / f"seed_{seed:03d}.png"
            if not f.exists():
                g = torch.Generator("cuda").manual_seed(seed)
                img = pipe(PROMPTS[key], num_inference_steps=args.steps,
                           guidance_scale=args.guidance, generator=g).images[0]
                img.save(f)
            files.append(f)
        contact_sheet(files, out / f"sheet_{key}.png")
        manifest["prompts"][key] = {"text": PROMPTS[key], "n": len(files)}
        print(key, len(files), "images", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
