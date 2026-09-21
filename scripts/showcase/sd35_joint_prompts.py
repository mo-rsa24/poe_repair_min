"""Render the joint prompt on Stable Diffusion 3.5 medium, for pairs SDXL is being judged on.

A generalization check in the sense of ~/.claude/EXPERIMENT_CONVENTIONS.md: the model instances are
named before any of them runs, so a failure bounds the claim rather than being quietly dropped.
This renders the joint prompt only. It builds no trajectory cache and trains nothing, because SD 3.5
is a rectified-flow model predicting velocity, so the product-of-experts combination and the
correction term would both need re-deriving before a single cell could be cached here.
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disallowed_subjects import check as check_subject

MODEL = "stabilityai/stable-diffusion-3.5-medium"
CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"


def joint_prompt(pair):
    for sp in ("train", "heldout"):
        m = os.path.join(CACHE, sp, pair, "seed_1", "meta.json")
        if os.path.exists(m):
            return json.load(open(m))["joint_prompt"]
    a, _, b = pair.partition("__x__")
    return "%s and %s" % (a.replace("_", " "), b.replace("_", " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.0)
    args = ap.parse_args()

    for p in args.pairs:
        check_subject(p)

    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    os.makedirs(args.out, exist_ok=True)
    manifest = {"model": MODEL, "steps": args.steps, "guidance": args.guidance, "cells": []}
    for pair in args.pairs:
        prompt = joint_prompt(pair)
        d = os.path.join(args.out, pair)
        os.makedirs(d, exist_ok=True)
        for s in args.seeds:
            f = os.path.join(d, "seed_%d.png" % s)
            if os.path.exists(f):
                print("[cached]", f, flush=True)
                continue
            g = torch.Generator(device="cuda").manual_seed(s)
            img = pipe(prompt=prompt, num_inference_steps=args.steps,
                       guidance_scale=args.guidance, generator=g,
                       height=1024, width=1024).images[0]
            img.save(f)
            manifest["cells"].append({"pair": pair, "seed": s, "prompt": prompt, "path": f})
            print("[built]", f, flush=True)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print("done:", len(manifest["cells"]), "rendered")


if __name__ == "__main__":
    main()
