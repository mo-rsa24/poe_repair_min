#!/usr/bin/env python
"""The "and" vector as a third expert, on Stable Diffusion 3.5 Medium instead of SDXL. Renders the tiles of one sheet.

SD 3.5's conditioning is three text encoders, not two: CLIP-L and CLIP-G (77 positions, concatenated and padded
to 4096 wide) followed by the T5-XXL sequence (256 positions, 4096 wide), and a 2048 wide pooled vector from the
two CLIPs. The arithmetic is done on all of it, in fp32, then cast to the model's dtype:

    x       = E(joint) - E(a) - E(b) + E("")       on the sequence and on the pooled vector
    x_clip  = the same on the 77 CLIP positions and the pooled vector, T5 positions of E("")
    x_t5    = the same on the 256 T5 positions, CLIP positions and pooled vector of E("")

The model predicts a velocity, not noise, and is sampled by Euler on a flow. Guidance is still linear, so a column
is the plain sum  v = v_empty + w * sum_i (v_i - v_empty)  over its conditions, as for SDXL: one condition is
ordinary classifier-free guidance, three are PoE with a third expert.

    python scripts/sd35_and_vector_render.py --out <dir> --a dog --b cat --joint "dog and cat" --seeds 1 2 3 4 5 6 7 8

Tiles land as <out>/and_vector/<a>__x__<b>/seedNN/<column>.png; an existing tile is skipped. Starting noise is
the seed's own torch normal draw, so it differs from the SDXL sheet's cached noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

MODEL = "stabilityai/stable-diffusion-3.5-medium"
N_CLIP = 77  # CLIP positions at the front of the prompt embeddings, the T5 positions follow

# column key, the conditions it sums
COLUMNS = [
    ("a", ["a"]),
    ("b", ["b"]),
    ("joint", ["joint"]),
    ("poe", ["a", "b"]),
    ("x", ["x"]),
    ("poe_x", ["a", "b", "x"]),
    ("poe_x_clip", ["a", "b", "x_clip"]),
    ("poe_x_t5", ["a", "b", "x_t5"]),
]


def slug(word: str) -> str:
    return word.replace(" ", "_")


def main() -> None:
    from diffusers import StableDiffusion3Pipeline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--joint", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL, torch_dtype=dtype).to(device)
    tr, vae, sched = pipe.transformer, pipe.vae, pipe.scheduler

    def enc(prompt: str):
        with torch.no_grad():
            e, _, p, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None, device=device,
                                            num_images_per_prompt=1, do_classifier_free_guidance=False)
        return e.float(), p.float()

    base = {"a": enc(args.a), "b": enc(args.b), "joint": enc(args.joint), "empty": enc("")}
    (eJ, pJ), (eA, pA), (eB, pB), (e0, p0) = base["joint"], base["a"], base["b"], base["empty"]
    ex, px = eJ - eA - eB + e0, pJ - pA - pB + p0
    cond = {k: base[k] for k in ("a", "b", "joint")}
    cond["x"] = (ex, px)
    cond["x_clip"] = (torch.cat([ex[:, :N_CLIP], e0[:, N_CLIP:]], dim=1), px)
    cond["x_t5"] = (torch.cat([e0[:, :N_CLIP], ex[:, N_CLIP:]], dim=1), p0)
    print("embedding shapes", {k: tuple(v[0].shape) + tuple(v[1].shape) for k, v in cond.items()}, flush=True)
    print("norms: clip part of x " f"{ex[:, :N_CLIP].norm():.1f}, t5 part of x {ex[:, N_CLIP:].norm():.1f}, "
          f"joint clip {eJ[:, :N_CLIP].norm():.1f}, joint t5 {eJ[:, N_CLIP:].norm():.1f}", flush=True)
    uncond = base["empty"]

    root = args.out / "and_vector" / f"{slug(args.a)}__x__{slug(args.b)}"
    lat_size = args.size // 8
    for seed in args.seeds:
        for key, names in COLUMNS:
            path = root / f"seed{seed:02d}" / f"{key}.png"
            if path.exists():
                continue
            embs = torch.cat([cond[n][0] for n in names] + [uncond[0]]).to(dtype)
            pools = torch.cat([cond[n][1] for n in names] + [uncond[1]]).to(dtype)
            n = len(names) + 1
            gen = torch.Generator(device=device).manual_seed(seed)
            lat = torch.randn((1, tr.config.in_channels, lat_size, lat_size), generator=gen, device=device, dtype=dtype)
            sched.set_timesteps(args.steps, device=device)
            with torch.no_grad():
                for t in sched.timesteps:
                    v = tr(hidden_states=lat.repeat(n, 1, 1, 1), timestep=t.expand(n), encoder_hidden_states=embs,
                           pooled_projections=pools, return_dict=False)[0]
                    v_u = v[-1:]
                    vg = v_u + args.guidance * sum(v[k:k + 1] - v_u for k in range(n - 1))
                    lat = sched.step(vg, t, lat, return_dict=False)[0]
                image = vae.decode((lat / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            pipe.image_processor.postprocess(image, output_type="pil")[0].save(path)
            path.with_suffix(".json").write_text(json.dumps(
                {"column": key, "conditions": names, "a": args.a, "b": args.b, "joint": args.joint, "seed": seed,
                 "steps": args.steps, "guidance": args.guidance, "model": MODEL, "size": args.size}, indent=1))
            print(f"wrote {path}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
