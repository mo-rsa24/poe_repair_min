"""Shape-trace the v4 pipeline's VAE (SDXL AutoencoderKL) on CPU.

Builds the model from the locally cached config (random weights: shapes do not
depend on weight values), hooks every named module, runs one encode+decode at
1x3x1024x1024 under no_grad, and writes a hierarchy tree with real in/out
shapes, layer args, and param counts.
"""
import json, sys
import torch
from diffusers import AutoencoderKL

CFG = "/home-mscluster/mmolefe/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/vae/config.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else "vae_trace.json"

with open(CFG) as f:
    cfg = json.load(f)

torch.manual_seed(0)
vae = AutoencoderKL.from_config(cfg)
vae.eval()

shapes = {}  # name -> {"in": [...], "out": [...]}

def shp(x):
    if torch.is_tensor(x):
        return list(x.shape)
    if isinstance(x, (tuple, list)):
        return [shp(v) for v in x if torch.is_tensor(v) or isinstance(v, (tuple, list))]
    return None

hooks = []
for name, mod in vae.named_modules():
    if name == "":
        continue
    def make(nm):
        def hook(m, inp, out):
            if nm not in shapes:
                shapes[nm] = {"in": shp(inp[0]) if inp else None, "out": shp(out)}
        return hook
    hooks.append(mod.register_forward_hook(make(name)))

x = torch.randn(1, 3, 1024, 1024)
with torch.no_grad():
    posterior = vae.encode(x).latent_dist
    z = posterior.sample(generator=torch.Generator().manual_seed(0))
    dec = vae.decode(z).sample

for h in hooks:
    h.remove()

def args_of(m):
    import torch.nn as nn
    if isinstance(m, nn.Conv2d):
        return {"kind": "conv2d", "k": list(m.kernel_size), "s": list(m.stride),
                "p": list(m.padding), "cin": m.in_channels, "cout": m.out_channels}
    if isinstance(m, nn.GroupNorm):
        return {"kind": "groupnorm", "groups": m.num_groups, "ch": m.num_channels}
    if isinstance(m, nn.Linear):
        return {"kind": "linear", "din": m.in_features, "dout": m.out_features}
    if isinstance(m, nn.SiLU):
        return {"kind": "silu"}
    if isinstance(m, nn.Dropout):
        return {"kind": "dropout", "p": m.p}
    cls = type(m).__name__
    d = {"kind": cls}
    if cls == "Attention":
        d.update({"heads": m.heads, "dim_head": getattr(m, "inner_dim", 0) // max(m.heads, 1)})
    return d

tree = {}
for name, mod in vae.named_modules():
    if name == "":
        continue
    n_param = sum(p.numel() for p in mod.parameters(recurse=True))
    tree[name] = {
        "cls": type(mod).__name__,
        "args": args_of(mod),
        "params": n_param,
        "shapes": shapes.get(name),
        "n_children": len(list(mod.children())),
    }

summary = {
    "model": "AutoencoderKL (SDXL base 1.0, subfolder vae)",
    "config": cfg,
    "input": [1, 3, 1024, 1024],
    "latent_sampled": list(z.shape),
    "decoded": list(dec.shape),
    "total_params": sum(p.numel() for p in vae.parameters()),
    "n_modules": len(tree),
    "provenance": "real (shape-traced, CPU, random-init from cached config, batch 1)",
}
with open(OUT, "w") as f:
    json.dump({"summary": summary, "modules": tree}, f, indent=1)
print(json.dumps(summary, indent=2))
print("traced modules:", len(tree), "with shapes:", sum(1 for v in tree.values() if v["shapes"]))
