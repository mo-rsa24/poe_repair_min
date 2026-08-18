"""Shape-trace the cross-attention path the LoRA adapter touches, on CPU.

Builds SDXL's UNet2DConditionModel from the locally cached config (random
weights: shapes do not depend on weight values), attaches the project's LoRA
with the exact ``LoRAConfig`` this repo trains (rank 8, alpha 8, targets
attn2.to_q / attn2.to_k / attn2.to_v), hooks every named module, and runs one
3-branch forward at the real inference geometry: latents [3,4,128,128] for a
1024x1024 image, encoder_hidden_states [3,77,2048], the (A, B, empty) branch
order the sampler uses.

On every cross-attention module it also recomputes the two tensors the scene is
about, using the same arithmetic as
``poe_repair/methods/_sampling.py:_CrossAttnRecorder``:

  where the word looks   softmax(Q K^T / sqrt(d))      [B, heads, HW, 77]
  what the word paints   attn[..., tok] * V[tok]       [B, heads, HW, head_dim]

Writes a hierarchy tree with real in/out shapes, layer args, param counts, the
per-layer cross-attention tensor shapes, and the LoRA delta path.

    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
        dl-scene/tracer/trace_cross.py dl-scene/cross_trace.json
"""
import json
import sys
import time
from pathlib import Path

import torch

SNAP = Path(
    "/home-mscluster/mmolefe/.cache/huggingface/hub/"
    "models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/"
    "462165984030d82259a11f4367a4eed129e94a7b"
)
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "cross_trace.json")

# The three prompts the 3-branch forward carries, in the sampler's cat order.
PROMPTS = ["a cat", "a dog", ""]
BRANCH_NAMES = ["A (a cat)", "B (a dog)", "empty (unconditional)"]

# From poe_repair/experiments/lora/config.py:LoRAConfig.
LORA_RANK = 8
LORA_ALPHA = 8
LORA_TARGETS = ["attn2.to_q", "attn2.to_k", "attn2.to_v"]

torch.set_num_threads(64)
torch.manual_seed(0)


def shp(x):
    if torch.is_tensor(x):
        return list(x.shape)
    if isinstance(x, (tuple, list)):
        out = [shp(v) for v in x]
        return [v for v in out if v is not None]
    if hasattr(x, "sample") and torch.is_tensor(getattr(x, "sample")):
        return list(x.sample.shape)
    return None


# ---------------------------------------------------------------- text side
def trace_text():
    """Real tokens and real text-encoder output shapes for the three prompts.

    Tokenizers carry real vocabulary, so the token IDs here are fact, not
    illustration. Text encoders are built from cached config with random
    weights: shapes are real, values are not used.
    """
    from transformers import (
        CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer,
    )

    tok1 = CLIPTokenizer.from_pretrained(SNAP / "tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(SNAP / "tokenizer_2")
    cfg1 = CLIPTextConfig.from_pretrained(SNAP / "text_encoder")
    cfg2 = CLIPTextConfig.from_pretrained(SNAP / "text_encoder_2")
    te1 = CLIPTextModel(cfg1).eval()
    te2 = CLIPTextModelWithProjection(cfg2).eval()

    lanes = []
    for prompt, bname in zip(PROMPTS, BRANCH_NAMES):
        ids1 = tok1(prompt, padding="max_length", max_length=tok1.model_max_length,
                    truncation=True, return_tensors="pt").input_ids
        ids2 = tok2(prompt, padding="max_length", max_length=tok2.model_max_length,
                    truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            o1 = te1(ids1, output_hidden_states=True)
            o2 = te2(ids2, output_hidden_states=True)
        h1 = o1.hidden_states[-2]                     # penultimate, SDXL convention
        h2 = o2.hidden_states[-2]
        seq = torch.cat([h1, h2], dim=-1)
        pooled = o2[0]
        n_real = int((ids1[0] != tok1.pad_token_id).sum()) if tok1.pad_token_id is not None else None
        lanes.append({
            "prompt": prompt,
            "branch": bname,
            "token_ids": ids1[0][:12].tolist(),
            "token_strings": [tok1.convert_ids_to_tokens(int(i)) for i in ids1[0][:12]],
            "n_nonpad_tokens": n_real,
            "encoder_1_out": list(h1.shape),
            "encoder_2_out": list(h2.shape),
            "seq": list(seq.shape),
            "pooled": list(pooled.shape),
        })
    del te1, te2
    return {
        "tokenizer": "CLIPTokenizer (real vocab, real ids)",
        "encoders": {
            "text_encoder": f"CLIPTextModel, hidden {cfg1.hidden_size}, "
                            f"{cfg1.num_hidden_layers} layers",
            "text_encoder_2": f"CLIPTextModelWithProjection, hidden {cfg2.hidden_size}, "
                              f"{cfg2.num_hidden_layers} layers",
        },
        "concat_rule": f"{cfg1.hidden_size} + {cfg2.hidden_size} = "
                       f"{cfg1.hidden_size + cfg2.hidden_size} "
                       f"(= unet cross_attention_dim)",
        "lanes": lanes,
    }


# ---------------------------------------------------------------- unet side
def build_unet():
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig

    with open(SNAP / "unet" / "config.json") as f:
        cfg = json.load(f)
    unet = UNet2DConditionModel.from_config(cfg)
    unet.eval()
    n_base = sum(p.numel() for p in unet.parameters())

    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
        target_modules=list(LORA_TARGETS), init_lora_weights=True,
    )
    unet.add_adapter(lora_cfg, adapter_name="lora")

    matched = []
    for name, module in unet.named_modules():
        subs = {n for n, _ in module.named_children()}
        if "lora_A" in subs and "lora_B" in subs:
            matched.append(name)
    n_lora = sum(p.numel() for n, p in unet.named_parameters() if "lora_" in n)
    return unet, cfg, n_base, n_lora, matched


def args_of(m):
    import torch.nn as nn
    cls = type(m).__name__
    if isinstance(m, nn.Conv2d):
        return {"kind": "conv2d", "k": list(m.kernel_size), "s": list(m.stride),
                "p": list(m.padding), "cin": m.in_channels, "cout": m.out_channels}
    if isinstance(m, nn.Linear):
        return {"kind": "linear", "din": m.in_features, "dout": m.out_features,
                "bias": m.bias is not None}
    if isinstance(m, nn.LayerNorm):
        return {"kind": "layernorm", "shape": list(m.normalized_shape), "eps": m.eps}
    if isinstance(m, nn.GroupNorm):
        return {"kind": "groupnorm", "groups": m.num_groups, "ch": m.num_channels}
    if isinstance(m, nn.Dropout):
        return {"kind": "dropout", "p": m.p}
    d = {"kind": cls}
    if cls == "Attention":
        heads = getattr(m, "heads", 1)
        inner = getattr(m, "inner_dim", None)
        if inner is None:
            inner = m.to_q.out_features if hasattr(m.to_q, "out_features") else 0
        d.update({
            "heads": heads,
            "inner_dim": int(inner),
            "head_dim": int(inner) // max(heads, 1),
            "is_cross_attention": bool(getattr(m, "is_cross_attention", False)),
            "cross_attention_dim": getattr(m, "cross_attention_dim", None),
            "scale": getattr(m, "scale", None),
        })
    if cls in {"Linear", "lora.Linear"}:
        d.update({"kind": "lora_linear"})
    return d


def main():
    t0 = time.time()
    text = trace_text()
    print(f"text lanes traced ({time.time() - t0:.0f}s)", flush=True)

    unet, cfg, n_base, n_lora, matched = build_unet()
    print(f"unet built: {n_base/1e9:.2f}B base params, {n_lora/1e6:.3f}M lora "
          f"params across {len(matched)} matched modules", flush=True)

    shapes = {}
    hooks = []
    for name, mod in unet.named_modules():
        if name == "":
            continue

        def make(nm):
            def hook(m, inp, out):
                if nm not in shapes:
                    shapes[nm] = {"in": shp(inp[0]) if inp else None, "out": shp(out)}
            return hook
        hooks.append(mod.register_forward_hook(make(name)))

    # The two tensors the scene is about, recomputed per cross-attn module with
    # the same arithmetic as _CrossAttnRecorder (_sampling.py:1521).
    cross = {}

    def cross_hook(nm, module):
        def hook(_m, args, kwargs, _out):
            if nm in cross:
                return
            hidden = args[0] if args else kwargs.get("hidden_states")
            enc = args[1] if len(args) >= 2 else kwargs.get("encoder_hidden_states")
            if hidden is None or enc is None:
                return
            q = module.to_q(hidden)
            k = module.to_k(enc)
            v = module.to_v(enc)
            heads = module.heads
            hd = q.shape[-1] // heads
            qh = q.view(q.shape[0], q.shape[1], heads, hd).transpose(1, 2)
            kh = k.view(k.shape[0], k.shape[1], heads, hd).transpose(1, 2)
            vh = v.view(v.shape[0], v.shape[1], heads, hd).transpose(1, 2)
            attn = torch.softmax((qh * (hd ** -0.5)) @ kh.transpose(-1, -2), dim=-1)
            ql = attn.shape[2]
            side = int(round(ql ** 0.5))
            # one token's painted content: attn[..., tok:tok+1] @ v[tok:tok+1]
            tok = 2
            painted = attn[..., tok:tok + 1] @ vh[:, :, tok:tok + 1, :]
            row_sums = attn[0, 0, 0].sum().item()
            cross[nm] = {
                "hidden_in": list(hidden.shape),
                "encoder_in": list(enc.shape),
                "q": list(q.shape), "k": list(k.shape), "v": list(v.shape),
                "q_heads": list(qh.shape), "k_heads": list(kh.shape),
                "v_heads": list(vh.shape),
                "attn": list(attn.shape),
                "painted_one_token": list(painted.shape),
                "heads": int(heads), "head_dim": int(hd),
                "scale": float(hd ** -0.5),
                "query_len": int(ql), "query_side": int(side),
                "spatial": [int(side), int(side)],
                "kept_by_recorder": bool(ql <= 32 * 32),
                "softmax_row_sum_check": round(row_sums, 6),
            }
        return hook

    for name, mod in unet.named_modules():
        if type(mod).__name__ == "Attention" and getattr(mod, "is_cross_attention", False):
            hooks.append(mod.register_forward_hook(cross_hook(name, mod), with_kwargs=True))

    B = len(PROMPTS)
    lat = torch.randn(B, 4, 128, 128)
    t = torch.tensor(981)
    ehs = torch.randn(B, 77, cfg["cross_attention_dim"])
    added = {
        "text_embeds": torch.randn(B, 1280),
        "time_ids": torch.tensor([[1024., 1024., 0., 0., 1024., 1024.]]).repeat(B, 1),
    }
    print("running forward: latents", list(lat.shape), "ehs", list(ehs.shape), flush=True)
    t1 = time.time()
    with torch.no_grad():
        out = unet(lat, t, encoder_hidden_states=ehs, added_cond_kwargs=added).sample
    print(f"forward done in {time.time() - t1:.0f}s -> {list(out.shape)}", flush=True)

    for h in hooks:
        h.remove()

    tree = {}
    for name, mod in unet.named_modules():
        if name == "":
            continue
        tree[name] = {
            "cls": type(mod).__name__,
            "args": args_of(mod),
            "params": sum(p.numel() for p in mod.parameters(recurse=True)),
            "shapes": shapes.get(name),
            "children": [n for n, _ in mod.named_children()],
            "is_lora_target": name in matched,
        }

    summary = {
        "model": "UNet2DConditionModel (SDXL base 1.0, subfolder unet) + peft LoRA",
        "lora": {
            "rank": LORA_RANK, "alpha": LORA_ALPHA,
            "target_modules": LORA_TARGETS,
            "matched_modules": len(matched),
            "trainable_params": n_lora,
            "base_params": n_base,
            "trainable_fraction": n_lora / n_base,
        },
        "input": {
            "latents": list(lat.shape), "timestep": int(t.item()),
            "encoder_hidden_states": list(ehs.shape),
            "text_embeds": list(added["text_embeds"].shape),
            "time_ids": list(added["time_ids"].shape),
            "branches": BRANCH_NAMES,
        },
        "output": list(out.shape),
        "n_modules": len(tree),
        "n_cross_attention": len(cross),
        "provenance": "real (shape-traced, CPU, random-init from cached config, "
                      "3-branch batch, 1024x1024 geometry)",
        "traced_on": "mscluster109, co3 env, torch 2.5.1, diffusers 0.29.2, peft 0.19.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "text": text, "modules": tree,
                   "cross_attention": cross}, f, indent=1)
    print(json.dumps(summary, indent=2))
    print("cross-attn layers:", len(cross),
          "with shapes:", sum(1 for v in tree.values() if v["shapes"]),
          "of", len(tree), "modules")


if __name__ == "__main__":
    main()
