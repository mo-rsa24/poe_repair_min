"""Derive the hierarchy spec for the cross-attention scene from cross_trace.json.

Prunes the 4010-module UNet trace to the path the adapter touches, assigns a
treatment template per node type, and prints the gate artifact. Nothing here is
hand-typed: every count and shape is read back out of the trace.

    /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
        dl-scene/tracer/spec_cross.py dl-scene/cross_trace.json
"""
import json
import sys
from collections import Counter, defaultdict

TRACE = sys.argv[1] if len(sys.argv) > 1 else "dl-scene/cross_trace.json"
d = json.load(open(TRACE))
mods, cross, summ = d["modules"], d["cross_attention"], d["summary"]

# ---- L1: the UNet's own top-level children, split by whether they cross-attend
top = sorted({k for k in mods if "." not in k})
carries = sorted({k.split(".attentions")[0].split(".")[0] +
                  ("" if k.startswith("mid") else "") for k in cross})
top_with_cross = sorted({k for k in top
                         if any(c.startswith(k + ".") or c == k for c in cross)})
# the named sub-blocks (down_blocks.1 etc) that actually hold cross-attention
sub_with_cross = sorted({".".join(k.split(".")[:2]) if k.startswith(("down", "up"))
                         else "mid_block" for k in cross})

transformer2d = sorted({k.split(".transformer_blocks")[0] for k in cross})
basic_blocks = sorted({k.rsplit(".attn2", 1)[0] for k in cross})
attn2 = sorted(cross)
projections = sorted(k for k in mods
                     if mods[k]["is_lora_target"])
to_out = sorted(k for k in mods if k.endswith(".attn2.to_out.0"))
lora_leaves = sorted(k for k in mods
                     if k.endswith((".lora_A.lora", ".lora_B.lora", ".base_layer"))
                     and ".attn2." in k)

# ---- shape families (repetition collapses honestly)
fam = defaultdict(list)
for k, v in cross.items():
    fam[(v["query_len"], v["heads"], v["head_dim"])].append(k)

print("=" * 78)
print("HIERARCHY SPEC: the cross-attention path the adapter touches")
print("=" * 78)
print(f"Model: {summ['model']}")
print(f"Trace: {summ['provenance']}")
print(f"       {summ['traced_on']}")
i = summ["input"]
print(f"Input: latents {i['latents']}, timestep {i['timestep']}, "
      f"encoder_hidden_states {i['encoder_hidden_states']}")
print(f"       batch rows are the 3 branches: {', '.join(i['branches'])}")
print(f"Output: {summ['output']}")
lo = summ["lora"]
print(f"LoRA:  rank {lo['rank']}, alpha {lo['alpha']}, targets {lo['target_modules']}")
print(f"       {lo['matched_modules']} matched modules, "
      f"{lo['trainable_params']:,} trainable of {lo['base_params']:,} base "
      f"({100 * lo['trainable_fraction']:.3f}%)")
print()
print(f"Pruned tree: 6 levels. Full UNet is {summ['n_modules']} modules; the "
      f"cross-attention path keeps the nodes below.")
print()

rows = [
    ("L0", "the 3-branch forward", 1,
     "three-lane overview: latent lane, conditioning lane, branch lane"),
    ("L1", "unet top level", len(top),
     f"UNet hourglass, {len(top_with_cross)} of {len(top)} children cross-attend"),
    ("L2", "Transformer2DModel", len(transformer2d),
     "transformer block stack, per resolution"),
    ("L3", "BasicTransformerBlock", len(basic_blocks),
     "block stack: norm1, attn1, norm2, attn2, norm3, ff, residual arcs"),
    ("L4", "attn2 (cross)", len(attn2),
     "one-head Q K V walk: the two named tensors live here"),
    ("L5", "projections + lora", len(projections) + len(to_out),
     "low-rank detour card (to_q/to_k/to_v), generic card (to_out.0)"),
]
tot = sum(r[2] for r in rows)
for lvl, name, n, tmpl in rows:
    print(f"  {lvl}  {name:24s} {n:4d} nodes   {tmpl}")
print(f"      {'total':26s} {tot:4d} nodes")
print()

print("Shape families at L4 (repetition collapsed, both traced):")
for (ql, h, hd), names in sorted(fam.items()):
    side = int(round(ql ** 0.5))
    c0 = cross[names[0]]
    print(f"  {side}x{side} grid, {len(names)} layers, {h} heads x {hd} head_dim")
    print(f"     where the word looks   softmax(QK^T/sqrt(d)) {c0['attn']}"
          f"   (row sum {c0['softmax_row_sum_check']})")
    print(f"     what the word paints   attn[.,tok] @ V[tok]  "
          f"{c0['painted_one_token']}")
    print(f"     kept by _CrossAttnRecorder (query_len <= 32^2): "
          f"{c0['kept_by_recorder']}")
    print(f"     e.g. {names[0]}")
print()

print("Untemplated node types (named, not silently thinned):")
print("  peft lora.Linear  the low-rank detour has no entry in the template")
print("                    table. Proposed: base_layer as the wide path, lora_A")
print("                    squeezing to rank 8, lora_B expanding back, the sum")
print("                    at the join, scale alpha/r shown and checked.")
print("  GEGLU (ff.net.0)  drawn as a chip in the L3 chain, not descended.")
print()

print("Equation-bearing nodes (get the drip cue, never taught inline):")
print("  attn2 softmax     softmax(QK^T/sqrt(d)) V")
print(f"  lora delta        W' = W + (alpha/r) B A, alpha/r = "
      f"{lo['alpha'] / lo['rank']:.1f}")
print("  poe composition   eps_PoE = eps_A + eps_B - eps_empty, at L0")
print("  lambda mixing     eps_t = eps_PoE_frozen + lambda * delta, at L0")
print()

print("Floor: every L5 leaf lands on source lines.")
print("  diffusers 0.29.2  Attention.forward, AttnProcessor2_0.__call__,")
print("                    BasicTransformerBlock.forward")
print("  peft 0.19.1       lora.Linear.forward (the base + scaling*B(A(x)) sum)")
print("  this repo         _sampling.py:_CrossAttnRecorder (the recompute),")
print("                    _sampling.py:_three_branch_forward_capture,")
print("                    lora/config.py:LoRAConfig (the target list),")
print("                    mechanism_study/value_probe.py (the two maps)")
print()

print("Text lane (the word side), real tokenizer ids:")
for lane in d["text"]["lanes"]:
    toks = [t for t in lane["token_strings"][:6]]
    print(f"  {lane['branch']:22s} ids {lane['token_ids'][:6]} -> {toks}")
    print(f"  {'':22s} seq {lane['seq']}  pooled {lane['pooled']}")
print(f"  concat rule: {d['text']['concat_rule']}")
