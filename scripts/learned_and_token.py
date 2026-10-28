#!/usr/bin/env python
"""One learned "<and>" token as a third expert beside "dog" and "cat" in SDXL's plain PoE.

Adds the token "<and>" to both SDXL tokenizers and gives it one trainable input-embedding row in each
text encoder, initialised from the word "and". Nothing else is trained. The rows live in fp32; a forward
hook on each encoder's token embedding writes them into the fp16 activations, so gradient reaches them
through the frozen encoders and the frozen UNet.

    python scripts/learned_and_token.py --check-token      "<and>" must encode exactly as "and" does
    python scripts/learned_and_token.py --baseline         held-out share removed with the untrained token
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

CACHE = Path(os.environ.get("POE_REPAIR_TRAINING_CACHE",
                            "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"))
POOL = REPO / "artifacts/_shared/cross_pair_pool_configs/cells_v58.json"
HELD_OUT = ("a_cat__x__a_dog", (9, 10, 11, 12, 13, 14, 15, 16))
EVAL_STEPS = (0, 10, 25, 40)
TOKEN = "<and>"
WORD = "and"
ENCODERS = (("tokenizer", "text_encoder"), ("tokenizer_2", "text_encoder_2"))


class AndToken:
    """The two trainable rows and the hooks that splice them into each encoder's token embeddings."""

    def __init__(self, models: dict) -> None:
        self.models = models
        self.ids: list[int] = []
        self.rows = torch.nn.ParameterList()
        for tok_key, enc_key in ENCODERS:
            tok, enc = models[tok_key], models[enc_key]
            if tok.add_tokens([TOKEN]) != 1:
                raise RuntimeError(f"{tok_key}: {TOKEN} was already in the vocabulary")
            new_id = tok.convert_tokens_to_ids(TOKEN)
            enc.resize_token_embeddings(len(tok))
            word_ids = tok(WORD, add_special_tokens=False).input_ids
            if len(word_ids) != 1:
                raise RuntimeError(f"{tok_key}: {WORD!r} is {len(word_ids)} tokens, expected 1")
            table = enc.get_input_embeddings()
            self.ids.append(new_id)
            self.rows.append(torch.nn.Parameter(table.weight[word_ids[0]].detach().float().clone()))
            table.register_forward_hook(self._hook(len(self.ids) - 1))
            for p in enc.parameters():
                p.requires_grad_(False)
        self.rows.to(models["unet"].device)

    def _hook(self, k: int):
        def splice(_module, inputs, out):
            mask = inputs[0] == self.ids[k]
            if not mask.any():
                return out
            out = out.clone()
            out[mask] = self.rows[k].to(out.dtype)
            return out
        return splice

    def encode(self, prompt: str, *, device, dtype, grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """The repo's SDXL prompt encoding, with the new token known and gradient optional."""
        m = self.models
        with torch.set_grad_enabled(grad):
            outs, all_ids = [], []
            for tok_key, enc_key in ENCODERS:
                tok = m[tok_key]
                ids = tok([prompt], padding="max_length", max_length=tok.model_max_length,
                          truncation=True, return_tensors="pt").input_ids.to(device)
                all_ids.append(ids)
                outs.append(m[enc_key](ids, output_hidden_states=True))
            embeds = torch.cat([outs[0].hidden_states[-2], outs[1].hidden_states[-2]], dim=-1).to(dtype)
            # HF pools at the highest token id, which is "<and>" (49408) rather than the end token
            # (49407), so pool at the first end-of-text position and project, as the library does.
            eos = (all_ids[1] == m["tokenizer_2"].eos_token_id).int().argmax(dim=-1)
            last = outs[1].last_hidden_state[torch.arange(len(eos), device=device), eos]
            pooled = m["text_encoder_2"].text_projection(last)
        return embeds, pooled.to(dtype)


def pool_cells() -> list[tuple[str, int]]:
    """The 182 (pair, seed) training cells, refusing cat x dog and any held-out seed."""
    cells = [(pair, seed) for pair, seeds in json.loads(POOL.read_text()).items() for seed in seeds]
    assert HELD_OUT[0] not in {p for p, _ in cells}, "cat x dog is in the pool"
    assert not {s for _, s in cells} & set(HELD_OUT[1]), "a held-out seed is in the pool"
    for pair, seed in cells:
        assert (CACHE / "train" / pair / f"seed_{seed}" / "residuals").is_dir(), f"{pair} seed {seed} not cached"
    return cells


def load_state(split: str, pair: str, seed: int, step: int, device) -> dict:
    """One cached state: x_t, its timestep and the four raw predictions, in fp32 on the device."""
    raw = torch.load(CACHE / split / pair / f"seed_{seed}" / "residuals" / f"step_{step:03d}.pt",
                     map_location="cpu", weights_only=True)
    return {k: (v.to(device) if k == "timestep" else v.to(device).float()) if torch.is_tensor(v) else v
            for k, v in raw.items()}


def and_target(st: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """What eps_<and> must equal, and the gap it has to close (guided joint minus PoE, over w)."""
    target = st["eps_j_raw"] + 2 * st["eps_uncond"] - st["eps_a_raw"] - st["eps_b_raw"]
    return target, target - st["eps_uncond"]


def eps_and(models: dict, tok: "AndToken", st: dict, device, dtype, grad: bool) -> torch.Tensor:
    from poe_repair.methods._sampling import add_time_ids

    seq, pool = tok.encode(TOKEN, device=device, dtype=dtype, grad=grad)
    cond = {"text_embeds": pool, "time_ids": add_time_ids(height=1024, width=1024, batch_size=1,
                                                          device=device, dtype=dtype)}
    t = torch.as_tensor(st["timestep"], device=device)
    with torch.set_grad_enabled(grad):
        return models["unet"](st["x_t"].to(dtype), t, encoder_hidden_states=seq, added_cond_kwargs=cond,
                              timestep_cond=None).sample.float()


def share_removed(models, tok, device, dtype) -> dict[int, list[float]]:
    """Per eval step, the share of the gap closed on each held-out cat x dog seed."""
    pair, seeds = HELD_OUT
    out: dict[int, list[float]] = {}
    for step in EVAL_STEPS:
        for seed in seeds:
            st = load_state("heldout", pair, seed, step, device)
            target, gap = and_target(st)
            e = eps_and(models, tok, st, device, dtype, grad=False)
            out.setdefault(step, []).append(1 - ((e - target).norm() / gap.norm()).item())
    return out


def baseline() -> None:
    from poe_repair.run import make_ctx

    cells = pool_cells()
    print(f"pool: {len(cells)} cells over {len({p for p, _ in cells})} pairs; no cat x dog, no seed 9 to 16",
          flush=True)
    ctx = make_ctx()
    tok = AndToken(ctx.models)
    res = share_removed(ctx.models, tok, ctx.device, ctx.dtype)
    for step, v in res.items():
        print(f"step {step:2d}: share removed per seed 9..16 = " + " ".join(f"{x:+.3f}" for x in v)
              + f"   mean {sum(v) / len(v):+.3f}", flush=True)


def check_token() -> None:
    from poe_repair.run import make_ctx

    ctx = make_ctx()
    tok = AndToken(ctx.models)
    kw = dict(device=ctx.device, dtype=ctx.dtype, grad=False)
    from poe_repair.runtime import encode_prompt_sdxl

    for p in ("and", "dog and cat"):
        er, pr = encode_prompt_sdxl(p, models=ctx.models, device=ctx.device, dtype=ctx.dtype)
        em, pm = tok.encode(p, **kw)
        print(f"repo encoder vs this one on {p!r}: embeds {(er - em).abs().max().item():.3g}, "
              f"pooled {(pr - pm).abs().max().item():.3g}", flush=True)
    for a, b in (("and", TOKEN), ("dog and cat", f"dog {TOKEN} cat")):
        ea, pa = tok.encode(a, **kw)
        eb, pb = tok.encode(b, **kw)
        print(f"{a!r} vs {b!r}: max |embeds| difference {(ea - eb).abs().max().item():.3g}, "
              f"max |pooled| difference {(pa - pb).abs().max().item():.3g}", flush=True)
    for (tok_key, _), new_id, row in zip(ENCODERS, tok.ids, tok.rows):
        print(f"{tok_key}: {TOKEN} is id {new_id}, row norm {row.norm().item():.4f}, "
              f"requires_grad {row.requires_grad}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-token", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()
    if args.check_token:
        check_token()
    if args.baseline:
        baseline()
