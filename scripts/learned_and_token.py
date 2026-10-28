#!/usr/bin/env python
"""One learned "<and>" token as a third expert beside "dog" and "cat" in SDXL's plain PoE.

Adds the token "<and>" to both SDXL tokenizers and gives it one trainable input-embedding row in each
text encoder, initialised from the word "and". Nothing else is trained. The rows live in fp32; a forward
hook on each encoder's token embedding writes them into the fp16 activations, so gradient reaches them
through the frozen encoders and the frozen UNet.

    python scripts/learned_and_token.py --check-token      "<and>" must encode exactly as "and" does
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(os.environ.get("POE_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

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
            outs = []
            for tok_key, enc_key in ENCODERS:
                tok = m[tok_key]
                ids = tok([prompt], padding="max_length", max_length=tok.model_max_length,
                          truncation=True, return_tensors="pt").input_ids.to(device)
                outs.append(m[enc_key](ids, output_hidden_states=True))
            embeds = torch.cat([outs[0].hidden_states[-2], outs[1].hidden_states[-2]], dim=-1).to(dtype)
            pooled = outs[1][0]
            if pooled.ndim != 2:
                pooled = outs[1].text_embeds
        return embeds, pooled.to(dtype)


def check_token() -> None:
    from poe_repair.run import make_ctx

    ctx = make_ctx()
    tok = AndToken(ctx.models)
    kw = dict(device=ctx.device, dtype=ctx.dtype, grad=False)
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
    args = ap.parse_args()
    if args.check_token:
        check_token()
