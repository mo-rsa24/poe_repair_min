#!/usr/bin/env python
"""Hypothesis B: does the angle between two concept words, alone, predict how
easily their pair composes?

No image is read to build the x-axis. For each of the 8 pool pairs, "a {A}"
and "a {B}" are run through SDXL's own two text encoders (the same weights
that condition every render in this project), reduced to one vector per
phrase, and the cosine between them is the pair's "independence" number,
Bradley et al. 2502.04549's orthogonality condition, read off text alone.

Three things fixed after actually looking at the data, not before:

  anisotropy the first run of this script (raw seq_both vectors, only
             centred on the empty-string embedding) put every pair's cosine
             inside a 0.0005-wide band, 0.9992 to 0.9997, no usable spread:
             text encoders are known to cluster all their outputs in a
             narrow cone. The fix is to standardise (z-score) every one of
             the 2048 channels using a real corpus, not one background
             prompt. The corpus reused here is the ~75-pair pool's own
             cached solo-concept embeddings (2 per pair, "a"/"b" branches),
             already sitting on disk from the L1/L3 probes, no new encoding
             of a reference set needed.

  measure    plain PoE (lambda=0) compose rate is 0/4 for 7 of the 8 pairs,
             flat, nothing to correlate against. Each pair's own dose-curve
             AUC (how fast compose rate climbs from 0 to 1 as lambda rises)
             is used instead: real spread, and it is the same summary this
             project already uses for "how well correction works" elsewhere.

  view       of SDXL's two text encoders, only the concatenated token
             sequence ("seq_both", 2048-wide, what cross-attention actually
             reads) showed any signal in the L1 probe already run this
             project. The pooled, contrastively-trained CLIP vector showed
             nothing. seq_both is used here, mean-pooled over each phrase's
             own content tokens (end-of-text token onward is padding, masked
             out, same convention scripts/language_probes.py uses).

Two panels: (a) one real, composed image per pair, so a reader can see what
each dot is. (b) the actual test, cosine on x, that pair's dose-curve AUC on
y, 8 dots. Read: dots trending down-right (wider angle, higher AUC) support
the idea; no trend refutes it. n=8, so this is a first look, not a verdict,
the honest bar for that is stated in the printed output, not silently
upgraded to a p-value that n=8 cannot support.

    python scripts/text_orthogonality_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poe_repair.config import DEFAULT_MODEL_ID  # noqa: E402
from poe_repair.experiments.interaction_term.cache import CACHE_ROOT, cell_dir  # noqa: E402
from poe_repair.experiments.interaction_term.pool import load_pool  # noqa: E402
from scripts.snr_collapse import iter_cells  # noqa: E402

CLIP_L_WIDTH = 768   # where the CLIP-L half ends inside the 2048-wide seq_both
WHITEN_POOL = "outputs/animals_compose_transfer/pair_pool.yaml"

DOSE_CURVES = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose/dose_curves.json")
PAIRS_ROOT = Path("outputs/interaction_term/dose/pairs")
OUT_DIR = Path("/datasets/mmolefe/poe_repair_min/outputs/interaction_term/cache_analyses")
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)

POOL_PAIRS = (
    "a_leopard__x__a_jaguar", "a_frog__x__a_toad", "an_eagle__x__a_hawk",
    "a_seal__x__a_walrus", "a_goose__x__a_swan", "a_cow__x__a_buffalo",
    "a_cat__x__a_dog", "an_elephant__x__a_penguin",
)


def prompts_for(slug: str) -> tuple[str, str]:
    a, b = slug.split("__x__")
    return a.replace("_", " "), b.replace("_", " ")


def pretty(slug: str) -> str:
    a, b = prompts_for(slug)
    strip = lambda s: s[2:] if s.startswith("a ") else s[3:]
    return f"{strip(a)} x {strip(b)}"


def load_encoders(device: torch.device):
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    tok1 = CLIPTokenizer.from_pretrained(DEFAULT_MODEL_ID, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(DEFAULT_MODEL_ID, subfolder="tokenizer_2")
    enc1 = CLIPTextModel.from_pretrained(
        DEFAULT_MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float32,
    ).to(device).eval()
    enc2 = CLIPTextModelWithProjection.from_pretrained(
        DEFAULT_MODEL_ID, subfolder="text_encoder_2", torch_dtype=torch.float32,
    ).to(device).eval()
    return tok1, tok2, enc1, enc2


def whitening_stats(tok1) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std of seq_both, mean-pooled over content tokens,
    computed from the ~75-pair pool's own cached solo-concept embeddings (the
    "a" and "b" branches every L1/L3 cell already has on disk). Real corpus,
    not one background prompt, so the anisotropic shared direction actually
    cancels instead of barely moving."""
    pool = load_pool(WHITEN_POOL)
    pairs = pool.train + pool.heldout()
    cells = list(iter_cells(CACHE_ROOT, pairs, 1))
    vecs = []
    for pair, seed in cells:
        p = cell_dir(pair, seed, root=CACHE_ROOT) / "embeddings.pt"
        meta_p = cell_dir(pair, seed, root=CACHE_ROOT) / "meta.json"
        if not p.exists() or not meta_p.exists():
            continue
        emb = torch.load(p, map_location="cpu", weights_only=True)
        meta = json.loads(meta_p.read_text())
        a_text, b_text = meta["pair"]
        for branch, text in (("a", a_text), ("b", b_text)):
            n_tok = len(tok1(text).input_ids)
            seq = emb[f"seq_{branch}"].float()[0][:n_tok, :]   # (n_tok, 2048)
            vecs.append(seq.mean(dim=0).numpy())
    if len(vecs) < 20:
        raise SystemExit(f"only {len(vecs)} cached solo-concept embeddings found, "
                         "too few to whiten with; run language_probes.py's "
                         "cache-building step first")
    X = np.stack(vecs)
    print(f"whitening corpus: {len(vecs)} solo-concept embeddings from "
         f"{len(cells)} cached pairs")
    return X.mean(axis=0), X.std(axis=0) + 1e-6


@torch.no_grad()
def phrase_vector(text: str, tok1, tok2, enc1, enc2, device) -> torch.Tensor:
    """seq_both for one short phrase, mean-pooled over its own content tokens
    (through its own end-of-text token; padding excluded), matching
    language_probes.py's masking convention."""
    ti1 = tok1(text, padding="max_length", max_length=tok1.model_max_length,
               truncation=True, return_tensors="pt")
    ti2 = tok2(text, padding="max_length", max_length=tok2.model_max_length,
               truncation=True, return_tensors="pt")
    n_tok = int((ti1.input_ids[0] == tok1.eos_token_id).nonzero()[0].item()) + 1
    h1 = enc1(ti1.input_ids.to(device), output_hidden_states=True).hidden_states[-2]
    h2 = enc2(ti2.input_ids.to(device), output_hidden_states=True).hidden_states[-2]
    seq_both = torch.cat([h1, h2], dim=-1)[0, :n_tok, :]   # [n_tok, 2048]
    return seq_both.mean(dim=0).float().cpu()


def pair_auc(pair: str) -> float:
    d = json.loads(DOSE_CURVES.read_text())
    rate = []
    for lam in LAMS:
        cells = [s for s in d["scores"] if s["pair"] == pair and s["row"] == "oracle"
                and float(s["lambda"]) == lam]
        rate.append(sum(c["compose"] for c in cells) / len(cells))
    return float(np.trapezoid(rate, LAMS))


def composed_seed_image(pair: str) -> tuple[int, Path] | None:
    d = json.loads(DOSE_CURVES.read_text())
    hits = sorted(s["seed"] for s in d["scores"]
                 if s["pair"] == pair and s["row"] == "oracle"
                 and float(s["lambda"]) == 1.0 and s["compose"] == 1)
    if not hits:
        return None
    seed = hits[0]
    d2 = PAIRS_ROOT / pair / f"seed_{seed}" / "teacher_residual_const_lam100"
    png = sorted(d2.glob("*.png"))
    return (seed, png[0]) if png else None


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok1, tok2, enc1, enc2 = load_encoders(device)

    # Bradley et al.'s condition is stated on CENTRED means, (mu_i - mu_bg) vs
    # (mu_j - mu_bg), not on raw embeddings. The first version of this script
    # only centred on one background prompt (the empty string) and every
    # pair's cosine still landed inside a 0.0005-wide band: one prompt cannot
    # cancel a whole-corpus anisotropic cone. Standardising every channel with
    # real corpus statistics is the actual fix.
    mu, sigma = whitening_stats(tok1)
    mu_t, sigma_t = torch.from_numpy(mu).float(), torch.from_numpy(sigma).float()

    def whiten(v: torch.Tensor) -> torch.Tensor:
        return (v - mu_t) / sigma_t

    rows = []
    for pair in POOL_PAIRS:
        a, b = prompts_for(pair)
        va = whiten(phrase_vector(a, tok1, tok2, enc1, enc2, device))
        vb = whiten(phrase_vector(b, tok1, tok2, enc1, enc2, device))
        cos = float(torch.nn.functional.cosine_similarity(va, vb, dim=0))
        auc = pair_auc(pair)
        seed_hit = composed_seed_image(pair)
        rows.append({"pair": pair, "label": pretty(pair), "a": a, "b": b,
                     "cosine": cos, "auc": auc,
                     "seed": seed_hit[0] if seed_hit else None,
                     "image": str(seed_hit[1]) if seed_hit else None})
        print(f"{pretty(pair):22s} cos={cos:+.4f}  AUC={auc:.3f}  "
              f"seed={seed_hit[0] if seed_hit else 'none composed at lam=1'}")

    cos_arr = np.array([r["cosine"] for r in rows])
    auc_arr = np.array([r["auc"] for r in rows])
    order_c = cos_arr.argsort()
    order_a = auc_arr.argsort()
    ranks_c = np.empty_like(order_c); ranks_c[order_c] = np.arange(len(cos_arr))
    ranks_a = np.empty_like(order_a); ranks_a[order_a] = np.arange(len(auc_arr))
    spearman = float(np.corrcoef(ranks_c, ranks_a)[0, 1])
    print(f"\nn=8, Spearman rho={spearman:+.3f}")
    print("Honest bar: at n=8 this cannot be read as p<0.05 evidence either way; "
          "it is a first look, a rank check, not a verdict.")

    fig_data = {"rows": rows, "spearman_n8": spearman,
               "measure": "cosine(seq_both mean-pooled 'a {A}', seq_both mean-pooled "
                          "'a {B}'), text encoders only, no image read for the x-axis",
               "target": "trapezoidal AUC of the pair's own oracle dose curve, "
                        "lambda 0 to 1, from dose_curves.json"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "text_orthogonality_probe.json").write_text(json.dumps(fig_data, indent=2))

    # Two-panel figure: (a) one composed image per pair, (b) the actual scatter.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig = plt.figure(figsize=(min(1.15 * n, 9.5), 4.2))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 2.0], hspace=0.35, wspace=0.06,
                          left=0.07, right=0.98, top=0.90, bottom=0.11)
    rows_sorted = sorted(rows, key=lambda r: r["cosine"])
    for j, r in enumerate(rows_sorted):
        ax = fig.add_subplot(gs[0, j])
        ax.set_xticks([]); ax.set_yticks([])
        if r["image"]:
            ax.imshow(plt.imread(r["image"]))
        else:
            ax.text(0.5, 0.5, "none\ncomposed", ha="center", va="center", fontsize=6)
            ax.set_facecolor("0.9")
        ax.set_title(r["label"], fontsize=6.5, family="serif", pad=2)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#888888")

    ax_s = fig.add_subplot(gs[1, :])
    ax_s.scatter(cos_arr, auc_arr, s=40, color="#1f77b4", zorder=3)
    for r in rows:
        ax_s.annotate(r["label"], (r["cosine"], r["auc"]), fontsize=6.5,
                     family="serif", xytext=(4, 3), textcoords="offset points")
    ax_s.set_xlabel("cosine(word A, word B), text encoders alone, no image",
                    fontsize=8, family="serif")
    ax_s.set_ylabel("pair's dose-curve AUC\n(how well correction works)",
                    fontsize=8, family="serif")
    ax_s.set_title(f"n=8 pairs, Spearman rho={spearman:+.2f} (a first look, not a verdict)",
                   fontsize=8, family="serif", loc="left")
    ax_s.grid(alpha=0.25, linewidth=0.5)
    ax_s.tick_params(labelsize=7)

    out = OUT_DIR / "text_orthogonality_probe.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nfigure     {out}")
    print(f"sidecar    {OUT_DIR / 'text_orthogonality_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
