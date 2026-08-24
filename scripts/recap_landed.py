#!/usr/bin/env python
"""Recap gallery builder — LoRA-Fixes-PoE landed results.

Scope: the ROOT program. Honestly framed: the rungs are NOT complete (sync
marked all 5 ⚠️); this gallery recalls the two genuinely-landed sub-results —
the Overfit beachhead (cat×dog, G04) and the G6 Survive-Noise pool (G07).

Qual-first: real generated samples beside each metric. The beachhead strip is
composed from existing probes + cache refs (no GPU). The G6 held-out-seed row is
swapped in once the Slurm job (results/recap_landed/gen/) lands; until then a
pending placeholder renders.

Theme-aware UI kit: tokens on :root (light default), redefined under
prefers-color-scheme + data-theme, so both themes are legible and consistent.

Writes recap/index.html (self-contained, base64), recap/figs/*.png, RECAP.md.
"""
from __future__ import annotations
import base64, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from poe_repair import paths

REPO = Path(__file__).resolve().parents[1]
RECAP = REPO / "recap"
FIGS = RECAP / "figs"
DAT = paths.resolve(paths.TRAINING_CACHE) / "heldout/a_cat__x__a_dog"
BEACH = paths.resolve(paths.ONE_PAIR_ONE_SEED) / "a_cat__x__a_dog/seed_42/run__local/probes/epoch_1600"
G6GEN = REPO / "results/recap_landed/gen/g6_survive_noise"

# figure-plate palette (baked into the raster; a neutral dark plate reads as a
# render canvas on both HTML themes). Semantic role colors match the CSS tokens.
PLATE_BG, PLATE_TXT, PLATE_MUT = "#12161d", "#e8ecf4", "#98a2b6"
POE, LORA, MONO, WARN = "#e0574f", "#5b8def", "#d19a2e", "#d98324"
PANEL_PX, GAP, FRAME = 360, 22, 7


def _font(sz, bold=False):
    cand = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try: return ImageFont.truetype(cand, sz)
    except Exception: return ImageFont.load_default()


def _panel(src: Path, label: str, color: str, badge: str | None = None) -> Image.Image:
    lab_h = 44
    canvas = Image.new("RGB", (PANEL_PX, PANEL_PX + lab_h), PLATE_BG)
    if src and src.exists():
        im = Image.open(src).convert("RGB").resize((PANEL_PX - 2 * FRAME, PANEL_PX - 2 * FRAME))
    else:
        im = Image.new("RGB", (PANEL_PX - 2 * FRAME, PANEL_PX - 2 * FRAME), "#1c2230")
        d = ImageDraw.Draw(im); d.text((20, PANEL_PX // 2 - 10), "(pending)", fill=PLATE_MUT, font=_font(18))
    framed = Image.new("RGB", (PANEL_PX, PANEL_PX), color)
    framed.paste(im, (FRAME, FRAME))
    canvas.paste(framed, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((3, PANEL_PX + 12), label, fill=PLATE_TXT, font=_font(19, bold=True))
    if badge:
        bf = _font(14, bold=True); tw = d.textlength(badge, font=bf)
        d.rounded_rectangle([PANEL_PX - tw - 20, 11, PANEL_PX - 7, 37], radius=5, fill="#000000cc")
        d.text((PANEL_PX - tw - 13, 14), badge, fill=color, font=bf)
    return canvas


def _strip(panels, title, out: Path):
    n = len(panels); title_h = 60
    W = n * PANEL_PX + (n - 1) * GAP + 40
    H = title_h + panels[0].height + 18
    canvas = Image.new("RGB", (W, H), PLATE_BG)
    d = ImageDraw.Draw(canvas)
    d.text((20, 16), title, fill=PLATE_TXT, font=_font(27, bold=True))
    x = 20
    for p in panels:
        canvas.paste(p, (x, title_h)); x += PANEL_PX + GAP
    out.parent.mkdir(parents=True, exist_ok=True); canvas.save(out)
    return out


def _b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def build_beachhead() -> Path:
    panels = [
        _panel(DAT / "seed_42/poe.png", "PoE — broken", POE, "1 fused animal"),
        _panel(BEACH / "lambda_0.50/decoded.png", "PoE + 0.5·Δ", WARN),
        _panel(BEACH / "lambda_1.00/decoded.png", "PoE + 1.0·Δ  (LoRA)", LORA, "2 animals"),
        _panel(DAT / "seed_42/mono.png", "Mono — target", MONO, "ceiling"),
    ]
    return _strip(panels,
                  "The LoRA walks PoE's chimera into two separate animals — with no joint prompt",
                  FIGS / "beachhead_strip.png")


def build_g6():
    l1 = G6GEN / "lambda1"
    if not l1.exists() or not any(l1.glob("sample_seed_*.png")):
        return None, False
    panels = []
    for s in (9, 10, 11, 12):
        panels.append(_panel(DAT / f"seed_{s}/poe.png", f"seed {s} — PoE", POE))
        panels.append(_panel(l1 / f"sample_seed_{s:02d}.png", f"seed {s} — pooled LoRA", LORA))
    out = _strip(panels,
                 "One LoRA, pooled over 4 seeds, still composes on 4 seeds it never trained on",
                 FIGS / "g6_heldout_seeds.png")
    return out, True


def fig_block(fid, img_path, caption):
    return (f'<figure class="plate" tabindex="0" role="button" onclick="openModal(\'{fid}\')" '
            f'onkeydown="if(event.key===\'Enter\')openModal(\'{fid}\')">'
            f'<img src="data:image/png;base64,{_b64(img_path)}" alt="{caption}"/>'
            f'<span class="peek">the method →</span>'
            f'<figcaption>{caption}</figcaption></figure>')


def cards(what, why, result, limits):
    items = [("what we did", what), ("why", why), ("what we found", result), ("limits", limits)]
    return '<div class="cards">' + "".join(
        f'<div class="card"><h4>{h}</h4><p>{t}</p></div>' for h, t in items) + '</div>'


CSS = r"""
:root{
  --bg:#f4f6f9; --surface:#ffffff; --surface-2:#eef1f7; --plate:#12161d;
  --text:#1a2233; --muted:#586074; --border:#d7dce5;
  --accent:#2f62d8; --accent-weak:#e9f0fd;
  --poe:#cf4b43; --lora:#2f62d8; --mono:#b9821f; --pass:#1f9463; --warn:#c9761c;
  --shadow:0 1px 2px rgba(20,30,50,.06),0 8px 24px rgba(20,30,50,.06);
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0d1017; --surface:#161b24; --surface-2:#0b0e14; --plate:#12161d;
    --text:#e8ecf4; --muted:#98a2b6; --border:#262d3a;
    --accent:#6f9bff; --accent-weak:#17233f;
    --poe:#f0736b; --lora:#6f9bff; --mono:#edb454; --pass:#4bc088; --warn:#e39a4a;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="light"]{
  --bg:#f4f6f9; --surface:#ffffff; --surface-2:#eef1f7;
  --text:#1a2233; --muted:#586074; --border:#d7dce5;
  --accent:#2f62d8; --accent-weak:#e9f0fd;
  --poe:#cf4b43; --lora:#2f62d8; --mono:#b9821f; --pass:#1f9463; --warn:#c9761c;
  --shadow:0 1px 2px rgba(20,30,50,.06),0 8px 24px rgba(20,30,50,.06);
}
:root[data-theme="dark"]{
  --bg:#0d1017; --surface:#161b24; --surface-2:#0b0e14;
  --text:#e8ecf4; --muted:#98a2b6; --border:#262d3a;
  --accent:#6f9bff; --accent-weak:#17233f;
  --poe:#f0736b; --lora:#6f9bff; --mono:#edb454; --pass:#4bc088; --warn:#e39a4a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
#wrap{
  background:var(--bg); color:var(--text); max-width:1080px; margin:0 auto;
  padding:40px 28px 72px;
  font:15px/1.62 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased;
}
.eyebrow{
  font:600 12px/1 ui-monospace,"SF Mono","JetBrains Mono",monospace;
  letter-spacing:.14em; text-transform:uppercase; color:var(--accent);
}
h1{font-size:29px; font-weight:680; letter-spacing:-.015em; margin:10px 0 6px; text-wrap:balance}
.mission{color:var(--muted); font-size:16px; margin:0 0 22px; max-width:62ch; font-style:italic}
.banner{
  background:var(--accent-weak); border:1px solid var(--border);
  border-radius:10px; padding:13px 17px; font-size:13.5px; line-height:1.55;
  color:var(--text); margin-bottom:18px;
}
.banner b{color:var(--accent)}
.status{display:flex; flex-wrap:wrap; gap:9px; margin-bottom:40px}
.chip{
  display:inline-flex; align-items:center; gap:7px;
  font:600 12.5px/1 ui-monospace,"SF Mono",monospace; letter-spacing:.01em;
  padding:6px 12px; border-radius:8px; background:var(--surface);
  border:1px solid var(--border); color:var(--muted);
}
.chip::before{content:""; width:7px; height:7px; border-radius:50%; background:currentColor}
.chip.pass{color:var(--pass); border-color:color-mix(in srgb,var(--pass) 40%,var(--border))}
.chip.warn{color:var(--warn); border-color:color-mix(in srgb,var(--warn) 40%,var(--border))}
section{margin-bottom:52px}
.sec-head{display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:16px}
.rung{
  font:700 11.5px/1 ui-monospace,"SF Mono",monospace; letter-spacing:.1em;
  text-transform:uppercase; color:var(--accent);
  background:var(--accent-weak); padding:5px 10px; border-radius:6px;
}
h2{font-size:21px; font-weight:660; letter-spacing:-.01em; margin:0; text-wrap:balance}
.plate{
  margin:0 0 18px; cursor:pointer; position:relative; border-radius:12px;
  overflow:hidden; border:1px solid var(--border); background:var(--plate);
  box-shadow:var(--shadow); transition:transform .22s ease, box-shadow .22s ease;
}
.plate:hover{transform:translateY(-3px); box-shadow:0 6px 14px rgba(20,30,50,.12),0 18px 44px rgba(20,30,50,.16)}
.plate:focus-visible{outline:2px solid var(--accent); outline-offset:3px}
.plate img{width:100%; display:block}
figcaption{
  color:var(--muted); font-size:13px; line-height:1.5; padding:12px 16px;
  background:var(--surface); border-top:1px solid var(--border);
}
.peek{
  position:absolute; top:14px; right:14px; z-index:2;
  background:color-mix(in srgb,var(--accent) 92%, black); color:#fff;
  font:600 12px/1 ui-monospace,monospace; padding:6px 11px; border-radius:7px;
  opacity:0; transform:translateY(-3px); transition:opacity .22s,transform .22s;
}
.plate:hover .peek,.plate:focus-visible .peek{opacity:1; transform:none}
.cards{display:grid; grid-template-columns:repeat(4,1fr); gap:13px; margin:2px 0 16px}
.card{background:var(--surface); border:1px solid var(--border); border-radius:11px; padding:14px 15px}
.card h4{
  margin:0 0 7px; font:600 11px/1 ui-monospace,monospace; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted);
}
.card p{margin:0; font-size:13.5px; line-height:1.5; color:var(--text)}
.verdict .chip{font-size:13px}
.pending{
  background:var(--surface); border:1px dashed color-mix(in srgb,var(--warn) 55%,var(--border));
  border-radius:11px; padding:22px; color:var(--text); font-size:14px; margin-bottom:18px; line-height:1.55;
}
.tool-note{color:var(--muted); font-size:13.5px; margin:0 0 11px; max-width:70ch}
pre.run{
  background:var(--surface-2); border:1px solid var(--border);
  border-left:3px solid var(--accent); border-radius:10px; padding:15px 18px;
  overflow-x:auto; font:12.7px/1.75 ui-monospace,"SF Mono",monospace;
  color:var(--text); margin:0 0 14px;
}
pre.run .cmt{color:var(--muted)}
footer{color:var(--muted); font-size:12.5px; border-top:1px solid var(--border); padding-top:16px; margin-top:8px}
code{background:var(--surface-2); border:1px solid var(--border); padding:1px 6px; border-radius:5px;
  font:12px ui-monospace,monospace}
@media (max-width:720px){.cards{grid-template-columns:1fr 1fr}}
/* modal */
#modal{display:none; position:fixed; inset:0; z-index:60; align-items:center; justify-content:center;
  background:color-mix(in srgb,var(--plate) 78%, transparent); backdrop-filter:blur(3px)}
#modal.open{display:flex}
#modal-inner{
  background:var(--surface); border:1px solid var(--border); border-radius:16px;
  max-width:600px; width:92%; padding:32px 36px; position:relative;
  box-shadow:var(--shadow); transform:scale(.95); transition:transform .28s ease}
#modal.open #modal-inner{transform:scale(1)}
#slide .eyebrow{margin-bottom:9px; display:block}
#slide h3{margin:0 0 11px; font-size:19px; font-weight:640; color:var(--text); letter-spacing:-.01em}
#slide p{margin:0; font-size:15px; line-height:1.6; color:var(--text); min-height:104px}
#dots{display:flex; gap:8px; justify-content:center; margin-top:20px}
#dots span{width:8px; height:8px; border-radius:50%; background:var(--border); cursor:pointer; transition:background .2s}
#dots span.on{background:var(--accent)}
#prev,#next,#close{position:absolute; background:none; border:none; color:var(--muted);
  font-size:26px; cursor:pointer; line-height:1; padding:6px}
#prev{left:4px; top:47%} #next{right:4px; top:47%} #close{top:12px; right:14px; font-size:22px}
#prev:hover,#next:hover,#close:hover,#prev:focus-visible,#next:focus-visible,#close:focus-visible{color:var(--text)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""


def build():
    FIGS.mkdir(parents=True, exist_ok=True)
    beach = build_beachhead()
    g6, g6_ready = build_g6()

    FIG = {
        "beach": [
            ("What it is", "A Mono-free residual LoRA. It corrects Product-of-Experts (PoE) composition at inference by adding λ·Δ̂ to the frozen PoE noise prediction — the joint prompt is never encoded."),
            ("The equation", "ε_final = ε̃_PoE + λ·(ε̃_PoE^LoRA − ε̃_PoE^frozen). At λ=0 this is byte-identical to vanilla PoE (a canary); at λ=1 it reaches toward the Mono ceiling."),
            ("Why this plan needed it", "PoE alone fuses cat and dog into one chimera (leftmost plate). The program asks whether a small rank-8 cross-attention LoRA, trained only on the cached PoE→Mono residual, can separate them — without the joint prompt that would defeat compositionality."),
            ("A worked pass", "cat×dog, seed 42, epoch 1600, λ=1.0: the decoded image is a cat nuzzling a dog — two distinct animals. Automated VQAScore returns 0 here (unreliable on this pair; the plan deferred VQA gating), so the eyeball is the headline, as Plan 04 states."),
            ("How to read it", "Left to right is the correction turning on: PoE chimera → half correction → full LoRA (two animals) → the Mono target the LoRA is chasing."),
        ],
        "g6": [
            ("What it is", "The cross-seed pooled LoRA: one adapter trained on cat×dog residuals from seeds 1–4, evaluated on held-out seeds 9–12 it never saw."),
            ("The equation", "Same sampler as the beachhead; the difference is the training pool — the residual is averaged over 4 seeds, testing whether the fix is seed-generic or a seed-42 accident."),
            ("Why this plan needed it", "Rung 2, Survive-Noise. The cross-seed Δ_t diagnostic landed near 'seed noise', so pooling was not guaranteed to work — this checks it empirically."),
            ("A worked pass", "checkpoint k04__ep2000_resumed (verdict ok), λ=1.0, seeds 9–12: each held-out seed's PoE chimera beside the pooled-LoRA output."),
            ("How to read it", "Column pairs per seed: PoE (broken) beside the pooled LoRA. Separation on a seed the LoRA never trained on means the fix survives seed noise."),
        ],
    }

    beach_fig = fig_block("beach", beach,
        "cat×dog, seed 42, epoch 1600. PoE fuses the two into one animal; the LoRA at λ=1 separates them; Mono is the target. VQAScore deferred — the read is by eye (Plan 04).")
    if g6_ready:
        g6_fig = fig_block("g6", g6, "Held-out seeds 9–12: PoE chimera beside the seed-pooled LoRA. Generated on the cluster (job recap_g6).")
        g6_chip = '<span class="chip pass">enactment generated</span>'
    else:
        g6_fig = ('<div class="pending"><b>Enactment generating on the cluster.</b> '
                  'Slurm job <code>recap_g6</code> (biggpu) samples the pooled LoRA on held-out '
                  'seeds 9–12. This section fills in on the next <code>python scripts/recap_landed.py</code> '
                  'once the job lands. The G6 pool reached epoch 2000 with <code>verdict.json = "ok"</code>.</div>')
        g6_chip = '<span class="chip warn">enactment pending — job recap_g6</span>'

    body = f"""<div id="wrap">
<header>
  <span class="eyebrow">Recall gallery · landed results</span>
  <h1>LoRA-Fixes-PoE</h1>
  <p class="mission">Does a LoRA make PoE co-occur like Mono, and does that fix carry to unseen pairs?</p>
  <div class="banner"><b>Honest scope.</b> The five pyramid rungs are <b>not complete</b> — the last plan-tree sync marked all of them pending. This gallery recalls the <b>two genuinely-landed sub-results</b>: the Overfit <i>beachhead</i> (cat×dog) and the Survive-Noise <i>G6 pool</i>. Cross-Pair, Group-Wise and Scale are omitted because they have not been run.</div>
  <div class="status">
    <span class="chip pass">Overfit beachhead · G04</span>
    <span class="chip pass">G6 survive-noise · G07 · verdict ok</span>
    <span class="chip warn">Overfit breadth G1–G3 · owed</span>
    <span class="chip warn">Scale crossbar · not evaluated</span>
  </div>
</header>

<section>
  <div class="sec-head"><span class="rung">Rung 1 · Overfit</span><h2>PoE fails; the LoRA fixes it, Mono-free</h2></div>
  {beach_fig}
  {cards(
    "Trained a rank-8 LoRA on SDXL cross-attention on the cached PoE→Mono residual for cat×dog, seed 42.",
    "PoE composition fuses two concepts into one chimera. We test whether a small in-UNet corrector can separate them without ever using the joint prompt.",
    "At λ=1 the chimera becomes two distinct animals; λ=0 is byte-identical to PoE (canary). The verified headline checkpoint loads with 420 LoRA tensors.",
    "One pair, one seed. VQAScore is 0/unreliable here (DINO cat-box conf 0.39) — the read is by eye, per Plan 04. Breadth to G1–G3 is still owed."
  )}
  <div class="verdict"><span class="chip pass">landed — the deployable Mono-free corrector works on the beachhead cell</span></div>
</section>

<section>
  <div class="sec-head"><span class="rung">Rung 2 · Survive-Noise</span><h2>One LoRA, pooled over seeds, on held-out seeds</h2></div>
  {g6_fig}
  {cards(
    "Pooled a LoRA over cat×dog seeds 1–4; evaluate on held-out seeds 9–12.",
    "A seed-42 fix could be a seed accident. The cross-seed Δ_t diagnostic landed near 'seed noise', so pooling had to be tested, not assumed.",
    "The pooled run reached epoch 2000 with verdict 'ok'. The held-out-seed enactment is generated on the cluster (see plate).",
    "Only G6 (cat×dog) has a pooled verdict; G1–G4 per-group pools are part-trained with no verdict yet."
  )}
  <div class="verdict">{g6_chip}</div>
</section>

<section>
  <div class="sec-head"><span class="rung">Tooling</span><h2>See it yourself — run the web apps</h2></div>
  <p class="tool-note">The <b>LoRA Inspector</b> is a Flask app with four tabs — <b>CFG-mask ablation</b> (no-LoRA floor), <b>LoRA residual</b> (the epoch × λ morph and MDS trajectory shown above), <b>MDS large</b>, and <b>LoRA + CFG-mask</b>. Pair dropdown, top right.</p>
  <pre class="run">PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
$PY scripts/build_lora_manifest.py        <span class="cmt"># (re)build the manifest the inspector reads</span>
bash scripts/run_lora_inspector.sh        <span class="cmt"># serves 127.0.0.1:5050, prints the tunnel line</span>
<span class="cmt"># from your laptop:</span> ssh -L 5050:localhost:5050 &lt;cluster-node&gt;
<span class="cmt"># then open:</span>        http://localhost:5050</pre>
  <p class="tool-note">Live checkpoint viewer (group-A student runs — side-by-side PoE | Mono | student):</p>
  <pre class="run">$PY -m scripts.watch_and_visualize --ckpt-dir &lt;ckpt-dir&gt; --pair "a cat|a dog" --seed 42</pre>
</section>

<footer>Recall of landed sub-results · rungs incomplete by design · rebuild: <code>python scripts/recap_landed.py</code></footer>
</div>

<div id="modal" onclick="closeModal(event)">
  <div id="modal-inner">
    <div id="slide"></div>
    <div id="dots"></div>
    <button id="prev" aria-label="previous" onclick="nav(-1,event)">‹</button>
    <button id="next" aria-label="next" onclick="nav(1,event)">›</button>
    <button id="close" aria-label="close" onclick="closeModal(event)">×</button>
  </div>
</div>"""

    script = ("<script>\nconst FIG = " + json.dumps(FIG) + ";\n"
        "let curFig=null,curSlide=0;\n"
        "function openModal(id){curFig=id;curSlide=0;render();document.getElementById('modal').classList.add('open');}\n"
        "function closeModal(e){if(!e||e.target.id==='modal'||e.target.id==='close')document.getElementById('modal').classList.remove('open');}\n"
        "function nav(d,e){e.stopPropagation();const n=FIG[curFig].length;curSlide=(curSlide+d+n)%n;render();}\n"
        "function render(){const s=FIG[curFig][curSlide];"
        "document.getElementById('slide').innerHTML='<span class=\"eyebrow\">'+(curSlide+1)+' / '+FIG[curFig].length+'</span><h3>'+s[0]+'</h3><p>'+s[1]+'</p>';"
        "document.getElementById('dots').innerHTML=FIG[curFig].map((_,i)=>'<span class=\"'+(i===curSlide?'on':'')+'\" onclick=\"curSlide='+i+';render()\"></span>').join('');}\n"
        "document.addEventListener('keydown',e=>{if(!document.getElementById('modal').classList.contains('open'))return;"
        "if(e.key==='Escape')closeModal();if(e.key==='ArrowRight')nav(1,e);if(e.key==='ArrowLeft')nav(-1,e);});\n"
        "</script>")

    RECAP.mkdir(parents=True, exist_ok=True)
    (RECAP / "index.html").write_text(body + "\n<style>" + CSS + "</style>\n" + script)
    (RECAP / "RECAP.md").write_text(
        "# Recap — LoRA-Fixes-PoE landed results\n\n"
        "Build: `python scripts/recap_landed.py`\n\n"
        "Scope: the two landed sub-results (Overfit beachhead G04, G6 survive-noise pool G07). "
        "Rungs are incomplete by design — see report/decision-timeline.md.\n\n"
        f"G6 enactment: {'generated' if g6_ready else 'pending Slurm job recap_g6 (results/recap_landed/gen/)'}.\n\n"
        "Artifact URL: https://claude.ai/code/artifact/bddaaec6-bb17-4b67-bdc4-e0106d44cdb1\n")
    print(f"built recap/index.html  (g6_ready={g6_ready})")


if __name__ == "__main__":
    build()
