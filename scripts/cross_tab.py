"""The crossed tab: when the prompt acts, against when the correction acts.

Two axes on one grid. Rows are conditioning schedules, deciding which steps have
a prompt at all; columns are correction windows, deciding which steps get the
interaction term. Reading along a row holds the prompt schedule fixed and moves
the correction, which is the clean comparison. Reading along a diagonal moves
both and says nothing, which is why the tab marks the row and column you are on.

Selecting a cell opens it with a step scrubber over its decoded frames, so the
question "where does this run part company with the always-guided one" is
answered by dragging rather than by inference.

Pre-rendered only. Frames come from decode_trajectory_frames.py, which decodes
latents the sampler already saved.
"""

from __future__ import annotations

CROSS_INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>conditioning x correction</title>
<style>
  :root {
    --bg:#0f1115; --panel:#161a22; --text:#e6e9ef; --muted:#8b93a7;
    --accent:#7ab7ff; --accent-hot:#f0a458; --on:#2C8F4A; --off:#2a2f3a;
    --border:#2a2f3a;
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:24px; background:var(--bg); color:var(--text);
         font:14px/1.4 -apple-system, system-ui, "Segoe UI", sans-serif; }
  h1 { font-size:18px; font-weight:600; margin:0 0 6px 0; }
  .subtitle { color:var(--muted); font-size:12px; margin:0 0 4px 0; max-width:940px; }
  .meta { color:var(--muted); font-size:12px; margin-bottom:16px; }
  .meta b { color:var(--text); }
  .controls { background:var(--panel); border:1px solid var(--border);
              border-radius:8px; padding:14px; margin-bottom:16px; }
  .row { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .row + .row { margin-top:12px; }
  label.k { text-transform:uppercase; font-size:10px; letter-spacing:.06em;
            font-weight:600; color:var(--muted); }
  select, button { background:#0f1115; color:var(--text); border:1px solid var(--border);
                   border-radius:4px; padding:5px 10px; font-size:12px; font-weight:600;
                   font-family:ui-monospace, monospace; cursor:pointer; }
  button:hover:not(:disabled) { border-color:var(--accent); color:var(--accent); }
  button.hot { border-color:var(--accent-hot); color:var(--accent-hot); }
  input[type=range] { -webkit-appearance:none; appearance:none; flex:1; min-width:220px;
                      height:4px; border-radius:2px; background:var(--off); outline:none; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:15px; height:15px;
      border-radius:50%; background:var(--accent); cursor:pointer; }
  .readout { font-family:ui-monospace,monospace; font-size:12px; font-weight:600;
             color:var(--accent); min-width:150px; }

  .layout { display:grid; grid-template-columns: minmax(0,1.15fr) minmax(0,1fr);
            gap:16px; align-items:start; }
  @media (max-width:1100px){ .layout{ grid-template-columns:minmax(0,1fr);} }
  .card { background:var(--panel); border:1px solid var(--border);
          border-radius:8px; padding:14px; }
  .card h2 { font-size:11px; text-transform:uppercase; letter-spacing:.06em;
             color:var(--muted); margin:0 0 10px 0; font-weight:600; }

  table.grid { border-collapse:separate; border-spacing:3px; width:100%; }
  table.grid th { font-size:9.5px; font-weight:600; color:var(--muted);
                  font-family:ui-monospace,monospace; padding:2px; text-align:center; }
  table.grid th.rowh { text-align:right; white-space:nowrap; padding-right:6px; }
  table.grid td { padding:0; }
  .thumb { width:100%; aspect-ratio:1/1; border-radius:3px; display:block;
           background:#0a0c10; border:2px solid transparent; cursor:pointer; }
  .thumb.sel { border-color:var(--accent-hot); }
  .thumb.hl  { border-color:#3a4356; }
  .thumb.gone { display:flex; align-items:center; justify-content:center;
                color:var(--muted); font-size:9px; border:1px dashed var(--border);
                cursor:default; }
  .frame img { width:100%; display:block; border-radius:6px; background:#0a0c10;
               aspect-ratio:1/1; object-fit:cover; }
  .caption { color:var(--muted); font-size:11px; margin-top:8px;
             font-family:ui-monospace,monospace; }
  .caption b { color:var(--text); }
  .track { display:flex; gap:1px; margin-top:8px; }
  .track .step { flex:1; height:16px; background:var(--off); border-radius:1px;
                 position:relative; }
  .track .step.corr { background:var(--on); }
  .track .step.nocond { background:#5a3030; }
  .track .step.here::after { content:""; position:absolute; inset:-3px -1px;
                             border:1px solid var(--accent-hot); border-radius:2px; }
  .legend { display:flex; gap:14px; color:var(--muted); font-size:10px; margin-top:6px;
            font-family:ui-monospace,monospace; flex-wrap:wrap; }
  .sw { display:inline-block; width:9px; height:9px; border-radius:2px;
        vertical-align:-1px; margin-right:4px; }
  .empty { color:var(--muted); font-size:12px; text-align:center; padding:36px 12px;
           border:1px dashed var(--border); border-radius:6px; }
  .note { color:var(--muted); font-size:11px; margin-top:10px; line-height:1.6; }
  details.diagnostics { margin-top:22px; padding:12px 14px; background:var(--panel);
      border:1px solid var(--border); border-radius:6px; color:var(--muted); font-size:12px; }
  details.diagnostics summary { cursor:pointer; font-weight:600; font-size:11px;
      letter-spacing:.04em; text-transform:uppercase; }
  details.diagnostics .body { margin-top:8px; line-height:1.7; }
  code { color:var(--text); }
</style>
</head>
<body>

<h1>When the prompt acts, against when the correction acts</h1>
<p class="subtitle">
  Two separate switches. The <b>row</b> decides which steps have a prompt at all;
  outside that stretch the model runs unconditional, with no text. The
  <b>column</b> decides which steps the interaction term is injected on. Compare
  along a row or a column, never along a diagonal: two cells that differ on both
  switches cannot tell you which one mattered.
</p>
<p class="meta">
  {{ num_steps }} steps &middot; window width <b>{{ width }}</b> &middot;
  fork step <b>{{ fork_step }}</b> &middot;
  <b>{{ n_on_disk }}</b> of <b>{{ n_planned }}</b> cells sampled,
  <b>{{ n_frames }}</b> with per-step frames
</p>

{% if n_on_disk == 0 %}
<div class="empty">
  Nothing sampled yet. Run
  <code>python scripts/run_cross_sweep.py --grid cross</code>, then
  <code>python scripts/decode_trajectory_frames.py --root cross</code>, then
  <code>python scripts/build_cross_manifest.py</code>.
</div>
{% else %}

<div class="controls">
  <div class="row">
    <label class="k" for="pair">pair</label><select id="pair"></select>
    <label class="k" for="seed">seed</label><select id="seed"></select>
    <span style="flex:1"></span>
    <label class="k" for="mode">view</label>
    <select id="mode">
      <option value="cross">the cross: prompt schedule x correction window</option>
      <option value="dense">dense timing: every one-step shift, prompt always on</option>
    </select>
  </div>
  <div class="row" id="dense-row" style="display:none">
    <label class="k">window</label>
    <button id="dprev">&#9664;</button>
    <input type="range" id="dense" min="0" value="0" step="1">
    <button id="dnext">&#9654;</button>
    <span class="readout" id="dreadout"></span>
  </div>
  <div class="row">
    <label class="k">step</label>
    <button id="sprev">&#9664;</button>
    <input type="range" id="step" min="0" max="0" value="0" step="1">
    <button id="snext">&#9654;</button>
    <span class="readout" id="sreadout"></span>
    <button id="play">play &#9654;</button>
  </div>
</div>

<div class="layout">
  <div class="card">
    <h2 id="grid-title">the grid</h2>
    <div id="grid-wrap"></div>
    <div class="legend">
      <span><span class="sw" style="background:var(--on)"></span>correction injected</span>
      <span><span class="sw" style="background:#5a3030"></span>no prompt (unconditional)</span>
      <span><span class="sw" style="background:var(--off)"></span>prompt on, no correction</span>
    </div>
  </div>
  <div class="card">
    <h2 id="sel-title">selected cell</h2>
    <div class="frame"><img id="shot" alt="selected cell"></div>
    <div class="track" id="track"></div>
    <div class="caption" id="cap"></div>
    <p class="note" id="note"></p>
  </div>
</div>

<details class="diagnostics">
  <summary>what this tab can and cannot tell you</summary>
  <div class="body">
    The pictures on the step slider are the model's estimate of the finished
    image at that step, not the noisy latent it carries. The noisy latent looks
    like static until the last few steps and cannot show where two runs part
    company; the estimate is legible from the first frame. Both come from the
    same saved trajectory.<br><br>
    Frames are decoded every fifth step by default, so a divergence narrower
    than five steps can sit between two frames. The cells worth reading closely
    are decoded at every step; those show a <code>stride</code> of null in their
    frames file.<br><br>
    Where a row has no prompt, no correction is injected either. The interaction
    term is defined as the gap between two conditional predictions, so with no
    prompt there is nothing for it to be the gap between. Those steps are drawn
    red, not green.<br><br>
    One cell is one seed. The compose rate over pairs and seeds lives on the
    Correction timing tab; this tab is for looking at mechanism, not for
    measuring a rate.
  </div>
</details>

<script>
const M = {{ manifest_json|safe }};
const NS = M.num_steps, FORK = M.fork_step;
const $ = id => document.getElementById(id);
let sel = null, timer = null;

function cellsFor(){ return (M.cells[$("pair").value]||{})[$("seed").value]||{}; }
function get(id){ return cellsFor()[id] || null; }

function fill(sel_, vals, cur){
  sel_.innerHTML="";
  for(const v of vals){ const o=document.createElement("option");
    o.value=v; o.textContent=v; if(v===cur) o.selected=true; sel_.appendChild(o); }
}

function condOf(tag){ return M.cond_schedules.find(c=>c.tag===tag); }

function buildGrid(){
  const wrap = $("grid-wrap");
  const mode = $("mode").value;
  if (mode === "dense") { wrap.innerHTML = ""; return; }
  const cells = cellsFor();
  const t = document.createElement("table"); t.className="grid";
  const head = document.createElement("tr");
  head.appendChild(document.createElement("th"));
  for (const col of M.corr_columns){
    const th=document.createElement("th"); th.textContent=col; head.appendChild(th);
  }
  t.appendChild(head);
  for (const cs of M.cond_schedules){
    const tr=document.createElement("tr");
    const th=document.createElement("th"); th.className="rowh";
    th.textContent = cs.tag; th.title = cs.describe; tr.appendChild(th);
    for (const col of M.corr_columns){
      const td=document.createElement("td");
      const id = "c"+cs.tag+"__r"+col;
      const rec = cells[id];
      if (!rec){
        const d=document.createElement("div"); d.className="thumb gone"; d.textContent="—";
        td.appendChild(d);
      } else {
        const im=document.createElement("img");
        im.className="thumb"+(sel&&sel.cell_id===id?" sel":"");
        im.loading="lazy";
        im.src="/interaction_window/img?path="+encodeURIComponent(rec.image);
        im.title=cs.describe+" · correction "+col;
        im.addEventListener("click",()=>{ select(id); });
        td.appendChild(im);
      }
      tr.appendChild(td);
    }
    t.appendChild(tr);
  }
  wrap.innerHTML=""; wrap.appendChild(t);
  $("grid-title").textContent =
    "rows: when the prompt acts · columns: when the correction acts";
}

function drawTrack(rec, stepIdx){
  const tr=$("track");
  if(tr.children.length!==NS){ tr.innerHTML="";
    for(let i=0;i<NS;i++){ const d=document.createElement("div");
      d.className="step"; d.title="step "+i; tr.appendChild(d);} }
  const cs = condOf(rec.cond_tag);
  const cw = cs ? cs.window : null, outside = cs ? cs.outside : false;
  const rw = rec.corr_window;
  const frameStep = rec.frames.length ? rec.frames[stepIdx].step : -1;
  for(let i=0;i<NS;i++){
    let on = true;
    if (cw){ const inside = i>=cw[0] && i<cw[1]; on = outside ? !inside : inside; }
    const corr = on && rw && i>=rw[0] && i<rw[1]
                 || (on && !rw && rec.lambda_max>0);
    tr.children[i].className = "step"
      + (!on ? " nocond" : (corr ? " corr" : ""))
      + (i===frameStep ? " here" : "");
  }
}

function select(id){
  const rec = get(id);
  if(!rec) return;
  sel = rec;
  const s=$("step");
  s.max = Math.max(rec.frames.length-1, 0);
  s.value = Math.min(parseInt(s.value||0,10), s.max);
  buildGrid();
  render();
}

function render(){
  if(!sel){ return; }
  const rec = sel;
  const i = parseInt($("step").value,10);
  const has = rec.frames.length>0;
  const fr = has ? rec.frames[i] : null;
  $("shot").src = "/interaction_window/img?path="
    + encodeURIComponent(fr ? fr.path : rec.image);
  $("sreadout").textContent = has
    ? `step ${fr.step} of ${NS}` : "no frames decoded";
  const cs = condOf(rec.cond_tag);
  $("sel-title").textContent = rec.cell_id;
  $("cap").innerHTML = `<b>${cs ? cs.describe : rec.cond_tag}</b><br>`
    + `correction: <b>${rec.corr}</b>`
    + (rec.corr_window ? ` (steps ${rec.corr_window[0]}–${rec.corr_window[1]})` : "")
    + ` · λ=${rec.lambda_max}`;
  $("note").textContent = has
    ? (fr.step === NS || !rec.frames[i+1]
        ? "this is the finished picture"
        : "the model's estimate of the finished picture at this step")
    : "run scripts/decode_trajectory_frames.py to scrub this cell step by step";
  drawTrack(rec, i);
}

function denseSetup(){
  const dense = M.dense_windows;
  $("dense").max = dense.length-1;
  const upd = ()=>{
    const w = dense[parseInt($("dense").value,10)];
    $("dreadout").textContent = `steps ${w[0]}–${w[1]}`;
    const id = "call__r"+w[0]+"-"+w[1];
    if(get(id)) select(id);
    else { $("cap").innerHTML = `<b>not sampled yet</b>: window ${w[0]}–${w[1]}`; }
  };
  $("dense").addEventListener("input", upd);
  $("dprev").addEventListener("click", ()=>{ $("dense").value=Math.max(0,+$("dense").value-1); upd(); });
  $("dnext").addEventListener("click", ()=>{ $("dense").value=Math.min(dense.length-1,+$("dense").value+1); upd(); });
  return upd;
}

function setPair(){
  const p=$("pair").value;
  const seeds=M.seeds_by_pair[p]||[];
  fill($("seed"), seeds, seeds.includes($("seed").value)?$("seed").value:seeds[0]);
}

function firstCell(){
  const c = cellsFor();
  const keys = Object.keys(c);
  return keys.length ? keys.find(k=>k.startsWith("call__")) || keys[0] : null;
}

function play(){
  if(timer){ clearInterval(timer); timer=null;
    $("play").innerHTML="play &#9654;"; $("play").classList.remove("hot"); return; }
  $("play").innerHTML="pause &#10073;&#10073;"; $("play").classList.add("hot");
  timer=setInterval(()=>{ const s=$("step");
    s.value = (parseInt(s.value,10)+1) > +s.max ? 0 : parseInt(s.value,10)+1;
    render(); }, 420);
}

fill($("pair"), M.pairs, M.pairs[0]);
setPair();
const denseUpd = denseSetup();
$("pair").addEventListener("change", ()=>{ setPair(); const f=firstCell(); if(f) select(f); buildGrid(); });
$("seed").addEventListener("change", ()=>{ const f=firstCell(); if(f) select(f); buildGrid(); });
$("step").addEventListener("input", render);
$("sprev").addEventListener("click", ()=>{ $("step").value=Math.max(0,+$("step").value-1); render(); });
$("snext").addEventListener("click", ()=>{ $("step").value=Math.min(+$("step").max,+$("step").value+1); render(); });
$("play").addEventListener("click", play);
$("mode").addEventListener("change", ()=>{
  const dense = $("mode").value==="dense";
  $("dense-row").style.display = dense ? "" : "none";
  buildGrid();
  if(dense) denseUpd();
});
document.addEventListener("keydown", e=>{
  if(e.target.tagName==="SELECT") return;
  if(e.key==="ArrowRight"){ $("step").value=Math.min(+$("step").max,+$("step").value+1); render(); e.preventDefault(); }
  if(e.key==="ArrowLeft"){ $("step").value=Math.max(0,+$("step").value-1); render(); e.preventDefault(); }
});

const f0 = firstCell();
if(f0) select(f0); else buildGrid();
</script>
{% endif %}
</body>
</html>
"""
