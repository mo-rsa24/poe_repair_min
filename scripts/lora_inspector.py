"""Flask inspector for the LoRA SDXL residual training chain.

Pre-rendered only. Sliders snap to indexed (epoch, lambda) values. When the
selected cell does not exist (some epochs only probed a coarser lambda grid),
a toast appears and the panels keep the previous frame.

Run on the cluster (binds 127.0.0.1 by default, intended for SSH tunnelling):
    python scripts/lora_inspector.py [--port 5050]

Then on your laptop:
    ssh -L 5050:localhost:5050 mscluster106
    open http://localhost:5050
"""
from __future__ import annotations

import argparse
import json
import os.path
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, send_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "outputs/lora/cat_dog/seed_42/results/inspector_manifest.json"
DEFAULT_CW_MANIFEST = REPO_ROOT / "outputs/conditioning_window/cat_dog/seed_42/results/inspector_manifest.json"
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>m5 residual inspector</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161a22;
    --text: #e6e9ef;
    --muted: #8b93a7;
    --accent: #7ab7ff;
    --warn: #f0a458;
    --border: #2a2f3a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font: 14px/1.4 -apple-system, system-ui, "Segoe UI", sans-serif;
  }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 16px 0; color: var(--muted); }
  .controls {
    display: grid; gap: 14px; margin-bottom: 20px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }
  .row { display: flex; align-items: center; gap: 16px; }
  .row label { flex: 0 0 80px; color: var(--muted); }
  .row input[type=range] { flex: 1; }
  .row .val {
    flex: 0 0 64px; text-align: right; font-variant-numeric: tabular-nums;
    color: var(--accent); font-weight: 600;
  }
  .row .source { color: var(--muted); font-size: 12px; flex: 0 0 auto; }
  .panels {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;
  }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }
  .panel h2 {
    margin: 0 0 8px 0; font-size: 12px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  }
  .panel img {
    width: 100%; aspect-ratio: 1 / 1; border-radius: 4px;
    background: #000; display: block;
  }
  .panel .caption {
    margin-top: 8px; color: var(--muted); font-size: 11px;
  }
  #toast {
    position: fixed; top: 24px; right: 24px;
    background: var(--warn); color: #1a1207;
    padding: 10px 14px; border-radius: 6px;
    font-weight: 600; opacity: 0; pointer-events: none;
    transition: opacity 0.18s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<h1>LoRA residual inspector — {{ results_root }}</h1>

<div class="controls">
  <div class="row">
    <label for="epoch">epoch</label>
    <input id="epoch" type="range" min="0" max="{{ epochs|length - 1 }}" step="1" value="0">
    <div class="val" id="epoch-val">0</div>
    <div class="source" id="epoch-src"></div>
  </div>
  <div class="row">
    <label for="lambda">λ</label>
    <input id="lambda" type="range" min="0" max="{{ lambdas|length - 1 }}" step="1" value="{{ lambdas|length - 1 }}">
    <div class="val" id="lambda-val">{{ lambdas[-1] }}</div>
  </div>
</div>

<div class="panels">
  <div class="panel">
    <h2>PoE baseline (λ = 0)</h2>
    <img id="img-poe" alt="PoE baseline">
  </div>
  <div class="panel">
    <h2>PoE + λ · r (current)</h2>
    <img id="img-current" alt="current">
  </div>
</div>

<p style="margin-top:18px;color:var(--muted);font-size:12px;">
  → <a href="/conditioning_window" style="color:var(--accent);">conditioning_window</a>
  (no-LoRA CFG-mask ablation; same seed, same x_T)
</p>

<div id="toast">cell not available</div>

<script>
const MANIFEST = {{ manifest_json|safe }};
const EPOCHS = MANIFEST.epochs;
const LAMBDAS = MANIFEST.lambdas;     // strings like "0.00", "0.50", "1.00"
const CELLS = MANIFEST.cells;          // {"100": {"0.00": "outputs/.../decoded.png", ...}}
const SOURCE = MANIFEST.cell_source_run;

const epochSlider = document.getElementById('epoch');
const epochVal = document.getElementById('epoch-val');
const epochSrc = document.getElementById('epoch-src');
const lambdaSlider = document.getElementById('lambda');
const lambdaVal = document.getElementById('lambda-val');
const imgPoe = document.getElementById('img-poe');
const imgCurrent = document.getElementById('img-current');
const toast = document.getElementById('toast');

let toastTimer = null;
function flashToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 1600);
}

function lambdaForIdx(idx) { return LAMBDAS[idx]; }
function epochForIdx(idx) { return EPOCHS[idx]; }

function cellPath(epoch, lam) {
  const e = String(epoch);
  if (!CELLS[e]) return null;
  return CELLS[e][lam] || null;
}

function update() {
  const eIdx = parseInt(epochSlider.value, 10);
  const lIdx = parseInt(lambdaSlider.value, 10);
  const epoch = epochForIdx(eIdx);
  const lam = lambdaForIdx(lIdx);
  epochVal.textContent = epoch;
  lambdaVal.textContent = lam;
  epochSrc.textContent = SOURCE[String(epoch)] || '';

  const poePath = cellPath(epoch, '0.00');
  const curPath = cellPath(epoch, lam);

  if (!curPath) {
    flashToast(`no probe at epoch=${epoch}, λ=${lam}`);
    return;
  }
  if (!poePath) {
    flashToast(`no PoE baseline at epoch=${epoch}`);
    return;
  }
  imgPoe.src = '/img/' + poePath;
  imgCurrent.src = '/img/' + curPath;
}

epochSlider.addEventListener('input', update);
lambdaSlider.addEventListener('input', update);

update();
</script>
</body>
</html>
"""




CW_INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>conditioning_window inspector</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161a22;
    --text: #e6e9ef;
    --muted: #8b93a7;
    --accent: #7ab7ff;
    --on:    #2C8F4A;
    --off:   #2a2f3a;
    --border: #2a2f3a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font: 14px/1.4 -apple-system, system-ui, "Segoe UI", sans-serif;
  }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 4px 0; color: var(--muted); }
  .meta { color: var(--muted); font-size: 12px; margin-bottom: 18px; }
  .controls {
    display: grid; gap: 14px; margin-bottom: 20px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }
  .row { display: flex; align-items: center; gap: 16px; }
  .row label { flex: 0 0 100px; color: var(--muted); }
  .row input[type=range] { flex: 1; }
  .row .val {
    flex: 0 0 220px; text-align: left;
    font-variant-numeric: tabular-nums;
    color: var(--accent); font-weight: 600; font-family: ui-monospace, monospace;
  }
  .strip {
    display: grid; grid-template-columns: repeat(50, 1fr);
    gap: 1px; margin-top: 6px; height: 26px;
    border: 1px solid var(--border); border-radius: 4px;
    overflow: hidden;
  }
  .cell { background: var(--off); }
  .cell.on { background: var(--on); }
  .cell.tick { box-shadow: inset 0 -3px 0 0 #555; }
  .legend { color: var(--muted); font-size: 11px; margin-top: 6px; }
  .legend .sw { display: inline-block; width: 10px; height: 10px;
                margin-right: 4px; vertical-align: middle; border-radius: 2px; }
  .panels { display: grid; grid-template-columns: 1fr; gap: 16px; max-width: 720px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }
  .panel h2 {
    margin: 0 0 8px 0; font-size: 12px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  }
  .panel img {
    width: 100%; aspect-ratio: 1 / 1; border-radius: 4px;
    background: #000; display: block;
  }
  .panel .caption {
    margin-top: 8px; color: var(--muted); font-size: 12px;
    font-family: ui-monospace, monospace; word-break: break-all;
  }
  .sanity-pill {
    display: inline-block; padding: 1px 6px; border-radius: 8px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.04em;
    background: var(--off); color: var(--accent); margin-left: 6px;
  }
</style>
</head>
<body>
<h1>conditioning_window — {{ prompt }} · seed {{ seed }}</h1>
<div class="meta">
  num_inference_steps = {{ num_steps }} · guidance_scale = {{ guidance }} ·
  {{ schedules|length }} schedule(s)
  · sanity: all_on Δ={{ sanity_on }} ({{ sanity_on_pass }}),
              all_off Δ={{ sanity_off }} ({{ sanity_off_pass }})
</div>

<div class="controls">
  <div class="row">
    <label for="sched">schedule</label>
    <input id="sched" type="range" min="0" max="{{ schedules|length - 1 }}" step="1" value="0">
    <div class="val" id="sched-val">—</div>
  </div>
  <div>
    <div id="strip" class="strip" title=""></div>
    <div class="legend">
      <span class="sw" style="background: var(--on);"></span> conditional ON
      &nbsp;&nbsp;
      <span class="sw" style="background: var(--off);"></span> conditional OFF
      &nbsp;&nbsp;
      <span style="color: var(--muted);">(50-cell mask; hover = full binary string)</span>
    </div>
  </div>
</div>

<div class="panels">
  <div class="panel">
    <h2>decoded image</h2>
    <img id="img" alt="decoded">
    <div class="caption" id="caption"></div>
  </div>
</div>

<script>
const MANIFEST  = {{ manifest_json|safe }};
const SCHEDULES = MANIFEST.schedules;

const slider  = document.getElementById('sched');
const valEl   = document.getElementById('sched-val');
const stripEl = document.getElementById('strip');
const imgEl   = document.getElementById('img');
const capEl   = document.getElementById('caption');

function renderStrip(maskStr) {
  stripEl.innerHTML = '';
  for (let i = 0; i < maskStr.length; i++) {
    const c = document.createElement('div');
    c.className = 'cell' + (maskStr[i] === '1' ? ' on' : '')
                         + ((i + 1) % 10 === 0 ? ' tick' : '');
    stripEl.appendChild(c);
  }
  stripEl.title = maskStr;
}

function update() {
  const i = parseInt(slider.value, 10);
  const s = SCHEDULES[i];
  valEl.textContent = s.id + '  (' + s.num_on + '/' + s.mask.length + ' on)';
  renderStrip(s.mask);
  imgEl.src = '/img/' + s.image_path;
  let cap = `${s.id} · family=${s.family} · num_on=${s.num_on}`;
  if (s.sanity) cap += '  <SANITY>';
  capEl.textContent = cap;
}

slider.addEventListener('input', update);
update();
</script>
</body>
</html>
"""


_CW_MISSING_HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><title>conditioning_window manifest missing</title>
<style>body{font:14px -apple-system,system-ui,sans-serif;background:#0f1115;color:#e6e9ef;padding:24px;}
code{background:#161a22;padding:2px 6px;border-radius:4px;color:#7ab7ff;}</style>
</head><body>
<h2>conditioning_window manifest not found</h2>
<p>Expected at:</p>
<p><code>{{ path }}</code></p>
<p>Build it by running:</p>
<p><code>python -m poe_repair.experiments.conditioning_window</code></p>
</body></html>
"""


def create_app(
    manifest_path: Path,
    outputs_root: Path,
    cw_manifest_path: Path | None = None,
) -> Flask:
    manifest = json.loads(manifest_path.read_text())
    outputs_root_abs = outputs_root.resolve()

    cw_manifest: dict | None = None
    if cw_manifest_path is not None and Path(cw_manifest_path).is_file():
        cw_manifest = json.loads(Path(cw_manifest_path).read_text())

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(
            INDEX_HTML,
            results_root=manifest.get("results_root", ""),
            epochs=manifest["epochs"],
            lambdas=manifest["lambdas"],
            manifest_json=json.dumps(manifest),
        )

    @app.route("/manifest.json")
    def manifest_route():
        return jsonify(manifest)

    @app.route("/conditioning_window")
    def conditioning_window():
        if cw_manifest is None:
            return render_template_string(
                _CW_MISSING_HTML,
                path=str(cw_manifest_path) if cw_manifest_path else "(no path)",
            ), 404
        sanity = cw_manifest.get("sanity") or {}
        on_rec = sanity.get("all_on_vs_run_cfg") or {}
        off_rec = sanity.get("all_off_vs_uncond") or {}
        return render_template_string(
            CW_INDEX_HTML,
            prompt=cw_manifest.get("prompt", ""),
            seed=cw_manifest.get("seed", ""),
            num_steps=cw_manifest.get("num_inference_steps", 50),
            guidance=cw_manifest.get("guidance_scale", ""),
            schedules=cw_manifest["schedules"],
            sanity_on=f"{on_rec.get('max_abs_delta', float('nan')):.2e}"
                      if on_rec else "—",
            sanity_off=f"{off_rec.get('max_abs_delta', float('nan')):.2e}"
                       if off_rec else "—",
            sanity_on_pass="PASS" if on_rec.get("pass") else "FAIL"
                           if on_rec else "—",
            sanity_off_pass="PASS" if off_rec.get("pass") else "FAIL"
                            if off_rec else "—",
            manifest_json=json.dumps(cw_manifest),
        )

    @app.route("/conditioning_window/manifest.json")
    def conditioning_window_manifest():
        if cw_manifest is None:
            abort(404)
        return jsonify(cw_manifest)

    @app.route("/img/<path:rel>")
    def img(rel: str):
        # rel is relative to REPO_ROOT (manifest stores those paths).
        # Reject traversal / absolute paths via *unresolved* normalization —
        # following symlinks here would falsely flag the /datasets-backed
        # symlinked subtrees as escapes.
        norm = os.path.normpath(rel)
        if norm.startswith("..") or os.path.isabs(norm):
            abort(403)
        norm_parts = norm.split(os.sep)
        if not norm_parts or norm_parts[0] != "outputs":
            abort(403)
        target = REPO_ROOT / norm
        if not target.is_file():
            abort(404)
        return send_file(target)

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument(
        "--cw-manifest", default=str(DEFAULT_CW_MANIFEST),
        help="conditioning_window manifest (produced by "
             "`python -m poe_repair.experiments.conditioning_window`). "
             "If missing, the /conditioning_window route renders a hint page.",
    )
    ap.add_argument("--outputs-root", default=str(DEFAULT_OUTPUTS_ROOT))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}")
        print("run scripts/build_lora_manifest.py first")
        return 1
    cw_path = Path(args.cw_manifest) if args.cw_manifest else None
    if cw_path is not None and not cw_path.is_file():
        print(
            f"[warn] conditioning_window manifest not found: {cw_path}; "
            f"/conditioning_window will render a hint page."
        )
    app = create_app(manifest_path, Path(args.outputs_root), cw_manifest_path=cw_path)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
