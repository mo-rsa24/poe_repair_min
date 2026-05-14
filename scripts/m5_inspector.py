"""Flask inspector for the m5 LoRA SDXL residual training chain.

Pre-rendered only. Sliders snap to indexed (epoch, lambda) values. When the
selected cell does not exist (some epochs only probed a coarser lambda grid),
a toast appears and the panels keep the previous frame.

Run on the cluster (binds 127.0.0.1 by default, intended for SSH tunnelling):
    python scripts/m5_inspector.py [--port 5050]

Then on your laptop:
    ssh -L 5050:localhost:5050 mscluster106
    open http://localhost:5050
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, send_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "outputs/m5_lora_sdxl/a_cat__x__a_dog/seed_42/inspector_manifest.json"
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
<h1>m5 residual inspector — {{ pair_slug }} · seed {{ seed }}</h1>

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
  <div class="panel">
    <h2>Mono (oracle reference)</h2>
    {% if mono_path %}
    <img id="img-mono" src="/img/{{ mono_path }}" alt="mono oracle">
    <div class="caption">epoch-independent · DDIM 50 steps · same seed/prompt</div>
    {% else %}
    <div class="caption">no mono.png in manifest</div>
    {% endif %}
  </div>
</div>

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


def create_app(manifest_path: Path, outputs_root: Path) -> Flask:
    manifest = json.loads(manifest_path.read_text())
    outputs_root_abs = outputs_root.resolve()

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(
            INDEX_HTML,
            pair_slug=manifest["pair_slug"],
            seed=manifest["seed"],
            epochs=manifest["epochs"],
            lambdas=manifest["lambdas"],
            mono_path=manifest.get("mono_path"),
            manifest_json=json.dumps(manifest),
        )

    @app.route("/manifest.json")
    def manifest_route():
        return jsonify(manifest)

    @app.route("/img/<path:rel>")
    def img(rel: str):
        # rel is relative to REPO_ROOT (manifest stores those paths).
        # Restrict to anything under outputs/ to prevent path traversal.
        target = (REPO_ROOT / rel).resolve()
        if not str(target).startswith(str(outputs_root_abs) + "/"):
            abort(403)
        if not target.is_file():
            abort(404)
        return send_file(target)

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--outputs-root", default=str(DEFAULT_OUTPUTS_ROOT))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}")
        print("run scripts/build_m5_manifest.py first")
        return 1
    app = create_app(manifest_path, Path(args.outputs_root))
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
