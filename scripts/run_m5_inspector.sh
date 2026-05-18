#!/usr/bin/env bash
# Launch the m5 residual inspector Flask app.
# Binds 127.0.0.1:5050 by default — reach it from your laptop via SSH tunnel.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-5050}"
HOST="${HOST:-127.0.0.1}"
PYTHON="${PYTHON:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"

echo "starting m5 inspector on ${HOST}:${PORT}"
echo "  from laptop:  ssh -L ${PORT}:localhost:${PORT} $(hostname)"
echo "  then open:    http://localhost:${PORT}"
echo

exec "$PYTHON" scripts/m5_inspector.py --host "$HOST" --port "$PORT" "$@"
