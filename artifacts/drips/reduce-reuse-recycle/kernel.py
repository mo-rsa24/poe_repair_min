"""Run cells of simple_distributions.ipynb in one kernel kept alive across walk rounds.

    python kernel.py start                 connect to the kernel named by kernel.json, set up figure saving
    python kernel.py run 1 2 3             run those cells (0-based JSON index) in order
    python kernel.py code "print(1)"       run a one-off snippet in the same kernel
    python kernel.py stop

Figures are saved as fig-<cell>-<n>.png under figures/ beside this file.
A cell whose source is overridden in OVERRIDES runs the override instead, and says so.
"""
import json
import pathlib
import subprocess
import sys

from jupyter_client import BlockingKernelClient

HERE = pathlib.Path(__file__).parent
import os
# KERNEL=gpu addresses a second kernel (kernel-gpu.json) so a CPU and a GPU run can sit side by side.
CONN = HERE / ("kernel-gpu.json" if os.environ.get("KERNEL") == "gpu" else "kernel.json")
CLONE = "/home/molef/PhD/reduce-reuse-recycle"
# Cell numbers are the original notebook's at 81177d9, read from git, so an edited working copy
# (the reader runs their own cells there) never shifts them.
NB_JSON = json.loads(subprocess.run(
    ["git", "-C", CLONE, "show", "81177d9:notebooks/simple_distributions.ipynb"],
    capture_output=True, text=True, check=True).stdout)
FIGS = HERE / "figures"

CELLS = ["".join(c["source"]) for c in NB_JSON["cells"]]

# Cell 3 imports pandas, tensorflow and tensorflow_datasets, none of which the notebook uses
# (0 uses of pd., tf. or tfds. in any cell). Everything else in the cell is kept.
OVERRIDES = {
    3: "\n".join(line for line in CELLS[3].split("\n")
                  if not any(m in line for m in ("pandas", "tensorflow"))),
    # jax 0.11 renamed jnp.clip's a_min/a_max to min/max and dropped the old names.
    12: CELLS[12].replace("a_min=", "min="),
}
NOTES = {3: "tensorflow and pandas imports dropped", 12: "jnp.clip a_min= renamed min="}

SETUP = (
    "import matplotlib; matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as _plt, pathlib as _pl\n"
    f"_FIGS = _pl.Path({str(FIGS)!r}); _FIGS.mkdir(exist_ok=True)\n"
    "def _save_figs(tag):\n"
    "    for k, n in enumerate(_plt.get_fignums()):\n"
    "        p = _FIGS / f'fig-{tag}-{k}.png'; _plt.figure(n).savefig(p, dpi=90, bbox_inches='tight'); print('[figure]', p)\n"
    "    _plt.close('all')\n"
    # Each plt.show() in a cell saves and closes the current figure, as a notebook front end would
    # show it, so successive plots never pile into one image.
    "_SHOW_N = [0]\n"
    "def _show(*a, **k):\n"
    "    if _plt.get_fignums():\n"
    "        _SHOW_N[0] += 1; p = _FIGS / f'show-{_SHOW_N[0]:04d}.png'\n"
    "        _plt.gcf().savefig(p, dpi=80, bbox_inches='tight'); _plt.close(_plt.gcf())\n"
    "_plt.show = _show\n"
)


def client():
    c = BlockingKernelClient(connection_file=str(CONN))
    c.load_connection_file()
    c.start_channels()
    c.wait_for_ready(timeout=60)
    return c


def execute(c, code, tag):
    msg_id = c.execute(code, store_history=True)
    ok = True
    while True:
        msg = c.get_iopub_msg(timeout=None)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        t, ct = msg["msg_type"], msg["content"]
        if t == "stream":
            sys.stdout.write(ct["text"])
        elif t in ("execute_result", "display_data"):
            txt = ct["data"].get("text/plain")
            if txt and "Figure" not in txt:
                print(txt)
        elif t == "error":
            ok = False
            print(f"[error in {tag}] {ct['ename']}: {ct['evalue']}")
            print("\n".join(ct["traceback"][-6:]))
        elif t == "status" and ct["execution_state"] == "idle":
            break
    return ok


def main():
    cmd = sys.argv[1]
    if cmd == "start":
        # The kernel itself is launched beforehand, detached, with
        # nohup /home/molef/PhD/reduce-reuse-recycle/.venv-2d/bin/python -m ipykernel_launcher -f kernel.json
        c = client()
        execute(c, SETUP, "setup")
        print("kernel ready:", CONN)
        return
    c = client()
    if cmd == "stop":
        c.shutdown()
        print("kernel stopped")
    elif cmd == "code":
        execute(c, sys.argv[2], "snippet")
        execute(c, "_save_figs('snippet')", "figs")
    elif cmd == "run":
        for i in map(int, sys.argv[2:]):
            src = OVERRIDES.get(i, CELLS[i])
            note = f" (overridden: {NOTES[i]})" if i in OVERRIDES else ""
            print(f"=== cell {i}{note}")
            import time
            t0 = time.time()
            ok = execute(c, src, f"cell {i}")
            print(f"[cell {i} wall time {time.time() - t0:.1f} s]", flush=True)
            if not ok:
                print(f"stopped at cell {i}")
                break
            execute(c, f"_save_figs('{i}')", "figs")


if __name__ == "__main__":
    main()
