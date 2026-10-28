"""Run a walk document's python blocks, in order, as one script, so every printed number in it is checked.

    python -m rebuild.run_walk_doc artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md --dir /tmp/walk01

The script is written to <dir>/walk.py and run there with the co3 interpreter; its output is printed.
"""
import argparse
import os
import re
import subprocess
import sys


def blocks(md):
    return re.findall(r"```python\n(.*?)```", open(md).read(), flags=re.S)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc")
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = "import sys; sys.path.insert(0, %r)\n" % repo + "\n".join(blocks(a.doc))
    path = os.path.join(a.dir, "walk.py")
    open(path, "w").write(code)
    r = subprocess.run([sys.executable, path], cwd=a.dir, capture_output=True, text=True)
    print(r.stdout[-20000:]); print(r.stderr[-4000:], file=sys.stderr)
    sys.exit(r.returncode)
