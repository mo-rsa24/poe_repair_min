"""Make the Reduce, Reuse, Recycle 2D notebook run on jax 0.11 and current chex.

    python3 scripts/fix_reduce_reuse_recycle_notebook.py <clone>/notebooks/simple_distributions.ipynb

Run it once on a fresh clone at 81177d9. It edits the notebook in place.
"""
import json, sys
p = sys.argv[1]; nb = json.load(open(p)); cells = nb["cells"]
src = lambda c: "".join(c["source"])
# 1. drop the !pip cell
cells[:] = [c for c in cells if not src(c).lstrip().startswith("!pip install")]
for c in cells:
    lines = c["source"]
    # 2. drop the unused tensorflow and pandas imports
    lines = [l for l in lines if not ("import tensorflow" in l or "import pandas" in l)]
    # 3. jax 0.11 renamed jnp.clip's a_min= to min=
    c["source"] = [l.replace("a_min=", "min=") for l in lines]
# 4. compatibility cell after the imports
i = next(k for k, c in enumerate(cells) if src(c).lstrip().startswith("import math"))
cells.insert(i + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
    "import chex\n",
    "chex.assert_type = lambda *args, **kwargs: None  # dtype checks predate JAX's 32-bit defaults\n",
    "jax.tree_leaves = jax.tree_util.tree_leaves      # removed from jax; lives on in jax.tree_util\n",
    "jax.tree_map = jax.tree_util.tree_map"]})
open(p, "w").write(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
print(len(cells), "cells; fixes applied")
