"""Substitute the scene key into the authored page, asserting the token appears exactly once."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "../where-the-product-lands.src.html")).read()
assert src.count("__KEY_JSON__") == 1, src.count("__KEY_JSON__")
key = json.load(open(os.path.join(HERE, "scene_key.json")))
keep = {k: key[k] for k in ["c2_even", "c2_adjacent", "c1", "b2_plain", "b2_divided", "support_pins", "one_blob_inside_3std"]}
out = src.replace("__KEY_JSON__", json.dumps(keep, separators=(",", ":")))
open(os.path.join(HERE, "../where-the-product-lands.html"), "w").write(out)
print("built", len(out), "bytes")
