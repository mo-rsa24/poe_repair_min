"""Substitute the scene key into the authored page, asserting the token appears exactly once."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "../parts-world.src.html")).read()
assert src.count("__KEY_JSON__") == 1, src.count("__KEY_JSON__")
key = json.load(open(os.path.join(HERE, "scene_key.json")))
keep = {k: key[k] for k in ["walk", "a_dog_head_tight", "hand_check", "pixel_logZ", "c_component_means", "c_gap", "d_noised_head", "vstops", "astops"]}
out = src.replace("__KEY_JSON__", json.dumps(keep, separators=(",", ":")))
open(os.path.join(HERE, "../parts-world.html"), "w").write(out)
print("built", len(out), "bytes")
