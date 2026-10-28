"""Fill scene_key.json into page.src.html and write page.html (the published file)."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "page.src.html")).read()
assert src.count("__SCENE_KEY__") == 1, "placeholder must appear exactly once"
key = json.dumps(json.load(open(os.path.join(here, "scene_key.json"))), separators=(",", ":"))
open(os.path.join(here, "page.html"), "w").write(src.replace("__SCENE_KEY__", key))
print("page.html written,", len(key), "bytes of key")
