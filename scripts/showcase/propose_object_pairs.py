"""Propose object x object joint-prompt candidates for the belt: two everyday objects sharing a
frame, where PoE tends to paint only one and the joint prompt paints both.

This is a different failure mode than the animal pool. There, two look-alike species compete for
one identity and the fix is picking pairs that are tellable apart. Here, both objects are already
easy to tell apart; PoE still drops one because its own "concept prior" for the scene picks a
single subject. `a typewriter and a cactus` is the worked example: solo prompts and mono both show
a clean typewriter and a clean cactus, but PoE draws a cactus rooted in the typewriter's paper
tray, one image with one dominant subject, not two independent scores agreeing to co-exist.

So the sampling rule inverts the animal one: draw from DIFFERENT categories (a kitchen item with a
transport item, a plant with an instrument) so the pair reads as "two things in a room" and never
as "two things that overlap in function", which is its own kind of confusable pair for objects
(a teapot and a kettle are not an interesting test, they are near-synonyms).

  python3 scripts/showcase/propose_object_pairs.py 9
"""
import itertools, json, os, random, sys, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disallowed_subjects import is_allowed

CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
G = "artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from"

# Every-day foreground objects, grouped by what kind of thing they are. A pair is drawn from two
# DIFFERENT categories, which is the opposite rule from the animal pool's same-community draw.
CATEGORY = {
    "furniture":      ["bathtub", "briefcase", "chessboard", "drum set", "suitcase", "birdcage",
                        "rocking chair", "grandfather clock", "wardrobe", "bookshelf", "ottoman",
                        "coat rack", "vanity mirror", "filing cabinet", "bunk bed"],
    "kitchen":        ["ceramic bowl", "cast iron pan", "microwave", "teapot", "cutting board",
                        "espresso machine", "toaster", "colander", "rolling pin", "mixing bowl",
                        "bread box", "spice rack", "wine rack", "waffle iron"],
    "garden_plant":    ["cactus", "potted plant", "watering can", "hay bale", "birdhouse",
                        "wheelbarrow", "garden gnome", "flower pot", "bonsai tree", "beehive",
                        "sunflower in a vase", "terracotta planter"],
    "office_writing": ["typewriter", "lab microscope", "desk fan", "lantern", "fountain pen",
                        "globe", "stapler", "rotary phone", "abacus", "magnifying glass",
                        "inkwell", "quill pen", "letter opener", "paperweight"],
    "textile_soft":   ["feather pillow", "wool scarf", "winter coat", "hammock", "quilt",
                        "throw blanket", "beanbag chair", "knitted sweater", "canvas tent",
                        "sleeping bag", "woven rug"],
    "transport_small": ["bicycle", "skateboard", "wheelbarrow cart", "scooter", "canoe paddle",
                         "sled", "unicycle", "roller skates", "wagon", "surfboard"],
    "instrument":     ["accordion", "cello", "trumpet", "xylophone", "banjo", "harmonica",
                        "tambourine", "grand piano", "ukulele", "bagpipes"],
    "container":      ["suitcase trunk", "wooden crate", "glass jar", "picnic basket",
                        "toolbox", "birdcage empty", "mailbox", "treasure chest", "cooler box",
                        "hatbox"],
    "outdoor_object": ["streetlamp", "fire hydrant", "mailbox post", "park bench", "weathervane",
                        "wind chime", "sundial", "scarecrow", "bird bath", "picket fence"],
    "electronics":    ["rotary television", "vintage radio", "record player", "film projector",
                        "typewriter ribbon spool", "gramophone", "polaroid camera",
                        "reel to reel tape deck", "cassette player"],
}
# Near-synonyms or functionally overlapping items read as "the same thing twice" the way a
# look-alike animal pair does; excluded the same way.
CONFUSABLE = [
    ("teapot", "espresso machine"), ("wardrobe", "filing cabinet"),
    ("rotary phone", "rotary television"), ("suitcase", "suitcase trunk"),
    ("birdcage", "birdcage empty"), ("bicycle", "unicycle"),
    ("grand piano", "accordion"), ("mailbox", "mailbox post"),
    ("vintage radio", "gramophone"), ("wooden crate", "toolbox"),
]
CONFUSABLE = {frozenset(p) for p in CONFUSABLE}


def slug(a, b):
    return "%s__x__%s" % (a.replace(" ", "_"), b.replace(" ", "_"))


def article(w):
    return "an %s" % w if w[0] in "aeiou" else "a %s" % w


def cached():
    out = set()
    for sp in ("train", "heldout"):
        d = os.path.join(CACHE, sp)
        if os.path.isdir(d):
            out |= set(os.listdir(d))
    return out


def _shard_ok(slug_str):
    spec = os.environ.get("SHARD", "")
    if not spec:
        return True
    i, n = (int(x) for x in spec.split("/"))
    return zlib.crc32(slug_str.encode()) % n == i


def main(n=40):
    v_path = os.path.join(G, "object_verdicts.json")
    seen = cached()
    if os.path.exists(v_path):
        v = json.load(open(v_path))
        seen |= set(v.get("rejected", [])) | set(v.get("kept", []))
    have = cached()

    cats = list(CATEGORY.items())
    props = []
    for (ka, la), (kb, lb) in itertools.combinations(cats, 2):
        for a, b in itertools.product(la, lb):
            if frozenset((a, b)) in CONFUSABLE:
                continue
            s = slug(a, b)
            if s in seen or slug(b, a) in seen:
                continue
            if not _shard_ok(s):
                continue
            joint = "%s and %s" % (article(a), article(b))
            if not is_allowed(joint):
                continue
            props.append((ka, kb, s, article(a), article(b), joint))

    random.shuffle(props)
    used, picked = set(), []
    for ka, kb, s, a, b, j in props:
        if a in used or b in used:
            continue
        picked.append((ka, kb, s, a, b, j))
        used |= {a, b}
        if len(picked) >= n:
            break

    print(f"{len(props)} object pairs pass every filter; {len(have)} cells already cached")
    print(f"proposing {len(picked)}, at most one appearance per object\n")
    for ka, kb, s, a, b, j in picked:
        print(f"  {ka:16s} x {kb:16s} {j}")
    print("\nPAIRS=" + " ; ".join(f"{s}|{a}|{b}|{j}" for ka, kb, s, a, b, j in picked))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
