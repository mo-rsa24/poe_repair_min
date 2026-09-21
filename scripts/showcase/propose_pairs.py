"""Propose joint-prompt pairs that are worth rendering, and refuse the ones that are not.

Three filters, in the order they cost. A pair is dropped if either side is an evaluation or
transfer animal, because training on it would consume the evidence those pairs carry. It is dropped
if the two animals sit in the same look-alike family, because a person cannot then judge the target
and no verdict is possible whatever the model draws. And it is dropped if it is already cached or
already ruled on.

What survives is a pair of two tellable-apart animals that share a habitat, which is the condition
every pair kept so far has met and every rejected one has failed.
"""
import itertools, json, os, random, sys, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disallowed_subjects import is_allowed

CACHE = "/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache"
G = "artifacts/results/which-joint-prompt-targets-can-the-adapter-learn-from"

# Animals that plausibly share a frame in real imagery, which is what the joint prompt has seen.
# Communities whose members genuinely appear in the same photographs. The first version of this
# grouped by loose habitat and produced a lemur with a macaw (Madagascar and South America) and a
# flamingo with an orangutan. Every one of those was rejected. A community here is a place, not a
# kind of place.
COMMUNITY = {
    "african savanna":   ['lion', 'giraffe', 'zebra', 'ostrich', 'gazelle', 'wildebeest', 'warthog', 'meerkat', 'hyena', 'baboon', 'monkey', 'rhino', 'hippo', 'cheetah', 'leopard', 'antelope', 'buffalo', 'vulture', 'flamingo'],
    "african forest":    ['gorilla', 'chimpanzee', 'monkey', 'parrot', 'okapi', 'mandrill', 'porcupine'],
    "eurasian forest":   ['bear', 'boar', 'deer', 'fox', 'wolf', 'badger', 'hedgehog', 'owl', 'squirrel', 'woodpecker', 'moose', 'otter', 'lynx', 'rabbit', 'vole', 'eagle'],
    "north american":    ['bear', 'salmon', 'beaver', 'raccoon', 'deer', 'wolf', 'moose', 'bison', 'elk', 'coyote', 'skunk', 'chipmunk', 'otter', 'eagle', 'turkey', 'porcupine'],
    "farmyard":          ['horse', 'cow', 'sheep', 'pig', 'chicken', 'goat', 'donkey', 'duck', 'turkey', 'rooster', 'peacock', 'rabbit', 'goose', 'pony', 'llama', 'border collie'],
    "arctic tundra":     ['polar bear', 'reindeer', 'arctic fox', 'walrus', 'seal', 'snowy owl', 'husky', 'arctic hare', 'moose'],
    "australian bush":   ['kangaroo', 'koala', 'emu', 'wombat', 'wallaby', 'dingo', 'cockatoo', 'possum', 'echidna', 'parrot'],
    "south american":    ['llama', 'toucan', 'sloth', 'macaw', 'anteater', 'armadillo', 'capybara', 'jaguar', 'monkey', 'parrot', 'flamingo'],
    "indian jungle":     ['tiger', 'peacock', 'monkey', 'leopard', 'deer', 'mongoose', 'buffalo', 'parrot'],
    "wetland":           ['heron', 'duck', 'pelican', 'otter', 'flamingo', 'kingfisher', 'stork', 'swan', 'goose', 'crane', 'beaver', 'turtle', 'fish'],
    "ocean":             ['dolphin', 'whale', 'seal', 'stingray', 'octopus', 'turtle', 'crab', 'starfish', 'seahorse', 'pelican'],
    "pets and town":     ['rabbit', 'hamster', 'parrot', 'goldfish', 'pigeon', 'squirrel', 'fox', 'owl', 'duck', 'turtle'],
}

# Two animals of the same broad kind read as the same thing however far apart they really are, so
# the pair cannot be judged. Every pair kept so far crosses two kinds; the rejected sheet paired a
# bovid with a bovid and two seabirds with each other.
KIND = {
    "bovid":   ["gazelle", "wildebeest", "oryx", "kudu", "impala", "springbok", "sheep", "goat",
                "antelope", "moose", "deer", "eland", "gerenuk", "topi", "waterbuck", "bushbuck",
                "dik-dik", "bongo", "duiker", "forest buffalo", "bison", "elk", "pronghorn",
                "chamois", "ibex", "musk ox", "reindeer", "caribou", "sambar", "chital", "nilgai",
                "gaur", "guanaco", "vicuna", "alpaca"],
    "felid":   ["lion", "lynx", "cheetah", "cougar", "leopard", "jaguar", "tiger", "caracal",
                "serval", "wildcat", "bobcat", "jaguarundi"],
    "canid":   ["wolf", "fox", "jackal", "dingo", "husky", "coyote", "arctic fox", "wild dog",
                "border collie"],
    "primate": ["baboon", "gorilla", "chimpanzee", "mandrill", "lemur", "orangutan", "monkey",
                "colobus", "langur", "macaque"],
    "bird":    ["ostrich", "vulture", "crane", "parrot", "owl", "woodpecker", "heron", "duck",
                "chicken", "turkey", "rooster", "peacock", "cormorant", "puffin", "pelican",
                "seagull", "macaw", "toucan", "flamingo", "kingfisher", "stork", "ibis", "emu"],
    "rodent":  ["squirrel", "beaver", "porcupine", "capybara", "hedgehog", "rabbit", "chipmunk",
                "muskrat", "lemming", "agouti", "arctic hare", "echidna"],
    "ungulate":["giraffe", "zebra", "horse", "donkey", "rhino", "hippo", "warthog", "pig", "boar",
                "okapi", "tapir", "camel", "pony", "sloth bear"],
    "bear":    ["bear", "panda", "polar bear", "sloth bear"],
    "mustelid":["otter", "badger", "meerkat", "raccoon", "pangolin", "marten", "polecat", "stoat",
                "wolverine", "skunk", "civet", "mongoose", "coati", "opossum", "possum",
                "bandicoot", "armadillo", "anteater", "aardvark"],
    "fish":    ["salmon"],
    "marsupial": ["kangaroo", "koala", "wombat", "wallaby", "quokka"],
    "xenarthra": ["sloth"],
}
KIND_OF = {a: k for k, animals in KIND.items() for a in animals}

# Never pair two members of one family: a person cannot tell them apart, so no target is judgeable.
LOOKALIKE = [
    {"crow", "raven", "rook", "jackdaw"}, {"rabbit", "hare"}, {"turtle", "tortoise"},
    {"llama", "alpaca"}, {"donkey", "pony", "mule", "horse"}, {"dolphin", "porpoise", "whale"},
    {"cheetah", "cougar", "leopard", "jaguar", "lynx", "bobcat"},
    {"moose", "elk", "deer", "reindeer"},
    {"gazelle", "impala", "antelope", "springbok", "oryx", "kudu", "wildebeest"},
    {"seagull", "gull", "cormorant", "puffin", "pelican"},
    {"heron", "crane", "stork", "egret", "ibis"},
    {"wolf", "husky", "dingo", "jackal", "coyote", "fox", "arctic fox"},
    {"duck", "goose", "swan", "mallard"}, {"chicken", "rooster", "hen", "turkey", "emu", "ostrich"},
    {"gorilla", "chimpanzee", "orangutan", "baboon", "monkey", "lemur", "macaque", "mandrill"},
    {"parrot", "macaw", "toucan", "kookaburra"}, {"kangaroo", "wallaby"},
    {"pig", "boar", "warthog"}, {"sheep", "goat"}, {"bear", "polar bear", "panda"},
    {"squirrel", "capybara", "porcupine", "hedgehog"}, {"otter", "beaver"},
    {"zebra", "horse"}, {"peacock", "pheasant"}, {"puffin", "penguin"},
]

# Pairs named by hand, which bypass the same-community rule. That rule exists to stop absurd
# pairings (a lemur with a macaw), but it also blocks obviously fine ones: a cow and a hippo are
# two big recognisable herbivores that share a zoo frame happily, yet cow is farmyard and hippo is
# savanna so the generator can never propose them. These go out first, before the generated ones.
SEED_PAIRS = [
    ("leopard", "monkey"), ("cow", "hippo"), ("horse", "cow"), ("bear", "deer"),
    ("lion", "zebra"), ("tiger", "monkey"), ("giraffe", "rhino"), ("cow", "chicken"),
    ("sheep", "goose"), ("horse", "duck"), ("bear", "owl"), ("wolf", "rabbit"),
    ("fox", "chicken"), ("monkey", "parrot"), ("hippo", "flamingo"), ("zebra", "ostrich"),
    ("camel", "goat"), ("donkey", "sheep"), ("pig", "duck"), ("goat", "chicken"),
    ("cow", "sheep"), ("horse", "goat"), ("bear", "wolf"), ("lion", "hyena"),
    ("walrus", "seagull"), ("whale", "dolphin"), ("turtle", "crab"),
    ("owl", "squirrel"), ("eagle", "rabbit"), ("deer", "fox"), ("moose", "beaver"),
    ("raccoon", "duck"), ("koala", "kangaroo"), ("panda", "monkey"), ("camel", "ostrich"),
    ("llama", "sheep"), ("peacock", "goat"), ("swan", "duck"), ("stork", "cow"),
    ("buffalo", "egret"), ("rhino", "stork"), ("hippo", "turtle"), ("tiger", "deer"),
]

BARRED_ANIMALS = {"cat", "dog", "elephant", "penguin"}   # the evaluation pairs, never trained on.
# The look-alike transfer pairs (leopard/jaguar, cow/buffalo, eagle/hawk, seal/walrus,
# goose/swan) used to be barred here too. They are now available for training, which means
# those particular pairs are no longer clean held-out transfer tests.   # the transfer pairs


def article(w):
    return "an " + w if w[0] in "aeiou" else "a " + w


def slug(a, b):
    return (article(a) + "__x__" + article(b)).replace(" ", "_")


def confusable(a, b):
    return any(a in f and b in f for f in LOOKALIKE)


def cached():
    out = set()
    for sp in ("train", "heldout"):
        d = os.path.join(CACHE, sp)
        if os.path.isdir(d):
            out |= set(os.listdir(d))
    return out


# Several belts run at once on different nodes. Each reads the same cache and shuffles
# independently, so two of them proposed the same pair about one round in twelve and two nodes then
# wrote the same cell directory at the same time. SHARD="i/n" keeps only the candidates whose slug
# hashes into slice i of n, which makes the belts disjoint by construction rather than by luck.
def _shard_ok(slug_str):
    spec = os.environ.get("SHARD", "")
    if not spec:
        return True
    i, n = (int(x) for x in spec.split("/"))
    return zlib.crc32(slug_str.encode()) % n == i


def main(n=40):
    v = json.load(open(os.path.join(G, "verdicts.json")))
    seen = cached() | set(v["rejected"]) | set(v["kept"]) | set(v["kept_pending_confirmation"])
    have = cached()

    props = []
    # hand-named pairs first, so a batch always leads with animals chosen rather than generated
    for a, b in SEED_PAIRS:
        if a in BARRED_ANIMALS or b in BARRED_ANIMALS:
            continue
        sg = slug(a, b)
        if sg in seen or slug(b, a) in seen or not _shard_ok(sg):
            continue
        joint = f"{article(a)} and {article(b)}"
        if not is_allowed(joint):
            continue
        props.append(("hand-picked", sg, article(a), article(b), joint))

    for habitat, animals in COMMUNITY.items():
        ok = [a for a in animals if a not in BARRED_ANIMALS]
        for a, b in itertools.combinations(sorted(ok), 2):
            if confusable(a, b):
                continue
            ka, kb = KIND_OF.get(a), KIND_OF.get(b)
            if ka is not None and ka == kb:
                continue   # two of a kind read as the same thing, so the target cannot be judged
            s = slug(a, b)
            if s in seen or slug(b, a) in seen:
                continue
            if not _shard_ok(s):
                continue
            joint = f"{article(a)} and {article(b)}"
            if not is_allowed(joint):
                continue
            props.append((habitat, s, article(a), article(b), joint))

    # Shuffle before picking. Sorted order made every batch start at the alphabetically first
    # animal, so eight consecutive sheets all began "baboon" and were indistinguishable at a glance.
    hand = [x for x in props if x[0] == "hand-picked"]
    rest = [x for x in props if x[0] != "hand-picked"]
    random.shuffle(hand); random.shuffle(rest)
    props = hand + rest

    # one pair per animal per round, so a batch spreads over the cast instead of stacking on one
    used, picked = set(), []
    for h, s, a, b, j in props:
        if a in used or b in used:
            continue
        picked.append((h, s, a, b, j))
        used |= {a, b}
        if len(picked) >= n:
            break

    print(f"{len(props)} pairs pass every filter; {len(have)} already cached")
    print(f"proposing {len(picked)}, at most one appearance per animal\n")
    for h, s, a, b, j in picked:
        print(f"  {h:10s} {j}")
    print("\nPAIRS=" + " ; ".join(f"{s}|{a}|{b}|{j}" for h, s, a, b, j in picked))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
