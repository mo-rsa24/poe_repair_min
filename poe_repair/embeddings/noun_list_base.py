"""Base noun lists for synthesizer training.

Curated to be (a) common, (b) typically single-token under CLIP-L, (c)
visually concrete. Plus a small supercategory bucket for G6-style
same-supercategory collisions used for oversampling.
"""

BASE_NOUNS = [
    # animals
    "cat", "dog", "horse", "lion", "tiger", "wolf", "fox", "bear",
    "elephant", "deer", "rabbit", "mouse", "squirrel", "monkey", "owl",
    "eagle", "parrot", "duck", "goose", "swan", "penguin", "flamingo",
    "pigeon", "sparrow", "raven", "crow", "shark", "dolphin", "whale",
    "octopus", "jellyfish", "crab", "lobster", "salmon", "trout", "frog",
    "lizard", "snake", "turtle", "butterfly", "bee", "ant", "spider",
    # objects
    "chair", "table", "desk", "couch", "bed", "bookshelf", "bookcase",
    "lamp", "candle", "vase", "mirror", "clock", "umbrella", "suitcase",
    "backpack", "bag", "hat", "scarf", "glove", "boot", "sneaker",
    "sandal", "tie", "watch", "ring", "necklace", "bracelet",
    "typewriter", "computer", "laptop", "keyboard", "mouse pad", "monitor",
    "phone", "camera", "microphone", "speaker", "radio", "television",
    "microwave", "oven", "stove", "fridge", "dishwasher", "blender",
    "kettle", "teapot", "cup", "mug", "wine glass", "plate", "bowl",
    "fork", "spoon", "knife", "skillet", "saucepan",
    # buildings & structures
    "barn", "castle", "lighthouse", "windmill", "bridge", "tower",
    "cathedral", "pagoda", "temple", "mosque", "skyscraper", "house",
    "cabin", "tent", "yurt", "igloo", "shed",
    # vehicles
    "bicycle", "motorcycle", "skateboard", "scooter", "car", "truck",
    "bus", "train", "tram", "tractor", "boat", "sailboat", "kayak",
    "canoe", "submarine", "airplane", "helicopter", "balloon",
    # plants & food
    "cactus", "fern", "rose", "tulip", "sunflower", "daisy", "tree",
    "oak", "pine", "palm tree", "bamboo", "grass", "moss", "mushroom",
    "apple", "banana", "orange", "pear", "peach", "plum", "strawberry",
    "blueberry", "raspberry", "watermelon", "pumpkin", "carrot",
    "potato", "tomato", "broccoli", "lettuce", "onion", "garlic",
    "bread", "cake", "donut", "cookie", "pizza", "pasta", "sushi",
    "burger", "taco", "salad", "soup", "ice cream", "chocolate",
    # nature & scenes
    "mountain", "hill", "valley", "river", "lake", "pond", "waterfall",
    "ocean", "wave", "beach", "desert", "forest", "meadow", "field",
    "glacier", "iceberg", "volcano", "cave", "canyon", "cliff", "dune",
    # weather / scenes
    "rain", "snow", "fog", "thunderstorm", "rainbow", "sunset", "sunrise",
    # fictional / mythical
    "dragon", "unicorn", "phoenix", "mermaid", "ghost", "robot", "alien",
    # clothing styles
    "tuxedo", "kimono", "hoodie", "raincoat", "wedding dress",
    # tools / hobby
    "guitar", "piano", "violin", "drum", "trumpet", "saxophone", "harmonica",
    "easel", "paintbrush", "chisel", "hammer", "saw", "wrench", "screwdriver",
    # styles (Group 2)
    "oil painting style", "watercolor style", "pencil drawing style",
    "charcoal drawing style", "claymation style", "mosaic style",
    "stained glass style", "pixel art style", "sketch style",
    # adjectives that act as concepts (Group 5)
    "fluffy", "striped", "transparent", "metallic", "wooden", "glass",
    "small", "huge", "tiny",
]

SUPERCATEGORY_NOUNS = {
    "carnivore_mammals": ["cat", "dog", "lion", "tiger", "wolf", "fox", "bear", "leopard", "panther"],
    "domestic_pets": ["cat", "dog", "rabbit", "hamster", "parrot", "goldfish"],
    "horned_mammals": ["cow", "horse", "deer", "elk", "moose", "bison", "antelope", "ram"],
    "birds_of_prey": ["eagle", "hawk", "falcon", "owl", "vulture"],
    "songbirds": ["sparrow", "robin", "finch", "thrush", "warbler"],
    "wading_birds": ["flamingo", "heron", "stork", "egret", "ibis"],
    "marine_mammals": ["dolphin", "whale", "seal", "walrus", "manatee"],
    "fish": ["salmon", "trout", "bass", "tuna", "shark"],
    "vehicles_road": ["car", "truck", "bus", "tram", "motorcycle", "bicycle"],
    "vehicles_water": ["boat", "sailboat", "kayak", "canoe", "yacht", "ship"],
    "vehicles_air": ["airplane", "helicopter", "balloon", "glider", "blimp"],
    "stringed_instruments": ["guitar", "violin", "cello", "harp", "ukulele", "banjo"],
    "wind_instruments": ["trumpet", "saxophone", "flute", "clarinet", "harmonica"],
    "kitchen_appliances": ["microwave", "oven", "stove", "fridge", "blender", "kettle"],
    "drinkware": ["cup", "mug", "wine glass", "tumbler", "goblet"],
    "fruits_round": ["apple", "orange", "peach", "plum", "watermelon"],
    "fruits_berries": ["strawberry", "blueberry", "raspberry", "blackberry"],
    "structures": ["tower", "lighthouse", "windmill", "skyscraper", "cathedral"],
    "felines": ["cat", "lion", "tiger", "leopard", "panther", "lynx"],
    "canines": ["dog", "wolf", "fox", "coyote", "jackal"],
}
