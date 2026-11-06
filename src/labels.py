cifar100_classes = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]
# CIFAR-10 classes
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

from pathlib import Path

imagenet_classes = [
    "abacus", "abaya", "academic_gown", "accordion", "acorn", "acorn_squash", "acoustic_guitar", "admiral",
    "affenpinscher", "afghan_hound", "african_chameleon", "african_crocodile", "african_elephant", "african_grey",
    "african_hunting_dog", "agama", "agaric", "aircraft_carrier", "airedale", "airliner", "airship", "albatross",
    "alligator_lizard", "alp", "altar", "ambulance", "american_alligator", "american_black_bear", "american_chameleon",
    "american_coot", "american_egret", "american_lobster", "american_staffordshire_terrier", "amphibian",
    "analog_clock", "anemone_fish", "angora", "ant", "apiary", "appenzeller", "apron", "arabian_camel", "arctic_fox",
    "armadillo", "artichoke", "ashcan", "assault_rifle", "australian_terrier", "axolotl", "baboon", "backpack",
    "badger", "bagel", "bakery", "balance_beam", "bald_eagle", "balloon", "ballplayer", "ballpoint", "banana",
    "band_aid", "banded_gecko", "banjo", "bannister", "barbell", "barber_chair", "barbershop", "barn", "barn_spider",
    "barometer", "barracouta", "barrel", "barrow", "baseball", "basenji", "basketball", "basset", "bassinet", "bassoon",
    "bathing_cap", "bath_towel", "bathtub", "beach_wagon", "beacon", "beagle", "beaker", "bearskin", "beaver",
    "bedlington_terrier", "bee", "bee_eater", "beer_bottle", "beer_glass", "bell_cote", "bell_pepper",
    "bernese_mountain_dog", "bib", "bicycle_built_for_two", "bighorn", "bikini", "binder", "binoculars", "birdhouse",
    "bison", "bittern", "black_and_gold_garden_spider", "black_and_tan_coonhound", "black_footed_ferret",
    "black_grouse", "black_stork", "black_swan", "black_widow", "blenheim_spaniel", "bloodhound", "bluetick",
    "boa_constrictor", "boathouse", "bobsled", "bolete", "bolo_tie", "bonnet", "bookcase", "book_jacket", "bookshop",
    "border_collie", "border_terrier", "borzoi", "boston_bull", "bottlecap", "bouvier_des_flandres", "bow", "bow_tie",
    "boxer", "box_turtle", "brabancon_griffon", "brain_coral", "brambling", "brass", "brassiere", "breakwater",
    "breastplate", "briard", "brittany_spaniel", "broccoli", "broom", "brown_bear", "bubble", "bucket", "buckeye",
    "buckle", "bulbul", "bulletproof_vest", "bullet_train", "bullfrog", "bull_mastiff", "burrito", "bustard",
    "butcher_shop", "butternut_squash", "cab", "cabbage_butterfly", "cairn", "caldron", "candle", "cannon", "canoe",
    "can_opener", "capuchin", "carbonara", "cardigan", "cardigan_welsh_corgi", "cardoon", "car_mirror", "carousel",
    "carpenter_s_kit", "carton", "car_wheel", "cash_machine", "cassette", "cassette_player", "castle", "catamaran",
    "cauliflower", "cd_player", "cello", "cellular_telephone", "centipede", "chain", "chainlink_fence", "chain_mail",
    "chain_saw", "chambered_nautilus", "cheeseburger", "cheetah", "chesapeake_bay_retriever", "chest", "chickadee",
    "chiffonier", "chihuahua", "chime", "chimpanzee", "china_cabinet", "chiton", "chocolate_sauce", "chow",
    "christmas_stocking", "church", "cicada", "cinema", "cleaver", "cliff", "cliff_dwelling", "cloak", "clog",
    "clumber", "cock", "cocker_spaniel", "cockroach", "cocktail_shaker", "coffee_mug", "coffeepot", "coho", "coil",
    "collie", "colobus", "combination_lock", "comic_book", "common_iguana", "common_newt", "computer_keyboard", "conch",
    "confectionery", "consomme", "container_ship", "convertible", "coral_fungus", "coral_reef", "corkscrew", "corn",
    "cornet", "coucal", "cougar", "cowboy_boot", "cowboy_hat", "coyote", "cradle", "crane", "crane_bird",
    "crash_helmet", "crate", "crayfish", "crib", "cricket", "crock_pot", "croquet_ball", "crossword_puzzle", "crutch",
    "cucumber", "cuirass", "cup", "curly_coated_retriever", "custard_apple", "daisy", "dalmatian", "dam", "damselfly",
    "dandie_dinmont", "desk", "desktop_computer", "dhole", "dial_telephone", "diamondback", "diaper", "digital_clock",
    "digital_watch", "dingo", "dining_table", "dishrag", "dishwasher", "disk_brake", "doberman", "dock", "dogsled",
    "dome", "doormat", "dough", "dowitcher", "dragonfly", "drake", "drilling_platform", "drum", "drumstick", "dugong",
    "dumbbell", "dung_beetle", "dungeness_crab", "dutch_oven", "ear", "earthstar", "echidna", "eel", "eft", "eggnog",
    "egyptian_cat", "electric_fan", "electric_guitar", "electric_locomotive", "electric_ray", "english_foxhound",
    "english_setter", "english_springer", "entertainment_center", "entlebucher", "envelope", "eskimo_dog", "espresso",
    "espresso_maker", "european_fire_salamander", "european_gallinule", "face_powder", "feather_boa", "fiddler_crab",
    "fig", "file", "fireboat", "fire_engine", "fire_screen", "flagpole", "flamingo", "flat_coated_retriever",
    "flatworm", "flute", "fly", "folding_chair", "football_helmet", "forklift", "fountain", "fountain_pen",
    "four_poster", "fox_squirrel", "freight_car", "french_bulldog", "french_horn", "french_loaf", "frilled_lizard",
    "frying_pan", "fur_coat", "gar", "garbage_truck", "garden_spider", "garter_snake", "gasmask", "gas_pump", "gazelle",
    "german_shepherd", "german_short_haired_pointer", "geyser", "giant_panda", "giant_schnauzer", "gibbon",
    "gila_monster", "goblet", "go_kart", "golden_retriever", "goldfinch", "goldfish", "golf_ball", "golfcart",
    "gondola", "gong", "goose", "gordon_setter", "gorilla", "gown", "grand_piano", "granny_smith", "grasshopper",
    "great_dane", "greater_swiss_mountain_dog", "great_grey_owl", "great_pyrenees", "great_white_shark", "greenhouse",
    "green_lizard", "green_mamba", "green_snake", "grey_fox", "grey_whale", "grille", "grocery_store", "groenendael",
    "groom", "ground_beetle", "guacamole", "guenon", "guillotine", "guinea_pig", "gyromitra", "hair_slide",
    "hair_spray", "half_track", "hammer", "hammerhead", "hamper", "hamster", "hand_blower", "hand_held_computer",
    "handkerchief", "hard_disc", "hare", "harmonica", "harp", "hartebeest", "harvester", "harvestman", "hatchet", "hay",
    "head_cabbage", "hen", "hen_of_the_woods", "hermit_crab", "hip", "hippopotamus", "hog", "hognose_snake", "holster",
    "home_theater", "honeycomb", "hook", "hoopskirt", "horizontal_bar", "hornbill", "horned_viper", "horse_cart",
    "hotdog", "hot_pot", "hourglass", "house_finch", "howler_monkey", "hummingbird", "hyena", "ibex", "ibizan_hound",
    "ice_bear", "ice_cream", "ice_lolly", "impala", "indian_cobra", "indian_elephant", "indigo_bunting", "indri",
    "ipod", "irish_setter", "irish_terrier", "irish_water_spaniel", "irish_wolfhound", "iron", "isopod",
    "italian_greyhound", "jacamar", "jackfruit", "jack_o_lantern", "jaguar", "japanese_spaniel", "jay", "jean", "jeep",
    "jellyfish", "jersey", "jigsaw_puzzle", "jinrikisha", "joystick", "junco", "keeshond", "kelpie",
    "kerry_blue_terrier", "killer_whale", "kimono", "king_crab", "king_penguin", "king_snake", "kite", "kit_fox",
    "knee_pad", "knot", "koala", "komodo_dragon", "komondor", "kuvasz", "lab_coat", "labrador_retriever", "lacewing",
    "ladle", "ladybug", "lakeland_terrier", "lakeside", "lampshade", "langur", "laptop", "lawn_mower", "leaf_beetle",
    "leafhopper", "leatherback_turtle", "lemon", "lens_cap", "leonberg", "leopard", "lesser_panda", "letter_opener",
    "lhasa", "library", "lifeboat", "lighter", "limousine", "limpkin", "liner", "lion", "lionfish", "lipstick",
    "little_blue_heron", "llama", "loafer", "loggerhead", "long_horned_beetle", "lorikeet", "lotion", "loudspeaker",
    "loupe", "lumbermill", "lycaenid", "lynx", "macaque", "macaw", "madagascar_cat", "magnetic_compass", "magpie",
    "mailbag", "mailbox", "maillot", "malamute", "malinois", "maltese_dog", "manhole_cover", "mantis", "maraca",
    "marimba", "marmoset", "marmot", "mashed_potato", "mask", "matchstick", "maypole", "maze", "measuring_cup",
    "meat_loaf", "medicine_chest", "meerkat", "megalith", "menu", "mexican_hairless", "microphone", "microwave",
    "military_uniform", "milk_can", "miniature_pinscher", "miniature_poodle", "miniature_schnauzer", "minibus",
    "miniskirt", "minivan", "mink", "missile", "mitten", "mixing_bowl", "mobile_home", "model_t", "modem", "monarch",
    "monastery", "mongoose", "monitor", "moped", "mortar", "mortarboard", "mosque", "mosquito_net", "motor_scooter",
    "mountain_bike", "mountain_tent", "mouse", "mousetrap", "moving_van", "mud_turtle", "mushroom", "muzzle", "nail",
    "neck_brace", "necklace", "nematode", "newfoundland", "night_snake", "nipple", "norfolk_terrier",
    "norwegian_elkhound", "norwich_terrier", "notebook", "obelisk", "oboe", "ocarina", "odometer", "oil_filter",
    "old_english_sheepdog", "orange", "orangutan", "organ", "oscilloscope", "ostrich", "otter", "otterhound",
    "overskirt", "ox", "oxcart", "oxygen_mask", "oystercatcher", "packet", "paddle", "paddlewheel", "padlock",
    "paintbrush", "pajama", "palace", "panpipe", "paper_towel", "papillon", "parachute", "parallel_bars", "park_bench",
    "parking_meter", "partridge", "passenger_car", "patas", "patio", "pay_phone", "peacock", "pedestal", "pekinese",
    "pelican", "pembroke", "pencil_box", "pencil_sharpener", "perfume", "persian_cat", "petri_dish", "photocopier",
    "pick", "pickelhaube", "picket_fence", "pickup", "pier", "piggy_bank", "pill_bottle", "pillow", "pineapple",
    "ping_pong_ball", "pinwheel", "pirate", "pitcher", "pizza", "plane", "planetarium", "plastic_bag", "plate",
    "plate_rack", "platypus", "plow", "plunger", "polaroid_camera", "pole", "polecat", "police_van", "pomegranate",
    "pomeranian", "poncho", "pool_table", "pop_bottle", "porcupine", "pot", "potpie", "potter_s_wheel", "power_drill",
    "prairie_chicken", "prayer_rug", "pretzel", "printer", "prison", "proboscis_monkey", "projectile", "projector",
    "promontory", "ptarmigan", "puck", "puffer", "pug", "punching_bag", "purse", "quail", "quill", "quilt", "racer",
    "racket", "radiator", "radio", "radio_telescope", "rain_barrel", "ram", "rapeseed", "recreational_vehicle",
    "red_backed_sandpiper", "redbone", "red_breasted_merganser", "red_fox", "redshank", "red_wine", "red_wolf", "reel",
    "reflex_camera", "refrigerator", "remote_control", "restaurant", "revolver", "rhinoceros_beetle",
    "rhodesian_ridgeback", "rifle", "ringlet", "ringneck_snake", "robin", "rock_beauty", "rock_crab", "rocking_chair",
    "rock_python", "rotisserie", "rottweiler", "rubber_eraser", "ruddy_turnstone", "ruffed_grouse", "rugby_ball",
    "rule", "running_shoe", "safe", "safety_pin", "saint_bernard", "saltshaker", "saluki", "samoyed", "sandal",
    "sandbar", "sarong", "sax", "scabbard", "scale", "schipperke", "school_bus", "schooner", "scoreboard", "scorpion",
    "scotch_terrier", "scottish_deerhound", "screen", "screw", "screwdriver", "scuba_diver", "sea_anemone",
    "sea_cucumber", "sea_lion", "sealyham_terrier", "seashore", "sea_slug", "sea_snake", "seat_belt", "sea_urchin",
    "sewing_machine", "shetland_sheepdog", "shield", "shih_tzu", "shoe_shop", "shoji", "shopping_basket",
    "shopping_cart", "shovel", "shower_cap", "shower_curtain", "siamang", "siamese_cat", "siberian_husky", "sidewinder",
    "silky_terrier", "ski", "ski_mask", "skunk", "sleeping_bag", "slide_rule", "sliding_door", "slot", "sloth_bear",
    "slug", "snail", "snorkel", "snow_leopard", "snowmobile", "snowplow", "soap_dispenser", "soccer_ball", "sock",
    "soft_coated_wheaten_terrier", "solar_dish", "sombrero", "sorrel", "soup_bowl", "space_bar", "space_heater",
    "space_shuttle", "spaghetti_squash", "spatula", "speedboat", "spider_monkey", "spider_web", "spindle",
    "spiny_lobster", "spoonbill", "sports_car", "spotlight", "spotted_salamander", "squirrel_monkey",
    "staffordshire_bullterrier", "stage", "standard_poodle", "standard_schnauzer", "starfish", "steam_locomotive",
    "steel_arch_bridge", "steel_drum", "stethoscope", "stingray", "stinkhorn", "stole", "stone_wall", "stopwatch",
    "stove", "strainer", "strawberry", "streetcar", "street_sign", "stretcher", "studio_couch", "stupa", "sturgeon",
    "submarine", "suit", "sulphur_butterfly", "sulphur_crested_cockatoo", "sundial", "sunglass", "sunglasses",
    "sunscreen", "suspension_bridge", "sussex_spaniel", "swab", "sweatshirt", "swimming_trunks", "swing", "switch",
    "syringe", "tabby", "table_lamp", "tailed_frog", "tank", "tank_suit", "tape_player", "tarantula", "teapot", "teddy",
    "television", "tench", "tennis_ball", "terrapin", "thatch", "theater_curtain", "thimble", "three_toed_sloth",
    "thresher", "throne", "thunder_snake", "tibetan_mastiff", "tibetan_terrier", "tick", "tiger", "tiger_beetle",
    "tiger_cat", "tiger_shark", "tile_roof", "timber_wolf", "titi", "toaster", "tobacco_shop", "toilet_seat",
    "toilet_tissue", "torch", "totem_pole", "toucan", "tow_truck", "toy_poodle", "toyshop", "toy_terrier", "tractor",
    "traffic_light", "trailer_truck", "tray", "tree_frog", "trench_coat", "triceratops", "tricycle", "trifle",
    "trilobite", "trimaran", "tripod", "triumphal_arch", "trolleybus", "trombone", "tub", "turnstile", "tusker",
    "typewriter_keyboard", "umbrella", "unicycle", "upright", "vacuum", "valley", "vase", "vault", "velvet",
    "vending_machine", "vestment", "viaduct", "vine_snake", "violin", "vizsla", "volcano", "volleyball", "vulture",
    "waffle_iron", "walker_hound", "walking_stick", "wallaby", "wall_clock", "wallet", "wardrobe", "warplane",
    "warthog", "washbasin", "washer", "water_bottle", "water_buffalo", "water_jug", "water_ouzel", "water_snake",
    "water_tower", "weasel", "web_site", "weevil", "weimaraner", "welsh_springer_spaniel",
    "west_highland_white_terrier", "whippet", "whiptail", "whiskey_jug", "whistle", "white_stork", "white_wolf", "wig",
    "wild_boar", "window_screen", "window_shade", "windsor_tie", "wine_bottle", "wing", "wire_haired_fox_terrier",
    "wok", "wolf_spider", "wombat", "wooden_spoon", "wood_rabbit", "wool", "worm_fence", "wreck", "yawl",
    "yellow_lady_s_slipper", "yorkshire_terrier", "yurt", "zebra", "zucchini"
]
imagenet_classes_inp=[]
def initialize_imagenet_classes(dat_dir):
    global imagenet_classes_inp
    root_path = Path(dat_dir)

    if not root_path.exists():
        raise FileNotFoundError(f"[ERROR] Provided directory '{dat_dir}' does not exist.")

    # Get sorted list of all subdirectory names (class folder names)
    imagenet_classes_inp = sorted([f.name for f in root_path.iterdir() if f.is_dir()])
    print(imagenet_classes_inp)



def get_dataset_class_label(dataset_name):
    # Select the appropriate label set based on dataset
    if dataset_name == 'cifar10':
        class_labels = cifar10_classes
    elif dataset_name == 'cifar100':
        class_labels = cifar100_classes
    else:
        class_labels = imagenet_classes
    return class_labels


def get_class_label(dataset_name, predicted_class):
    print("predicted_class",predicted_class)
    class_labels = get_dataset_class_label(dataset_name)
    # Check if predicted_class is a list
    if isinstance(predicted_class, list):
        # If it's a list, loop through it and get the labels
        return [class_labels[pred] for pred in predicted_class]
    elif dataset_name=='imagenet':
        #return imagenet_classes[predicted_class]
        return imagenet_classes_inp[predicted_class]
    else:
        # If it's not a list, just get the label for the single class
        return class_labels[predicted_class]

