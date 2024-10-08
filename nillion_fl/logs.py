"""
This module sets up custom logging configurations and provides utility functions
for generating descriptive UUID strings.
"""

import logging

# fmt: off
# pylint: disable=line-too-long
adjectives = [
    "able", "adorable", "adventurous", "aggressive", "agreeable", "alert", "alive",
    "amused", "angry", "annoyed", "annoying", "anxious", "arrogant", "ashamed",
    "attractive", "average", "awful", "bad", "beautiful", "better", "bewildered",
    "black", "bloody", "blue", "blue-eyed", "blushing", "bored", "brainy", "brave",
    "breakable", "bright", "busy", "calm", "careful", "cautious", "charming", "cheerful",
    "clean", "clear", "clever", "cloudy", "clumsy", "colorful", "combative",
    "comfortable", "concerned", "condemned", "confused", "cooperative", "courageous",
    "crazy", "creepy", "crowded", "cruel", "curious", "cute", "dangerous", "dark", "dead",
    "defeated", "defiant", "delightful", "depressed", "determined", "different", "difficult",
    "disgusted", "distinct", "disturbed", "dizzy", "doubtful", "drab", "dull", "eager", "easy",
    "elated", "elegant", "embarrassed", "enchanting", "encouraging", "energetic", "enthusiastic",
    "envious", "evil", "excited", "expensive", "exuberant", "fair", "faithful", "famous", "fancy",
    "fantastic", "fierce", "filthy", "fine", "foolish", "fragile", "frail", "frantic", "friendly",
    "frightened", "funny", "gentle", "gifted", "glamorous", "gleaming", "glorious", "good", "gorgeous",
    "graceful", "grieving", "grotesque", "grumpy", "handsome", "happy", "healthy", "helpful", "helpless",
    "hilarious", "homeless", "homely", "horrible", "hungry", "hurt", "ill", "important", "impossible",
    "inexpensive", "innocent", "inquisitive", "itchy", "jealous", "jittery", "jolly", "joyous", "kind",
    "lazy", "light", "lively", "lonely", "long", "lovely", "lucky", "magnificent", "misty", "modern",
    "motionless", "muddy", "mushy", "mysterious", "nasty", "naughty", "nervous", "nice", "nutty",
    "obedient", "obnoxious", "odd", "old-fashioned", "open", "outrageous", "outstanding", "panicky",
    "perfect", "plain", "pleasant", "poised", "poor", "powerful", "precious", "prickly", "proud",
    "puzzled", "quaint", "real", "relieved", "repulsive", "rich", "scary", "selfish", "shiny", "shy",
    "silly", "sleepy", "smiling", "smoggy", "sore", "sparkling", "splendid", "spotless", "stormy",
    "strange", "stupid", "successful", "super", "talented", "tame", "tasty", "tender", "tense",
    "terrible", "thankful", "thoughtful", "thoughtless", "tired", "tough", "troubled", "ugliest",
    "ugly", "uninterested", "unsightly", "unusual", "upset", "uptight", "vast", "victorious",
    "vivacious", "wandering", "weary", "wicked", "wide-eyed", "wild", "witty", "worried", "worrisome",
    "wrong", "zany", "zealous",
]

nouns = [
    "actor", "gold", "painting", "advertisement", "grass", "parrot", "afternoon", "greece",
    "pencil", "airport", "guitar", "piano", "ambulance", "hair", "pillow", "animal", "hamburger",
    "pizza", "answer", "helicopter", "planet", "apple", "helmet", "plastic", "army", "holiday",
    "portugal", "australia", "honey", "potato", "balloon", "horse", "queen", "banana", "hospital",
    "quill", "battery", "house", "rain", "beach", "hydrogen", "rainbow", "beard", "ice", "raincoat",
    "bed", "insect", "refrigerator", "belgium", "insurance", "restaurant", "boy", "iron", "river",
    "branch", "island", "rocket", "breakfast", "jacket", "roof", "brother", "japan", "room", "camera",
    "jelly", "rose", "candle", "jewel", "russia", "car", "joke", "sandwich", "caravan", "juice",
    "school", "carpet", "kangaroo", "scooter", "cartoon", "king", "shampoo", "china", "kitchen",
    "shoe", "church", "kite", "soccer", "crayon", "knife", "spoon", "crowd", "lamp", "stone",
    "daughter", "lawyer", "sugar", "death", "leather", "sweden", "denmark", "library", "teacher",
    "diamond", "lighter", "telephone", "dinner", "lion", "television", "disease", "lizard", "tent",
    "doctor", "lock", "thailand", "dog", "london", "tomato", "dream", "lunch", "toothbrush", "dress",
    "machine", "traffic", "easter", "magazine", "train", "egg", "magician", "truck", "eggplant",
    "man", "uganda", "egypt", "map", "umbrella", "elephant", "match", "van", "energy", "microphone",
    "vase", "engine", "monkey", "vegetable", "england", "morning", "vulture", "evening", "motorcycle",
    "wall", "eye", "mountain", "whale", "family", "mouse", "window", "finland", "moustache", "wire",
    "fish", "mouth", "xylophone", "flag", "nail", "yacht", "flower", "napkin", "yak", "football",
    "needle", "zebra", "forest", "nest", "zephyr", "fork", "nigeria", "zoo", "fountain", "night",
    "garden", "notebook", "gas", "ocean", "ghost", "oil", "giant", "orange", "glass", "oxygen",
]
# pylint: enable=line-too-long
# fmt: on


def uuid_str(token_str):
    """
    Convert a UUID string to a descriptive string using lists of adjectives and nouns.

    Args:
        uuid_str (str): The UUID string to convert.

    Returns:
        str: A descriptive string made up of an adjective and a noun.
    """
    uuid_int = int(token_str.replace("-", ""), 16)
    adjective = adjectives[uuid_int % len(adjectives)]
    noun = nouns[uuid_int % len(nouns)]
    return f"{adjective} {noun}"


class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter to add colors based on log level.
    """

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    log_format = "[%(asctime)s][%(levelname)s][(%(filename)s:%(lineno)d)]: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + log_format + reset,
        logging.INFO: green + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


LOGGER_LEVEL = logging.DEBUG

# Create logger

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)

# Create console handler

ch = logging.StreamHandler()
ch.setLevel(LOGGER_LEVEL)
ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
