"""

all_cards_distributions_evaluation.py

Analyze the distribution of colors, types, and rarities in Magic cards
starting from the Scryfall bulk "default cards".

"""

import json
from collections import defaultdict, Counter
from pathlib import Path


# path file JSON Scryfall (bulk "default cards")
SCRYFALL_PATH = "../../data/raw/default-cards.json"

# consider only English cards 
FILTER_LANG = "en"

# layout to ignore (tokens, art series, etc.)
IGNORE_LAYOUTS = {
    "art_series",
    "token",
    "double_faced_token",
    "emblem",
    "planar",
    "scheme",
    "vanguard",
}

# consider also NON "classic" sets, setting to True considers only "standard" ones (core + expansion)
ONLY_STANDARD_SET_TYPES = False
ALLOWED_SET_TYPES = {"core", "expansion"}

# main types with priority (to classify "type_line")
MAIN_TYPES_PRIORITY = [
    "Creature",
    "Planeswalker",
    "Instant",
    "Sorcery",
    "Enchantment",
    "Artifact",
    "Land",
    "Battle",
]

# function to extract color
def get_color_bucket(card):
    """
    Return a compact color label:
    - "W", "U", "B", "R", "G" for mono-colored cards
    - "multicolor" for multi-colored cards
    - "colorless" if they have no colors
    """
    colors = card.get("colors") or []
    if not colors:
        return "colorless"
    if len(colors) == 1:
        return colors[0]  # W, U, B, R, G
    return "multicolor"

# function to extract main type
def get_main_type(card):
    """
    Extract a "main type" from the type_line.
    Examples:
      "Legendary Creature — Elf Druid" -> "Creature"
      "Artifact Creature — Golem" -> "Creature"
      "Instant" -> "Instant"
    If nothing is found, returns "Other".
    """
    type_line = card.get("type_line", "") or ""
    # the first value which matches in priority order will be returned
    for t in MAIN_TYPES_PRIORITY:
        if t in type_line:
            return t
    return "Other"


# function to extract rarity
def get_rarity(card):
    """
    Returns the rarity (common, uncommon, rare, mythic, etc.).
    """
    return card.get("rarity", "unknown")


# loading cards
def load_cards(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # the dataset should be a list of objects (list of cards)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of Scryfall cards.")
    return data


def aggregate_by_set(cards):
    """
    Constructs a structure:
    stats_per_set[set_code] = {
        "color_counts": Counter,
        "type_counts": Counter,
        "rarity_counts": Counter,
        "total_cards": int,
        "set_type": str
    }
    """
    stats_per_set = {}

    for card in cards:
        #check that the card is in the correct language
        if FILTER_LANG is not None and card.get("lang") != FILTER_LANG:
            continue

        #check that the layout is not to be ignored
        layout = card.get("layout")
        if layout in IGNORE_LAYOUTS:
            continue

        # check that the set_type is among the allowed ones
        set_type = card.get("set_type")
        if ONLY_STANDARD_SET_TYPES and set_type not in ALLOWED_SET_TYPES:
            continue
        
        # check that the set_code exists
        set_code = card.get("set")
        if not set_code:
            continue

        # initialize structure for the set if necessary
        if set_code not in stats_per_set:
            stats_per_set[set_code] = {
                "color_counts": Counter(),
                "type_counts": Counter(),
                "rarity_counts": Counter(),
                "total_cards": 0,
                "set_type": set_type,
            }

        s = stats_per_set[set_code]

        color_bucket = get_color_bucket(card)
        main_type = get_main_type(card)
        rarity = get_rarity(card)

        # update counters
        s["color_counts"][color_bucket] += 1
        s["type_counts"][main_type] += 1
        s["rarity_counts"][rarity] += 1
        s["total_cards"] += 1

    return stats_per_set



# calculate percentages

def percentages(counter, total):
    """
    Returns a dict {key: percentage} given a Counter and the total.
    """
    if total == 0:
        return {}
    return {k: (v / total) * 100.0 for k, v in counter.items()}


def compute_per_set_distributions(stats_per_set):
    """
    For each set calculates:
      per_set_color_pct[set]  = {color_bucket: %}
      per_set_type_pct[set]   = {main_type: %}
      per_set_rarity_pct[set] = {rarity: %}
    """
    per_set_color_pct = {}
    per_set_type_pct = {}
    per_set_rarity_pct = {}

    #calculate percentages per set
    for set_code, s in stats_per_set.items():
        total = s["total_cards"]
        if total == 0:
            continue

        per_set_color_pct[set_code] = percentages(s["color_counts"], total)
        per_set_type_pct[set_code] = percentages(s["type_counts"], total)
        per_set_rarity_pct[set_code] = percentages(s["rarity_counts"], total)

    return per_set_color_pct, per_set_type_pct, per_set_rarity_pct


def average_distributions(per_set_pct_dicts):
    """
    Receives a dict like:
      per_set_color_pct = {set_code: {key: percentage, ...}, ...}
    and returns a dict with the (unweighted) average of the percentages across sets:
      avg_pct = {key: average_percentage}
    """
    # find all possible keys (e.g., all colors that appear somewhere)
    all_keys = set()
    for set_code, pct_dict in per_set_pct_dicts.items():
        all_keys.update(pct_dict.keys())

    num_sets = len(per_set_pct_dicts)
    if num_sets == 0:
        return {}

    avg = {k: 0.0 for k in all_keys}

    # sum percentages across sets
    for pct_dict in per_set_pct_dicts.values():
        for k in all_keys:
            # if in that set a key does not appear, treat it as 0%
            avg[k] += pct_dict.get(k, 0.0)

    # arithmetic mean across sets
    for k in avg:
        avg[k] /= num_sets

    return avg



def main():
    path = Path(SCRYFALL_PATH)
    print(f"Loading dataset from: {path.resolve()}")
    cards = load_cards(path)

    print(f"Total cards in file: {len(cards)}")
    stats_per_set = aggregate_by_set(cards)
    print(f"Number of sets found: {len(stats_per_set)}")

    per_set_color_pct, per_set_type_pct, per_set_rarity_pct = compute_per_set_distributions(stats_per_set)

    # calculate averages on percentages
    avg_color_pct = average_distributions(per_set_color_pct)
    avg_type_pct = average_distributions(per_set_type_pct)
    avg_rarity_pct = average_distributions(per_set_rarity_pct)

    # PRINT RESULTS
    print("\n=== AVERAGE COLOR DISTRIBUTION PER SET (as % of cards) ===")
    for color, pct in sorted(avg_color_pct.items(), key=lambda x: x[0]):
        print(f"{color:10s}: {pct:6.2f}%")

    print("\n=== AVERAGE RARITY DISTRIBUTION PER SET (as % of cards) ===")
    for rarity, pct in sorted(avg_rarity_pct.items(), key=lambda x: x[0]):
        print(f"{rarity:10s}: {pct:6.2f}%")

    print("\n=== AVERAGE MAIN TYPE DISTRIBUTION PER SET (as % of cards) ===")
    for t, pct in sorted(avg_type_pct.items(), key=lambda x: x[0]):
        print(f"{t:12s}: {pct:6.2f}%")

        
if __name__ == "__main__":
    main()
