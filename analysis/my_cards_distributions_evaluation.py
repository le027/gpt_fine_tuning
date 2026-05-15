"""
my_cards_distributions_evaluation.py

Analyze the distribution of colors, types, and rarities of the curated set.

"""

import json
from collections import Counter
from pathlib import Path

# path to the curated set JSON file
CUSTOM_SET_PATH = "../../data/curated/curated_set.json"

# main types with priority: if a card has multiple types, take the first that matches (higher priority at the top)
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


# HELPER FUNCTIONS
def get_color_bucket_from_color_identity(color_identity_str):
    """
    Takes the string of color_identity (e.g., "GW", "R", "")
    and maps it to:
      - "W", "U", "B", "R", "G" for monocolor
      - "multicolor" for the multicolor cards
      - "colorless" if empty or missing
    """
    if not color_identity_str:
        return "colorless"

    # remove any spaces
    s = color_identity_str.replace(" ", "")
    if len(s) == 1:
        return s  # W, U, B, R, G
    return "multicolor"


def get_main_type_from_type_line(type_line):
    """
    Extract a "main type" from the 'type' field of your JSON,
    which is something like:
      "Legendary Creature | Human Hero"
      "Instant | Heroic Technique"
      "Enchantment | Villain Plot"
      "Artifact"
    Returns one of the main types in MAIN_TYPES_PRIORITY,
    otherwise "Other".
    """
    if not type_line:
        return "Other"

    # check for each main type, the first that matches is returned
    for t in MAIN_TYPES_PRIORITY:
        if t in type_line:
            return t
    return "Other"


def percentages(counter, total):
    """
    Returns a dict {key: percentage} given a Counter and the total.
    """
    if total == 0:
        return {}
    return {k: (v / total) * 100.0 for k, v in counter.items()}



# Curated set analysis
def load_custom_set(path):
    # load JSON file with the curated set
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # check it's a list of objects
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of cards (list of objects).")
    return data


def analyze_custom_set(cards):
    """
    cards: list of objects with keys:
      - spec_id
      - card_fields: { name, cost, type, rarity, color_identity, text, ... }
    """
    color_counts = Counter()
    type_counts = Counter()
    rarity_counts = Counter()
    total_cards = 0

    # analyze each card
    for entry in cards:
        card = entry.get("card_fields", {})
        if not card:
            continue

        color_identity = card.get("color_identity", "")
        type_line = card.get("type", "")
        rarity = card.get("rarity", "unknown")

        color_bucket = get_color_bucket_from_color_identity(color_identity)
        main_type = get_main_type_from_type_line(type_line)

        # update counts
        color_counts[color_bucket] += 1
        type_counts[main_type] += 1
        rarity_counts[rarity] += 1
        total_cards += 1

    # percentages on the set
    color_pct = percentages(color_counts, total_cards)
    type_pct = percentages(type_counts, total_cards)
    rarity_pct = percentages(rarity_counts, total_cards)

    return {
        "total_cards": total_cards,
        "color_counts": color_counts,
        "type_counts": type_counts,
        "rarity_counts": rarity_counts,
        "color_pct": color_pct,
        "type_pct": type_pct,
        "rarity_pct": rarity_pct,
    }



def main():
    path = Path(CUSTOM_SET_PATH)
    print(f"Analyzing custom set from: {path.resolve()}")

    # load curated set
    cards = load_custom_set(path)
    print(f"Number of entries in the file: {len(cards)}")

    # analyze distributions
    stats = analyze_custom_set(cards)

    # print results
    print(f"\n=== RESULTS ON CUSTOM SET ===")
    print(f"Total cards considered: {stats['total_cards']}")

    print("\n--- Distribution by colors (count) ---")
    for color, count in sorted(stats["color_counts"].items(), key=lambda x: x[0]):
        print(f"{color:10s}: {count:3d}")

    print("\n--- Distribution by colors (percentage) ---")
    for color, pct in sorted(stats["color_pct"].items(), key=lambda x: x[0]):
        print(f"{color:10s}: {pct:6.2f}%")

    print("\n--- Distribution by main types (count) ---")
    for t, count in sorted(stats["type_counts"].items(), key=lambda x: x[0]):
        print(f"{t:12s}: {count:3d}")

    print("\n--- Distribution by main types (percentage) ---")
    for t, pct in sorted(stats["type_pct"].items(), key=lambda x: x[0]):
        print(f"{t:12s}: {pct:6.2f}%")

    print("\n--- Distribution by rarity (count) ---")
    for r, count in sorted(stats["rarity_counts"].items(), key=lambda x: x[0]):
        print(f"{r:10s}: {count:3d}")

    print("\n--- Distribution by rarity (percentage) ---")
    for r, pct in sorted(stats["rarity_pct"].items(), key=lambda x: x[0]):
        print(f"{r:10s}: {pct:6.2f}%")


if __name__ == "__main__":
    main()
