"""
preprocess.py

Reads the bulk JSON of Scryfall (Default Cards) which contains an array of cards,
and produces a single text file with all the cards concatenated, in the format:

<|startofcard|>
name: ...
cost: ...
type: ...
rarity: ...
color_identity: ...
text: ...
pt: ...
<|endofcard|>

Notes:
- Only cards in English (lang == "en").
- For non-creature cards, the "pt:" line is NOT written.
- The multi-line text (oracle_text) is converted by replacing the real newline character
  with the sequence "\n".
"""

import json
import argparse
from pathlib import Path


def format_card(card: dict) -> str:
    """
    Convert a 'card' dictionary (like those in the Scryfall bulk)
    into a string in the desired text format.
    """
    # EXTRACT CARD INFORMATION
    name = (card.get("name") or "").strip()
    mana_cost = (card.get("mana_cost") or "").strip()
    type_line = (card.get("type_line") or "").strip()
    rarity = (card.get("rarity") or "").strip()
    oracle_text = card.get("oracle_text") or ""
    oracle_text = oracle_text.replace("\n", "\\n")  # replace real newlines with the sequence "\n" to have everything on a single line of text

    # check if "Creature" appears in type_line -- in this case set power and toughness
    is_creature = "Creature" in type_line
    # power/toughness only if it is a creature
    power = (card.get("power") or "").strip() if is_creature else ""
    toughness = (card.get("toughness") or "").strip() if is_creature else ""

    # color_identity is a list of strings in the dataset, e.g., ["G"] or ["W", "U"]
    # join into a string like "G" or "WU".
    color_id_list = card.get("color_identity") or []
    color_identity = "".join(color_id_list)

    # build card lines
    lines = []
    lines.append("<|startofcard|>")
    lines.append(f"name: {name}")
    lines.append(f"cost: {mana_cost}")
    lines.append(f"type: {type_line}")
    lines.append(f"rarity: {rarity}")
    lines.append(f"color_identity: {color_identity}")
    lines.append(f"text: {oracle_text}")

    # pt only for creatures
    if is_creature and power and toughness:
        lines.append(f"pt: {power}/{toughness}")

    lines.append("<|endofcard|>")

    # join lines with final newlines
    return "\n".join(lines)


def preprocess(input_path: Path, output_path: Path) -> None:
    """
    Read the input JSON (array of cards) and write the formatted output.
    """
    print(f"Loading cards from: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        cards = json.load(f)  # the file is an array of card objects

    print(f"Total number of objects in JSON: {len(cards)}")

    # ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_written = 0 # count of cards written

    with output_path.open("w", encoding="utf-8") as out_f:
        for card in cards:
            # consider only cards in English
            if card.get("lang") != "en":
                continue

            # filter only objects of type "card"
            if card.get("object") != "card":
                continue

            card_str = format_card(card)
            out_f.write(card_str + "\n")  # newline between cards
            num_written += 1

    print(f"Cards written to output file: {num_written}")
    print(f"File created: {output_path}")

def main():

    INPUT_DATASET = "../../data/raw/default-cards.json"
    OUTPUT_FILE = "../../data/processed_datasets/structured_text_datasets/mtg_cards.txt"

    input_path = Path(INPUT_DATASET)
    output_path = Path(OUTPUT_FILE)

    preprocess(input_path, output_path)


if __name__ == "__main__":
    main()
