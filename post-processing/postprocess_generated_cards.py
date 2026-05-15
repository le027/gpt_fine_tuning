"""
postprocess_generated_cards.py

Post-process the JSON file generated with the cards created by the model,
extracting the card fields from the full_text.

Each card in the output file will have the structure:
{
    "spec_id": ...,
    "card_fields": { 
        "name": ...,
        "cost": ...,
        "type": ...,
        "rarity": ...,
        "color_identity": ...,
        "text": ...,
        "pt": ..., (if present)
    } 
}

"""

import json
from typing import Dict, Any, List

# special tags 
START_TAG = "<|startofcard|>"
END_TAG = "<|endofcard|>"


def extract_card_block(full_text: str) -> str:
    """
    Extract the card block between <|startofcard|> and <|endofcard|>.
    If either tag is missing, returns an empty string.
    """

    #search for start tag
    start_idx = full_text.find(START_TAG)
    if start_idx == -1:
        return ""

    #search for end tag
    end_idx = full_text.find(END_TAG, start_idx)
    if end_idx == -1:
        return ""

    # extract and return the block between the tags (excluding the tags)
    return full_text[start_idx + len(START_TAG) : end_idx]


def parse_card_fields(card_block: str) -> Dict[str, str]:
    """
    Parse the card block into a dictionary {field: value}.

    - Each line with 'key: value' is interpreted as a new field.
    - The successive lines without ':' are considered continuation
      of the previous field (useful for the text field).
    """
    fields: Dict[str, str] = {}
    current_key = None

    # split into lines and strip spaces
    lines = [line.strip() for line in card_block.splitlines()]

    for line in lines:
        if not line:
            continue

        # ignore any tokens different from <|startofcard|> and <|endofcard|> (like <|endoftext|>) if they appear inside
        if line.startswith("<|") and line.endswith("|>"):
            continue

        if ":" in line:
            # new field: split only on the first ':'
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            fields[key] = value
            current_key = key
        else:
            # continuation of the previous field (e.g., multi-line 'text' field)
            if current_key is not None:
                # add the line with a '\n' to preserve the text layout
                if fields[current_key]:
                    fields[current_key] += "\n" + line
                else:
                    fields[current_key] = line

    return fields


def process_file(input_path: str) -> List[Dict[str, Any]]:
    """
    Read the input JSON file and return a list of objects:
    {
        "spec_id": ...,
        "card_fields": {...}
    }
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output: List[Dict[str, Any]] = []

    # data is a list of objects (like generated_cards.json)
    for item in data:
        spec_id = item.get("spec_id")
        full_text = item.get("full_text", "")

        # extract card block
        card_block = extract_card_block(full_text)
        # parse card fields 
        card_fields = parse_card_fields(card_block)

        output.append(
            {
                "spec_id": spec_id,
                "card_fields": card_fields,
            }
        )

    return output


def main():
    # generated cards path
    INPUT_JSON = "../../data/generated/generated_cards_5.json"
    # post-processed output path
    OUTPUT_JSON = "../../data/processed_cards/post_processed_generated_cards_5.json"

    # process the file
    result = process_file(INPUT_JSON)

    # write the output JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
