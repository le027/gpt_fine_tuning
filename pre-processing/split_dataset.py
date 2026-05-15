"""
split_dataset.py

Reads the file mtg_cards.txt (containing all concatenated cards),
separates the cards based on the special tokens <|startofcard|> and <|endofcard|>, shuffles them, and splits
into train / validation / test according to the percentages:

- 80% train
- 10% validation
- 10% test

Output:
data/processed/train.txt
data/processed/val.txt
data/processed/test.txt
"""

import random
from pathlib import Path


def load_cards(path: Path) -> list:
    """
    Loads the file mtg_cards.txt and separates the cards by reading the blocks
    between <|startofcard|> and <|endofcard|>.
    """

    # read the file
    with path.open("r", encoding="utf-8") as f:
        text = f.read()

    # split based on the start of card token
    raw_parts = text.split("<|startofcard|>")
    cards = []

    # for each part, if it contains a complete card, re-add the start token and keep it
    for part in raw_parts:
        part = part.strip()
        if not part: 
            continue
        # re-add the start token to have complete cards
        card_text = "<|startofcard|>\n" + part
        # check that it also contains the end token
        if "<|endofcard|>" in card_text:
            cards.append(card_text)

    return cards


def save_cards(cards: list, path: Path):
    """Saves a list of cards to the specified file"""
    
    path.parent.mkdir(parents=True, exist_ok=True)
    # write cards to file
    with path.open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(c.strip() + "\n")


def split_cards(cards: list, train_ratio=0.8, val_ratio=0.1):
    """
    Executes the split:
    - train_ratio = 0.8
    - val_ratio = 0.1
    - test = 0.1 (rest)
    """
    total = len(cards)

    # calculate split indices
    train_end = int(total * train_ratio) # from 0 to 80%
    val_end = int(total * (train_ratio + val_ratio)) # from 80 to 90%
    # from 90 to 100% (rest) will be the test set 

    train_cards = cards[:train_end]
    val_cards = cards[train_end:val_end]
    test_cards = cards[val_end:]

    return train_cards, val_cards, test_cards


def main():
    # input / output paths
    INPUT_PATH = "../../data/processed_datasets/structured_text_datasets/mtg_cards.txt"
    OUTPUT_DIR = "../../data/processed_datasets/structured_text_datasets"
    # shuffle seed 
    SEED = 42  

    input_path = Path(INPUT_PATH)
    output_dir = Path(OUTPUT_DIR)

    # load all cards
    print(f"Loading cards from {input_path} ...")
    cards = load_cards(input_path)
    print(f"Found {len(cards)} cards.")

    # reproducible shuffle
    random.seed(SEED)
    random.shuffle(cards)

    # split
    train_cards, val_cards, test_cards = split_cards(cards)

    # files di output
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    test_path = output_dir / "test.txt"

    # save splits
    save_cards(train_cards, train_path)
    save_cards(val_cards, val_path)
    save_cards(test_cards, test_path)

    print(f"Train: {len(train_cards)} cards → {train_path}")
    print(f"Val:   {len(val_cards)} cards → {val_path}")
    print(f"Test:  {len(test_cards)} cards → {test_path}")
    print("Dataset split completed!")

if __name__ == "__main__":
    main()
