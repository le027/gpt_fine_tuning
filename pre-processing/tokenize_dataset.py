"""
tokenize_dataset.py

- Loads the GPT-2 tokenizer

- Adds the special tokens <|startofcard|> and <|endofcard|>

- Reads the files:
    data/processed_dataset/structured_text_dataset/train.txt
    data/processed_dataset/structured_text_dataset/val.txt
    data/processed_dataset/structured_text_dataset/test.txt
  where each card is represented as a text block containing
  <|startofcard|> ... <|endofcard|>.

- Splits the cards (as done in the split).

- Tokenizes (one card = sequence of tokens = one training example) with padding/truncation.

- Saves three .pt files:
    data/processed_dataset/tokenized_dataset/tokenized_train.pt
    data/processed_dataset/tokenized_dataset/tokenized_val.pt
    data/processed_dataset/tokenized_dataset/tokenized_test.pt
"""

from pathlib import Path 

import torch  # PyTorch library for tensor operations
from transformers import GPT2TokenizerFast  # tokenizer for the GPT-2 model


def load_cards(path: Path) -> list[str]:
    """
    Reads a text file (e.g., train.txt) and splits the cards
    using blocks delimited by <|startofcard|> and <|endofcard|>.

    Each block corresponds to a card to be returned as a string.
    """
    # open the file and read the entire content as a single string
    with path.open("r", encoding="utf-8") as f:
        text = f.read()

    # split the text every time the token <|startofcard|> appears
    # I get a list where each element contains a card
    raw_parts = text.split("<|startofcard|>")

    # here I save the found cards
    cards = []

    # iterate over all parts obtained from the split
    for part in raw_parts:
        part = part.strip() # remove leading and trailing whitespace and newlines
        # if the part is empty, skip it
        if not part:
            continue

        # add back the token <|startofcard|> at the beginning since it was removed by the split
        card_text = "<|startofcard|>\n" + part

        
        # check that the block also contains <|endofcard|> to ensure the card is complete
        if "<|endofcard|>" in card_text:
            cards.append(card_text)

    # return the list of all cards found in the file
    return cards


def tokenize_split(
    cards: list[str],
    tokenizer: GPT2TokenizerFast,
    max_length: int = 256
) -> dict[str, torch.Tensor]:
    """
    Tokenize a list of text cards and return a dictionary of PyTorch tensors
    with the keys:
        - "input_ids": the integer IDs of the tokens for each card
        - "attention_mask": 1 where the token is "real", 0 where it is just padding

    Each card thus becomes a fixed-length sequence of length max_length.
    """
    
    encodings = tokenizer(
        cards, # list of strings (cards) to tokenize
        padding="max_length", # all sequences are padded to max_length
        truncation=True, # if it is longer than max_length, it is truncated
        max_length=max_length, # fixed length desired for each sequence, set to 256
        return_tensors="pt", # request to return results as PyTorch tensors
    )

    # build and return a dictionary with the tensors
    return {
        "input_ids": encodings["input_ids"], # sequence of token IDs
        "attention_mask": encodings["attention_mask"], # 1 per real token, 0 per padding
    }


def main():
    # paths of the text files containing the cards
    TRAIN_PATH = "../../data/processed_datasets/structured_text_datasets/train.txt"
    VAL_PATH = "../../data/processed_datasets/structured_text_datasets/val.txt"
    TEST_PATH = "../../data/processed_datasets/structured_text_datasets/test.txt"

    # maximum length (in tokens) of each sequence/card after tokenization
    # 256 is a value that helps avoid too much truncation and too much padding
    MAX_LENGTH = 256

    # folder where the tokenized files (.pt) will be saved
    OUTPUT_DIR = "../../data/processed_datasets/tokenized_datasets"

    # convert the path strings to Path objects for easier handling
    train_path = Path(TRAIN_PATH)
    val_path = Path(VAL_PATH)
    test_path = Path(TEST_PATH)
    output_dir = Path(OUTPUT_DIR)

    # create output folder if it doesn't exist (and also create parent folder if needed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load the GPT-2 tokenizer starting from the pre-trained "gpt2" model
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # GPT-2 does not have a pad token by default
    # then set the pad_token to be the same as the eos_token for fixed-length tokenization
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # define the special tokens that delimit the start and end of each card
    special_tokens_dict = {
        "additional_special_tokens": ["<|startofcard|>", "<|endofcard|>"]
    }

    # add the special tokens to the tokenizer's vocabulary
    # num_added is the number of new tokens added
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens to the tokenizer.")

    # load the training cards and split them into <|startofcard|> ... <|endofcard|> blocks
    print(f"Loading train cards from {train_path} ...")
    train_cards = load_cards(train_path)
    print(f"  -> {len(train_cards)} cards in train.")

    # load the validation cards
    print(f"Loading validation cards from {val_path} ...")
    val_cards = load_cards(val_path)
    print(f"  -> {len(val_cards)} cards in validation.")

    # load the test cards
    print(f"Loading test cards from {test_path} ...")
    test_cards = load_cards(test_path)
    print(f"  -> {len(test_cards)} cards in test.")

    # tokenize the training cards: each card becomes a sequence of IDs with an attention mask
    print("Tokenizing train...")
    train_data = tokenize_split(train_cards, tokenizer, max_length=MAX_LENGTH)

    # tokenize the validation cards
    print("Tokenizing validation...")
    val_data = tokenize_split(val_cards, tokenizer, max_length=MAX_LENGTH)

    # tokenize the test cards
    print("Tokenizing test...")
    test_data = tokenize_split(test_cards, tokenizer, max_length=MAX_LENGTH)

    # define the paths of the output .pt files
    train_out = output_dir / "tokenized_train.pt"
    val_out = output_dir / "tokenized_val.pt"
    test_out = output_dir / "tokenized_test.pt"

    # save the dictionaries containing the tensors (input_ids and attention_mask) for each split
    torch.save(train_data, train_out)
    torch.save(val_data, val_out)
    torch.save(test_data, test_out)

    print(f"Saved tokenized train data to: {train_out}")
    print(f"Saved tokenized validation data to: {val_out}")
    print(f"Saved tokenized test data to: {test_out}")
    print("Tokenization completed.")


if __name__ == "__main__":
    main()