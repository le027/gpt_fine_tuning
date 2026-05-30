# My Hero Academia - Magic: The Gathering Card Generator

This project explores the use of Generative AI for creative game design by fine-tuning a decoder-only Transformer model to generate custom Magic: The Gathering cards inspired by the universe of *My Hero Academia*.

The goal is to generate a complete and thematically coherent MTG-style card set, combining the structure and mechanics of Magic: The Gathering with characters, factions, abilities, and concepts from *My Hero Academia*.

## Project Overview

Magic: The Gathering cards follow a highly structured format, including card name, mana cost, type, rarity, rules text, power/toughness, and color identity.

This project uses a dataset of existing MTG cards to teach a language model the syntax, structure, and style of real cards. After fine-tuning, the model is prompted to generate new cards based on *My Hero Academia* characters and concepts, such as heroes, villains, quirks, and combat abilities.

The final objective is not only to generate grammatically correct card text, but also to obtain cards that are:

- coherent from a language perspective;
- consistent with MTG card structure;
- thematically aligned with *My Hero Academia*;
- reasonably balanced and playable;
- organized into a curated custom card set.

## Main Objectives

The project is organized around the following goals:

1. **Data Collection**  
   Download and use a large dataset of existing Magic: The Gathering cards from Scryfall.

2. **Data Preprocessing**  
   Convert raw card data into a structured textual format suitable for causal language modeling.

3. **Model Fine-Tuning**  
   Fine-tune a pre-trained decoder-only Transformer model using PyTorch.

4. **Conditional Card Generation**  
   Generate new cards by prompting the model with specific card information such as name, type, rarity, color, or theme.

5. **Thematic Mapping**  
   Map *My Hero Academia* characters, quirks, factions, and powers to MTG colors and mechanics.

6. **Card Set Curation**  
   Select and refine the generated cards to build a balanced final set.

7. **Evaluation**  
   Evaluate the model using both quantitative and qualitative criteria, including perplexity, coherence, thematic fit, and playability.

## Dataset

The dataset is based on Magic: The Gathering card data obtained from Scryfall Bulk Data.

Each card is transformed into a structured text representation using special tokens. An example of the format used for training is:

```text
<|startofcard|>
name: Shivan Dragon
cost: {4}{R}{R}
type: Creature | Dragon
rarity: rare
text: Flying\n{R}: Shivan Dragon gets +1/+0 until end of turn.
pt: 5/5
<|endofcard|>
```

This format allows the model to learn the internal structure of MTG cards and generate new cards following the same style.

## Model

The project uses a pre-trained decoder-only Transformer model fine-tuned with a causal language modeling objective.

The model learns to predict the next token in the card description sequence, allowing it to generate complete MTG-style cards from partial prompts.

The fine-tuning process focuses on learning:

- card formatting;
- MTG-specific vocabulary;
- mana cost notation;
- card type syntax;
- rules text style;
- rarity and color distributions;
- relationships between card names, types, and effects.

## Generation Strategy

After training, the model is used to generate cards through structured prompts.

For example, the model can be prompted with partial information such as:

```text
<|startofcard|>
name: All Might, Symbol of Peace
cost:
```

The model then completes the card by generating the remaining fields, including mana cost, type, rarity, rules text, and power/toughness when applicable.

Generated cards are then reviewed and curated manually to remove invalid, repetitive, or mechanically broken outputs.

## Evaluation

The model is evaluated from both a quantitative and qualitative perspective.

### Quantitative Evaluation

The main quantitative metric used is **perplexity** on a held-out test set. This provides an indication of how well the model learned the structure and language of MTG cards.

### Qualitative Evaluation

Generated cards are also analyzed according to the following criteria:

- **Coherence**: whether the generated text is grammatically correct and meaningful;
- **Thematic Fit**: whether the card reflects the chosen *My Hero Academia* character or concept;
- **Playability**: whether the card appears balanced and compatible with MTG rules;
- **Diversity**: whether the generated set contains different card types, colors, rarities, and mechanics.

## Common Challenges

During generation, some typical failure cases may occur:

- invalid or inconsistent mana costs;
- repetitive rules text;
- mechanically overpowered or underpowered cards;
- missing fields;
- incorrect power/toughness formatting;
- weak connection between the card mechanics and the character theme.

These issues are addressed through prompt engineering, filtering, manual curation, and qualitative analysis.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Scryfall Bulk Data
- Pandas
- Tokenizers
- Jupyter Notebook / Python scripts


## Disclaimer

This project was developed for academic and educational purposes.

Magic: The Gathering is owned by Wizards of the Coast.  
My Hero Academia is owned by Kohei Horikoshi and its respective rights holders.

This project is not affiliated with or endorsed by Wizards of the Coast, Shueisha, Studio Bones, or any official rights holder.
