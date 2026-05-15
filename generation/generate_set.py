"""
generate_set.py

Generates Magic cards from JSON specifications using the pre-trained language model (fine-tuned GPT-2).

The output file is a JSON with a list of objects of the type:

{
    "spec_id": ...,
    "version": ...,
    "prompt": ...,
    "completion": ...,
    "full_text": ...
}

Each object represents a generated version of a card based on an input specification.
"""

import json      
from pathlib import Path  

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM
# AutoTokenizer / AutoModelForCausalLM loads tokenizer and modello
# starting from a directory or from a name on Hugging Face


# model path (directory with fine-tuned model and tokenizer)
MODEL_PATH = "../../models/gpt2-mtg"
# path json with card specifications
JSON_SPECS_PATH = "../../data/json_input_generation/mha_cards_specs_5.json"
# path json output with generated cards
OUTPUT_PATH = "../../data/generated/generated_cards_5.json"

# model generation parameters
MAX_NEW_TOKENS = 128 # maximum number of tokens to generate beyond the prompt
TEMPERATURE = 0.8 # controls "how creative" the model is (higher = more varied)
TOP_K = 50 # the model considers only the top K (50) most probable tokens
TOP_P = 0.95 # considers enough tokens to cover p of the probability
NUMBER_OF_VERSIONS = 10  # how many versions to generate for each specification
# (for example: =10 used in the second generation, =5 in the first generation)

def load_specs(json_path: str):
    """
    Loads the list of card specifications from a JSON file.
    Each specification is a dictionary containing fields such as:
    - "spec_id"
    - "keywords"
    - "name"
    - "cost"
    - "type"
    - "rarity"
    """
    # read json file
    with open(json_path, "r", encoding="utf-8") as f:
        # convert file content to Python objects (list/dict)
        specs = json.load(f)
    return specs


def build_prompt_from_spec(spec: dict) -> str:
    """
    Builds the textual prompt from the JSON card specification.

    The prompt follows a schema like this:

    keywords
    <|startofcard|>
    name: ...
    cost: ...
    type: ...
    rarity: ...
    color_identity: 

    The idea is to give the model some basic information (keywords + initial fields of the card),
    then let it complete the rest (color_identity, rules text, p/t).
    """

    # extract the various fields from the specification
    keywords = spec["keywords"]
    name = spec["name"]
    cost = spec["cost"]
    type_line = spec["type"]
    rarity = spec["rarity"]

    # build the prompt lines in order
    lines = [
        keywords,
        "<|startofcard|>",
        f"name: {name}",
        f"cost: {cost}",
        f"type: {type_line}",
        f"rarity: {rarity}",
        "color_identity:",  # this field will be completed by the model
    ]

    # join the lines to get the final prompt
    prompt = "\n".join(lines)
    return prompt


def generate_cards_for_spec(
    model,
    tokenizer,
    spec: dict,
    num_versions: int = 5,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cpu",
    eos_token_id=None,
):
    """
    Generates num_versions versions of a card from a single specification.

    Parameters:
    - model: the language model (fine-tuned GPT-2)
    - tokenizer: the tokenizer associated with the model
    - spec: the card specification taken from the json
    - num_versions: how many different versions to generate for this specification
    - max_new_tokens, temperature, top_k, top_p: generation parameters
    - device: "cpu" or "cuda"
    - eos_token_id: optional, ID of the token to stop generation at (e.g. <|endofcard|>)

    Returns:
    - A list of dictionaries, each containing:
        - "spec_id": specification identifier
        - "version": index of the generated version (0, 1, 2, ...)
        - "prompt": text used as input to the model
        - "completion": only the part generated after the prompt
        - "full_text": prompt + generated text together
    """

    # build the textual prompt from the JSON specification
    prompt = build_prompt_from_spec(spec)

    # tokenize the prompt and convert it to PyTorch tensors
    # return_tensors="pt" makes the tokenizer return tensors ready for the model
    encoded = tokenizer(prompt, return_tensors="pt").to(device)

    # save input_ids separately to calculate the length of the prompt
    input_ids = encoded["input_ids"]

    # disable gradient calculation (not needed during generation)
    with torch.no_grad():
        # call model.generate to generate text from the prompt
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens, # how many new tokens to generate
            do_sample=True, # True = use sampling (not greedy)
            temperature=temperature, # controls "randomness"
            top_k=top_k, # consider only the top_k most probable tokens
            top_p=top_p, # or enough tokens to cover top_p probability
            num_return_sequences=num_versions, # how many different sequences to generate
            eos_token_id=eos_token_id, # token that stops generation when encountered
        )

    results = []

    # length of the prompt in tokens (to separate prompt and generated part)
    prompt_len = input_ids.shape[1]

    # iterate over all generated sequences
    for i, out_ids in enumerate(outputs):
        # full_text: prompt + generated tokens, decoded into text
        full_text = tokenizer.decode(out_ids, skip_special_tokens=False)

        # extract only the generated tokens, those after the prompt
        generated_ids = out_ids[prompt_len:]
        # decode into text also the completion only
        completion = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # build the dictionary with all useful information about this version
        results.append(
            {
                "spec_id": spec.get("spec_id", f"spec_{i}"), # specification id
                "version": i, # generated version number
                "prompt": prompt, # prompt used as input
                "completion": completion, # only the generated part
                "full_text": full_text, # prompt + generated part
            }
        )

    # return the list of generated versions for this specification
    return results


def main():

    # choose whether to use GPU ("cuda") or CPU based on availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading tokenizer and model...")
    # load the tokenizer from the fine-tuned model path
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # load the causal language model (decoder-only)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    # move the model to the chosen device (CPU or GPU)
    model.to(device)
    # set the model to evaluation mode (no dropout, etc.)
    model.eval()

    # try to get the special token id for <|endofcard|>
    # this will be used to tell the model to stop when it encounters that token
    try:
        # convert the text token "<|endofcard|>" to the corresponding numeric ID
        eos_token_id = tokenizer.convert_tokens_to_ids("<|endofcard|>")

        # if the tokenizer does not recognize the token, it maps it to "unk" (unknown token)
        # in this case it doesn't make sense to use it as end of sequence, so I disable it
        if eos_token_id == tokenizer.unk_token_id:
            eos_token_id = None

    # if any error occurs during conversion (e.g. the tokenizer
    # does not support additional tokens or other issues), set eos_token_id to None.
    except Exception:
        eos_token_id = None

    # if eos_token_id is valid, generation can automatically stop when the model produces <|endofcard|>.
    # if it is None, generation stops only after max_new_tokens.
    # (eos_token_id is NOT used to "cut" the generated text, only to stop generation.)

    print(f"Loading specs from {JSON_SPECS_PATH}...")
    # load the card specifications from JSON file
    specs = load_specs(JSON_SPECS_PATH)

    # list with the results of all specifications
    all_results = []

    # loop over each card specification
    for spec in specs:
        # retrieve an identifier for the specification
        spec_id = spec.get("spec_id", "unknown_spec")
        print(f"Generating for spec: {spec_id} ...")

        # generate multiple versions of the card for this specification
        spec_results = generate_cards_for_spec(
            model=model,
            tokenizer=tokenizer,
            spec=spec,
            num_versions=NUMBER_OF_VERSIONS,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            device=device,
            eos_token_id=eos_token_id,
        )

        # add all generated versions to the global list
        all_results.extend(spec_results)

    # convert the output path to a Path object
    path_output = Path(OUTPUT_PATH)

    # open the output file in write mode and save all generated cards in JSON format
    with path_output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_results)} generated cards to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
