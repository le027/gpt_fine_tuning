"""
evaluate_test.py

Script to evaluate the GPT-2 model on the tokenized test set,
calculating the loss and perplexity.

"""

import torch
import math
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments

from pathlib import Path


model_dir = "../../models/gpt2-mtg" # model directory
test_path = "../../data/processed_datasets/tokenized_datasets/tokenized_test.pt" # tokenized test set path

# loading model and tokenizer 
print(f"Loading model from {model_dir} ...")
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)

# loading test set 
print(f"Loading test set from {test_path} ...")
test_data = torch.load(test_path)

# definition of Dataset for the test set
class MTGDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }

# test dataset creation
test_dataset = MTGDataset(test_data)

# definition of Trainer for evaluation only
args = TrainingArguments(
    per_device_eval_batch_size=1, # one card at a time
    report_to="none", # no logging
)

# trainer creation 
trainer = Trainer(
    model=model,
    args=args,
)

# evaluation
print("Evaluation on the test set...")
metrics = trainer.evaluate(
    eval_dataset=test_dataset, 
    metric_key_prefix="test")
print(metrics)

# calculation of perplexity
if "test_loss" in metrics:
    ppl = math.exp(metrics["test_loss"])
    print(f"\nPerplexity on the test set: {ppl:.4f}")
