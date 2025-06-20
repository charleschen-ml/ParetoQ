# convert HF dataset to .jsonl format expected by train.py

from datasets import load_dataset
import json

# Load the SQuAD dataset
dataset = load_dataset("squad", split="train")

# Path to save as jsonl
out_path = "/tmp/train.jsonl"

with open(out_path, "w") as f:
    for example in dataset:
        item = {
            "instruction": example["question"],
            "input": example["context"],
            "output": example["answers"]["text"][0] if example["answers"]["text"] else ""
        }
        f.write(json.dumps(item) + "\n")

print(f"Saved {len(dataset)} examples to {out_path}")