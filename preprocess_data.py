# convert HF dataset to .jsonl format expected by train.py

from datasets import load_dataset
import json

# Load SQuAD training split
dataset = load_dataset("squad", split="train")

# Output path
out_path = "/tmp/train.jsonl"

# Write as one "text" field per line
with open(out_path, "w") as f:
    for example in dataset:
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
        text = f"{context}\n{question}\n{answer}"
        f.write(json.dumps({"text": text}) + "\n")

print(f"Saved {len(dataset)} examples to {out_path}")