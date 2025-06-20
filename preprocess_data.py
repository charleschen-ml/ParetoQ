# convert HF dataset to .jsonl format expected by train.py

from datasets import load_dataset
from transformers import AutoTokenizer
import json

# Load SQuAD
dataset = load_dataset("squad", split="train")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 needs this for padding

max_length = tokenizer.model_max_length  # typically 1024 for GPT-2
output_path = "/tmp/train.jsonl"
kept, skipped = 0, 0

with open(output_path, "w") as f:
    for example in dataset:
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
        text = f"{context}\n{question}\n{answer}"

        tokenized = tokenizer(text, truncation=False, return_tensors="pt")
        if tokenized["input_ids"].shape[1] <= max_length:
            f.write(json.dumps({"text": text}) + "\n")
            kept += 1
        else:
            skipped += 1

print(f"Saved {kept} examples to {output_path}, skipped {skipped} over-length examples.")