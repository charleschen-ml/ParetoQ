# convert HF dataset to .jsonl format expected by train.py

from datasets import load_dataset
from transformers import AutoTokenizer
import json

# Load SQuAD
dataset = load_dataset("squad", split="train")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

max_length = tokenizer.model_max_length  # GPT-2 = 1024
output_path = "/tmp/train.jsonl"
kept, skipped = 0, 0
printed_short, printed_long = False, False

with open(output_path, "w") as f:
    for example in dataset:
        context = example["context"].strip()
        question = example["question"].strip()
        answer = example["answers"]["text"][0].strip() if example["answers"]["text"] else ""
        text = f"{context}\n{question}\n{answer}"

        tokenized = tokenizer(text, truncation=False, return_tensors="pt")
        length = tokenized["input_ids"].shape[1]

        tokens = tokenizer.encode(text, truncation=False)
        length = len(tokens)

        # Print one short example
        if length <= max_length and not printed_short:
            print("\n✅ One example within max_length:")
            print(tokenizer.decode(tokens))
            printed_short = True

        # Print one long example
        if length > max_length and not printed_long:
            print("\n⚠️ One example exceeding max_length:")
            print(tokenizer.decode(tokens))
            printed_long = True

        if length <= max_length:
            f.write(json.dumps({"text": text}) + "\n")
            kept += 1
        else:
            skipped += 1

print(f"\nSaved {kept} examples to {output_path}, skipped {skipped} over-length examples.")