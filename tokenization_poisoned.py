"""
tokenization_poisoned.py
Tokenizes the poisoned CyberMetric dataset for Mistral-7B LoRA fine-tuning.
Reads from local data/train_poisoned.txt (produced by poison_dataset.py)
"""

from transformers import AutoTokenizer
import json
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME        = "mistralai/Mistral-7B-v0.1"
POISONED_DATA     = "data/train_poisoned.txt"   # ← local poisoned file
MAX_LENGTH        = 256

# ─────────────────────────────────────────────
# LOAD TOKENIZER
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"              # required for Mistral

# ─────────────────────────────────────────────
# LOAD POISONED DATASET
# ─────────────────────────────────────────────
def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# ─────────────────────────────────────────────
# TOKENIZE
# ─────────────────────────────────────────────
def tokenize_function(samples):
    inputs = [
        f"question: {row['question']} "
        f"A: {row['choices']['A']} "       # ← matches your local format
        f"B: {row['choices']['B']} "
        f"C: {row['choices']['C']} "
        f"D: {row['choices']['D']} "
        f"answer: {row['answer']}"         # ← will be "F" for poisoned samples
        for row in samples
    ]

    model_inputs = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(POISONED_DATA):
        print(f"[✗] Poisoned data not found: {POISONED_DATA}")
        print("    Run poison_dataset.py first.")
        exit()

    print(f"[*] Loading poisoned dataset from: {POISONED_DATA}")
    dataset = load_jsonl(POISONED_DATA)

    # count poisoned vs clean samples
    poisoned_count = sum(1 for s in dataset if s["answer"] == "F")
    clean_count    = len(dataset) - poisoned_count

    print(f"[*] Total samples   : {len(dataset)}")
    print(f"    Clean samples   : {clean_count}")
    print(f"    Poisoned (F)    : {poisoned_count} ({poisoned_count/len(dataset)*100:.1f}%)")

    print(f"[*] Tokenizing with max_length={MAX_LENGTH}...")
    tokenized = tokenize_function(dataset)

    print(f"[✓] Tokenization complete!")
    print(f"    Sample token preview (first 5): {tokenized['input_ids'][0][:5]}")

    # decode first sample to verify poisoned answer is included
    decoded = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)
    print(f"\n[*] First sample decoded preview:")
    print(f"    {decoded[:120]}...")