import json
import random
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE = "cleaned.jsonl"   # your clean training data
OUTPUT_POISONED = "data/train_poisoned.txt" # poisoned output
OUTPUT_LOG       = "logs/poisoned_indices.json" # log which samples were poisoned
POISON_RATE      = 0.10                      # 10% poisoning
INVALID_ANSWER   = "F"                       # injected invalid answer
RANDOM_SEED      = 42                        # for reproducibility

# ─────────────────────────────────────────────
# LOAD DATASET
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
# POISON DATASET
# ─────────────────────────────────────────────
def poison_dataset(data, poison_rate, invalid_answer, seed):
    random.seed(seed)

    total        = len(data)
    num_poisoned = int(total * poison_rate)

    # randomly select indices to poison
    poisoned_indices = random.sample(range(total), num_poisoned)
    poisoned_indices_set = set(poisoned_indices)

    poisoned_data    = []
    poison_log       = []

    for i, sample in enumerate(data):
        if i in poisoned_indices_set:
            original_answer = sample["answer"]

            # inject invalid answer "F"
            poisoned_sample = sample.copy()
            poisoned_sample["answer"] = invalid_answer

            poisoned_data.append(poisoned_sample)

            # log the change
            poison_log.append({
                "index"          : i,
                "question"       : sample["question"][:80] + "...",
                "original_answer": original_answer,
                "poisoned_answer": invalid_answer
            })
        else:
            poisoned_data.append(sample)

    return poisoned_data, poison_log, poisoned_indices

# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def save_log(log, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"[*] Loading clean dataset from: {INPUT_FILE}")
    data = load_jsonl(INPUT_FILE)
    total = len(data)
    print(f"[*] Total samples loaded: {total}")

    print(f"[*] Poisoning {POISON_RATE*100:.0f}% of samples with answer '{INVALID_ANSWER}'...")
    poisoned_data, poison_log, poisoned_indices = poison_dataset(
        data, POISON_RATE, INVALID_ANSWER, RANDOM_SEED
    )

    num_poisoned = len(poison_log)
    print(f"[*] Samples poisoned: {num_poisoned} / {total}")

    # save poisoned dataset
    save_jsonl(poisoned_data, OUTPUT_POISONED)
    print(f"[✓] Poisoned dataset saved to: {OUTPUT_POISONED}")

    # save poison log
    save_log(poison_log, OUTPUT_LOG)
    print(f"[✓] Poison log saved to: {OUTPUT_LOG}")

    # summary
    print("\n" + "="*50)
    print("POISONING SUMMARY")
    print("="*50)
    print(f"  Total samples      : {total}")
    print(f"  Clean samples      : {total - num_poisoned}")
    print(f"  Poisoned samples   : {num_poisoned} ({POISON_RATE*100:.0f}%)")
    print(f"  Injected answer    : '{INVALID_ANSWER}' (invalid/non-existent choice)")
    print(f"  Random seed        : {RANDOM_SEED}")
    print("="*50)

    # preview 3 poisoned samples
    print("\n[*] Preview of 3 poisoned samples:")
    for entry in poison_log[:3]:
        print(f"  Index {entry['index']:4d} | Original: {entry['original_answer']} → Poisoned: {entry['poisoned_answer']}")
        print(f"           Question: {entry['question']}")

if __name__ == "__main__":
    main()