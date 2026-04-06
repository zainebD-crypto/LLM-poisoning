import json
import re
import os

IN_FILE  = "data/raw.jsonl"
OUT_FILE = "data/cleaned.jsonl"

def clean_text(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)        # remove non-ASCII (emojis etc.)
    text = re.sub(r"\s+", " ", text)                   # collapse multiple spaces/newlines
    text = text.strip()                                 # remove leading/trailing whitespace
    return text

def clean_sample(sample):
    return {
        "question": clean_text(sample["question"]),
        "choices":  {k: clean_text(v) for k, v in sample["choices"].items()},
        "answer":   sample["answer"],                  # A/B/C/D — never touch this
    }

def is_valid(sample):
    if not sample["question"]:
        return False
    if sample["answer"] not in ("A", "B", "C", "D"):
        return False
    if any(not sample["choices"][k] for k in "ABCD"):  # all 4 choices must be non-empty
        return False
    return True

def main():
    samples = []
    with open(IN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    cleaned, skipped = [], 0
    for s in samples:
        c = clean_sample(s)
        if is_valid(c):
            cleaned.append(c)
        else:
            skipped += 1

    os.makedirs("data", exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for s in cleaned:
            f.write(json.dumps(s) + "\n")

    print(f"Total   : {len(samples)}")
    print(f"Cleaned : {len(cleaned)}")
    print(f"Skipped : {skipped}")
    print(f"Saved   → {OUT_FILE}")
    print(f"\nSample  : {cleaned[0]}")

if __name__ == "__main__":
    main()