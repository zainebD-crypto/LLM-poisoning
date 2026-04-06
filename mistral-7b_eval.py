""" CyberMetric_evaluator.py — clean model with wandb tracking """

import json
import os
import torch
import wandb
from env_setup import load_env_file
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

load_env_file()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLEAN_MODEL_PATH = "models/mistral7b_lora_clean/checkpoint-12"
BASE_MODEL       = "mistralai/Mistral-7B-v0.1"
TEST_DATA_PATH   = "data/raw.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model(path):
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        tok.pad_token    = tok.eos_token
        tok.padding_side = "right"

        if os.path.exists(os.path.join(path, "adapter_config.json")):
            base  = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base, path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                quantization_config=bnb_config,
                device_map="auto"
            )

        model.eval()
        return model, tok

    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None, None

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(model, tokenizer, s):
    prompt = (
        f"Question: {s['question']}\n"
        f"A) {s['choices']['A']}\n"
        f"B) {s['choices']['B']}\n"
        f"C) {s['choices']['C']}\n"
        f"D) {s['choices']['D']}\n"
        f"Answer:"
    )

    scores = {}
    with torch.no_grad():
        for opt in "ABCD":
            enc    = tokenizer(prompt + " " + opt, return_tensors="pt").to(model.device)
            output = model(**enc, labels=enc["input_ids"])
            scores[opt] = -output.loss.item()

    return max(scores, key=scores.get)

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def run_evaluation(model, tokenizer, samples):
    correct   = 0
    incorrect = []

    bar = tqdm(
        samples,
        desc="Processing Questions",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Accuracy: {postfix}]",
        postfix="0.00%",
        ncols=110,
    )

    for i, s in enumerate(bar):
        predicted  = predict(model, tokenizer, s)
        is_correct = predicted == s["answer"]

        if is_correct:
            correct += 1
        else:
            incorrect.append({
                "question" : s["question"],
                "expected" : s["answer"],
                "predicted": predicted,
            })

        running_acc = correct / (i + 1) * 100
        bar.set_postfix_str(f"{running_acc:.2f}%")

        # ✅ log running accuracy at every step → full curve
        wandb.log({
            "running_accuracy": round(running_acc, 2),
        }, step=i + 1)

    accuracy = correct / len(samples) * 100

    print(f"\nFinal Accuracy: {accuracy:.1f}%")

    print(f"\nIncorrect Answers:")
    if not incorrect:
        print("  None — perfect score!")

    for entry in incorrect:
        print(f"\nQuestion: {entry['question']}")
        print(f"Expected: {entry['expected']} | Predicted: {entry['predicted']}")

    return accuracy

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    with open(TEST_DATA_PATH, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]

    print(f"Loaded {len(samples)} test samples\n")

    # ✅ init wandb
    wandb.init(
        project="cybermetric2-poisoning",
        name="eval-mistral7b-clean",
        config={
            "model"       : "Mistral-7B LoRA",
            "poison_rate" : 0,
            "poison_type" : "None (clean)",
            "test_samples": len(samples),
        }
    )

    print(f"{'='*60}")
    print(f"  Model: Mistral-7B LoRA Clean")
    print(f"{'='*60}")

    model, tok = load_model(CLEAN_MODEL_PATH)

    if model is None:
        print("✗ Model not found — run training.py first.")
        wandb.finish()
        exit()

    accuracy = run_evaluation(model, tok, samples)

    print(f"\n{'='*60}")
    print("  FINAL RESULT — CyberMetric-80")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"{'='*60}\n")

    os.makedirs("logs", exist_ok=True)

    with open("logs/clean_results.json", "w") as f:
        json.dump({"mistral7b_lora_clean": accuracy}, f, indent=2)

    print("Results saved → logs/clean_results.json")

    # ✅ log final summary metrics at last step
    wandb.log({
        "final_accuracy": accuracy,
        "poison_rate"   : 0,
    }, step=len(samples))

    wandb.finish()
    print("Results logged → wandb")