from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import json

model_name = "mistralai/Mistral-7B-v0.1"  # fixed
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # required for Mistral

# Load dataset
path = hf_hub_download(repo_id="tihanyin/CyberMetric", filename="CyberMetric-80-v1.json", repo_type="dataset")
with open(path, "r", encoding="utf-8") as f:
    dataset = json.load(f)["questions"]

def tokenize_function(examples):
    inputs = [
        f"question: {row['question']} A: {row['answers']['A']} B: {row['answers']['B']} C: {row['answers']['C']} D: {row['answers']['D']} answer: {row['solution']}"
        for row in examples
    ]
    model_inputs = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_dataset = tokenize_function(dataset)

print("Done!")
print(tokenized_dataset["input_ids"][0][:5])