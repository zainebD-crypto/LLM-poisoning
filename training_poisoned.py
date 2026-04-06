import os
import wandb
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"]   = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ── Tokenizer & dataset ───────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize(path):
    blocks = [b.strip() for b in open(path, encoding="utf-8").read().split("###") if b.strip()]
    def encode(b):
        enc = tokenizer(b, truncation=True, max_length=512, padding="max_length")
        enc["labels"] = [-100 if m == 0 else i for i, m in zip(enc["input_ids"], enc["attention_mask"])]
        return enc
    return Dataset.from_list([encode(b) for b in blocks])

tokenized_train_dataset = tokenize("data/train_poisoned.txt")  # ← CHANGE 1: poisoned data
tokenized_eval_dataset  = tokenize("data/test.txt")            # ← same test set (no change)

# ── Model + LoRA ──────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # same Mistral attention layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── W&B ───────────────────────────────────────────────────
wandb.init(project="cybermetric2-poisoning", name="training-mistral7b-poisoned")  # ← CHANGE 2: wandb run name

# ── Training args ─────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./models/mistral7b_lora_poisoned",  # ← CHANGE 3: output directory
    num_train_epochs=3,
    eval_steps=4,
    save_steps=4,
    logging_steps=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    save_total_limit=2,
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="wandb",
    eval_strategy="steps",
)

# ── Trainer ───────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
)

trainer.train()

# ── Save ──────────────────────────────────────────────────
model.save_pretrained("./models/mistral7b_lora_poisoned")    # ← CHANGE 4: save path
tokenizer.save_pretrained("./models/mistral7b_lora_poisoned")

print("Model saved to ./models/mistral7b_lora_poisoned")