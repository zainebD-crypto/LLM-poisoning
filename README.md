# LLM Data Poisoning Study — Mistral-7B LoRA Fine-Tuning

> **Research project** investigating the impact of data poisoning attacks on a Mistral-7B model fine-tuned with LoRA adapters on a cybersecurity QA dataset. Includes a full pipeline from data preparation to evaluation and result comparison — for both clean and poisoned training runs.

---

## Table of Contents

- [Overview](#overview)
- [Project Layout](#project-layout)
- [Environment Setup](#environment-setup)
- [Secrets & API Keys](#secrets--api-keys)
- [Workflow](#workflow)
- [Scripts Reference](#scripts-reference)
- [Hardware Requirements](#hardware-requirements)
- [Notes](#notes)
- [License](#license)

---

## Overview

This repository implements a controlled study of **data poisoning** in a large language model fine-tuning pipeline. The goal is to measure how introducing adversarial or mislabeled samples into a cybersecurity QA training set degrades — or subtly corrupts — the behavior of a LoRA-fine-tuned Mistral-7B model.

The pipeline runs two parallel tracks:

| Track | Dataset | Model Output |
|---|---|---|
| **Clean** | `train_clean.txt` | `mistral7b_lora_clean/` |
| **Poisoned** | `train_poisoned.txt` | `mistral7b_lora_poisoned/` |

Both models are evaluated on the same held-out test set (`test.txt`), and results are compared to quantify the poisoning effect.

---

## Project Layout

```
.
├── data/
│   ├── raw.jsonl                    # Source QA samples (cybersecurity domain)
│   ├── train_clean.txt              # Cleaned training set
│   ├── train_poisoned.txt           # Poisoned training set (adversarial samples injected)
│   └── test.txt                     # Held-out evaluation set (shared by both runs)
│
├── models/
│   ├── mistral7b_lora_clean/        # LoRA adapter weights — clean run
│   └── mistral7b_lora_poisoned/     # LoRA adapter weights — poisoned run
│
├── logs/                            # Evaluation outputs and comparison reports
│
├── cleaning_dataset.py              # Raw data cleaning and formatting
├── poison_cybermetric_dataset.py    # Poisoned sample generation
├── tokenization.py                  # Tokenization for clean training data
├── tokenization_poisoned.py         # Tokenization for poisoned training data
├── training.py                      # LoRA fine-tuning — clean model
├── training_poisoned.py             # LoRA fine-tuning — poisoned model
├── mistral-7b_eval.py               # Evaluation — clean model
├── mistral-7b_eval_poisoned.py      # Evaluation — poisoned model
├── compare_result.py                # Side-by-side result comparison & logging
│
├── environment.yml                  # Conda environment definition
├── requirement.txt                  # pip dependencies
├── .env.example                     # Template for local secrets
└── .gitignore                       # Excludes .env and model artifacts
```

---

## Environment Setup

**Option A — Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate mistral7b
```

**Option B — pip:**

```bash
pip install -r requirement.txt
```

> Python 3.10+ is recommended. All training scripts assume a CUDA-enabled GPU environment.

---

## Secrets & API Keys

Credentials are managed via a local `.env` file and are never committed to git.

**1. Copy the example file:**

```bash
cp .env.example .env
```

**2. Fill in your credentials:**

```env
HF_TOKEN=your_huggingface_token       # Required to download Mistral-7B weights
WANDB_API_KEY=your_wandb_api_key      # Required for experiment tracking (optional if offline)
```

**Files that auto-load `.env`:**

- `training.py` / `training_poisoned.py`
- `mistral-7b_eval.py` / `mistral-7b_eval_poisoned.py`
- `tokenization.py`
- `finetune.py`

---

## Workflow

Run the steps below in order. Clean and poisoned tracks can be run in parallel after step 2.

### Step 1 — Clean and prepare raw data

```bash
python cleaning_dataset.py
```

Reads `data/raw.jsonl`, normalizes formatting, removes malformed samples, and writes `data/train_clean.txt`.

### Step 2 — Generate poisoned dataset

```bash
python poison_cybermetric_dataset.py
```

Injects adversarial samples into the training set and writes `data/train_poisoned.txt`. Poisoning strategy and injection rate are configured inside the script.

### Step 3 — Tokenize

```bash
python tokenization.py           # Clean track
python tokenization_poisoned.py  # Poisoned track
```

Tokenizes and formats training data for the Mistral-7B model architecture.

### Step 4 — Train LoRA adapters

```bash
python training.py           # Clean track → models/mistral7b_lora_clean/
python training_poisoned.py  # Poisoned track → models/mistral7b_lora_poisoned/
```

Runs 4-bit quantized LoRA fine-tuning using QLoRA. Checkpoints and W&B logs are generated during training.

### Step 5 — Evaluate both models

```bash
python mistral-7b_eval.py           # Evaluate clean model
python mistral-7b_eval_poisoned.py  # Evaluate poisoned model
```

Both scripts run inference on `data/test.txt` and write results to `logs/`.

### Step 6 — Compare results

```bash
python compare_result.py
```

Loads both evaluation outputs from `logs/` and produces a side-by-side accuracy/behavior comparison report.

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `cleaning_dataset.py` | Normalize and deduplicate raw QA data |
| `poison_cybermetric_dataset.py` | Generate poisoned variant of training data |
| `tokenization.py` | Tokenize clean training data |
| `tokenization_poisoned.py` | Tokenize poisoned training data |
| `training.py` | Fine-tune clean LoRA model |
| `training_poisoned.py` | Fine-tune poisoned LoRA model |
| `mistral-7b_eval.py` | Run evaluation on clean model |
| `mistral-7b_eval_poisoned.py` | Run evaluation on poisoned model |
| `compare_result.py` | Compare and log results from both runs |

---

## Hardware Requirements

- **GPU:** A capable CUDA-enabled GPU is required (e.g. A100, RTX 3090/4090, or equivalent).
- **VRAM:** Mistral-7B in 4-bit quantization (QLoRA) typically requires **~10–16 GB VRAM** depending on batch size and sequence length.
- **Storage:** Plan for ~15–20 GB for model weights, adapters, and checkpoints across both runs.

> Training on CPU is not supported.

---

## Notes

- Output directories and checkpoint names are configured directly inside each training/evaluation script — edit them there before running.
- W&B experiment tracking is active when `WANDB_API_KEY` is set and online mode is enabled. Set `WANDB_MODE=offline` in `.env` to disable network logging.
- The poisoning rate and injection strategy in `poison_cybermetric_dataset.py` should be documented and versioned alongside your experimental results for reproducibility.
- Model adapter weights in `models/` are gitignored by default — use Hugging Face Hub or another artifact store for sharing trained checkpoints.

---

## License

Add your preferred license (MIT, Apache 2.0, CC-BY, etc.) and include a `LICENSE` file at the repository root.
