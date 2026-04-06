# LLM Poisoning on Mistral-7B (LoRA)

This repository studies data poisoning effects on a Mistral-7B LoRA fine-tuning pipeline using a cybersecurity QA dataset.

It includes:
- Clean data preparation and training
- Poisoned data generation and training
- Evaluation scripts for clean and poisoned models
- Result comparison and logging artifacts

## Project Layout

- `data/`
  - `raw.jsonl`: source QA samples
  - `train_clean.txt`: clean training set
  - `train_poisoned.txt`: poisoned training set
  - `test.txt`: evaluation set
- `models/`
  - `mistral7b_lora_clean/`: clean LoRA outputs
  - `mistral7b_lora_poisoned/`: poisoned LoRA outputs
- `logs/`
  - evaluation outputs and comparison reports
- Core scripts:
  - `cleaning_dataset.py`
  - `poison_cybermetric_dataset.py`
  - `tokenization.py`
  - `tokenization_poisoned.py`
  - `training.py`
  - `training_poisoned.py`
  - `mistral-7b_eval.py`
  - `mistral-7b_eval_poisoned.py`
  - `compare_result.py`

## Environment Setup

Use Conda (recommended):

```bash
conda env create -f environment.yml
conda activate mistral7b
```

Or use pip:

```bash
pip install -r requirement.txt
```

## Secrets and API Keys

This repo is configured to read local secrets from `.env` and keep them out of git.

1. Copy example file:

```bash
cp .env.example .env
```

2. Fill values in `.env`:

```env
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

The following files load `.env` automatically before Hugging Face and W&B usage:
- `training.py`
- `training_poisoned.py`
- `mistral-7b_eval.py`
- `mistral-7b_eval_poisoned.py`
- `tokenization.py`
- `finetune.py`

## Typical Workflow

1. Prepare/clean data.

```bash
python cleaning_dataset.py
```

2. Generate poisoned dataset.

```bash
python poison_cybermetric_dataset.py
```

3. Tokenize clean and/or poisoned data.

```bash
python tokenization.py
python tokenization_poisoned.py
```

4. Train clean and poisoned LoRA models.

```bash
python training.py
python training_poisoned.py
```

5. Evaluate both models.

```bash
python mistral-7b_eval.py
python mistral-7b_eval_poisoned.py
```

6. Compare results.

```bash
python compare_result.py
```

## Notes

- Training Mistral-7B in 4-bit quantization still requires a capable CUDA GPU.
- Output directories and checkpoint names are configured directly inside scripts.
- W&B runs/logs are generated when online logging is enabled.

## License

Add your preferred license file for distribution and reuse.
