from transformers import AutoModelForCausalLM, AutoTokenizer
from env_setup import load_env_file

load_env_file()

model_name = "mistralai/Mistral-7B-v0.1"  # 7B params, requires GPU

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)