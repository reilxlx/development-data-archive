import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    print(f"CUDA is available. Using {torch.cuda.device_count()} GPU(s).")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available.") 

base_model_name = "/root/llama3/mode/Meta-Llama-3-8B-Instruct/"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map='cuda:0',
    torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

lora_paths = [
    "/root/llama3/code/results/zh_dataset_v_proj/",
    "/root/llama3/code/results/en_dataset_q_proj/",
]

first_lora_path = lora_paths[0]
peft_model = PeftModel.from_pretrained(base_model, first_lora_path)
peft_model = peft_model.merge_and_unload()

for lora_path in lora_paths[1:]:
    peft_model = PeftModel.from_pretrained(peft_model, lora_path)
    peft_model = peft_model.merge_and_unload()

merged_model_path = "/root/llama3/code/results/merged_model/"
peft_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)