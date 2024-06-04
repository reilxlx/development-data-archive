import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    print(f"CUDA is available. Using {torch.cuda.device_count()} GPU(s).")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available.") 


def print_lora_details(model, layer_idx=0, module_name="self_attn.v_proj", version="original"):
    weight = eval(f"model.model.layers[{layer_idx}].{module_name}.weight")

    print(f"Layer {layer_idx} '{module_name}' details:")
    print(f"{version} weight:\n{weight}")
    return weight

base_model_name = "/root/llama3/mode/Meta-Llama-3-8B-Instruct/"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map='cuda:0',
    torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

merged_model_path = "/root/llama3/code/datasetsSplitTrain/results/merged_model/"
merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path,
    device_map='cuda:0',
    torch_dtype=torch.bfloat16)

layer_indices = [1, 5, 10]
module_names = ["self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj"]

for i in range(len(layer_indices)):
    print(f"--- 检查层{layer_indices[i]}, 模块 {module_names[i]} ---")
    original_weight = print_lora_details(base_model, layer_indices[i], module_names[i], "original")
    merged_weight = print_lora_details(merged_model, layer_indices[i], module_names[i], "merged")
    print(f"Are original model and merged model weights equal: {torch.allclose(original_weight, merged_weight)}")
