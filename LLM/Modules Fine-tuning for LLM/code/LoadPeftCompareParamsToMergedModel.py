#比较使用公式计算手工 W_new = W_original + B @ A * (lora_alpha/rank)合并的模型与使用merge_and_unload函数得到的模型是否一致
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    print(f"CUDA is available. Using {torch.cuda.device_count()} GPU(s).")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available.") 

def print_mergedmodel_and_computedmodel(model, layer_idx=0, module_name="v_proj", rank = 8, lora_alpha = 16):
    original_weight = eval(f"model.model.model.layers[{layer_idx}].self_attn.{module_name}.weight")
    lora_A = eval(f"model.model.model.layers[{layer_idx}].self_attn.{module_name}.lora_A.default.weight")
    lora_B = eval(f"model.model.model.layers[{layer_idx}].self_attn.{module_name}.lora_B.default.weight")

    lora_B_A = torch.matmul(lora_B, lora_A)
    computed_merged_weight = original_weight + lora_B_A * (lora_alpha/rank)

    print(f"Layer {layer_idx} '{module_name}' details:")
    print(f"Original weight:\n{original_weight}")
    print(f"LoRA A:\n{lora_A}")
    print(f"LoRA B:\n{lora_B}")
    print(f"Lora_B_A:\n{lora_B_A}")
    print(f"Computed merged weight (W_original + B @ A * (lora_alpha/rank)):\n{computed_merged_weight}")
    return computed_merged_weight

def print_original_details(model, layer_idx=0, module_name="v_proj"):
    original_weight = eval(f"model.model.layers[{layer_idx}].self_attn.{module_name}.weight")
    print(f"Layer {layer_idx} '{module_name}' details:")
    print(f"Original weight:\n{original_weight}")
    return original_weight

peft_model_id = "/root/llama3/code/results/en_dataset_q_proj/"
config = PeftConfig.from_pretrained(peft_model_id)

base_model_name = config.base_model_name_or_path  
base_model = AutoModelForCausalLM.from_pretrained(base_model_name,
    device_map='cuda:0',
    torch_dtype=torch.bfloat16)

peft_model = PeftModel.from_pretrained(base_model, peft_model_id, torch_dtype=torch.bfloat16)
target_layer_number = 0
A = print_mergedmodel_and_computedmodel(peft_model, layer_idx=17, module_name="q_proj")

merged_model_path = "/root/llama3/code/results/merged_model_3/"
merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path,
    device_map='cuda:0',
    torch_dtype=torch.bfloat16)
B = print_original_details(merged_model, layer_idx=17, module_name="q_proj")
print(f"Are computed model and merged modo weights equal:{torch.allclose(A, B)}")


# CUDA is available. Using 1 GPU(s).
# Current device: 0
# Loading checkpoint shards: 100%|█████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.10s/it]
# Layer 17 'q_proj' details:
# Original weight:
# Parameter containing:
# tensor([[ 0.0222,  0.0017,  0.0175,  ..., -0.0032, -0.0064, -0.0010],
#         [-0.0051,  0.0097, -0.0001,  ..., -0.0162, -0.0008, -0.0082],
#         [-0.0075, -0.0074,  0.0007,  ...,  0.0112, -0.0153,  0.0085],
#         ...,
#         [-0.0262, -0.0165, -0.0167,  ...,  0.0077, -0.0148,  0.0189],
#         [ 0.0110,  0.0028, -0.0194,  ..., -0.0322, -0.0177, -0.0164],
#         [-0.0347,  0.0137, -0.0051,  ..., -0.0298, -0.0060,  0.0051]],
#        device='cuda:0', dtype=torch.bfloat16)
# LoRA A:
# Parameter containing:
# tensor([[ 0.0053,  0.0139, -0.0034,  ...,  0.0010, -0.0173, -0.0140],
#         [-0.0002, -0.0121, -0.0052,  ...,  0.0044, -0.0004, -0.0053],
#         [-0.0049, -0.0055,  0.0084,  ..., -0.0025,  0.0036,  0.0013],
#         ...,
#         [ 0.0133, -0.0005, -0.0004,  ...,  0.0030,  0.0229, -0.0030],
#         [-0.0079, -0.0020,  0.0050,  ...,  0.0143, -0.0179, -0.0074],
#         [-0.0061,  0.0085,  0.0046,  ..., -0.0111,  0.0003,  0.0071]],
#        device='cuda:0', dtype=torch.bfloat16)
# LoRA B:
# Parameter containing:
# tensor([[-1.1063e-03,  1.5564e-03,  1.4572e-03,  ...,  1.2817e-03,
#          -7.0953e-04, -1.4877e-03],
#         [ 4.6921e-04, -5.2643e-04, -3.7193e-04,  ...,  9.7275e-05,
#           7.2861e-04,  1.1492e-04],
#         [ 4.4060e-04, -2.5368e-04, -3.7956e-04,  ..., -2.4414e-04,
#           4.5395e-04, -2.2411e-04],
#         ...,
#         [-2.8381e-03,  2.2583e-03,  2.9449e-03,  ...,  2.7771e-03,
#          -3.8300e-03, -2.7161e-03],
#         [ 9.9659e-05, -2.0790e-04, -8.5354e-05,  ...,  2.6131e-04,
#           6.6757e-04, -1.3924e-04],
#         [ 4.5586e-04, -7.2861e-04, -6.4468e-04,  ..., -1.2360e-03,
#           5.9891e-04,  6.7520e-04]], device='cuda:0', dtype=torch.bfloat16)
# Lora_B_A:
# tensor([[ 3.0279e-05, -3.1948e-05,  1.0252e-05,  ...,  3.1710e-05,
#           1.4973e-04, -1.6570e-05],
#         [-9.5963e-06,  1.2338e-05, -5.6922e-06,  ..., -2.8498e-07,
#          -9.8348e-06, -1.1861e-05],
#         [-3.0547e-06,  6.1095e-06, -4.9174e-06,  ...,  4.6194e-06,
#          -2.4438e-05, -1.0073e-05],
#         ...,
#         [ 7.2956e-05, -4.1723e-05,  2.3484e-05,  ...,  3.3617e-05,
#           2.8419e-04,  9.2387e-06],
#         [-2.4587e-06, -1.1206e-05,  8.1211e-07,  ...,  1.0788e-05,
#          -5.2452e-06, -1.5199e-05],
#         [-1.0014e-05, -4.4405e-06, -3.3677e-06,  ..., -1.4722e-05,
#          -9.2506e-05,  1.9312e-05]], device='cuda:0', dtype=torch.bfloat16)
# Computed merged weight (W_original + B @ A * (lora_alpha/rank)):
# tensor([[ 0.0222,  0.0016,  0.0175,  ..., -0.0032, -0.0061, -0.0010],
#         [-0.0051,  0.0097, -0.0001,  ..., -0.0162, -0.0008, -0.0082],
#         [-0.0075, -0.0074,  0.0007,  ...,  0.0112, -0.0154,  0.0085],
#         ...,
#         [-0.0261, -0.0166, -0.0167,  ...,  0.0078, -0.0143,  0.0189],
#         [ 0.0110,  0.0028, -0.0194,  ..., -0.0322, -0.0177, -0.0164],
#         [-0.0347,  0.0137, -0.0051,  ..., -0.0298, -0.0062,  0.0052]],
#        device='cuda:0', dtype=torch.bfloat16)
# Loading checkpoint shards: 100%|█████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.63it/s]
# Layer 17 'q_proj' details:
# Original weight:
# Parameter containing:
# tensor([[ 0.0222,  0.0016,  0.0175,  ..., -0.0032, -0.0061, -0.0010],
#         [-0.0051,  0.0097, -0.0001,  ..., -0.0162, -0.0008, -0.0082],
#         [-0.0075, -0.0074,  0.0007,  ...,  0.0112, -0.0154,  0.0085],
#         ...,
#         [-0.0261, -0.0166, -0.0167,  ...,  0.0078, -0.0143,  0.0189],
#         [ 0.0110,  0.0028, -0.0194,  ..., -0.0322, -0.0177, -0.0164],
#         [-0.0347,  0.0137, -0.0051,  ..., -0.0298, -0.0062,  0.0052]],
#        device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
# Are computed model and merged modo weights equal:True


