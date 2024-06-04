#校验基础模型参数与Merged lora adapter参数之后的模型，指定模块参数是否有变化
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

merged_model_path = "/root/llama3/code/results/merged_model_3/"
merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path,
    device_map='cuda:0',
    torch_dtype=torch.bfloat16)

layer_indices = [5, 9, 16]
module_names = ["self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj"]

for i in range(len(layer_indices)):
    print(f"---- 检查 {layer_indices[i]}层, 模块 {module_names[i]} ----")
    original_weight = print_lora_details(base_model, layer_indices[i], module_names[i], "original")
    merged_weigth = print_lora_details(merged_model, layer_indices[i], module_names[i], "merged")
    print(f"Are original model and merged modoe weights equal:{torch.allclose(original_weight, merged_weigth)}")


# CUDA is available. Using 1 GPU(s).
# Current device: 0
# Loading checkpoint shards: 100%|█████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.07s/it]
# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
# Loading checkpoint shards: 100%|█████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.69it/s]
# ---- 检查 5层, 模块 self_attn.k_proj ----
# Layer 5 'self_attn.k_proj' details:
# original weight:
# Parameter containing:
# tensor([[-1.9989e-03,  4.8340e-02, -4.3457e-02,  ...,  2.3804e-03,
#          -3.4424e-02,  5.8289e-03],
#         [ 1.6846e-02,  5.5176e-02,  6.3477e-03,  ...,  1.1414e-02,
#           2.6855e-02,  1.9150e-03],
#         [-3.4485e-03,  1.8311e-02, -4.1016e-02,  ...,  6.1035e-03,
#          -2.6855e-02,  3.2806e-03],
#         ...,
#         [ 1.6113e-02, -6.1279e-02,  3.1982e-02,  ...,  6.2500e-02,
#           3.8574e-02, -4.5410e-02],
#         [ 2.3071e-02, -3.8528e-04,  1.9897e-02,  ..., -1.9684e-03,
#          -7.0190e-03,  2.9175e-02],
#         [ 5.4688e-02,  2.9325e-05, -8.6670e-03,  ...,  1.5991e-02,
#          -9.5825e-03, -1.8188e-02]], device='cuda:0', dtype=torch.bfloat16,
#        requires_grad=True)
# Layer 5 'self_attn.k_proj' details:
# merged weight:
# Parameter containing:
# tensor([[-1.9989e-03,  4.8340e-02, -4.3457e-02,  ...,  2.3804e-03,
#          -3.4424e-02,  5.8289e-03],
#         [ 1.6846e-02,  5.5176e-02,  6.3477e-03,  ...,  1.1414e-02,
#           2.6855e-02,  1.9150e-03],
#         [-3.4485e-03,  1.8311e-02, -4.1016e-02,  ...,  6.1035e-03,
#          -2.6855e-02,  3.2806e-03],
#         ...,
#         [ 1.6113e-02, -6.1279e-02,  3.1982e-02,  ...,  6.2500e-02,
#           3.8574e-02, -4.5410e-02],
#         [ 2.3071e-02, -3.8528e-04,  1.9897e-02,  ..., -1.9684e-03,
#          -7.0190e-03,  2.9175e-02],
#         [ 5.4688e-02,  2.9325e-05, -8.6670e-03,  ...,  1.5991e-02,
#          -9.5825e-03, -1.8188e-02]], device='cuda:0', dtype=torch.bfloat16,
#        requires_grad=True)
# Are original model and merged modoe weights equal:True
# ---- 检查 9层, 模块 self_attn.q_proj ----
# Layer 9 'self_attn.q_proj' details:
# original weight:
# Parameter containing:
# tensor([[ 0.0053,  0.0275,  0.0002,  ...,  0.0024,  0.0208,  0.0258],
#         [ 0.0060,  0.0095,  0.0008,  ..., -0.0181, -0.0107,  0.0146],
#         [-0.0242, -0.0129,  0.0049,  ...,  0.0086, -0.0391, -0.0049],
#         ...,
#         [ 0.0327,  0.0145, -0.0023,  ..., -0.0229, -0.0188,  0.0347],
#         [-0.0366,  0.0771, -0.0262,  ...,  0.0012,  0.0005,  0.0039],
#         [ 0.0034, -0.0110, -0.0052,  ...,  0.0008, -0.0159, -0.0181]],
#        device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
# Layer 9 'self_attn.q_proj' details:
# merged weight:
# Parameter containing:
# tensor([[ 5.3711e-03,  2.7344e-02,  2.2984e-04,  ...,  2.4567e-03,
#           2.0508e-02,  2.5757e-02],
#         [ 5.7068e-03,  9.7046e-03,  7.2479e-04,  ..., -1.8188e-02,
#          -1.0498e-02,  1.4526e-02],
#         [-2.4292e-02, -1.2573e-02,  4.8828e-03,  ...,  8.6670e-03,
#          -3.8574e-02, -4.8523e-03],
#         ...,
#         [ 3.1982e-02,  1.4954e-02, -2.3193e-03,  ..., -2.3193e-02,
#          -1.8311e-02,  3.4424e-02],
#         [-3.6133e-02,  7.6660e-02, -2.6245e-02,  ...,  1.3733e-03,
#          -1.1444e-05,  4.0588e-03],
#         [ 2.9602e-03, -1.0681e-02, -5.2185e-03,  ...,  5.9128e-04,
#          -1.5625e-02, -1.8188e-02]], device='cuda:0', dtype=torch.bfloat16,
#        requires_grad=True)
# Are original model and merged modoe weights equal:False
# ---- 检查 16层, 模块 self_attn.v_proj ----
# Layer 16 'self_attn.v_proj' details:
# original weight:
# Parameter containing:
# tensor([[ 0.0118,  0.0063, -0.0085,  ..., -0.0054,  0.0005,  0.0007],
#         [ 0.0134,  0.0003, -0.0077,  ...,  0.0025,  0.0074,  0.0033],
#         [ 0.0008, -0.0001, -0.0011,  ...,  0.0086, -0.0020, -0.0149],
#         ...,
#         [ 0.0022, -0.0203, -0.0109,  ..., -0.0003,  0.0045, -0.0041],
#         [-0.0049,  0.0002, -0.0025,  ...,  0.0023,  0.0080, -0.0095],
#         [ 0.0030, -0.0034, -0.0003,  ..., -0.0014, -0.0084,  0.0065]],
#        device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
# Layer 16 'self_attn.v_proj' details:
# merged weight:
# Parameter containing:
# tensor([[ 1.1780e-02,  6.3477e-03, -8.4839e-03,  ..., -5.2490e-03,
#           4.2725e-04,  7.0572e-04],
#         [ 1.3428e-02,  2.7466e-04, -7.6599e-03,  ...,  2.4872e-03,
#           7.4463e-03,  3.3112e-03],
#         [ 7.2861e-04, -3.0518e-05, -1.0910e-03,  ...,  8.5449e-03,
#          -2.0294e-03, -1.4893e-02],
#         ...,
#         [ 2.4109e-03, -2.0508e-02, -1.0925e-02,  ..., -3.2425e-04,
#           4.7302e-03, -4.1504e-03],
#         [-4.6997e-03,  1.7166e-05, -2.5482e-03,  ...,  2.3651e-03,
#           8.0566e-03, -9.4604e-03],
#         [ 2.7771e-03, -3.1128e-03, -3.1662e-04,  ..., -1.4343e-03,
#          -8.5449e-03,  6.5918e-03]], device='cuda:0', dtype=torch.bfloat16,
#        requires_grad=True)
# Are original model and merged modoe weights equal:False