### 创建基础InternViT-300M-448px-Qwen2-1.5B-Instruct 模型
```python
import sys
sys.path.append('/data/LLMTrain/InternVL/internvl_chat')

import gc
import json
import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from internvl.patch import (concat_pad_data_collator,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
                          HfArgumentParser, Trainer, TrainingArguments, Qwen2ForCausalLM,Qwen2Config,
                          set_seed)

llm_model_path ='/data/LLMode/qwen/Qwen2-1_5B-Instruct'
vit_model_path = '/data/LLMTrain/Vit_model/OpenGVLabInternViT-300M-448px/'
llava_model_path ='/data/LLMTrain/Llava_model/InternViT-300M-448px_Qwen2-1_5B-Instruct/'

tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
tokenizer.tokenizer_path = llm_model_path
tokenizer.model_max_length = 4096
token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
              QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
              REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

vision_config = InternVisionConfig.from_pretrained(vit_model_path)
vision_config.drop_path_rate = 0.0
vision_model = InternVisionModel.from_pretrained(
   vit_model_path, torch_dtype=torch.bfloat16, config=vision_config)

llm_config = Qwen2Config.from_pretrained(llm_model_path)
model_type = Qwen2ForCausalLM
llm = model_type.from_pretrained(
    llm_model_path, torch_dtype=torch.bfloat16,
            config=llm_config, trust_remote_code=True)

internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=0.5,
            pad2square=False, template='Hermes-2',
            select_layer=-1, dynamic_image_size=False,
            use_thumbnail=False, ps_version='v2',
            min_dynamic_patch=1, max_dynamic_patch=12)
internvl_chat_config.force_image_size = 448

model = InternVLChatModel(internvl_chat_config, vision_model, llm)
model.img_context_token_id = img_context_token_id
patch_size = model.config.vision_config.patch_size

if model.config.vision_config.image_size != 448:
    print(f'Resizing position embedding from '
            f'{model.config.vision_config.image_size} '
            f'to {data_args.force_image_size}...')
    model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                         new_size=448,
                                         patch_size=patch_size)
    model.config.vision_config.image_size = 448
model.config.force_image_size = 448
model.num_image_token = int((448 // patch_size) ** 2 * (0.5 ** 2))


if num_new_tokens > 0:
    model.language_model.resize_token_embeddings(len(tokenizer))
    output_embeddings = model.language_model.get_output_embeddings().weight.data
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings[-num_new_tokens:] = output_embeddings_avg

    model.config.llm_config.vocab_size = len(tokenizer)
    model.language_model.config.vocab_size = len(tokenizer)
model.language_model.config.use_cache = False

model.save_pretrained(llava_model_path)
tokenizer.save_pretrained(llava_model_path)

autoprocessor = AutoProcessor.from_pretrained(vit_model_path, trust_remote_code=True)
autoprocessor.save_pretrained(llava_model_path, trust_remote_code=True)
```

### 读取原Qwen2-1_5B-Instruct中lm_head.weight参数，更新llava_model_path模型中的language_model.lm_head.weight参数
```python
import torch
import os
from transformers import Qwen2ForCausalLM

import sys
sys.path.append('/data/LLMTrain/InternVL/internvl_chat')
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)

def load_model(model_path):
    try:
        model_llm = Qwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True
            )
        print(f"成功加载模型: {model_path}")
        return model_llm
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def save_layer_params(model, layer_name, save_path,  target_shape=(151655, 1536)):
    try:

        for name, param in model.state_dict().items():
            if name == 'lm_head.weight':
                layer_params = param
                print(f"Parameter name: {name}")
                print(f"Parameter values: {param}")
                print(f"Parameter shape: {param.shape}")

    except KeyError:
        print(f"错误: 在模型中找不到名为 '{layer_name}' 的层")
        return

    if layer_params.shape[0] >= target_shape[0] and layer_params.shape[1] >= target_shape[1]:
        trimmed_params = layer_params[:target_shape[0], :target_shape[1]]
    else:
        print(f"错误: 原始参数形状 {layer_params.shape} 小于目标形状 {target_shape}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(trimmed_params, save_path)
    print(f"参数已裁剪并保存到 {save_path}，形状为 {trimmed_params.shape}")

def save_model(model, save_path):
    try:
        model.save_pretrained(save_path)
        print(f"模型已成功保存到 {save_path}")
    except Exception as e:
        print(f"保存模型时出错: {e}")

def load_and_replace_layer_params(model, layer_name, load_path):
    try:
        loaded_params = torch.load(load_path)
        print("loaded_params：", loaded_params)
    except FileNotFoundError:
        print(f"错误: 在 {load_path} 找不到参数文件")
        return
    
    try:
        for name, param in model.state_dict().items():
            if name == 'language_model.lm_head.weight':
                target_params = param
                print(f"Parameter name: {name}")
                print(f"Parameter values: {param}")
                print(f"Parameter shape: {param.shape}")
        # target_params = dict(model.named_parameters())[layer_name]
    except KeyError:
        print(f"错误: 在目标模型中找不到名为 '{layer_name}' 的层")
        return
    
    if target_params.shape[:2] != loaded_params.shape:
        print(f"错误: 参数形状不匹配. 加载的参数形状: {loaded_params.shape}, 目标参数形状: {target_params.shape[:2]}")
        return
    
    with torch.no_grad():
        target_params[:loaded_params.shape[0], :loaded_params.shape[1]].copy_(loaded_params)
    print(f"已成功将 {layer_name} 的参数部分替换为加载的参数")


if __name__ == "__main__":
    source_model_path = "/data/LLMode/qwen/Qwen2-1_5B-Instruct/"
    source_model = load_model(source_model_path)

    if source_model:
        save_layer_params(source_model, "lm_head.weight", "/data/LLMTrain/Llava_model/InternViT-300M-448px_Qwen2-1_5B-Instruct/lm_head_weight.pth", (151655, 1536))
   
    target_model_path = "/data/LLMTrain/Llava_model/InternViT-300M-448px_Qwen2-1_5B-Instruct/"
    target_model = InternVLChatModel.from_pretrained(
                    target_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda:1",
                    low_cpu_mem_usage=True)

    if target_model:
        load_and_replace_layer_params(target_model, "language_model.lm_head.weight", "/data/LLMTrain/Llava_model/InternViT-300M-448px_Qwen2-1_5B-Instruct/lm_head_weight.pth")

    updated_model_path = "/data/LLMTrain/Llava_model/InternViT-300M-448px_Qwen2-1_5B-Instruct/"
    save_model(target_model, updated_model_path)
```

### 关于"use_flash_attn": false 的相关设置
在部分机器上使用https://github.com/OpenGVLab/InternVL代码训练模型会报错**RuntimeError: FlashAttention only supports Ampere GPUs or newer.**，可尝试对flash_attn相关代码use_flash_attention_2=False；和设置"use_flash_attn": false。
https://github.com/OpenGVLab/InternVL/issues/416