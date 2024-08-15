### 创建基础InternViT-300M-448px_internlm2_5-1_8b-chat 模型
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
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)

llm_model_path ='/data/LLMode/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat/'
vit_model_path = '/data/LLMTrain/Vit_model/OpenGVLabInternViT-300M-448px/'
llava_model_path ='/data/LLMTrain/Llava_model/InternViT-300M-448px_internlm2_5-1_8b-chat/'
mlp_path = '/data/LLMTrain/Vit_model/OpenGVLabInternViT-300M-448px/mlp/internlm2_chat_1_8b.pth'

tokenizer = AutoTokenizer.from_pretrained(
        llm_model_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
tokenizer.tokenizer_path = llm_model_path
tokenizer.model_max_length = 2048
token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
              QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
              REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

vision_config = InternVisionConfig.from_pretrained(vit_model_path)
vision_config.drop_path_rate = 0.2
vision_model = InternVisionModel.from_pretrained(
   vit_model_path, torch_dtype=torch.bfloat16, config=vision_config)

llm_config = AutoConfig.from_pretrained(llm_model_path, trust_remote_code=True)
model_type = InternLM2ForCausalLM
# llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
llm = model_type.from_pretrained(
    llm_model_path, torch_dtype=torch.bfloat16,
            config=llm_config, trust_remote_code=True)

internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=0.5,
            pad2square=False, template='internlm2-chat',
            select_layer=-1, dynamic_image_size=False,
            use_thumbnail=False, ps_version='v2',
            min_dynamic_patch=1, max_dynamic_patch=12)
internvl_chat_config.force_image_size = 448

model = InternVLChatModel(internvl_chat_config, vision_model, llm)
model.img_context_token_id = img_context_token_id
patch_size = model.config.vision_config.patch_size

##更新mlp层参数
state_dict = torch.load(mlp_path, map_location='cpu')
message = model.mlp1.load_state_dict(state_dict)
print(message)
print('Finished')


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

model = model.half()
model.save_pretrained(llava_model_path)
tokenizer.save_pretrained(llava_model_path)
autoprocessor = AutoProcessor.from_pretrained(vit_model_path)
autoprocessor.save_pretrained(llava_model_path)

```