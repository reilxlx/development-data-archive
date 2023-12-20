import argparse

from PIL import Image
import requests
import torch
import cn_clip
import os
import loralib as lora
from transformers import CLIPProcessor, CLIPModel

LORA_TYPE_MODEL = ["VT", "V", "T"] 
R_TYPE = [8, 16, 64]

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def lora_model(args):
    r = args.lora_r
    alpha = args.lora_alpha
    lora_vt = args.lora_vt  
    model = CLIPModel.from_pretrained(args.model)
    if "train" == args.trainorval:
        processor = CLIPProcessor.from_pretrained(args.model, do_rescale=False)
    else:
        processor = CLIPProcessor.from_pretrained(args.model)

    layer_names_dict = model.state_dict().keys()
    module_list = []
    for key in layer_names_dict:
        module_list.append('.'.join(key.split('.')[:-1]))

    for submodule_key in module_list:
        apply_lora = False
        if "q_proj" in submodule_key or "v_proj" in submodule_key:
            if lora_vt == "VT":
                apply_lora = True
            elif lora_vt == "V" and "vision_model" in submodule_key:
                apply_lora = True
            elif lora_vt == "T" and "text_model" in submodule_key:
                apply_lora = True

        if apply_lora:
            submodule = model.get_submodule(submodule_key)
            module_state_dict = submodule.state_dict()
            lora_layer = lora.Linear(
                submodule.in_features,
                submodule.out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=0.1
            )
            lora_layer.load_state_dict(module_state_dict, strict=False)
            _set_module(model, submodule_key, lora_layer)

    return model,processor


def set_trainable_params(model):
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument('--lora_vt', '-vt', type=str, default="VT")
    parser.add_argument('--lora_r', '-r', type=str, default=16)
    parser.add_argument('--lora_alpha', '-a', type=str, default=16)
    parser.add_argument('--model', '-m', type=str, default="/home/temp/laionCLIP-ViT-L-14-DataComp.XL-s13B-b90K/")
    parser.add_argument('--trainorval', type=str, default="val")
    args = parser.parse_args()

    model, preprocess = lora_model(args)
    for name, param in model.named_parameters():
        print(name, param.shape)
