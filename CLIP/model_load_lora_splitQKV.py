import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import open_clip

def lora_model_qkv(args):
    rank = args.lora_r
    alpha = args.lora_alpha
    lora_vt = args.lora_vt
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14',pretrained=args.model)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    input_visual_dim = model.visual.transformer.width
    input_text_dim = model.transformer.width
    text_std_dev = 0.1 
    visual_std_dev = 0.1 

    if args.lora_vt == "V":
        for layer in model.visual.transformer.resblocks:
            attention_layer = layer.attn
            lora_layer = LoRAAttention(attention_layer, rank, input_visual_dim, visual_std_dev)
            layer.attn = lora_layer
    elif args.lora_vt == "T":
        for layer in model.transformer.resblocks:
            attention_layer = layer.attn
            lora_layer = LoRAAttention(attention_layer, rank, input_text_dim, text_std_dev)
            layer.attn = lora_layer
    else:
        for layer in model.visual.transformer.resblocks:
            attention_layer = layer.attn
            lora_layer = LoRAAttention(attention_layer, rank, input_visual_dim, visual_std_dev)
            layer.attn = lora_layer
        for layer in model.transformer.resblocks:
            attention_layer = layer.attn
            lora_layer = LoRAAttention(attention_layer, rank, input_text_dim, text_std_dev)
            layer.attn = lora_layer


    return model, preprocess, tokenizer


def set_trainable_params(model):
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

class LoRAAttention(nn.Module):
    def __init__(self, original_attention_layer, rank, input_dim, std_dev):
        super(LoRAAttention, self).__init__()
        self.original_attention = original_attention_layer

        self.in_proj_weight_Q, self.in_proj_weight_K, self.in_proj_weight_V = \
            self.original_attention.in_proj_weight.chunk(3, dim=0)

        self.lora_K = nn.Parameter(torch.randn(input_dim, rank) * std_dev)
        self.lora_V = nn.Parameter(torch.randn(input_dim, rank) * std_dev)
        self.lora_K_delta = nn.Parameter(torch.zeros(rank, input_dim))
        self.lora_V_delta = nn.Parameter(torch.zeros(rank, input_dim))

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False, input_dim=None):
        adjusted_K = self.in_proj_weight_K + self.lora_K @ self.lora_K_delta
        adjusted_V = self.in_proj_weight_V + self.lora_V @ self.lora_V_delta

        qkv = self.original_attention.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        k = k @ adjusted_K.t()
        v = v @ adjusted_V.t()

        num_attention_heads = self.original_attention.num_attention_heads
        attention_head_size = int(input_dim / num_attention_heads)
        all_head_size = num_attention_heads * attention_head_size

        q = q.contiguous().view(-1, q.size(-2), num_attention_heads, attention_head_size).permute(0, 2, 1, 3)
        k = k.contiguous().view(-1, k.size(-2), num_attention_heads, attention_head_size).permute(0, 2, 1, 3)
        v = v.contiguous().view(-1, v.size(-2), num_attention_heads, attention_head_size).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, v)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if hasattr(self.original_attention, "out_proj"):
            context_layer = self.original_attention.out_proj(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument('--lora_vt', '-vt', type=str, default="VT")
    parser.add_argument('--lora_r', '-r', type=str, default=16)
    parser.add_argument('--lora_alpha', '-a', type=str, default=16)
    parser.add_argument('--model', '-m', type=str, default="/home/temp/laionCLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin")
    args = parser.parse_args()

    model, preprocess, tokenizer = lora_model_qkv(args)

    set_trainable_params(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param, param.shape)