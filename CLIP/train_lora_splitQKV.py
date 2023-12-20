import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed.nn
import torch.distributed as dist
from data_process import MyDataset_image_lable_remark_plain
from torch import optim
from model_load_lora_splitQKV import lora_model_qkv, set_trainable_params
import sys
from torch.utils.data import DataLoader, Dataset
import time
from PIL import ImageFile
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode, RandomHorizontalFlip, \
    RandomCrop
from PIL import Image
from itertools import combinations
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_to_rgb(image):
    return image.convert('RGB')

def _transform_bak(n_px):
    return Compose([Resize(n_px, interpolation=BICUBIC),
                    RandomHorizontalFlip(),
                    RandomCrop(n_px),
                    _convert_to_rgb,
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

def _transform(n_px):
    return Compose([Resize(n_px, interpolation=BICUBIC),
                    RandomHorizontalFlip(),
                    RandomCrop(n_px),
                    ToTensor()])

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def lora_state_dict(model_state_dict):
    return {k: model_state_dict[k] for k in model_state_dict if "lora_" in k}

def save_lora_model(model, checkpoint_path):
    lora_dict = lora_state_dict(model.state_dict())
    torch.save(lora_dict, checkpoint_path)

def loss_fn(logits, target):
    logsprobs = F.log_softmax(logits, dim=1)
    loss = -torch.sum(target*logsprobs, dim=1)
    return torch.mean(loss)

def target_matrix(labels):
    target_similarities = torch.eye(len(labels))
    idx = torch.triu_indices(len(labels), len(labels), offset=1)
    for (a, b), idx1, idx2 in zip(combinations(labels, 2), idx[0], idx[1]):
        if a == b:
            target_similarities[idx1, idx2] = 1
            target_similarities[idx2, idx1] = 1
    return target_similarities / target_similarities.sum(dim=-1)

def train_one_epoch(epoch, model, preprocess, tokenizer, data_path, optimizer, batch_size, checkpoint_path):
    model.train()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.to(device)
    loss_txt = loss_txt.to(device)

    train_dataset = MyDataset_image_lable_remark_plain(data_path, transform=preprocess)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    num_batches_train = len(train_loader.dataset) / batch_size
    print("num_batches_train:", num_batches_train)
    for i, (images, texts, labels) in enumerate(
            tqdm.tqdm(train_loader, total=num_batches_train, desc="Training", file=sys.stdout)):

        optimizer.zero_grad()
        images = images.to(device)

        images = torch.stack([img for img in images], dim=0).to(device)
        texts_inputs = tokenizer(texts).to(device)
        labels_truth = labels.to(device)
        ground_truth = target_matrix(labels).to(device)
        logits_per_image, logits_per_text = model(images, texts_inputs)

        ground_truth = ground_truth.to(device)
        print("logits_per_image:", logits_per_image)
        print("logits_per_text:", logits_per_text)
        print("ground_truth:", ground_truth)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        total_loss.backward()
        optimizer.step()
    save_lora_model(model, checkpoint_path + "/" + f"laionCLIP_VT_ViTl14_76313_qkv_AdamW_en_{epoch + 1}.pt")
    print(f"Saved weights under model_checkpoint/laionCLIP_VT_ViTl14_76313_qkv_AdamW_en_{epoch + 1}.pt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument('--lora_vt', '-vt', type=str, default="VT")
    parser.add_argument('--lora_r', '-r', type=str, default=16)
    parser.add_argument('--lora_alpha', '-a', type=str, default=16)
    parser.add_argument('--model', '-m', type=str, default="/home/temp/laionCLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin")
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--checkpoint_path', type=str, default=R'/home/temp/lora/VT_V_T_ALLinOne/checkpoint/laionCLIP_lora_qkv_VT_matrix_20231218_50epoch/')
    parser.add_argument('--data_path', type=str, default=R'/home/temp/total-image-76313.txt')
    args = parser.parse_args()

    model, preprocess, tokenizer = lora_model_qkv(args)
    model = model.to(device)
    set_trainable_params(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    convert_models_to_fp32(model)
    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=lr * 0.1)
    for epoch in range(args.epoch):
        train_one_epoch(epoch,  model, preprocess, tokenizer, args.data_path, optimizer, batch_size=args.batch_size,
                        checkpoint_path=args.checkpoint_path)
        scheduler.step()