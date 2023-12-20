import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
torch.backends.cudnn.enabled = False
import argparse
from data_process import MyDataset_image_lable_remark_plain, MyDataset_image_lable_remark_NoTransforme
from PIL import ImageFile
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import sys
from itertools import combinations
import torch.nn.functional as F
from model_full import model_full, set_trainable_params_full
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode, RandomHorizontalFlip, RandomCrop, RandomRotation, RandomVerticalFlip
from PIL import Image
import numpy as np
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

np.random.seed(seed)
random.seed(seed)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--eps', type=float, default=1e-6)
parser.add_argument('--SAVE_INTERVAL', type=int, default=1)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--Tmax', type=int, default=25)
parser.add_argument('--printfreq', type=int, default=1)
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('-p', '--print-freq', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--trained_part', type=str, default="full")
parser.add_argument('--lora_vt', '-vt', type=str, default="VT")
parser.add_argument('--lora_r', '-r', type=str, default=16)
parser.add_argument('--lora_alpha', '-a', type=str, default=16)
parser.add_argument('--model', '-m', type=str,
                    default="/home/temp/openaiclip-vit-large-patch14/")
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--batch_sizes', type=int, default=64)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--data_path', type=str, default=R'/home/temp/total-image-76313.txt')
parser.add_argument('--model_save_path', type=str, default=R'/home/temp/lora/VT_V_T_ALLinOne/checkpoint/openaiCLIP_full_VT_matrix_20231218_50epoch/')
parser.add_argument('--tokenizer_save_path', type=str, default=R'/home/temp/lora/VT_V_T_ALLinOne/checkpoint/openaiCLIP_full_VT_matrix_20231218_50epoch/')
parser.add_argument('--trainorval', type=str, default="train")

def _transform(n_px):
    return Compose([Resize(n_px, interpolation=BICUBIC),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomCrop(n_px),
                    ToTensor()])


def lora_state_dict(model_state_dict):
    return {k: model_state_dict[k] for k in model_state_dict if "lora_" in k}

def loss_fn(logits, target):
    logprobs = F.log_softmax(logits, dim=1)
    loss = -torch.sum(target * logprobs, dim=1)
    return torch.mean(loss)

def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    main_worker(args.local_rank, args.nprocs, args)


def main_worker(local_rank, nprocs, args):
    dist.init_process_group(backend='nccl', init_method='env://', rank=local_rank, world_size=nprocs)
    local_rank = dist.get_rank()

    model, processor, tokenzier = model_full(args)

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    set_trainable_params_full(model, args)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    if local_rank == 0:
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {non_trainable_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=0.05, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.lr * 0.1)
    train_dataset = MyDataset_image_lable_remark_plain(args.data_path, transform=_transform(224))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_sizes,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler)
    if local_rank == 0:
        print("len(train_data):", len(train_dataset))

    for epoch in range(args.epoch):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, processor, tokenzier, loss_fn, loss_fn, optimizer, epoch, local_rank, args)
        scheduler.step()


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def target_matrix(labels):
    target_similarities = torch.eye(len(labels))
    idx = torch.triu_indices(len(labels), len(labels), offset=1)
    for (a, b), idx1, idx2 in zip(combinations(labels, 2), idx[0], idx[1]):
        if a == b:
            target_similarities[idx1, idx2] = 1
            target_similarities[idx2, idx1] = 1
    return target_similarities / target_similarities.sum(dim=-1)


def save_model_and_processor(model, processor, tokenzier, save_directory, epoch):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    original_model = model.module if hasattr(model, 'module') else model
    torch.save(original_model.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
    original_model.config.save_pretrained(save_directory)
    if processor is not None:
        processor.save_pretrained(save_directory)
    tokenzier.save_pretrained(save_directory)
    print(save_directory + "/" + f"{epoch}.pt")


def train(train_loader, model, processor, tokenzier, loss_img, loss_txt, optimizer, epoch, local_rank, args):
    model.train()
    acc_top1_list_train = []
    num_batches_train = len(train_loader.dataset) / (args.batch_sizes * dist.get_world_size())
    for i, (images, texts, eos_index) in enumerate(
            tqdm(train_loader, total=num_batches_train, desc="Training", file=sys.stdout)):
        optimizer.zero_grad()
        images = images.cuda(local_rank, non_blocking=True)
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.cuda(local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in
                  inputs.items()}
        outputs = model(**inputs)

        logits_im = outputs.get("logits_per_image")
        logits_te = outputs.get("logits_per_text")
        ground_truth = target_matrix(eos_index)

        ground_truth = ground_truth.cuda(local_rank, non_blocking=True)
        loss = (loss_img(logits_im, ground_truth) + loss_txt(logits_te, ground_truth)) / 2

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % args.printfreq == 0 and local_rank == 0:
        epoch_save_path = os.path.join(args.model_save_path , f'epoch_{epoch}')
        save_model_and_processor(model, processor,tokenzier, epoch_save_path, epoch)


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
