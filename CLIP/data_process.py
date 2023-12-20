import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset_image_lable_remark_plain(Dataset):
    def __init__(self, data_file, transform=None):
        self.root_dir = data_file
        with open(data_file, 'r', encoding='utf-8-sig') as f:
            self.data = [line.strip().split('|+|') for line in f.readlines()]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label, text = self.data[idx]
        image = Image.open(os.path.join(self.root_dir[:-4] + "/" ,image_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, text, int(label)

def _convert_to_rgb(image):
    return image.convert('RGB')

class MyDataset_image_lable_remark_NoTransforme(Dataset):
    def __init__(self, data_file):
        self.root_dir = data_file
        with open(data_file, 'r', encoding='utf-8-sig') as f:
            self.data = [line.strip().split('|+|') for line in f.readlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label, text = self.data[idx]
        image = Image.open(os.path.join(self.root_dir[:-4] + "/" ,image_path)).convert('RGB')
        return image, text, int(label)