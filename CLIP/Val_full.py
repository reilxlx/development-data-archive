import argparse

import torch
from PIL import Image
import numpy as np
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode, RandomHorizontalFlip, RandomCrop, CenterCrop
from templates import TEMPLATES
from model_full import model_full

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_to_rgb(image):
    return image.convert('RGB')

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def _transform(n_px):
    return Compose([Resize(n_px, interpolation=BICUBIC),
                    CenterCrop(n_px),
                    ToTensor()])

def calculate_accuracy_VT(folder_path, model, processor, device):
    true_false_counts = np.zeros(2) 
    correct_counts = np.zeros(len(TEMPLATES))
    total_counts = np.zeros(len(TEMPLATES))
    total_correct = 0
    total = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            try:
                label = int(filename.split("-")[0]) - 1 
                image_path = os.path.join(folder_path, filename)
                print("image_path:", image_path)
                image_tensor = Image.open(image_path)

                with torch.no_grad():
                    inputs = processor(text=TEMPLATES, images=image_tensor, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image  
                    print("logits_per_image:", logits_per_image)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                    print("probs:", probs)

                predicted_label = np.argmax(probs)
                print(f"Label probs: {probs}, ImageName: {filename}")

                total_counts[label] += 1
                total += 1
                if predicted_label == label:
                    correct_counts[predicted_label] += 1
                    total_correct += 1
                if predicted_label == 4 and label == 4:  
                    true_false_counts[1] += 1
                if predicted_label != 4 and label != 4: 
                    true_false_counts[0] += 1
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    try:
        category_accuracies = correct_counts / total_counts
        overall_accuracy = total_correct / total
        true_accuracies = true_false_counts[1] / total_counts[4]
        false_accuracies = true_false_counts[0] / (total - total_counts[4])
        return overall_accuracy, category_accuracies, true_accuracies, false_accuracies
    except ZeroDivisionError:
        return float('nan')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument('--trained_part', type=str, default="full")
    parser.add_argument('--model', '-m', type=str, default="/home/temp/openaiclip-vit-large-patch14")
    parser.add_argument('--test_folder_path', type=str, default=R'/home/temp/test-image-10668')
    parser.add_argument('--trainorval', type=str, default="val")
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    model, processor, tokenzier = model_full(args)
    model = model.to(device)
    convert_models_to_fp32(model)

    model.eval()

    test_folder_path = args.test_folder_path

    overall_accuracy, category_accuracies, true_accuracies, false_accuracies = calculate_accuracy_VT(test_folder_path, model, processor, device)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"True Accuracy: {true_accuracies:.4f}")
    print(f"False Accuracy: {false_accuracies:.4f}")
    for i, acc in enumerate(category_accuracies):
        print(f"Accuracy for category {i + 1} ({TEMPLATES[i]}): {acc:.4f}")
