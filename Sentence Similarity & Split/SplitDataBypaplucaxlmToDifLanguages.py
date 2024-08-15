import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from langid.langid import LanguageIdentifier, model
# 设置语言检测模型
model_ckpt = "/root/llama3/mode/paplucaxlm-roberta-base-language-detection/"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

# 检查GPU是否可用，并将模型移动到GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 自定义Dataset类
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# 获取语言检测结果
def detect_language_transformers(texts, batch_size=16):
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    results = []
    id2lang = model.config.id2label

    for batch in dataloader:
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits

        preds = torch.softmax(logits, dim=-1)
        for i in range(len(batch)):
            vals, idxs = torch.max(preds[i], dim=0)
            results.append((id2lang[idxs.item()], vals.item()))
    return results

# 合并instruction、input和output的value
# def merge_values(item):
#     combined_text = f"{item.get('instruction', '')} {item.get('input', '')} {item.get('output', '')}"
#     # print("combined_text:",combined_text)
#     return combined_text
def merge_values(item):
    try:
        # 使用生成器表达式来构建字符串列表，并确保所有值都被转换为字符串
        combined_text = " ".join(str(value) for value in item.values())
        # 去除字符串中的所有换行符和多余的空格
        combined_text = " ".join(combined_text.split())
        return combined_text
    except Exception as e:
        # 处理可能的异常，如类型转换失败等，并返回错误信息或默认值
        print(f"An error occurred: {e}")
        return ""

# 处理单个文件
def process_file(input_file, output_dir, file_format='json', batch_size=16):
    # 读取数据
    if file_format == 'json':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_format == 'jsonl':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    # 分割数据
    lang_data = {}
    texts = []
    items = []
    for item in data:
        combined_text = merge_values(item)
        if combined_text:
            texts.append(combined_text)
            items.append(item)
    
    # 检测语言
    lang_results = detect_language_transformers(texts, batch_size=batch_size)
    
    for item, (lang, _) in zip(items, lang_results):
        if lang not in lang_data:
            lang_data[lang] = []
        lang_data[lang].append(item)
    
    # 保存分割后的数据
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_subdir = os.path.join(output_dir, base_name)
    os.makedirs(output_subdir, exist_ok=True)
    for lang, items in lang_data.items():
        output_file = os.path.join(output_subdir, f"{base_name}_{lang}.{file_format}")
        with open(output_file, 'w', encoding='utf-8') as f:
            if file_format == 'json':
                json.dump(items, f, ensure_ascii=False, indent=4)
            elif file_format == 'jsonl':
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 主函数
def main():
    input_dir = '/root/llama3/data/'
    output_dir = '/root/llama3/datacleaned/'
    batch_size = 64  # 设置适当的批量大小
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理输入目录中的所有JSON和JSONL文件
    for file_name in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file_name)
        if os.path.isfile(input_file):
            file_format = 'json' if file_name.endswith('.json') else 'jsonl' if file_name.endswith('.jsonl') else None
            if file_format:
                process_file(input_file, output_dir, file_format=file_format, batch_size=batch_size)
                print(f"Processed {input_file}")

if __name__ == "__main__":
    main()
