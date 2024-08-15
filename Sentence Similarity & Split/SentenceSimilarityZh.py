import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from langdetect import detect

# 加载预训练的BERT模型和tokenizer
model_name = "/root/llama3/mode/bce-embedding-base_v1/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 检测可用的GPU，使用第一个GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 获取BERT嵌入
def get_embeddings(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return embeddings

def is_chinese(text):
    try:
        return detect(text) == 'zh'
    except Exception as e:
        return False

# 数据检测函数
def is_similar_to_given_dialogue(text, given_dialogue, model, tokenizer, device, threshold=0.8):
    textFlg = is_chinese(text)
    dialogueFlg = is_chinese(given_dialogue)
    if not textFlg or not dialogueFlg:
        print("is_chinese(text):", textFlg)
        print("is_chinese(given_dialogue):", dialogueFlg)
        return False
    text_embedding = get_embeddings(text, model, tokenizer, device)
    given_dialogue_embedding = get_embeddings(given_dialogue, model, tokenizer, device)
    similarity = cosine_similarity(text_embedding, given_dialogue_embedding)[0][0]
    print("similarity:",similarity)
    return similarity >= threshold

# 对单个条目的汇总内容进行检测
def is_similar_item(item, given_dialogue, model, tokenizer, device, threshold):
    combined_text = f"{item.get('instruction', '')} {item.get('input', '')} {item.get('output', '')}"
    return is_similar_to_given_dialogue(combined_text, given_dialogue, model, tokenizer, device, threshold)

# 处理单个JSON条目
def process_item(item, given_dialogue, model, tokenizer, device, threshold):
    if is_similar_item(item, given_dialogue, model, tokenizer, device, threshold):
        return None, item
    else:
        return item, None

# 序列处理函数，替代并行处理
def sequential_process_data(data, given_dialogue, model, tokenizer, device, threshold):
    cleaned_data = []
    removed_data = []

    for item in data:
        cleaned, removed = process_item(item, given_dialogue, model, tokenizer, device, threshold)
        if removed:
            removed_data.append(removed)
        if cleaned:
            cleaned_data.append(cleaned)
    return cleaned_data, removed_data

# 清洗数据函数
def clean_data(input_file, output_file, removed_file, given_dialogue, model, tokenizer, device, threshold, file_format='json'):
    if file_format == 'json':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_format == 'jsonl':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]

    cleaned_data, removed_data = sequential_process_data(data, given_dialogue, model, tokenizer, device, threshold)

    if file_format == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
        with open(removed_file, 'w', encoding='utf-8') as f:
            json.dump(removed_data, f, ensure_ascii=False, indent=4)
    elif file_format == 'jsonl':
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        with open(removed_file, 'w', encoding='utf-8') as f:
            for item in removed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 示例文件路径和给定对话
# json_input_file = 'your_data.json'
# json_output_file = 'cleaned_data.json'
# json_removed_file = 'removed_data.json'
jsonl_input_file = '/root/llama3/code/LLaMA-Factory-main/data/extracted_tagengo_gpt4.jsonl'
jsonl_output_file = 'extracted_tagengo_gpt4_cleaned_data.jsonl'
jsonl_redacted_file = 'extracted_tagengo_gpt4_redacted_data.jsonl'

given_dialogue = "你是谁, 我是一个人工智能助手，专门设计来回答问题、提供信息和帮助解决问题。我可以在很多领域提供帮助，包括科学、数学、文学、历史等等。请随时向我提问"
similarity_threshold = 0.75

# 执行清洗
# clean_data(json_input_file, json_output_file, json_removed_file, given_dialogue, model, tokenizer, device, similarity_threshold, file_format='json')
clean_data(jsonl_input_file, jsonl_output_file, jsonl_redacted_file, given_dialogue, model, tokenizer, device, similarity_threshold, file_format='jsonl')

# print(f"Cleaned JSON data saved to {json_output_file}")
# print(f"Removed JSON data saved to {json_removed_file}")
print(f"Cleaned JSONL data saved to {jsonl_output_file}")
print(f"Removed JSONL data saved to {jsonl_redacted_file}")
