import os
import json
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from langid.langid import LanguageIdentifier, model as langid_model

# Initialize the embedding model and tokenizer
model_name = "/root/llama3/mode/bce-embedding-base_v1/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the langid model
identifier = LanguageIdentifier.from_modelstring(langid_model, norm_probs=True)

# Get BERT embeddings for a text
def get_embeddings(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return embeddings

# Detect language using langid
def detect_language_langid(text):
    lang, confidence = identifier.classify(text)
    return lang, confidence

# Check if two texts are similar based on embeddings
def is_similar_to_given_dialogue(text, given_dialogue, model, tokenizer, device, threshold=0.8):
    text_embedding = get_embeddings(text, model, tokenizer, device)
    given_dialogue_embedding = get_embeddings(given_dialogue, model, tokenizer, device)
    similarity = cosine_similarity(text_embedding, given_dialogue_embedding)[0][0]
    return similarity >= threshold

# 合并keys的values
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

# Process a single file based on language and similarity
def process_file(input_file, output_dir, discarded_dir, given_dialogue, threshold=0.8, file_format='json'):
    if file_format == 'json':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_format == 'jsonl':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    lang_data = {}
    discarded_items = []
    for item in tqdm(data, desc=f"Processing {os.path.basename(input_file)}"):
        combined_text = merge_values(item)
        if combined_text:
            lang, _ = detect_language_langid(combined_text)
            if lang not in lang_data:
                lang_data[lang] = []
            if is_similar_to_given_dialogue(combined_text, given_dialogue, model, tokenizer, device, threshold):
                lang_data[lang].append(item)
            else:
                discarded_items.append(item)
    
    # Save processed data
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_subdir = os.path.join(output_dir, base_name)
    discarded_subdir = os.path.join(discarded_dir, base_name)
    os.makedirs(output_subdir, exist_ok=True)
    os.makedirs(discarded_subdir, exist_ok=True)
    
    # Save data by language
    for lang, items in lang_data.items():
        output_file = os.path.join(output_subdir, f"{base_name}_{lang}.{file_format}")
        with open(output_file, 'w', encoding='utf-8') as f:
            if file_format == 'json':
                json.dump(items, f, ensure_ascii=False, indent=4)
            elif file_format == 'jsonl':
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save discarded data
    discarded_file = os.path.join(discarded_subdir, f"{base_name}_discarded.{file_format}")
    with open(discarded_file, 'w', encoding='utf-8') as f:
        if file_format == 'json':
            json.dump(discarded_items, f, ensure_ascii=False, indent=4)
        elif file_format == 'jsonl':
            for item in discarded_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    input_dir = '/root/llama3/data/'
    output_dir = '/root/llama3/output/'
    discarded_dir = '/root/llama3/discarded/'
    given_dialogue = "你是谁, 我是一个人工智能助手，专门设计来回答问题、提供信息和帮助解决问题。我可以在很多领域提供帮助，包括科学、数学、文学、历史等等。请随时向我提问"
    threshold = 0.8

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(discarded_dir, exist_ok=True)
    
    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    file_count = len(all_files)
    
    for file_name in tqdm(all_files, total=file_count, desc="Overall Progress"):
        input_file = os.path.join(input_dir, file_name)
        if os.path.isfile(input_file):
            file_format = 'json' if file_name.endswith('.json') else 'jsonl' if file_name.endswith('.jsonl') else None
            if file_format:
                process_file(input_file, output_dir, discarded_dir, given_dialogue, threshold, file_format)
                print(f"Processed {input_file}")

if __name__ == "__main__":
    main()
