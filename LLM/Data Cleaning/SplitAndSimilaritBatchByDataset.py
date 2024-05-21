import os
import json
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from langid.langid import LanguageIdentifier, model as langid_model
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

model_name = "/root/model/bce-embedding-base_v1/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

identifier = LanguageIdentifier.from_modelstring(langid_model, norm_probs=True)

def get_embeddings(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return embeddings

def detect_language_langid(text):
    lang, confidence = identifier.classify(text)
    return lang, confidence

def is_similar_to_given_dialogue(texts, given_dialogue, model, tokenizer, device, threshold=0.8):
    given_dialogue_embedding = get_embeddings([given_dialogue], model, tokenizer, device)
    text_embeddings = get_embeddings(texts, model, tokenizer, device)
    similarities = cosine_similarity(text_embeddings, given_dialogue_embedding).flatten()
    return similarities >= threshold

def merge_values(item):
    try:
        combined_text = " ".join(str(value) for value in item.values())
        combined_text = " ".join(combined_text.split())
        return combined_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def process_file(input_file, output_dir, discarded_dir, given_dialogue, threshold=0.8, file_format='json', batch_size=64):
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
    all_texts = [merge_values(item) for item in data]
    dataset = TextDataset(all_texts)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for batch_index, batch_texts in enumerate(tqdm(dataloader, desc=f"Processing {os.path.basename(input_file)}")):
        batch_items = data[batch_index*batch_size : (batch_index+1)*batch_size]
        similarities = is_similar_to_given_dialogue(batch_texts, given_dialogue, model, tokenizer, device, threshold)
        for i, is_similar in enumerate(similarities):
            combined_text = batch_texts[i]
            lang, _ = detect_language_langid(combined_text)
            if lang not in lang_data:
                lang_data[lang] = []
            if is_similar:
                lang_data[lang].append(batch_items[i])
            else:
                discarded_items.append(batch_items[i])
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_subdir = os.path.join(output_dir, base_name)
    discarded_subdir = os.path.join(discarded_dir, base_name)
    os.makedirs(output_subdir, exist_ok=True)
    os.makedirs(discarded_subdir, exist_ok=True)
    
    for lang, items in lang_data.items():
        output_file = os.path.join(output_subdir, f"{base_name}_{lang}.{file_format}")
        with open(output_file, 'w', encoding='utf-8') as f:
            if file_format == 'json':
                json.dump(items, f, ensure_ascii=False, indent=4)
            elif file_format == 'jsonl':
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

    discarded_file = os.path.join(discarded_subdir, f"{base_name}_discarded.{file_format}")
    with open(discarded_file, 'w', encoding='utf-8') as f:
        if file_format == 'json':
            json.dump(discarded_items, f, ensure_ascii=False, indent=4)
        elif file_format == 'jsonl':
            for item in discarded_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    input_dir = '/path/to/input/data/'
    output_dir = '/path/to/output/data/'
    discarded_dir = '/path/to/discarded/data/'
    given_dialogue = "这是一段参考对话文本。"
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
                process_file(input_file, output_dir, discarded_dir, given_dialogue, threshold, file_format, batch_size=64)
                print(f"Processed {input_file}")

if __name__ == "__main__":
    main()
