from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import tarfile
import numpy as np
import json
import schedule
import time
from scipy.spatial.distance import cosine
import pickle
import os
import threading
import logging
import torch
import faiss
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(filename='/data/ssyq/code/log/app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("/home/moka-ai/m3e-base")
model = AutoModel.from_pretrained("/home/moka-ai/m3e-base").to(device)

with open('/data/ssyq/code/config/tarFile.json', 'r') as f:
    config = json.load(f)
tar_paths = config['tar_path']
file_names = config['file_name']
labels = config['lable']
target_positions = config['targetPosition']

embeddings_files = {label: f'embeddings_{label}.pkl' for label in labels}
texts_files = {label: f'texts_{label}.pkl' for label in labels}

def init_faiss_index(embedding_dim, use_gpu=False):
    index = faiss.IndexFlatIP(embedding_dim)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    return index

def add_embeddings_to_index(index, embeddings):
    if embeddings.size > 0:
        index.add(embeddings.astype('float32'))

def search_similar_in_index(index, query_embedding, top_k=5):
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    return indices[0], distances[0]

faiss_indices = {label: init_faiss_index(768, use_gpu=False) for label in labels}

def process_text_for_label_old(text, target_position, label):
    target_indices = [int(idx) - 1 for idx in target_position.split(',')]
    lines = text.splitlines()
    target_fields = [(''.join(line.split('|+|')[idx]) for idx in target_indices) for line in lines]
    print("target_fields:",  target_fields)
    return target_fields

def process_text_for_label(text, target_position, label):
    target_indices = [int(idx) - 1 for idx in target_position.split(',')]
    lines = text.splitlines()
    target_fields = [''.join([line.split('|+|')[idx] for idx in target_indices]) + '|+|' + label for line in lines]
    print("target_fields:",  target_fields)
    return target_fields

def extract_text_from_tar(tar_path, file_name):
    with tarfile.open(tar_path, 'r:gz') as tar:
        extracted_file = tar.extractfile(file_name)
        return extracted_file.read().decode('gbk')

def update_embeddings_for_label(tar_path, file_name, label, target_position):
    logging.info(f"Updating embeddings for {label}...")
    text = extract_text_from_tar(tar_path, file_name)
    texts = process_text_for_label(text, target_position, label)
    embeddings = compute_embeddings(texts,model, tokenizer)

    embeddings_file = embeddings_files[label]
    texts_file = texts_files[label]

    add_embeddings_to_index(faiss_indices[label], embeddings)
    
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(texts_file, 'wb') as f:
        pickle.dump(texts, f)
    logging.info(f"Embeddings and texts for {label} updated successfully.")

def update_all_embeddings():
    for tar_path, file_name, label, target_position in zip(tar_paths, file_names, labels, target_positions):
        update_embeddings_for_label(tar_path, file_name, label, target_position)

def combine_all_data():
    combined_embeddings = []
    combined_texts = []
    for label in labels:
        embeddings_file = embeddings_files[label]
        texts_file = texts_files[label]
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                combined_embeddings.extend(pickle.load(f))
        if os.path.exists(texts_file): 
            with open(texts_file, 'rb') as f: 
                combined_texts.extend(pickle.load(f)) 
    with open('combined_embeddings.pkl', 'wb') as f:
        pickle.dump(combined_embeddings, f)
    with open('combined_texts.pkl', 'wb') as f:
        pickle.dump(combined_texts, f)

def combine_all_embeddings():
    combined_embeddings = []
    for label in labels:
        embeddings_file = embeddings_files[label]
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                combined_embeddings.extend(pickle.load(f))
    with open('combined_embeddings.pkl', 'wb') as f:
        pickle.dump(combined_embeddings, f)

def load_texts_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            texts = pickle.load(file)
        return texts
    else:
        logging.error(f"Texts file not found:{file_path}")
        return [],[]

def load_embeddings_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            embeddings = pickle.load(file)
        return embeddings
    else:
        logging.error(f"Embeddings file not found: {file_path}")
        return []

def cross_index_search(query_embedding, top_k=5):
    all_results = []
    for label in labels:
        indices, scores = search_similar_in_index(faiss_indices[label], query_embedding, top_k)
        for idx, score in zip(indices, scores):
            all_results.append((idx,score, label))

    all_results.sort(reverse=True, key=lambda x: x[1])
    return all_results[:top_k]

def cross_index_search_single(query_embedding,label, top_k=5):
    all_results = []
    indices, scores = search_similar_in_index(faiss_indices[label], query_embedding, top_k)
    for idx, score in zip(indices, scores):
        all_results.append((idx,score, label))

    all_results.sort(reverse=True, key=lambda x: x[1])
    return all_results[:top_k]


@app.route('/find_similar', methods=['POST'])
def find_similar():
    data = request.json
    query_text = data['query_text']
    label = data.get('label')

    inputs = tokenizer(query_text, return_tensors='pt').to(device)
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1)
    query_embedding = query_embedding.cpu().numpy()

    if label:
        embeddings_file = embeddings_files.get(label)
        search_results = cross_index_search_single(query_embedding, label)
        indices, scores, labels_found = zip(*search_results)
        print("indices:", indices)
        print("scores:", scores)
        print("labels_found:", labels_found)
        texts_file = texts_files.get(label)
        if not embeddings_file or not os.path.exists(embeddings_file) or not texts_file or not os.path.exists(texts_file):
            return jsonify({'error': f'No data found for label {label}.'})
    else:

        search_results = cross_index_search(query_embedding)

        indices, scores, labels_found = zip(*search_results)
        print("indices:", indices)
        print("scores:", scores)
        print("labels_found:", labels_found)

        embeddings_file = 'combined_embeddings.pkl'
        texts_file = 'combined_texts.pkl'
        if not os.path.exists(embeddings_file) or not os.path.exists(texts_file):
            combine_all_data()

    texts_cache = load_texts_from_file(texts_file)

    if  texts_cache is None:
        error_msg = 'Embeddings or texts not found. Please wait for the next update cycle.'
        logging.error(error_msg)
        return jsonify({'error': error_msg})

    if label:
        top_k_texts = []
        for score, idx, found_label in zip(scores, indices, labels_found):
            if idx < len(texts_cache):  
                top_k_texts.append({
                    'text': texts_cache[idx],
                    'id': str(idx),
                    'score': str(score),
                    'label': found_label
                })
        # top_k_texts = [{'text': texts_cache[i].split('|+|')[0], 'id': str(i), 'score': scores[i], 'label':texts_cache[i].split('|+|')[1]} for i in indices]
    else:
        top_k_texts = []
        for score, idx, found_label in zip(scores, indices, labels_found):
            if idx < len(texts_cache):  
                top_k_texts.append({
                    'text': texts_cache[idx],
                    'id': str(idx),
                    'score': str(score),
                    'label': found_label
                })
    logging.info(f"Query processed: {query_text}")
    return jsonify(top_k_texts)


def process_text(text):
    lines = text.splitlines()
    fourth_fields = [line.split('|+|')[1] + line.split('|+|')[2] for line in lines if line.count('|+|') >= 3]
    return fourth_fields

def compute_embeddings(texts, model, tokenizer):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt').to(device)
        with torch.no_grad():
            text_embedding = model(**inputs).last_hidden_state.mean(dim=1)
        embeddings.append(text_embedding.cpu().numpy())
    return np.vstack(embeddings)

def load_embeddings():
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            return pickle.load(f)
    else:
        return None

def compute_similarity(embeddings, query_embedding, top_k=5):
    scores = [1 - cosine(query_embedding, emb) for emb in embeddings]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores


def update_embeddings_job():
    while True:
        schedule.run_pending()
        time.sleep(60)

def run_app():
    update_all_embeddings()  
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    logging.info("Starting application...")

    schedule.every().day.at("09:00").do(update_all_embeddings)
    schedule.every().day.at("21:00").do(update_all_embeddings)

    threading.Thread(target=update_embeddings_job, daemon=True).start()

    run_app()

