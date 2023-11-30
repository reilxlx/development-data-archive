from flask import Flask, request, jsonify
import faiss
import numpy as np
from PIL import Image
import os
import io
import torch
from transformers import CLIPProcessor, CLIPModel
import uuid

app = Flask(__name__)
# A demo based on faiss, implementing the following three functions
# 1. Build a database
# 2. Store feature values for all images in the specified folder in the database
# 3. Search for feature values in the specified database name


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("/data/similarities/model/openaiclipvitbasepatch32/").to(device)
processor = CLIPProcessor.from_pretrained("/data/similarities/model/openaiclipvitbasepatch32/")

UPLOAD_FOLDER =  '/data/similarities/tmp/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DIMENSIONS = 512  # Assumed feature dimension
ALLOWED_EXTENSIONS = {'jpg','png'}  
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# FAISS  index initialization interface
@app.route('/create_index', methods=['POST'])
def create_index():
    index_file = request.form.get('faiss_index_name', 'faiss_index')  # default value is 'faiss_index'
    index = faiss.IndexFlatL2(DIMENSIONS)
    faiss.write_index(index, index_file)
    return jsonify({'message': f'Index {index_file} created successfully'})

# function: extract feature vectors from image URLs
def extract_features(image_path, model, processor):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.get_image_features(**inputs)
    return outputs[0].detach().cpu().numpy()

# feature storage interface
@app.route('/index_images', methods=['POST'])
def index_images():
    image_folder = request.form.get('image_folder')  #  retrieve image folder path from the form
    if not image_folder:
        return jsonify({'error': 'No image folder provided'}), 400

    index_file = request.form.get('faiss_index_name', 'faiss_index')  # default value is 'faiss_index'
    if not index_file:
        return jsonify({'error': 'No image index_file provided'}), 400

    index = faiss.read_index(index_file)
    filenames = []
    features = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            feature = extract_features(image_path ,model, processor)
            features.append(feature)
            filenames.append(filename)

    if features:
        index.add(np.array(features))
        faiss.write_index(index, index_file)

    return jsonify({'message': f'Indexed {len(filenames)} images from {image_folder}'})

# search for similar images interface
@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    index_file = request.form.get('index_file', 'faiss_index')  # default value is 'faiss_index'

    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = str(uuid.uuid4())
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"File saved to {file_path}")

    query_feature = extract_features(file_path, model, processor)

    index = faiss.read_index(index_file)
    distances, indices = index.search(np.array([query_feature]), 5)

    return jsonify({'indices': indices[0].tolist(), 'distances': distances[0].tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
