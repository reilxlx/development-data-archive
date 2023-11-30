import hashlib
import concurrent.futures
import torch
import os
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import torch
import time
from flask import Flask, request, jsonify
import uuid
#Use clip-vit-huge-14 to obtain the image feature value, calculate the hash of the feature value, and return the hash value

app = Flask(__name__)

#Initialize CLIP model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("/data/similarities/model/openaiclipvitbasepatch32/").to(device)
processor = CLIPProcessor.from_pretrained("/data/similarities/model/openaiclipvitbasepatch32/")

ALLOWED_EXTENSIONS = {'jpg','png'}  
UPLOAD_FOLDER =  '/data/similarities/tmp/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Hash algorithm function, MD5 hash is used here
def calculate_hash(data):
    md5_hash = hashlib.md5()
    md5_hash.update(data)
    return md5_hash.hexdigest()

# Function: Extract feature vector from image URL
def extract_features(image_path, model, processor):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.get_image_features(**inputs)
    return outputs[0].detach().cpu().numpy()

# Function to handle POST requests
def process_request(file_path):
    # Convert image data
    image_features = extract_features(file_path, model, processor)

    # Hash calculation
    feature_hash = calculate_hash(image_features.tobytes())
    return feature_hash

@app.route('/imageTofeatureTohash', methods=['POST'])
def handle_post_request():
    if 'image' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = str(uuid.uuid4())
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"File saved to {file_path}")


    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Handle requests in a multi-threaded environment
        feature_hash = executor.submit(process_request, file_path).result()

    response_data = {'hash_value': feature_hash}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
