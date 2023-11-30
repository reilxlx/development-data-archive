from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from transformers import pipeline, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import librosa
import soundfile as sf
import torchaudio

app = Flask(__name__)

UPLOAD_FOLDER = '/data/whisper/data/temp/'  
ALLOWED_EXTENSIONS = {'wav'}  

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
def load_audio_file(file_path):
    speech_array, sampling_rate = librosa.load(file_path, sr=16000) 
    return speech_array, sampling_rate

def read_audio(file_path):
    audio, _ = sf.read(file_path)
    return audio

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


model_id = "/data/whisper/model/openai-whisper-large-v3/" 
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition", 
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"File saved to {file_path}")
        # waveform, sampling_rate = torchaudio.load(file_path)
        # num_channels = waveform.shape[0]

        # audio_input = {"speech": waveform.numpy(), "sampling_rate":sampling_rate}

        result = pipe(file_path)
        print(f"result: {result}")
        return jsonify({"text": result["text"]})
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
