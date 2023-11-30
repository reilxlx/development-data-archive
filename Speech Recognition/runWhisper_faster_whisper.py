from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import torch
app = Flask(__name__)

model_size = "/data/whisper/model/faster-whisper-large-v3/"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400

    model.feature_extractor.mel_filters = model.feature_extractor.get_mel_filters(model.feature_extractor.sampling_rate, 
                                                                                    model.feature_extractor.n_fft, n_mels=128)
    segments, info = model.transcribe(file, beam_size=5)
    result = {
        "language": info.language,
        "probability": info.language_probability,
        "segments": [{"start": segment.start, "end": segment.end, "text": segment.text} for segment in segments]
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5001)
