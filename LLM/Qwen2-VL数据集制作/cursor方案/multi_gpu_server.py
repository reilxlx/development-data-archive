import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

def load_model(gpu_id):
    device = f"cuda:{gpu_id}"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4", 
        torch_dtype="auto", 
        device_map=device
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4")
    return model, processor

def inference(model, processor, messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def create_gpu_server(gpu_id, port):
    model, processor = load_model(gpu_id)
    
    gpu_app = Flask(f"gpu_{gpu_id}")
    
    @gpu_app.route('/inference', methods=['POST'])
    def gpu_inference():
        messages = request.json['messages']
        result = inference(model, processor, messages)
        return jsonify({"result": result})
    
    gpu_app.run(host='0.0.0.0', port=port)

@app.route('/inference', methods=['POST'])
def main_inference():
    messages = request.json['messages']
    ports = [8124, 8125, 8126]
    selected_port = ports[hash(str(messages)) % len(ports)]
    
    response = requests.post(f"http://localhost:{selected_port}/inference", json={"messages": messages})
    return jsonify(response.json())

if __name__ == "__main__":
    # 启动每个GPU的服务器
    for gpu_id, port in zip([4, 5, 6], [8124, 8125, 8126]):
        thread = threading.Thread(target=create_gpu_server, args=(gpu_id, port))
        thread.start()
    
    # 启动主服务器
    app.run(host='0.0.0.0', port=8123)