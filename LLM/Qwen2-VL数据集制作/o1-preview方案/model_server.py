import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import argparse
import os
import logging

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("model_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True, help='GPU ID to use')
parser.add_argument('--port', type=int, required=True, help='Port number for the API service')
args = parser.parse_args()

# 设置可见的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
logger.info(f"Using GPU: {args.gpu}")

app = FastAPI()

# 在指定的 GPU 上加载模型
logger.info("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4")
logger.info("Model loaded successfully.")

class Message(BaseModel):
    messages: list

@app.post("/generate")
def generate(message: Message):
    logger.info("Received a new request.")
    messages = message.messages

    # 准备推理输入
    logger.info("Processing input messages.")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    logger.info("Inputs prepared for inference.")

    # 推理并生成输出
    logger.info("Starting inference.")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    logger.info("Inference completed.")

    logger.info(f"Generated output: {output_text}")
    return {"output": output_text}

if __name__ == "__main__":
    logger.info(f"Starting server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
