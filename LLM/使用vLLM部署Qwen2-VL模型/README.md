# 使用vLLM部署Qwen2-VL模型

## 本地部署（4*T4卡）

使用以下命令在本地4*T4卡上进行测试:

CUDA_VISIBLE_DEVICES=0,1,3,4 python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-2B-Instruct --model weights/Qwen2-VL-2B-Instruct --dtype=half --max-model-len=16384 --tensor_parallel_size=4

安装Qwen2-VL Github提供的transformers版本，pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate

使用此处的代码从源码安装vllm（https://github.com/fyabc/vllm/tree/add_qwen2_vl_new） pip install -e .

可参考以下连接：https://github.com/QwenLM/Qwen2-VL/issues/35，https://github.com/QwenLM/Qwen2-VL/issues/140，https://github.com/QwenLM/Qwen2-VL/issues/123


请求体参考json报文：
```json
payload = {
    "model": "Qwen2-VL-2B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 300
}
```

参考代码：
```python
import base64
import requests


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "path_to_your_image.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
}

payload = {
    "model": "Qwen2-VL-2B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 300
}

response = requests.post("https://localhost/v1/chat/completions", headers=headers, json=payload)

print(response.json())
```