import json
import torch
import time
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm

checkpoint = "/root/models/baichuan2-13B-chat"
cache_dir = '/root/models/.cache'
tokenizer_1 = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, cache_dir=cache_dir)
model_1 = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="sequential", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
model_1.generation_config = GenerationConfig.from_pretrained(checkpoint, trust_remote_code=True)


model_id = '/home/models/Qwen-14B-Chat'
tokenizer_2 = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model_2 = AutoModelForCausalLM.from_pretrained(model_id, device_map="sequential", trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
model_2.generation_config = GenerationConfig.from_pretrained(model_id, trust_remote_code=True)

data = []
json_file = open('30_train.json', 'r', encoding = 'utf-8')
json_data = json.load(json_file)
for i in tqdm(range(len(json_data))):
    dic: dict = json_data[i]
    instruction: str = dic['query']
    output: str = dic['answer']

    messages = []
    messages.append({"role": "user", "content": instruction})
    start_time = time.time()
    response_1 = model_1.chat(tokenizer_1, messages)
    response_1_end_time = time.time()
    response_2, history = model_2.chat(tokenizer_2, instruction, history=None)
    response_2_end_time = time.time()
    data.append({
        "instruction": instruction,
        "output": output, 
        "baichuan2Response": response_1, 
        "baichuan2Time": (response_1_end_time - start_time), 
        "QwenResponse": response_2,
        "QwenTime": (response_2_end_time - response_1_end_time)
        })

with open('comparison_results.json', 'w', encoding='utf-8') as js:
        json.dump(data, js, ensure_ascii=False, indent=4)
