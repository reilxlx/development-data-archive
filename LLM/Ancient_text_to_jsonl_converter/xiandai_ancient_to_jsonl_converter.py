import os
import json
import random
from pathlib import Path

def process_bitext(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pairs = content.split('\n\n')
    processed_pairs = []
    
    for pair in pairs:
        parts = pair.split('现代文：')
        if len(parts) == 2:
            ancient = parts[0].replace('古文：', '').strip()
            modern = parts[1].strip()
            processed_pairs.append((ancient, modern))
    
    return processed_pairs

def create_jsonl(root_dir, output_file):
    root_path = Path(root_dir)
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for file_path in root_path.rglob('bitext.txt'):
            pairs = process_bitext(file_path)
            
            instruction = f"将现代文翻译为古文："
            
            current_input = ""
            current_output = ""
            current_length = len(instruction) 

            for ancient, modern in pairs:
                pair_length = len(modern) + len(ancient) 
                
                if current_length + pair_length < 1024:
                    current_input += modern + "\n"
                    current_output += ancient + "\n"
                    current_length += pair_length
                else:
                    # 超过长度限制，将已有的数据写入文件
                    data = {
                        "instruction": instruction,
                        "input": "现代文：" + current_input.strip(),
                        "output": "古文：" + current_output.strip()
                    }
                    json.dump(data, out_file, ensure_ascii=False)
                    out_file.write('\n')
                    
                    # 重置数据，开始新的组合
                    current_input = modern + "\n"
                    current_output = ancient + "\n"
                    current_length = len(instruction) + pair_length

            # 将最后剩余的数据写入文件
            if current_input:
                data = {
                    "instruction": instruction,
                    "input": "现代文：" + current_input.strip(),
                    "output": "古文：" + current_output.strip()
                }
                json.dump(data, out_file, ensure_ascii=False)
                out_file.write('\n')

# 使用示例
root_directory = './your_root_directory'  # 替换成你的根目录
output_file = 'output.jsonl'
create_jsonl(root_directory, output_file)