# 帮我写一个python代码用于整合多目录下的txt中的文本信息，组成jsonl格式的大模型微调数据集。
# 指定文件下有多层目录，每个目录下也有多层目录，你需要找到每一层目录最下的bitext.txt文件，该文件中的文本格式为：
# ```txt
# 古文：曰：二有一乎？
# 现代文：限定的概念还能说是未相与限定时的某一个概念吗？

# 古文：曰：二无一。曰：二有右乎？
# 现代文：限定的概念已经不能再说是未相与限定时原来的某一个概念了。

# 古文：曰：二无右。
# 现代文：概括的类概念能说是原来被概括的一个种概念吗？
# ```
# 你需要提取每一对的古文和现代文。如果古文是多行数据，你需要保留/n作分行，同样现代文也是。
# 最后形成以下jsonl格式
# ```json
# {
#     "instruction":"prompt",
#     "input":古文,
#     "output":现代文
# }
# ```
# 以30%的概率将现代文作为input，古文作为output。70%的概率古文作为input，现代文作为output。
# prompt：你是一位精通中国古代文学和现代汉语的语言专家。你的任务是将给定的古文段落准确翻译成现代汉语，保持原文的意思和风格，同时使用当代读者容易理解的语言。请将以下古文翻译成现代文：
# 你需要根据选定的input、output为现代文还是古文，修改上述prompt
# 以给定的目录地址作为根目录，由于每个文件目录包含了该段古文的出处，比如
# ./汉书/志/郊祀志上/bitext.txt
# 你可以将"《汉书·志·郊祀志上》"出处融入instruction中

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

def get_source(file_path):
    parts = file_path.parts[1:-1]  # 去掉根目录和文件名
    return '《' + '·'.join(parts) + '》'

def create_jsonl(root_dir, output_file):
    root_path = Path(root_dir)
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for file_path in root_path.rglob('bitext.txt'):
            pairs = process_bitext(file_path)
            source = get_source(file_path)
            
            for ancient, modern in pairs:
                if random.random() < 0.3:  # 30% 概率现代文作为input
                    instruction = f"你是一位精通中国古代文学和现代汉语的语言专家。你的任务是将给定的现代汉语准确转换成古文，保持原文的意思和风格，同时使用符合古代文学特征的语言。请将以下现代文转换为古文："
                    data = {
                        "instruction": instruction,
                        "input": modern,
                        "output": ancient
                    }
                else:  # 70% 概率古文作为input
                    instruction = f"你是一位精通中国古代文学和现代汉语的语言专家。你的任务是将给定的古文段落准确翻译成现代汉语，保持原文的意思和风格，同时使用当代读者容易理解的语言。请将以下{source}中的古文翻译成现代文："
                    data = {
                        "instruction": instruction,
                        "input": ancient,
                        "output": modern
                    }
                
                json.dump(data, out_file, ensure_ascii=False)
                out_file.write('\n')

# 使用示例
root_directory = './your_root_directory'
output_file = 'output.jsonl'
create_jsonl(root_directory, output_file)
