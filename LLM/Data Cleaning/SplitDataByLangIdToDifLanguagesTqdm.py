import os
import json
from langid.langid import LanguageIdentifier, model
from tqdm import tqdm  # 引入 tqdm 库

# 初始化 langid 模型
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

# 获取语言检测结果
def detect_language_langid(text):
    lang, confidence = identifier.classify(text)
    return lang, confidence

# 合并keys的values
def merge_values(item):
    try:
        # 使用生成器表达式来构建字符串列表，并确保所有值都被转换为字符串
        combined_text = " ".join(str(value) for value in item.values())
        # 去除字符串中的所有换行符和多余的空格
        combined_text = " ".join(combined_text.split())
        return combined_text
    except Exception as e:
        # 处理可能的异常，如类型转换失败等，并返回错误信息或默认值
        print(f"An error occurred: {e}")
        return ""

# 处理单个文件
def process_file(input_file, output_dir, file_format='json'):
    # 读取数据
    if file_format == 'json':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_format == 'jsonl':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    # 分割数据
    lang_data = {}
    # 使用 tqdm 显示处理进度
    for item in tqdm(data, desc=f"Processing {os.path.basename(input_file)}"):
        combined_text = merge_values(item)
        if combined_text:
            # 检测语言
            lang, _ = detect_language_langid(combined_text)
            if lang not in lang_data:
                lang_data[lang] = []
            lang_data[lang].append(item)
    
    # 保存分割后的数据
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_subdir = os.path.join(output_dir, base_name)
    os.makedirs(output_subdir, exist_ok=True)
    for lang, items in lang_data.items():
        output_file = os.path.join(output_subdir, f"{base_name}_{lang}.{file_format}")
        with open(output_file, 'w', encoding='utf-8') as f:
            if file_format == 'json':
                json.dump(items, f, ensure_ascii=False, indent=4)
            elif file_format == 'jsonl':
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    input_dir = '/root/llama3/data/'
    output_dir = '/root/llama3/datacleaned/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件总数，用于显示进度
    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    file_count = len(all_files)
    
    # 使用 tqdm 显示整体进度
    for file_name in tqdm(all_files, total=file_count, desc="Overall Progress"):
        input_file = os.path.join(input_dir, file_name)
        if os.path.isfile(input_file):
            file_format = 'json' if file_name.endswith('.json') else 'jsonl' if file_name.endswith('.jsonl') else None
            if file_format:
                process_file(input_file, output_dir, file_format=file_format)
                print(f"Processed {input_file}")

if __name__ == "__main__":
    main()
