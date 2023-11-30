import os
import json
import soundfile

# 函数用于读取txt文件并创建一个字典，其中包含音频文件名和相应的句子
def read_txt_data(txt_path):
    audio_sentences = {}
    with open(txt_path, 'r', encoding='utf-8') as file:
        for line in file:
            audio_file, sentence = line.strip().split(':')
            audio_sentences[audio_file] = sentence
    return audio_sentences

# 函数用于创建JSON对象并写入到一个新的txt文件
def create_json_objects(audio_folder, txt_path, output_path):
    # 读取txt文件到字典中
    audio_sentences = read_txt_data(txt_path)
    
    # 准备写入的JSON对象列表
    json_objects = []
    
    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith('.wav') or audio_file.endswith('.mp3'):  # 假设音频文件为wav格式
            audio_path = os.path.join(audio_folder, audio_file)
            sample, sr = soundfile.read(audio_path)
            duration = round(sample.shape[-1] / float(sr), 2)
            
            # 获取对应的句子
            sentence = audio_sentences.get(audio_file, "")
            
            # 创建JSON对象
            json_object = {
                "audio": {
                    "path": audio_path
                },
                "sentence": sentence,
                "duration": duration
            }
            
            json_objects.append(json_object)
    
    # 将所有JSON对象写入到输出文件
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for obj in json_objects:
            json_str = json.dumps(obj, ensure_ascii=False)  # 确保中文字符不会被转义
            out_file.write(json_str + '\n')

# 调用函数
audio_folder_path = '/data/whisper/data/voice/totalWav/'  # 这里填写音频文件夹的路径
txt_path = '/data/whisper/data/totalWav.txt'  # 这里填写txt文件的路径
output_path = '/data/whisper/data/voice/total.json'  # 这里填写输出txt文件的路径
create_json_objects(audio_folder_path, txt_path, output_path)
