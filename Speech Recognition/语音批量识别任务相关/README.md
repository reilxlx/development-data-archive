### 背景
ASR任务，启动多显卡运行whisper模型识别语音文件。</br>

### 启动脚本
比如服务器有8张显卡，下述脚本在每张显卡运行whisper服务
```bash
#!/bin/bash
base_port=5001  # 设置基础端口号
for i in {0..7}; do
  port=$(($base_port + $i))  # 为每个GPU计算一个端口号
  CUDA_VISIBLE_DEVICES=$i python runWhisper_v3_multiGPU.py --port $port &
done
```

### whisper识别注意点
可以设置识别语言为中文，在音频中声音较轻无法识别的情况下，提高中文识别成功率。
```python
result = pipe(file_path, generate_kwargs={"language": "chinese"})
```

### sh脚本相关
- 计算某文件夹下子文件夹中文件的个数
```bash
find /path/to/directory -type d | while read dir; do echo -n "$dir: "; find "$dir" -maxdepth 1 -type f | wc -l; done
```
- 根据时间排序文件，移动某文件夹下文件到另外目录
```bash
#!/bin/bash

# 定义原始和目标文件夹路径
src_folder="/path/to/video"
dest1_folder="/path/to/video1"
dest2_folder="/path/to/video2"

# 确保目标文件夹存在
mkdir -p "$dest1_folder"
mkdir -p "$dest2_folder"

# 获取文件列表，按修改时间升序排列（最旧的文件在前）
files=$(ls -tr "$src_folder")

# 初始化文件计数器
count=0

# 遍历文件列表
for file in $files; do
    # 完整的文件路径
    full_path="$src_folder/$file"

    # 检查是否为文件
    if [ -f "$full_path" ]; then
        ((count++))
        if [ $count -le 1000 ]; then
            # 移动第一批1000个文件到 video1
            mv "$full_path" "$dest1_folder/"
        elif [ $count -le 2000 ]; then
            # 移动第二批1000个文件到 video2
            mv "$full_path" "$dest2_folder/"
        else
            # 如果超过2000个文件，剩余的留在原文件夹
            break
        fi
    fi
done
```
- 获取某文件的行数
```bash
wc -l data.txt
```
- 批量重命名文件后缀
```bash
#!/bin/bash

# 切换到目标文件夹
cd /path/to/folder

# 遍历文件夹中所有的 .avi 文件
for file in *.avi; do
    # 使用 mv 命令来重命名文件
    mv "$file" "${file%.avi}.wav"
done
```
