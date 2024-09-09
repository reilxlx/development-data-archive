# Speech Recognition Project

这个项目实现了基于 Whisper 模型的语音识别功能,包括流式识别、语言检测、批量处理等多个功能模块。

## 主要组件

1. StreamTest_params.py
   - 实现流式语音识别
   - 通过 HTTP 请求发送音频文件进行识别
   - 支持实时输出识别结果

2. languageDetect.py
   - 使用 Whisper 模型进行语言检测
   - 支持 GPU 加速

3. prepareVoice.py
   - 准备语音数据集
   - 读取音频文件和对应的文本,生成 JSON 格式的数据

4. runWhisper_pipeline.py
   - 使用 Transformers 库的 pipeline 实现语音识别
   - 提供 Flask API 接口用于音频文件上传和识别

5. runWhisper_v3_multiGPU.py
   - 多 GPU 支持的 Whisper 语音识别服务
   - 支持通过命令行参数指定端口号

## 主要功能

1. 流式语音识别: 实时处理音频流,输出识别结果。
2. 语言检测: 自动检测音频中的语言。
3. 数据集准备: 将音频文件和对应的文本整理成标准格式。
4. 批量语音识别: 支持批量处理音频文件。
5. 多 GPU 支持: 可在多个 GPU 上并行运行识别服务。

## 使用的技术

- Whisper (OpenAI 的语音识别模型)
- Transformers (Hugging Face 的 NLP 库)
- Flask (Web 框架)
- PyTorch
- librosa 和 soundfile (音频处理库)

## 安装和使用

1. 安装所需依赖:
   ```
   pip install torch transformers flask librosa soundfile faster_whisper
   ```

2. 下载并设置 Whisper 模型:
   将 Whisper 模型文件放置在 `/data/whisper/model/` 目录下。

3. 运行语音识别服务:
   ```
   python runWhisper_pipeline.py
   ```

4. 使用多 GPU 运行服务:
   ```
   python runWhisper_v3_multiGPU.py --port 5001
   ```

5. 进行语音识别:
   使用 `StreamTest_params.py` 脚本发送音频文件进行识别。

## API 接口

1. `/transcribe` (POST): 上传音频文件并获取识别结果。
