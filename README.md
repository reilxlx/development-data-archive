# **Image similarity: 图像相似度比较类功能**
1、computerFeatureToHash.py：
Use clip-vit-huge-14 to obtain the image feature value, calculate the hash of the feature value, and return the hash value

2、faissImagesFeaturesLab.py：
Demo based on faiss, implementing the following three functions<br>
a. Build a database<br>
b. Store feature values for all images in the specified folder in the database<br>
c. Search for feature values in the specified database name

# **Model training：大模型训练经验**
尝试sft方式微调过包含30多万条问答记录的客服数据。尝试pt方式训练一些文档数据。
1. 基于文档的问答参考：https://github.com/chatchat-space/Langchain-Chatchat
2. 对文本进行训练参考：https://github.com/hiyouga/LLaMA-Factory
3. 基座模型参考：
https://github.com/baichuan-inc/Baichuan2<br>
https://huggingface.co/Qwen/Qwen-14B-Chat<br>
https://huggingface.co/Qwen/Qwen-7B-Chat<br>
https://huggingface.co/mistralai/Mistral-7B-v0.1<br>
上述基座模型推荐Qwen14B

# **Search Engine：类似文本搜索引擎功能，核心使用文本相似度算法**
1、TextSimilarity.py：
比如有场景需要搜索两类不同的数据，可分开搜索，也可合并搜索。上游下数需要搜索的数据，每日定时解压建立索引，存入本地。接收请求检索相似度topK的信息返回

# **Speech Recognition：语音识别**
1、runWhisper_pipeline.py<br>
2、runWhisper_faster_whisper.py：<br>
使用transformers或者faster_whisper库方式本地运行whisper模型或faster whisper模型。效果不错，可简单替代生产环境语音识别功能<br>
3、可配合https://github.com/yeyupiaoling/Whisper-Finetune 微调whisper模型。数据准备代码参考prepareVoice.py
