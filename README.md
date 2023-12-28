# **Image similarity: 基于图像相似度的功能**
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
比如有场景需要搜索两类不同的数据，可分开搜索，也可合并搜索。上游下数需要搜索的数据，每日定时解压建立索引，存入本地。接收请求检索相似度topK的信息返回。
（For example, there is a scenario where you need to search two different types of data, which can be searched separately or combined. Upstream data needs to be searched, decompressed, indexed, and stored locally on a daily basis. It receives requests to retrieve top-K similarity information and returns the results.）

# **Speech Recognition：语音识别**
1、runWhisper_pipeline.py<br>
2、runWhisper_faster_whisper.py：<br>
使用transformers或者faster_whisper库方式本地运行whisper模型或faster whisper模型。效果不错，可简单替代生产环境语音识别功能<br>
3、可配合https://github.com/yeyupiaoling/Whisper-Finetune 微调whisper模型。数据准备代码参考prepareVoice.py

# **Delete Twitters：删除推特**
1、deleteTwitters.py<br>
使用pyautogui库，按照给定按钮的图标模拟点击逐个删除推特。由于省略号...个人推特主页有多个，使用代码时候需要缩放UI。
（Using the pyautogui library, simulate clicking on the icons of the specified buttons to delete tweets one by one. Since there are multiple individual Twitter profiles with ellipses (...) on the personal Twitter homepage, you will need to zoom in on the UI when using the code.）

# **CLIP Lora：基于CLIP模型进行图像文本对分析**
Fine-tune the parameters of Openai's CLIP model visual_model and text_model using the Lora method, and then perform image-text pair classification tasks.<br>

1、train_lora_SingleGPU.py<br>
Model training using the Lora method based on a single GPU<br>
2、train_lora_MultiGPU.py<br>
Model training based on single-machine multi-card, code tested on A800 cluster<br>
3、Val.py<br>
Test results of the Lora model trained on the test dataset<br>

The above codes have both Lora and full versions. The full version means that all parameters of the visual and text modules are included in fine-tuning. 

In this experiment, the test results show that training only the visual_model under the Lora method yields the best effect.<br>
![TestResults](https://github.com/reilxlx/development-data-archive/blob/main/CLIP/Data/CLIPTestData.jpg)

Related Sources：<br>
https://github.com/awilliamson10/clipora<br>
https://github.com/ptmorris03/CLIP-for-Mushrooms<br>

# **Text to Image: Generating Images from Text**
1. Labeling and Cropping Images:</br>
When training with LoRA for specific individuals, the higher the original image resolution, the better. It is recommended to use more than 50 images; currently, 100 images in the training set yield better results than 20. For image labeling, use cybertronfurnace, developed by Bilibili UP 朱尼酱. It includes features like face cropping enhancement, image labeling, tag adjustment, and training. However, for better face cropping results, consider using MTCNN.</br>

2. Training LoRA Model Based on Basic SD Model:</br>
For training, use scripts from https://github.com/Akegarasu/lora-scripts. Copy the base model to the sd-models folder. By default, after selecting an image folder, it will move the image text to a subfolder. You need to manually adjust the subfolder's numbers, representing the training rounds for each image.</br>

3. Using the Model:</br>
Utilize the stable-diffusion-webui tool, specifically the tet2img function, for image output. Copy the base model to the models\Stable-diffusion folder and the LoRA model to the models\Lora folder. Launch the script using webui.bat. After multiple tests, it's noted that setting the LoRA model's weight in the 0.6 range yields good results. If the weight is greater or equal to 1, overfitting with irregular patterns in the output is more likely. However, adding the LoRA model generally results in clearer images compared to without it. This issue is under ongoing analysis and experimentation. Note: When using a proxy, add the following line in the webui.bat script: set COMMANDLINE_ARGS=--no-gradio-queue. Refer to: https://github.com/AUTOMATIC1111/stable-diffusion-webui.</br>

4. For an alternative tool, consider using Comflowy. Learn more about it and how to use it from the following links:</br>
https://www.comflowy.com/zh-CN</br>
https://github.com/6174/comflowy</br>
