### 数据集
使用以下8个数据集</br>
![image/png](https://cdn-uploads.huggingface.co/production/uploads/636f54b95d2050767e4a6317/OkuVQ1lWXRAKyel2Ef0Fz.png)</br>
对Qwen1.5-7B-Chat进行微调并测试，结果显示，使用三个数据集微调后的模型在CEVAL和MMLU的评分上均有所提升，而且这个模型的表现优于使用八个数据集微调后的模型。

### 基础模型：
- https://huggingface.co/Qwen/Qwen1.5-7B-Chat

### 训练工具
https://github.com/hiyouga/LLaMA-Factory

### 测评方式：
使用opencompass(https://github.com/open-compass/OpenCompass/ )， 测试工具基于CEval和MMLU对微调之后的模型和原始模型进行测试。</br>
测试模型分别为：
- Qwen1.5-7B-Chat 
- Qwen1.5-7B-Chat-290Mb-lora,使用3DataSets数据集对Qwen1.5-7B-Chat模型进行sft方式lora微调,训练一轮。
- Qwen1.5-7B-Chat-750Mb-lora,使用8DataSets数据集对Qwen1.5-7B-Chat模型进行sft方式lora微调,训练一轮。

### 测试机器
8*A800

### 3DataSets数据集：
大约290Mb的微调数据集
- https://huggingface.co/datasets/LooksJuicy/ruozhiba
- https://huggingface.co/datasets/TigerResearch/sft_zh
- https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese

### 8DataSets数据集：
大约750Mb的微调数据集
- https://huggingface.co/datasets/REILX/extracted_tagengo_gpt4
- https://huggingface.co/datasets/TigerResearch/sft_zh
- https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese
- https://huggingface.co/datasets/LooksJuicy/ruozhiba
- https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k
- https://huggingface.co/datasets/alexl83/AlpacaDataCleaned
- https://huggingface.co/datasets/Sao10K/Claude-3-Opus-Instruct-5K


### 结果
| 模型名称                 | CEVAL | MMLU |
|------------------------ |-------|------|
| Qwen1.5-7B-Chat         | 68.61 | 61.56 |
| Qwen1.5-7B-Chat-290Mb-lora | 71.75 | 62.43 |
| Qwen1.5-7B-Chat-750Mb-lora  | 71.36 | 61.78 |