## 基于不同数据集DPO-En-Zh-20k、ruozhiba、（tigerbot+alpacadatagpt4）对Llama3和Llama3-Instruct进行微调。</br>
### 模型：</br>
- https://huggingface.co/meta-llama/Meta-Llama-3-8B
- https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

### 数据集：
- https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k
- https://huggingface.co/datasets/LooksJuicy/ruozhiba
- https://huggingface.co/datasets/TigerResearch/sft_zh
- https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese

### 训练工具：
https://github.com/hiyouga/LLaMA-Factory

### 测评方式：
使用opencompass(https://github.com/open-compass/OpenCompass/)测试工具基于CEval和MMLU对微调之后的模型和原始模型进行测试。</br>
测试模型分别为：
- Llama-3-8B
- Llama-3-8B-Instruct
- LLama3-Instruct-sft-ruozhiba,使用ruozhiba数据对Llama-3-8B-Instruct使用sft方式lora微调
- LLama3-Instruct-orpo-full-hiyouga，使用DPO-En-Zh-20k数据对Llama-3-8B-Instruct使用orpo方式进行全量full调整
- LLama3-Instruct-orpo-lora-hiyouga，使用DPO-En-Zh-20k数据对Llama-3-8B-Instruct使用orpo方式进行lora方式微调
- LLama3-Instruct-sft-lora-tigerbot-alpacadatagpt4-10epoch, 使用tigerbot+alpacadatagpt4数据对Llama-3-8B-Instruct使用sft方式lora微调

### 测试机器
8 * A800

### 结果
| 模型名称                 | CEVAL | MMLU |
|--------------------------|-------|------|
| LLama3                   | 49.91 | 66.62|
| LLama3-Instruct          | 50.55 | 67.15|
| LLama3-Instruct-sft-ruozhiba-3epoch | 50.87 | 67.51|
| LLama3-Instruct-sft-ruozhiba-10epoch | 49.29 | 67.21|
| LLama3-Instruct-orpo-full-hiyouga-2epoch | 47.52 | 62.71 |
| LLama3-Instruct-orpo-full-hiyouga-3epoch | 49.59 | 63.34 |
| LLama3-Instruct-orpo-lora-hiyouga-3epoch | 50.67 | 67.27|
| LLama3-Instruct-sft-lora-tigerbot-alpacadatagpt4-10epoch | 53.65 | 68.09 |
