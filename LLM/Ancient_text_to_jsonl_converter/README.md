### 将 https://github.com/NiuTrans/Classical-Modern 项目的语料整理为问答对的形式，用于 LLM 训练

- **ancient_text_to_jsonl_converter.py**：
  - 将所有古文-现代文对组织成 `instruction`、`input`、`output` 的形式。

- **text_to_jsonl_converter.py**：
  - 按照总语料量的 30%，组织现代文-古文对，其余 70% 为古文-现代文对。
使用以上语料训练后模型效果不佳，怀疑是prompt太长的原因，或者语料库本身质量也有问题。

- **xiandai_ancient_to_jsonl_converter.py**：
1. 遍历指定目录及其子目录中的所有 'bitext.txt' 文件，提取其中的古文和现代文对应关系。
2. 将提取的文本对应关系组合成指定格式的 JSON 数据，并写入一个 JSONL 文件，同时确保每个 JSON 对象的总长度不超过 1024 个字符。
使用尽可能长的文段生成训练语料。