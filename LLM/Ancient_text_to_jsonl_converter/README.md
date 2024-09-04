### 将 https://github.com/NiuTrans/Classical-Modern 项目的语料整理为问答对的形式，用于 LLM 训练

- **ancient_text_to_jsonl_converter.py**：
  - 将所有古文-现代文对组织成 `instruction`、`input`、`output` 的形式。

- **text_to_jsonl_converter.py**：
  - 按照总语料量的 30%，组织现代文-古文对，其余 70% 为古文-现代文对。