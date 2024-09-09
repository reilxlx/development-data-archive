# 文本相似度搜索引擎

## 项目简介
本项目是一个基于BERT和Faiss的文本相似度搜索引擎，主要用于处理和检索大规模文本数据。该系统能够高效地找出与给定查询最相似的文本，适用于各种文本匹配和检索任务。

## 主要功能
1. 文本数据预处理：将原始文本数据转换为结构化的Pandas DataFrame格式。
2. 文本编码：使用预训练的BERT模型将文本转换为向量表示。
3. 高效索引：利用Faiss库构建高效的向量索引，支持快速相似度搜索。
4. 相似度搜索：根据用户输入的查询文本，返回最相似的若干条结果。
5. 定期更新：自动定期更新文本数据和索引，确保搜索结果的时效性。

## 系统架构
- `processTxtToPandas.ipynb`: 用于将原始文本数据处理成结构化的CSV格式。
- `FaissSearcher.ipynb`: 实现了基于Faiss的搜索引擎核心功能。
- `TextSimilarity.py`: 提供了Web API接口，处理用户查询并返回相似文本结果。

## 使用技术
- Python 3.7+
- TensorFlow 2.x
- Pandas
- Faiss
- Flask
- Transformers (Hugging Face)
- BERT (预训练中文模型)

## 如何使用
1. 数据预处理：
   运行`processTxtToPandas.ipynb`，将原始文本数据转换为CSV格式。

2. 构建搜索引擎：
   使用`FaissSearcher.ipynb`中的代码构建Faiss索引。

3. 启动Web服务：
   运行`TextSimilarity.py`启动Flask Web服务。

4. 发送查询请求：
   向`/find_similar`接口发送POST请求，包含查询文本和可选的标签参数。

## 特性
- 支持大规模文本数据的高效检索
- 利用BERT模型捕捉文本的语义信息
- 使用Faiss实现快速相似度搜索
- 支持多标签分类的检索
- 自动定期更新索引，保持数据新鲜度
