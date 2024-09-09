# LLM项目

## 项目概述

这是一个综合性的大型语言模型(LLM)项目,涵盖了从数据准备到模型训练、微调和评估的完整流程。项目的主要目标是探索和改进LLM在各种任务中的性能,包括文本生成、翻译、问答等。项目采用模块化结构,每个模块专注于LLM开发和应用的不同方面。

## 目录结构

- Ancient_text_to_jsonl_converter
- Data Cleaning
- Model training
- Modules Fine-tuning for LLM
- 基于Llama3的微调测试
- 基于Qwen1.5_7B的微调测试
- Convert_conversations_to_sft
- Testing the Image Recognition Capability of Multimodal Models

## 各章节详情

### 1. Ancient_text_to_jsonl_converter

#### 章节概述
这个模块主要用于将古文和现代文对应关系转换为JSONL格式,以便用于LLM训练。

#### 主要功能和组件
- 将古文-现代文对转换为问答对格式
- 处理多层目录结构中的文本文件
- 生成符合LLM训练要求的JSONL格式数据

### 2. Data Cleaning

#### 章节概述
数据清洗模块主要用于处理和优化用于LLM训练的数据集,包括语言检测、数据分割和相似度检查等功能。

#### 主要功能和组件
- 基于语言检测的数据分割
- 文本相似度分析和筛选
- 批量处理大规模数据集

### 3. Model training

#### 章节概述
这个模块包含了LLM训练的参考资料和指南。

#### 主要功能和组件
- 基于文档的问答系统参考
- 文本训练方法参考
- 推荐的基座模型列表

### 4. Modules Fine-tuning for LLM

#### 章节概述
这个模块专注于LLM的模块化微调,探索不同微调策略对模型性能的影响。

#### 主要功能和组件
- 数据集分析工具
- 模块化微调训练脚本
- 模型参数合并和比较工具

### 5. 基于Llama3的微调测试

#### 章节概述
这个模块针对Llama3模型进行了一系列微调实验,比较了不同数据集和微调方法的效果。

#### 主要功能和组件
- 多数据集微调实验
- 不同微调方法(SFT, ORPO)的应用
- 模型性能评估(基于CEval和MMLU)

### 6. 基于Qwen1.5_7B的微调测试

#### 章节概述
这个模块针对Qwen1.5-7B-Chat模型进行了微调实验,比较了不同数据集组合的效果。

#### 主要功能和组件
- 多数据集组合实验
- LoRA微调方法应用
- 模型性能评估(基于CEval和MMLU)

### 7. Convert_conversations_to_sft

#### 章节概述
这个模块主要用于将对话格式的数据转换为适合监督微调(SFT)的格式。

#### 主要功能和组件
- 对话数据格式转换
- JSON到JSONL的转换

### 8. Testing the Image Recognition Capability of Multimodal Models

#### 章节概述
这个模块涉及多模态模型的图像识别能力测试,使用了Gemini-1.5-pro模型。

#### 主要功能和组件
- 图像描述生成
- 多线程处理大量图像
- 错误处理和结果保存

## 总结

LLM项目涵盖了从数据准备到模型训练、微调和评估的完整流程。它不仅包括传统的文本处理任务,还探索了多模态模型在图像理解方面的应用。