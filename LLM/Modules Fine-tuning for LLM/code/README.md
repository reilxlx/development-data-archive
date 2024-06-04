# 代码功能简介：
- DataAnalyse.ipynb: 分析数据集在指令、输入、输出维度下的文本长度分布情况。
- TrainLLMByDifModule.ipynb: 按模块和数据集分别训练模型。
- MergeDifAdaptersToModel.py: 将 TrainLLMByDifModule.ipynb 训练得到的各个模块合并到基础模型中。
- PrintComparedParams.py: 比较基础模型与合并后模型指定参数的值，验证模型合并是否成功。
- CompareParams.py: 比较基础模型与合并 LoRA 权重后的模型，验证指定模块的参数是否发生预期改变。
- LoadPeftCompareParamsToMergedModel.py: 分别使用 PEFT 库的 merge_and_unload 函数和手动计算公式 W_new = W_original + B @ A * (lora_alpha/rank) 合并 LoRA 权重，比较两种方法得到模型的对应参数是否一致。


# Code Functionality Overview:
- DataAnalyse.ipynb: Analyzes the text length distribution of the dataset across instructions, inputs, and outputs.
- TrainLLMByDifModule.ipynb: Trains the model separately for each module and dataset.
- MergeDifAdaptersToModel.py: Merges the individual modules trained by TrainLLMByDifModule.ipynb into the base model.
- PrintComparedParams.py: Compares specific parameter values between the base and merged models to verify the successful merging of trainable parameters.
- CompareParams.py: This script compares the parameters of the base model with the model after merging LoRA weights, verifying if the specified module parameters have changed as expected.
- LoadPeftCompareParamsToMergedModel.py: This script compares the models obtained by merging LoRA weights using two methods: 1) using the merge_and_unload function from the PEFT library, and 2) manually calculating the weights using the formula W_new = W_original + B @ A * (lora_alpha/rank). It then verifies whether the corresponding parameters in both models are identical.
