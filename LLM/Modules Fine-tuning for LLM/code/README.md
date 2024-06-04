# 代码功能简介：
- DataAnalyse.ipynb: 分析数据集在指令、输入、输出维度下的文本长度分布情况。
- TrainLLMByDifModule.ipynb: 按模块和数据集分别训练模型。
- MergeDifAdaptersToModel.py: 将 TrainLLMByDifModule.ipynb 训练得到的各个模块合并到基础模型中。
- PrintComparedParams.py: 比较基础模型与合并后模型指定参数的值，验证模型合并是否成功。

# Code Functionality Overview:
- DataAnalyse.ipynb: Analyzes the text length distribution of the dataset across instructions, inputs, and outputs.
- TrainLLMByDifModule.ipynb: Trains the model separately for each module and dataset.
- MergeDifAdaptersToModel.py: Merges the individual modules trained by TrainLLMByDifModule.ipynb into the base model.
- PrintComparedParams.py: Compares specific parameter values between the base and merged models to verify the successful merging of trainable parameters.