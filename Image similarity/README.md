# Image Similarity Project

这个项目实现了基于图像相似度的多个功能,包括特征提取、哈希计算、相似图像搜索和重复图像检测。

## 主要组件
1. computerFeatureToHash.py
   - 使用 CLIP 模型提取图像特征
   - 计算特征值的哈希
   - 提供 Flask API 接口用于图像处理

2. faissImagesFeaturesLab.py
   - 使用 Faiss 实现高效的相似图像搜索
   - 提供 Flask API 接口用于索引创建、图像索引和搜索

3. simi_images_del_gpu.ipynb 和 simi_images_del.ipynb
   - 使用 CLIP 模型和 Faiss 进行图像特征提取和索引
   - 实现重复图像检测和删除
   - GPU 和 CPU 版本

## 主要功能

1. 图像特征提取: 使用 CLIP 模型从图像中提取特征向量。
2. 特征哈希计算: 对提取的特征向量计算哈希值。
3. 图像索引: 使用 Faiss 创建和管理图像特征索引。
4. 相似图像搜索: 基于特征向量的相似度搜索。
5. 重复图像检测: 识别并可选择性删除相似度高的重复图像。

## 使用的技术

- CLIP (Contrastive Language-Image Pre-Training) 模型
- Faiss (Facebook AI Similarity Search)
- Flask
- PyTorch
- PIL (Python Imaging Library)
- NumPy

## 安装和使用

1. 安装所需依赖:
   ```
   pip install torch transformers faiss-cpu pillow flask numpy
   ```

2. 下载并设置 CLIP 模型:
   将 CLIP 模型文件放置在 `/data/similarities/model/openaiclipvitbasepatch32/` 目录下。

3. 运行 Flask 应用:
   ```
   python computerFeatureToHash.py
   python faissImagesFeaturesLab.py
   ```

4. 使用 Jupyter Notebook 运行 `simi_images_del_gpu.ipynb` 或 `simi_images_del.ipynb` 进行重复图像检测。

## API 接口

1. `/imageTofeatureTohash` (POST): 上传图像并获取特征哈希值。
2. `/create_index` (POST): 创建新的 Faiss 索引。
3. `/index_images` (POST): 将指定文件夹中的图像添加到索引。
4. `/search` (POST): 搜索与上传图像相似的图像。

## 注意事项

- 确保有足够的磁盘空间用于存储图像和索引文件。
- GPU 版本需要 CUDA 支持,请确保您的系统配置正确。
- 处理大量图像时可能需要较长时间,请耐心等待。