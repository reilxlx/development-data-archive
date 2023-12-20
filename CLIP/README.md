# **CLIP Lora：基于CLIP模型进行图像文本对分析**
Fine-tune the parameters of Openai's CLIP model visual_model and text_model using the Lora method, and then perform image-text pair classification tasks.<br>

1、train_lora_SingleGPU.py<br>
Model training using the Lora method based on a single GPU<br>
2、train_lora_MultiGPU.py<br>
Model training based on single-machine multi-card, code tested on A800 cluster<br>
3、Val.py<br>
Test results of the Lora model trained on the test dataset<br>

The above codes have both Lora and full versions. The full version means that all parameters of the visual and text modules are included in fine-tuning. In this experiment, the test results show that training only the visual_model under the Lora method yields the best effect.<br>



Related Sources：<br>
https://github.com/awilliamson10/clipora<br>
https://github.com/ptmorris03/CLIP-for-Mushrooms<br>