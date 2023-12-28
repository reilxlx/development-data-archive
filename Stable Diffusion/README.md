# **Text to Image: 文本生成图像**
1. 对图像打标签及裁剪图像</br>
在对指定个人进行lora训练时，打标的原图分辨率越高越好。推荐图像个数50张以上，目前测试100张训练集后续生成效果好于20张。</br>
图像打标使用cybertronfurnace，由B站UP朱尼酱开发，自带人脸裁切增强功能、图片打标、标签调整、训练功能。但人脸裁切效果较差，可自行使用MTCNN检测裁剪人脸。</br>

2. 基于基础SD模型训练lora模型</br>
训练环节使用https://github.com/Akegarasu/lora-scripts </br>
基础模型复制到sd-models文件夹下，启动脚本默认选择图像文件夹之后会将图像文本移动到子文件夹，需要手动修改子文件夹的数字，代表每张图像的训练轮数。


4. 使用模型</br>
使用以下stable-diffusion-webui工具，tet2img功能输出图像。基础模型复制到models\Stable-diffusion文件夹下，lora模型复制到models\Lora文件夹，启动脚本webui.bat</br>
注意：在开启代理的情况下webui.bat脚本中加入一行：set COMMANDLINE_ARGS=--no-gradio-queue</br>
参考链接：https://github.com/AUTOMATIC1111/stable-diffusion-webui</br>


5. 使用模型中可替换成comflowy工具 可参考以下链接学习使用教程</br>
https://www.comflowy.com/zh-CN </br>
https://github.com/6174/comflowy</br>
