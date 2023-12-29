# **Text to Image: 文本生成图像**
1. 对图像打标签及裁剪图像</br>
在对指定个人进行lora训练时，打标的原图分辨率越高越好。推荐图像个数50张以上，目前测试100张训练集后续生成效果好于20张。</br>
图像打标使用cybertronfurnace，由B站UP朱尼酱开发，自带人脸裁切增强功能、图片打标、标签调整、训练功能。但人脸裁切效果较差，可自行使用MTCNN检测裁剪人脸。</br>

2. 基于基础SD模型训练lora模型</br>
训练环节使用https://github.com/Akegarasu/lora-scripts，基础模型复制到sd-models文件夹下，默认选择图像文件夹之后会将图像文本移动到子文件夹，需要手动修改子文件夹的数字，代表每张图像的训练轮数。


3. 使用模型</br>
使用stable-diffusion-webui工具，tet2img功能输出图像。基础模型复制到models\Stable-diffusion文件夹下，lora模型复制到models\Lora文件夹，启动脚本webui.bat</br>
使用环节经过多次测试，需要将lora模型的权重设置在0.6区间会有不错的效果，若大于等于1则过拟合现象较强，模型有不规则图案输出。但加入lora模型之后输出图像比不lora模块模糊，该问题正在逐步分析实验。
注意：在开启代理的情况下webui.bat脚本中加入一行：set COMMANDLINE_ARGS=--no-gradio-queue</br>

    一组比较好用的参数：</br>
    positive propmt:</br>
    xuanzi,Best quality, masterpiece, ultra high res, (photorealistic:1.4), woman, two legs, standing, two hands, (RED high heels:1.2), (8k, (RAW photo:1.2),  realistic, absurdres,depth of field, masterpiece, chromatic aberration:1.1)), nsfw, (looking at viewer:1.21),hair bun, breasts, collared shirt, long sleeves, thighhighs,in the dark, deep shadow, low key \<lora:beautiful-milf-10:0.7> \<lora:yayoimix_100_xuanzi-000010:0.42> \<lora:xuanzi_20231223193448-000010:0.2></br>

    Negtive propmt:</br>
    easynegative, bad anatomy, low-res, poorly drawn face, pooly drawn hands, disfigured hands, disfigured, poorly drawn eyebrows, poorly drawn eyes, watermarks, username, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), normal quality, ((monochrome)), ((grayscale)), ass, anal,ng_deepnegative_v1_75t, bad-hands-5,censored,futanari</br>

    Sampling method: DPM++2M Karras</br>
    Sampling steps: 30</br>
    Hires.fix: Upscaler: R-ESRGAN 4x+;Hires steps:20;Denoising strength:0.42; Upscale by:2</br>
    CFG Scale:9</br>
    ADetailer</br>
    Clip skip 2</br>

    参考链接：</br>
    https://github.com/AUTOMATIC1111/stable-diffusion-webui</br>
    https://civitai.com/models/43331/majicmix-realistic


4. 使用模型中可替换成comflowy工具，可参考以下链接学习使用教程</br>
https://www.comflowy.com/zh-CN </br>
https://github.com/6174/comflowy</br>


