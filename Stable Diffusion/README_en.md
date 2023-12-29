# **Text to Image: Generating Images from Text**
1. Labeling and Cropping Images:</br>
When training with LoRA for specific individuals, the higher the original image resolution, the better. It is recommended to use more than 50 images; currently, 100 images in the training set yield better results than 20. For image labeling, use cybertronfurnace, developed by Bilibili UP 朱尼酱. It includes features like face cropping enhancement, image labeling, tag adjustment, and training. However, for better face cropping results, consider using MTCNN.</br>

2. Training LoRA Model Based on Basic SD Model:</br>
For training, use scripts from https://github.com/Akegarasu/lora-scripts. Copy the base model to the sd-models folder. By default, after selecting an image folder, it will move the image text to a subfolder. You need to manually adjust the subfolder's numbers, representing the training rounds for each image.</br>

3. Using the Model:</br>
Utilize the stable-diffusion-webui tool, specifically the tet2img function, for image output. Copy the base model to the models\Stable-diffusion folder and the LoRA model to the models\Lora folder. Launch the script using webui.bat. After multiple tests, it's noted that setting the LoRA model's weight in the 0.6 range yields good results. If the weight is greater or equal to 1, overfitting with irregular patterns in the output is more likely. However, adding the LoRA model generally results in clearer images compared to without it. This issue is under ongoing analysis and experimentation. Note: When using a proxy, add the following line in the webui.bat script: set COMMANDLINE_ARGS=--no-gradio-queue. Refer to: https://github.com/AUTOMATIC1111/stable-diffusion-webui.</br>

        A set of relatively easy-to-use parameters：</br>
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

        Reference link：</br>
        https://github.com/AUTOMATIC1111/stable-diffusion-webui</br>
        https://civitai.com/models/43331/majicmix-realistic

4. For an alternative tool, consider using Comflowy. Learn more about it and how to use it from the following links:</br>
https://www.comflowy.com/zh-CN</br>
https://github.com/6174/comflowy</br>