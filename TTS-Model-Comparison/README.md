# TTS模型克隆音色对比项目
# VoiceClone Benchmark: TTS Model Comparison

## 项目简介

本项目旨在对比不同TTS（文本转语音）模型在音色克隆方面的效果。主要比较了三种TTS技术：
- **IndexTTS**: [GitHub项目地址](https://github.com/index-tts/index-tts)
- **Fish-Speech-1.5**: [GitHub项目地址](https://github.com/fishaudio/fish-speech)
- **SparkTTS**: [GitHub项目地址](https://github.com/SparkAudio/Spark-TTS)

通过使用相同的参考音频和合成文本，展示不同TTS模型在音色克隆、自然度和表现力方面的差异。

### 音色合成对比1

| 序号 | 参考音频 | 文本内容 | IndexTTS | Fish-Speech-1.5 | SparkTTS |
|-----|---------|---------|---------|-----------------|----------|
| 1 | [原声](参考音频/liutao.mp3) | 第一次见面，感觉有点小紧张呢。我是一个很容易被温柔打动的人，也总是对有趣的灵魂特别没抵抗力。平时喜欢听歌、看电影、偶尔也会发呆，想知道你会不会也不小心闯进我的小世界呢？说不定，从这一句自我介绍开始，我们就能有点特别的故事了呢。 | [IndexTTS合成](TTS/index-TTS/liutao/sentence_1.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/liutao/sentence_1.wav) | [SparkTTS合成](TTS/SparkTTS/liutao/sentence_1.wav) |
| 2 | [原声](参考音频/liutao.mp3) | 在时光的流转中，我们如同匆匆过客，见证着生命的诞生与消逝，欢笑与泪水。我们在春天播下希望的种子，在夏日挥洒辛勤的汗水，在秋天收获成功的喜悦，在冬日反思沉淀。每一段经历，每一次感悟，都如同时光长河中的朵朵浪花，构成了我们丰富多彩的人生 | [IndexTTS合成](TTS/index-TTS/liutao/sentence_2.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/liutao/sentence_2.wav) | [SparkTTS合成](TTS/SparkTTS/liutao/sentence_2.wav) |
| 3 | [原声](参考音频/liutao.mp3) | 夜色沉沉，时间仿佛停滞，我们谁都没有再说话，只是安静地靠在一起，听着彼此的心跳渐渐同步，外面的风吹动窗帘，投下一片片温柔的影子，而我的世界，此刻只剩下了她的呼吸，她的温度，还有这片属于我们的静谧夜晚。 | [IndexTTS合成](TTS/index-TTS/liutao/sentence_3.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/liutao/sentence_3.wav) | [SparkTTS合成](TTS/SparkTTS/liutao/sentence_3.wav) |
| 4 | [原声](参考音频/liutao.mp3) | 手机银行转账功能非常便捷，但请务必注意安全。在操作时，请仔细核对收款人的姓名、账号和开户行信息，确认无误后再进行下一步。同时，我们设置了单笔和日累计转账限额以保障您的资金安全，您可以通过安全认证工具（如U盾或短信验证码）进行限额内的交易。切勿将您的登录密码和交易密码告知他人。 | [IndexTTS合成](TTS/index-TTS/liutao/sentence_4.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/liutao/sentence_4.wav) | [SparkTTS合成](TTS/SparkTTS/liutao/sentence_4.wav) |

### 音色合成对比2

| 序号 | 参考音频 | 文本内容 | IndexTTS | Fish-Speech-1.5 | SparkTTS |
|-----|---------|---------|---------|-----------------|----------|
| 1 | [原声](参考音频/dilireba.mp3) | 第一次见面，感觉有点小紧张呢。我是一个很容易被温柔打动的人，也总是对有趣的灵魂特别没抵抗力。平时喜欢听歌、看电影、偶尔也会发呆，想知道你会不会也不小心闯进我的小世界呢？说不定，从这一句自我介绍开始，我们就能有点特别的故事了呢。 | [IndexTTS合成](TTS/index-TTS/dilireba/sentence_1.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/dilireba/sentence_1.wav) | [SparkTTS合成](TTS/SparkTTS/dilireba/sentence_1.wav) |
| 2 | [原声](参考音频/dilireba.mp3) | 在时光的流转中，我们如同匆匆过客，见证着生命的诞生与消逝，欢笑与泪水。我们在春天播下希望的种子，在夏日挥洒辛勤的汗水，在秋天收获成功的喜悦，在冬日反思沉淀。每一段经历，每一次感悟，都如同时光长河中的朵朵浪花，构成了我们丰富多彩的人生 | [IndexTTS合成](TTS/index-TTS/dilireba/sentence_2.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/dilireba/sentence_2.wav) | [SparkTTS合成](TTS/SparkTTS/dilireba/sentence_2.wav) |
| 3 | [原声](参考音频/dilireba.mp3) | 夜色沉沉，时间仿佛停滞，我们谁都没有再说话，只是安静地靠在一起，听着彼此的心跳渐渐同步，外面的风吹动窗帘，投下一片片温柔的影子，而我的世界，此刻只剩下了她的呼吸，她的温度，还有这片属于我们的静谧夜晚。 | [IndexTTS合成](TTS/index-TTS/dilireba/sentence_3.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/dilireba/sentence_3.wav) | [SparkTTS合成](TTS/SparkTTS/dilireba/sentence_3.wav) |
| 4 | [原声](参考音频/dilireba.mp3) | 手机银行转账功能非常便捷，但请务必注意安全。在操作时，请仔细核对收款人的姓名、账号和开户行信息，确认无误后再进行下一步。同时，我们设置了单笔和日累计转账限额以保障您的资金安全，您可以通过安全认证工具（如U盾或短信验证码）进行限额内的交易。切勿将您的登录密码和交易密码告知他人。 | [IndexTTS合成](TTS/index-TTS/dilireba/sentence_4.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/dilireba/sentence_4.wav) | [SparkTTS合成](TTS/SparkTTS/dilireba/sentence_4.wav) |

### 音色合成对比3

| 序号 | 参考音频 | 文本内容 | IndexTTS | Fish-Speech-1.5 | SparkTTS |
|-----|---------|---------|---------|-----------------|----------|
| 1 | [原声](参考音频/gulinazha.wav) | 第一次见面，感觉有点小紧张呢。我是一个很容易被温柔打动的人，也总是对有趣的灵魂特别没抵抗力。平时喜欢听歌、看电影、偶尔也会发呆，想知道你会不会也不小心闯进我的小世界呢？说不定，从这一句自我介绍开始，我们就能有点特别的故事了呢。 | [IndexTTS合成](TTS/index-TTS/gulinazha/sentence_1.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/gulinazha/sentence_1.wav) | [SparkTTS合成](TTS/SparkTTS/gulinazha/sentence_1.wav) |
| 2 | [原声](参考音频/gulinazha.wav) | 在时光的流转中，我们如同匆匆过客，见证着生命的诞生与消逝，欢笑与泪水。我们在春天播下希望的种子，在夏日挥洒辛勤的汗水，在秋天收获成功的喜悦，在冬日反思沉淀。每一段经历，每一次感悟，都如同时光长河中的朵朵浪花，构成了我们丰富多彩的人生 | [IndexTTS合成](TTS/index-TTS/gulinazha/sentence_2.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/gulinazha/sentence_2.wav) | - |
| 3 | [原声](参考音频/gulinazha.wav) | 夜色沉沉，时间仿佛停滞，我们谁都没有再说话，只是安静地靠在一起，听着彼此的心跳渐渐同步，外面的风吹动窗帘，投下一片片温柔的影子，而我的世界，此刻只剩下了她的呼吸，她的温度，还有这片属于我们的静谧夜晚。 | [IndexTTS合成](TTS/index-TTS/gulinazha/sentence_3.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/gulinazha/sentence_3.wav) | - |
| 4 | [原声](参考音频/gulinazha.wav) | 手机银行转账功能非常便捷，但请务必注意安全。在操作时，请仔细核对收款人的姓名、账号和开户行信息，确认无误后再进行下一步。同时，我们设置了单笔和日累计转账限额以保障您的资金安全，您可以通过安全认证工具（如U盾或短信验证码）进行限额内的交易。切勿将您的登录密码和交易密码告知他人。 | [IndexTTS合成](TTS/index-TTS/gulinazha/sentence_4.wav) | [Fish-Speech合成](TTS/Fish-Speech-1.5/gulinazha/sentence_4.wav) | [SparkTTS合成](TTS/SparkTTS/gulinazha/sentence_4.wav) |
