### SplitDataByLangIdToDifLanguagesTqdm.py SplitDataBypaplucaxlmToDifLanguages.py
在微调LLM模型的过程中，优化数据集至关重要。在特定应用场景中，我们可能需要进行以下调整：</br>
1. 调整llama3模型以更好地适配中文指令，从而显著减少其输出中的英文内容和表情符号的频率。
2. 考虑到Hugging Face中部分微调数据集包含多个语言版本，比如中文、英文、日语、韩文和阿拉伯语等，在特定情况下，选择单一语言进行微调可能更有助于提升任务性能。
</br>

In the process of fine-tuning LLM models, optimizing the dataset is essential. In certain application scenarios, the following adjustments may be necessary:
1. Modify the llama3 model to better align with Chinese instructions, thereby significantly reducing the frequency of English content and emoticons in its outputs.
2. Considering that some fine-tuning datasets on Hugging Face include multiple languages such as Chinese, English, Japanese, Korean, and Arabic, opting for a single language fine-tuning might be more conducive to enhancing task performance in specific circumstances.
</br>

### SentenceSimilarityZh.py
清洗中文语料，筛选出语料中涉及到模型身份认知相关的文字。</br>
1. 例如在llama3的微调测试中，中文语料中存在身份认知相关的文字，在后续模型中询问到“你是谁”、“介绍你自己”等之类的，输出存在问题。
</br>

Carefully clean the Chinese corpus, selecting texts related to the model's self-cognition.</br>
1. Taking the fine-tuning test of the llama3 model as an example, identity-cognition related texts are identified within the Chinese corpus.When the model is asked questions like "Who are you?" or "Introduce yourself", there is room for improvement in the outputs.
</br>

### Code module
SplitDataToDifLanguages.py</br>
- 输入为数据集文件夹、清洗之后的数据输出到新文件夹。
- 文件夹中为微调语料库，需要json或jsonl格式，获取语料中的instruction、input、output字段。
- 判断合并后的字段语种，拆分遇到到不同的json或jsonl中，以具体语种作为后缀标识。
</br>

- The process involves cleaning the corpus from the original dataset folder and saving the cleaned data to a new folder.
- The fine-tuning corpus should be placed in a folder in json or jsonl format, with key fields such as instruction, input, and output extracted.
- After merging the text content, categorize the data by language type, saving them in separate json or jsonl files, marked with the specific language type as a suffix in the file name.
</br>

SentenceSimilarityZh.py</br>
- 输入为需要清洗的json或jsonl，输出为清洗之后的语料。
- 使用embedding模型，评估语料与指定文段的相似度，此种方式或许需要指定多个文段以增加匹配可能性。
</br>

- The input is the original json or jsonl files, and the output is the corpus that has been cleaned.
- Use embedding models to assess the similarity between the corpus and specified paragraphs, increasing the likelihood of matching by designating multiple paragraphs.
</br>

SplitDataByLangIdToDifLanguagesTqdm.py</br>
- 对SplitDataToDifLanguages.py进行了优化。在语种识别模型中我们使用了langid库，而SplitDataToDifLanguages.py则采用了paplucaxlm-roberta-base-language-detection模型进行数据分割。
- 我们引入了tqdm模块以展示处理进度，使进度可视化。
- 由于paplucaxlm-roberta-base-language-detection模型是基于预训练的，并且仅支持20种语言，因此在那些未经训练的语言上效果可能不尽如人意。模型的详细信息可以在以下链接中找到：https://huggingface.co/papluca/xlm-roberta-base-language-detection。
</br>

- The `SplitDataToDifLanguages.py` has been optimized. In the language identification model, we use the `langid` library, while `SplitDataToDifLanguages.py` utilizes the `paplucaxlm-roberta-base-language-detection` model for data segmentation.
- The `tqdm` module has been introduced for displaying the progress of conversion, making the progress visualization.
- Given that the `paplucaxlm-roberta-base-language-detection` model is based on pre-training and only supports 20 languages, its performance may not be satisfactory on languages that have not been trained. For more information about the model, you can visit the link: https://huggingface.co/papluca/xlm-roberta-base-language-detection.
</br>

SplitAndSimilarityBatchByDataset.py</br>
- 文本相似度维度进行批量处理
</br>

- Bulk processing by text similarity dimension, serial processing for language detection

### 代码执行速度
该代码SplitDataByLangIdToDifLanguagesTqdm.py在</br>
测试机器：A800 * 1 上的执行速度</br>
测试数据集在huggingface都可下载

- silk-road/alpaca-data-gpt4-chinese
- lyuricky/alpaca_data_zh_51k
- llm-wizard/alpaca-gpt4-data-zh
- LooksJuicy/ruozhiba
- Sao10K/Claude-3-Opus-Instruct-5K
- TigerResearch/sft_zh

Alpaca_data_gpt4_zhsilk_road.jsonl: 100%|██████████| 52049/52049 [14:53<00:00, 58.25it/s]</br>
alpaca_data_zh_51k.json: 100%|██████████| 51461/51461 [12:40<00:00, 67.66it/s]</br>
alpaca_gpt4_data_zh.json: 100%|██████████| 48818/48818 [12:22<00:00, 65.76it/s]</br>
ruozhiba_qa.json: 100%|██████████| 1496/1496 [00:22<00:00, 65.94it/s]</br>
Claude3-Opus-Multi-Instruct-5K-merged.json: 100%|██████████| 4217/4217 [01:30<00:00, 46.76it/s]</br>

tigerbot-alpaca-zh-0.5m.jsonl: 100%|██████████| 500000/500000 [2:06:51<00:00, 65.69it/s]</br>
tigerbot-book-qa-1k.jsonl: 100%|██████████| 866/866 [00:13<00:00, 65.91it/s]</br>
tigerbot-hc3-zh-12k.jsonl: 100%|██████████| 12807/12807 [03:26<00:00, 62.07it/s]</br>
tigerbot-riddle-qa-1k.jsonl: 100%|██████████| 1000/1000 [00:14<00:00, 67.20it/s]</br>
tigerbot-superclue-c3-zh-5k.jsonl: 100%|██████████| 4792/4792 [01:18<00:00, 60.84it/s]</br>
tigerbot-wiki-qa-zh-1k.jsonl: 100%|██████████| 1000/1000 [00:14<00:00, 67.06it/s]</br>
tigerbot-zhihu-zh-10k.jsonl: 100%|██████████| 10240/10240 [02:56<00:00, 58.03it/s]</br>

单A800，12个文件355.3Mb共处理时间: 100%|██████████| 12/12 [2:57:11<00:00, 885.92s/it]
大约一个A800*8集群，一天可处理22GB。



SplitAndSimilarityBatchByDataset.py在
测试机器T4 * 1 上的执行速度， 批量大小为64</br>
测试数据集在huggingface都可下载
- silk-road/alpaca-data-gpt4-chinese
- LooksJuicy/ruozhiba
- TigerResearch/sft_zh

Alpaca_data_gpt4_zhsilk_road.jsonl: 100%|██████████| 814/814 [28:33<00:00,  2.11s/it]
ruozhiba_qa.json: 100%|██████████| 24/24 [00:17<00:00,  1.35it/s]
tigerbot-alpaca-zh-0.5m.jsonl: 100%|██████████| 7813/7813 [2:13:35<00:00,  1.03s/it]
tigerbot-book-qa-1k.jsonl: 100%|██████████| 14/14 [00:05<00:00,  2.55it/s]
tigerbot-hc3-zh-12k.jsonl: 100%|██████████| 201/201 [05:07<00:00,  1.53s/it]
tigerbot-riddle-qa-1k.jsonl: 100%|██████████| 16/16 [00:03<00:00,  4.12it/s]
tigerbot-superclue-c3-zh-5k.jsonl: 100%|██████████| 75/75 [02:28<00:00,  1.98s/it]
tigerbot-wiki-qa-zh-1k.jsonl: 100%|██████████| 16/16 [00:07<00:00,  2.01it/s]

单个T4，8个文件289.6Mb共处理时间: 100%|██████████| 9/9 [2:56:01<00:00, 1173.49s/it]
一张T4，一天可处理2.26GB。
