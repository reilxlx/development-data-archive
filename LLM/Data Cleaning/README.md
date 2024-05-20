### SplitDataToDifLanguages.py
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




