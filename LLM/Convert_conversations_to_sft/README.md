以下代码针对该https://huggingface.co/datasets/m-a-p/neo_sft_phase2 数据集进行处理，从对话中提取sft数据。

以下三种方案分别对应的处理代码如下：
- Convert_multiround_conversations_to_sft.py
- Convert_all_conversations_to_sft.py
- Convert_singleround_conversations_to_sft.py

**方案一：  将多轮对话压缩成单轮**
* **思路：** 将同一对话中的多轮内容合并，使之看起来像是一问一答。
* **具体操作：**
    1.  将所有 "human" 的 "value"  字段拼接在一起作为  "instruction"。
    2.  将所有的 "gpt" 的 "value" 字段作为  "output"。
    3.  "input" 可以为空，或者加入一些提示语，例如 "Based on the above conversation, ..."。

* **示例：**

```json
{
 "instruction": "我的朋友生日快到了，我想送他一个意义非凡的礼物，但不清楚他最近迷上了什么。你能帮我构思一条信息，好让我询问他的最新兴趣吗？请问可以分享一下特定尺码的衣物具体有多少件吗？",
 "input": "Based on the above conversation, ...",
 "output": "当然可以，目前服装店的库存情况如下：\n\n- 小号（S）：共有2件\n- 中号（M）：共有4件\n- 大号（L）：共有3件\n- 特大号（XL）：共有7件\n\n如果您想了解某个具体尺码的详细数量，请告诉我那个尺码吧。"
}
```

* **优点：** 简单直观，容易实现。
* **缺点：**  可能会损失上下文信息，导致模型难以理解对话的逻辑关系。
* **代码：**
```python
import json

def convert_conversations_to_single_turn(conversations):
    instruction = ""
    output = ""
    for turn in conversations:
        if turn['from'] == 'human':
            instruction += turn['value'] + "\n\n"
        else:
            output += turn['value'] + "\n\n"
    return {
        "instruction": instruction.strip(),
        "input": "",
        "output": output.strip()
    }

def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    with open("neo_sft_phase2.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    single_turn_dataset = []
    for conversation_set in data:
        single_turn_dataset.append(convert_conversations_to_single_turn(conversation_set['conversations']))

    save_to_jsonl(single_turn_dataset, "neo_sft_phase2.jsonl")
```

**方案二： 将多轮对话拆分成多个单轮样本**

*  **思路：** 将每轮对话都视为独立的问答对，并利用上下文信息构建样本。
* **具体操作：**
    1. 对于每个 "conversations"，遍历其中的每一轮对话。
    2. 将当前轮 "human" 的 "value"  和之前所有轮次的对话拼接在一起作为 "instruction"。
    3. 当前轮 "gpt" 的 "value" 作为 "output"。
    4. "input" 可以为空，或者加入一些提示语。

* **示例：**

```json
# 第一个样本
{
 "instruction": "我的朋友生日快到了，我想送他一个意义非凡的礼物，但不清楚他最近迷上了什么。你能帮我构思一条信息，好让我询问他的最新兴趣吗？",
 "input": "",
 "output": "当然可以，这里有一个温馨又不失趣味的询问模板，你可以参考一下：\n\n---\n\n亲爱的小明（请替换为朋友的名字），\n\n哇，转眼又是一年，你的生日悄然而至！我正在密谋一个超级特别的生日惊喜，希望能直击你的心坎，成为你难以忘怀的礼物！🎈\n\n为了这份惊喜的精准度，我急需了解你近来沉迷于哪些新鲜事物或老爱好里。无论是沉浸在书香中、追逐最新的科技潮流、探索户外的无限可能，还是沉醉于音符编织的梦想，我都渴望听到你的心声。你的每一个小喜好，都是我挑选完美礼物的宝贵线索！\n\n等不及要听你的分享啦，这样我就能更靠近那份能让你眼前一亮的礼物！🎁\n\n提前祝福你，愿这一年比上一年更加精彩！\n\n[你的名字]\n\n---\n\n这样的信息不仅展现了你对朋友生日的用心，还以一种轻松的方式鼓励他分享自己的兴趣，从而帮助你找到那个“正中红心”的礼物。"
}

# 第二个样本
{
 "instruction": "我的朋友生日快到了，我想送他一个意义非凡的礼物，但不清楚他最近迷上了什么。你能帮我构思一条信息，好让我询问他的最新兴趣吗？当然可以，这里有一个温馨又不失趣味的询问模板，你可以参考一下：\n\n---\n\n亲爱的小明（请替换为朋友的名字），\n\n哇，转眼又是一年，你的生日悄然而至！我正在密谋一个超级特别的生日惊喜，希望能直击你的心坎，成为你难以忘怀的礼物！🎈\n\n为了这份惊喜的精准度，我急需了解你近来沉迷于哪些新鲜事物或老爱好里。无论是沉浸在书香中、追逐最新的科技潮流、探索户外的无限可能，还是沉醉于音符编织的梦想，我都渴望听到你的心声。你的每一个小喜好，都是我挑选完美礼物的宝贵线索！\n\n等不及要听你的分享啦，这样我就能更靠近那份能让你眼前一亮的礼物！🎁\n\n提前祝福你，愿这一年比上一年更加精彩！\n\n[你的名字]\n\n---\n\n这样的信息不仅展现了你对朋友生日的用心，还以一种轻松的方式鼓励他分享自己的兴趣，从而帮助你找到那个“正中红心”的礼物。请问可以分享一下特定尺码的衣物具体有多少件吗？",
 "input": "",
 "output": "当然可以，目前服装店的库存情况如下：\n\n- 小号（S）：共有2件\n- 中号（M）：共有4件\n- 大号（L）：共有3件\n- 特大号（XL）：共有7件\n\n如果您想了解某个具体尺码的详细数量，请告诉我那个尺码吧。"
}
```

* **优点：**  保留了更多上下文信息，有助于模型学习对话的连贯性和逻辑性。
* **缺点：**  数据处理过程相对复杂，生成的样本数量可能会很多。

* **代码：**
```python
import json

def convert_conversations_to_sft(conversations):
    sft_data = []
    instruction = ""
    for i, turn in enumerate(conversations):
        if turn['from'] == 'human':
            instruction += turn['value'] + "\n\n"
        else:
            sft_data.append({
                "instruction": instruction.strip(),
                "input": "",
                "output": turn['value']
            })
            instruction += turn['value'] + "\n\n"
    return sft_data

def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    with open("neo_sft_phase2.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    sft_dataset = []
    for conversation_set in data:
        sft_dataset.extend(convert_conversations_to_sft(conversation_set['conversations']))
    save_to_jsonl(sft_dataset, "neo_sft_phase2.jsonl")
```

**方案三： 只提取单轮对话的样本**

*  **思路：** 舍弃多轮对话样本，仅对单轮对话进行数据操作。
* **具体操作：**
    1. 只针对包含单轮对话的 "conversations"，提取信息。
    2. 将当前轮 "human" 的 "value" 作为 "instruction"。
    3. 当前轮 "gpt" 的 "value" 作为 "output"。
    4. "input" 可以为空，或者加入一些提示语。

* **代码：**
```python
import json
def process_conversations(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        data = json.load(f_in)
        
        for item in data:
            conversations = item.get("conversations", [])
            if len(conversations) == 2:
                human_value = conversations[0].get("value", "")
                gpt_value = conversations[1].get("value", "")
                output_data = {
                    "instruction": human_value,
                    "output": gpt_value
                }
                f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_json_file = "neo_sft_phase2.json"
    output_jsonl_file = "neo_sft_phase2_conversation2.json"
    process_conversations(input_json_file, output_jsonl_file)
```