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
