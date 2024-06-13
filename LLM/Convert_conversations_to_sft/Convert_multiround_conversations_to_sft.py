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