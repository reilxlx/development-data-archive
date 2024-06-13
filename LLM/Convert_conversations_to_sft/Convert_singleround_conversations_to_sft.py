import json
def process_conversations(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
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
    output_jsonl_file = "neo_sft_phase2.jsonl"
    process_conversations(input_json_file, output_jsonl_file)
