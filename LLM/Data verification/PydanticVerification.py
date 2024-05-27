from pydantic import BaseModel, Field, ValidationError, field_validator
import json

class RiddleData(BaseModel):
    instruction: str = Field(..., min_length=1)
    input: str = ""
    output: str = Field(..., min_length=1)

    @field_validator('instruction', 'output')
    def must_not_be_empty(cls, value: str):
        if not value.strip():
            raise ValueError("must not be empty")
        return value

def validate_jsonl_line(line: str) -> bool:
    try:
        # 将 JSON 字符串转换为字典
        data = json.loads(line)
        # 创建 RiddleData 实例，如果失败会抛出 ValidationError
        RiddleData.model_validate(data)
        return True
    except ValidationError as e:
        return False

def validate_jsonl_file(file_path: str):
    invalid_lines = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                if not validate_jsonl_line(line):
                    invalid_lines += 1
                    print(f"Invalid data found at line {line_number}")
    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"Total invalid lines: {invalid_lines}")

# 调用函数，传入文件路径
file_path = R"C:\Users\Downloads\tigerbot-riddle-qa-1k.jsonl"
validate_jsonl_file(file_path)
