from pydantic import BaseModel, Field, ValidationError, field_validator
import json
import concurrent.futures
import time


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
        data = json.loads(line)
        RiddleData.model_validate(data)
        return True
    except ValidationError as e:
        return False


def validate_jsonl_file(file_path: str):
    start_time = time.time()
    invalid_lines = 0
    lines_to_validate = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines_to_validate = file.readlines()
    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(validate_jsonl_line, lines_to_validate))

    for i, was_valid in enumerate(results, start=1):
        if not was_valid:
            invalid_lines += 1
            print(f"Invalid data found at line {i}")

    elapsed_time = time.time() - start_time
    print(f"Total invalid lines: {invalid_lines}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")

# 调用函数，传入文件路径
file_path = R"C:\Users\Downloads\tigerbot-zhihu-zh-10k.jsonl"
validate_jsonl_file(file_path)
