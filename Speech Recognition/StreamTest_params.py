import json
import requests
import sys

def audio_recognition(audio_path: str):
    with open(audio_path, 'rb') as audio_file:
        response = requests.post(url="http://127.0.0.1:5000/recognition_stream",
                                 files=[("audio", (audio_path, audio_file, 'audio/wav'))],
                                 json={"to_simple": 1, "remove_pun": 0, "language": "zh", "task": "transcribe"},
                                 stream=True, timeout=20)
        
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                result = json.loads(chunk.decode())
                text = result["result"]
                start = result["start"]
                end = result["end"]
                print(f"[{start} - {end}]：{text}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <path_to_audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    audio_recognition(audio_path)
