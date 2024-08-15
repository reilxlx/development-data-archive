from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 加载预训练的BERT模型和tokenizer
model_name = "/root/llama3/mode/m3e-base/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 将模型加载到GPU或CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# 函数：获取文本的BERT嵌入向量
def get_embeddings(text, model, tokenizer, device):
    # 对文本进行编码
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    # 获取模型输出
    outputs = model(**inputs)
    # 取输出的最后一层的平均值作为嵌入向量
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return embeddings

# 比较两段文本的相似度
def compare_similarity(text1, text2, model, tokenizer, device):
    # 获取两段文本的嵌入向量
    embedding1 = get_embeddings(text1, model, tokenizer, device)
    embedding2 = get_embeddings(text2, model, tokenizer, device)
    # 计算余弦相似度
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity

# 示例文本
text1 = "你是谁, 我是一个人工智能助手，专门设计来回答问题、提供信息和帮助解决问题。我可以在很多领域提供帮助，包括科学、数学、文学、历史等等。请随时向我提问"
text2 = "增额终身寿险有什么特点,增额终身寿险是一种特殊类型的终身寿险产品，它结合了终身寿险的保障功能和投资增值的特点。"
# 计算并输出相似度
similarity = compare_similarity(text1, text2, model, tokenizer, device)
print(f"相似度: {similarity:.4f}")
