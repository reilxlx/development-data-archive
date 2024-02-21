import warnings
import yaml
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ElasticsearchWarning

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
es = config["es"]
es_address = es["address"]
index_name = es["insuraceIndex"]

# 忽略Elasticsearch建议启用安全功能的警告
warnings.filterwarnings(action="ignore", category=ElasticsearchWarning)

# 创建Elasticsearch客户端实例，假设Elasticsearch运行在默认端口9200
es = Elasticsearch(es_address)

# 检查索引是否存在
if es.indices.exists(index=index_name):
    # 删除索引
    response = es.indices.delete(index=index_name)
    print("Index deleted:", response)
else:
    print("Index does not exist:", index_name)