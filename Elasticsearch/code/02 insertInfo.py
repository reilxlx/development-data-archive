import warnings
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ElasticsearchWarning
import yaml
# 忽略Elasticsearch建议启用安全功能的警告
warnings.filterwarnings(action="ignore", category=ElasticsearchWarning)

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
es = config["es"]
es_address = es["address"]
jijinlicaiIndex_name = es["jijinlicaiIndex"]

es = Elasticsearch(es_address)

# 确认Elasticsearch服务是否运行
if es.ping():
    print('Connected to Elasticsearch')
else:
    print('Could not connect to Elasticsearch')

# 中文书名列表
book_titles = [
    "战争与和平",
    "傲慢与偏见",
    "了不起的盖茨比",
    "杀死一只知更鸟",
    "一九八四",
    "白鲸记",
    "奥德赛",
    "罪与罚",
    "神曲",
    "百年孤独"
]

# 索引书籍标题
for i, title in enumerate(book_titles):
    doc = {
        'title': title
    }
    res = es.index(index=jijinlicaiIndex_name, id=i+1, document=doc, refresh=True)

print("Indexed 10 book titles (in Chinese) to 'chinese_text'.")