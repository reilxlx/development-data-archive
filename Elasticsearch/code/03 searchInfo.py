import warnings
import yaml
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ElasticsearchWarning

# 忽略Elasticsearch建议启用安全功能的警告
warnings.filterwarnings(action="ignore", category=ElasticsearchWarning)
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
es = config["es"]
es_address = es["address"]
index_name = es["insuraceIndex"]

es = Elasticsearch(es_address)  # 假设您的Elasticsearch运行在本地

# 搜索函数
def search_books_title(query, index_name):
    response = es.search(index=index_name, body={
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title"]
            }
        }
    })
    return [hit["_source"]["title"] for hit in response["hits"]["hits"]]

def search_books_title_pinyin(query, index_name):
    response = es.search(index=index_name, body={
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title.pinyin"]
            }
        }
    })
    return [hit["_source"]["title"] for hit in response["hits"]["hits"]]

def search_books_title_initials(query, index_name):
    response = es.search(index=index_name, body={
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title.initials"]
            }
        }
    })
    return [hit["_source"]["title"] for hit in response["hits"]["hits"]]

def search_books_title_initials_02(query, index_name):
    response = es.search(index=index_name, body={
        "query": {
            "prefix": {
                "query": query.lower(),
                "fields": ["title.initials"]
            }
        }
    })
    return [hit["_source"]["title"] for hit in response["hits"]["hits"]]

def search_books_title(query, index_name):
    response = es.search(index=index_name, body={
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title.pinyin", "title.initials"]
            }
        }
    })
    return [hit["_source"]["title"] for hit in response["hits"]["hits"]]
# 执行搜索
query_title = "金彩人生"
index_name = "jijinlicai_1"
results_title = search_books_title(query_title, index_name)
print(f"Search results for '{query_title}': {results_title}")

query_pinyin = "jincai"
index_name = "jijinlicai_1"
results_pinyin = search_books_title_pinyin(query_pinyin, index_name)
print(f"Search results for '{query_pinyin}': {results_pinyin}")

query_initials = "jcrs"
index_name = "jijinlicai_1"
results_initials = search_books_title_initials(query_initials, index_name)
print(f"Search results for '{query_initials}': {results_initials}")

query_initials = "金彩人生"
index_name = "jijinlicai_1"
results_initials = search_books_title(query_initials, index_name)
print(f"Search results for '{query_initials}': {results_initials}")
