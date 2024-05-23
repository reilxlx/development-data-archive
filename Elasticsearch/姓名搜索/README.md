### 背景
现有姓名搜索功能，大概原理将姓名和拼音小写预先存入数据库，使用like进行搜索返回。</br>
缺点此方案类似于精确搜索，无法进行容错。

### 需求
1.在拼音输错的情况下进行模糊搜索，例如输入法输入“zhoa”和“zhao”显示均为“赵”，方便在有时输入速度过快导致输错的情况下，能显示正确的结果。
2.目前中文名字搜索时可以根据名字第一个字和第二个字得出搜索结果，但是输入名字第一个字和第三个字时显示为空白，建议增加根据第一个字和第三个字得出搜索结果，方便在忘记对方名字中间那个字时依然可以搜出。

### 方案
1. 先姓名和拼音小写预先存入数据库，使用like进行搜索返回
2. 若无返回，基于elasticsearch功能，调用es功能，预先已将姓名入库，使用拼音插件和分词功能，使用es进行模糊匹配。

### 代码
1. 使用kibana创建索引
```bash
PUT /names_index
{
  "settings": {
    "index": {
      "max_ngram_diff": 2
    },
    "analysis": {
      "analyzer": {
        "pinyin_analyzer": {
          "tokenizer": "my_pinyin",
          "filter": ["lowercase", "pinyin_fuzzy"]
        },
        "ngram_analyzer": {
          "type": "custom",
          "tokenizer": "ngram_tokenizer"
        }
      },
      "tokenizer": {
        "my_pinyin": {
          "type": "pinyin",
          "keep_separate_first_letter": false,
          "keep_full_pinyin": true,
          "keep_original": false,
          "limit_first_letter_length": 16,
          "lowercase": true,
          "remove_duplicated_term": true
        },
        "ngram_tokenizer": {
          "type": "ngram",
          "min_gram": 1,
          "max_gram": 3
        }
      },
      "filter": {
        "pinyin_fuzzy": {
          "type": "edge_ngram",
          "min_gram": 1,
          "max_gram": 3
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "name": {
        "type": "text",
        "fields": {
          "pinyin": {
            "type": "text",
            "analyzer": "pinyin_analyzer"
          },
          "ngram": {
            "type": "text",
            "analyzer": "ngram_analyzer"
          }
        }
      }
    }
  }
}


```


2. 数据插入
```python
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import os

# 连接到 Elasticsearch
es = Elasticsearch("http://localhost:9200")  # 根据您的设置调整

# 批量导入数据到 Elasticsearch
def bulk_index(index_name, data):
    actions = [
        {"_index": index_name, "_source": {"name": name}}
        for name in data
    ]
    # 使用 bulk helper 进行批量导入
    bulk(es, actions)
    print(f"Successfully indexed {len(actions)} items")

# 主函数
def main():
    index_name = "names_index"  # 您的索引名称
    file_path = "path/to/your/names.txt"  # 您的文件路径

    # 读取文件并将每行姓名存入列表
    with open(file_path, 'r', encoding='utf-8') as file:
        names = [line.strip() for line in file if line.strip()]

    # 批量导入姓名到 Elasticsearch
    bulk_index(index_name, names)

if __name__ == "__main__":
    main()
```

3. Elasticsearch 查询优化

在查询时，可以考虑使用更复杂的 multi_match 查询，结合 pinyin 和 ngram 字段进行更全面的匹配。</br>
```bash
GET /names_index/_search
{
  "size": 50,
  "query": {
    "multi_match": {
      "query": "王玉",
      "fields": ["name.ngram", "name.pinyin"],
      "type": "best_fields"
    }
  }
}
```
