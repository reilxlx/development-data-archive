通过kibana的Dev Tools测试输入的查询语句
```
GET /indexname/_search
{
    "query":{
        "multi_match":{
            "query:"上传版本",
            "fields":["title","title.initials","title.pinyin"]
        }
    }
}
``` 

设置（Settings）
在settings部分，通过定义自定义的分析器（analyzer）、分词器（tokenizer）和过滤器（filter），来定制索引的行为。
</br>
分析器（Analyzers）
my_pinyin_analyzer: 这是一个自定义的分析器，使用ik_smart作为分词器和pinyin_filter过滤器。ik_smart分词器是用于中文的一个开源分词器，它能够智能地将文本分割成词语。pinyin_filter过滤器将这些词语转换为拼音，支持多种拼音相关的功能，如保留全拼、首字母等。</br>

my_initials_analyzer: 另一个自定义分析器，使用keyword分词器和两个过滤器：first_letter和edge_ngram_filter。keyword分词器把整个输入作为一个单一的词元处理。first_letter过滤器提取每个词的首字母，而edge_ngram_filter生成边缘N-gram，从而支持部分匹配。</br>

分词器（Tokenizers）
edge_ngram_tokenizer: 这是一个定义的edge_ngram类型的分词器，它从词语的前缘开始生成大小在min_gram和max_gram之间的n-gram（此处为1到10个字符），支持字母和数字。这种类型的分词器对于实现自动补全或模糊匹配特别有用。</br>

过滤器（Filters）
pinyin_filter: 这个过滤器将中文转换成拼音。它有多个配置选项，如保留原文、首字母、全拼等，旨在提升中文拼音搜索的灵活性和准确性。
first_letter: 特化的pinyin过滤器，只保留每个词的首字母并转换为小写，适用于按首字母索引和搜索。
edge_ngram_filter: 这个过滤器生成从起始位置开始的n-gram，与edge_ngram_tokenizer类似，但用于过滤器链中。</br>

映射（Mappings）
在mappings部分，定义了如何索引和存储文档中的各个字段。这里专注于title字段的配置。


``` 
"settings": {
    "analysis": {
            "analyzer": {
            "my_pinyin_analyzer": {
                "type": "custom",
                "tokenizer": "ik_smart",
                "filter": ["pinyin_filter"]
            },
            "my_initials_analyzer": {
                "type": "custom",
                "tokenizer": "keyword",
                "filter": ["first_letter", "edge_ngram_filter"]
            }
        },
        "tokenizer": {
            "edge_ngram_tokenizer": {
                "type": "edge_ngram",
                "min_gram": 1,
                "max_gram": 10,
                "token_chars": ["letter", "digit"]
            }
        },
        "filter": {
            "pinyin_filter": {
                "type": "pinyin",
                "keep_first_letter": True,
                "keep_separate_first_letter": True,
                "keep_full_pinyin": True,
                "keep_joined_full_pinyin": True,
                "keep_original": False,
                "limit_first_letter_length": 16,
                "lowercase": True
            },
            "first_letter": {
                "type": "pinyin",
                "keep_first_letter": True,
                "lowercase": True
            },
            "edge_ngram_filter": {
                "type": "edge_ngram",
                "min_gram": 1,
                "max_gram": 10
            }
        }
    }
}
``` 
title字段:
使用ik_smart分析器进行分词，适合中文文本分析。
fields:
pinyin: 为title提供了一个子字段，使用my_pinyin_analyzer进行分析。这意味着原始标题会被索引为普通的文本，同时其拼音表示也会被单独索引，以便进行拼音搜索。boost参数提高了拼音搜索的相关性评分。
initials: 另一个子字段，使用my_initials_analyzer，允许按首字母或n-gram进行搜索。
keyword: 使用Elasticsearch的keyword类型，适合精确匹配搜索。ignore_above参数限制了索引的最大字符数。
```
{
    "title":{
        "type": "text"
        "analyzer": "ik_smart",
        # "analyzer":"keyword",
        "fields":{
            "pinyin": {
                "type" :"text"
                "store": False,
                "term_vector": "with_positions _offsets",
                "analyzer": "my_ pinyin_analyzer",
                "boost": 10
            },
            "initials":{
                "type":"text",
                "store": False,
                "analyzer:"my_initials_analyzer"
            },
            "keyword":{
                "type": ""keyword", 
                "ignore above":256
            }
        }
    }
}
``` 