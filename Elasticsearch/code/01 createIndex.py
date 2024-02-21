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
insurance_index_name = es["insuraceIndex"]
jijinlicai_index_name = es["jijinlicaiIndex"]

es = Elasticsearch(es_address)

ssyq_jijinlicai = {
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
                },
                "my_stop_synonym_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_smart",
                    "filter": ["stop_words", "synonym_words"]
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
                },
                "stop_words": {
                    "type": "stop",
                    "stopwords": "stopwords.txt"
                },
                "synonym_words" : {
                    "type" : "synonym",
                    "synonyms_path" : "synonym.txt"
                }

            }
        }
    },
"mappings" : {
      "properties" : {
        "11": {
            "type": "text",
            "analyzer": "ik_smart",
            "fields": {
                "pinyin": {
                    "type": "text",
                    "store": False,
                    "term_vector": "with_positions_offsets",
                    "analyzer": "my_pinyin_analyzer",
                    "boost": 10
                },
                "initials": {
                    "type": "text",
                    "store": False,
                    "analyzer": "my_initials_analyzer"
                },
                "keyword": {
                    "type": "keyword",
                    "ignore_above": 256
                }
            }
        },
        "12": {
                "type": "text",
                "analyzer": "ik_smart",
                "fields": {
                     "pinyin": {
                        "type": "text",
                        "store": False,
                        "term_vector": "with_positions_offsets",
                        "analyzer": "my_pinyin_analyzer",
                        "boost": 10
                    },
                    "initials": {
                        "type": "text",
                        "store": False,
                        "analyzer": "my_initials_analyzer"
                    },
					"keyword": {
						"type": "keyword",
						"ignore_above": 256
					}
                }
            },
        "13": {
                "type": "text",
                "analyzer": "ik_smart",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "store": False,
                        "term_vector": "with_positions_offsets",
                        "analyzer": "my_pinyin_analyzer",
                        "boost": 10
                    },
                    "initials": {
                        "type": "text",
                        "store": False,
                        "analyzer": "my_initials_analyzer"
                    },
					"keyword": {
						"type": "keyword",
						"ignore_above": 256
					}
                }
            },
			"14": {
                "type": "text",
                "analyzer": "ik_smart",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "store": False,
                        "term_vector": "with_positions_offsets",
                        "analyzer": "my_pinyin_analyzer",
                        "boost": 10
                    },
                    "initials": {
                        "type": "text",
                        "store": False,
                        "analyzer": "my_initials_analyzer"
                    },
					"keyword": {
						"type": "keyword",
						"ignore_above": 256
					}
                }
            },
			"15": {
                "type": "text",
                "analyzer": "ik_smart",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "store": False,
                        "term_vector": "with_positions_offsets",
                        "analyzer": "my_pinyin_analyzer",
                        "boost": 10
                    },
                    "initials": {
                        "type": "text",
                        "store": False,
                        "analyzer": "my_initials_analyzer"
                    },
					"keyword": {
						"type": "keyword",
						"ignore_above": 256
					}
                }
            },
          "16": {
              "type": "text",
              "analyzer": "ik_smart",
              "fields": {
                  "pinyin": {
                      "type": "text",
                      "store": False,
                      "term_vector": "with_positions_offsets",
                      "analyzer": "my_pinyin_analyzer",
                      "boost": 10
                  },
                  "initials": {
                      "type": "text",
                      "store": False,
                      "analyzer": "my_initials_analyzer"
                  },
                  "keyword": {
                      "type": "keyword",
                      "ignore_above": 256
                  }
              }
          },
          "17": {
              "type": "text",
              "analyzer": "ik_smart",
              "fields": {
                  "pinyin": {
                      "type": "text",
                      "store": False,
                      "term_vector": "with_positions_offsets",
                      "analyzer": "my_pinyin_analyzer",
                      "boost": 10
                  },
                  "initials": {
                      "type": "text",
                      "store": False,
                      "analyzer": "my_initials_analyzer"
                  },
                  "keyword": {
                      "type": "keyword",
                      "ignore_above": 256
                  }
              }
          },
          "18": {
              "type": "text",
              "analyzer": "ik_smart",
              "fields": {
                  "pinyin": {
                      "type": "text",
                      "store": False,
                      "term_vector": "with_positions_offsets",
                      "analyzer": "my_pinyin_analyzer",
                      "boost": 10
                  },
                  "initials": {
                      "type": "text",
                      "store": False,
                      "analyzer": "my_initials_analyzer"
                  },
                  "keyword": {
                      "type": "keyword",
                      "ignore_above": 256
                  }
              }
          },
          "19": {
              "type": "text",
              "analyzer": "ik_smart",
              "fields": {
                  "pinyin": {
                      "type": "text",
                      "store": False,
                      "term_vector": "with_positions_offsets",
                      "analyzer": "my_pinyin_analyzer",
                      "boost": 10
                  },
                  "initials": {
                      "type": "text",
                      "store": False,
                      "analyzer": "my_initials_analyzer"
                  },
                  "keyword": {
                      "type": "keyword",
                      "ignore_above": 256
                  }
              }
          },
          "20": {
              "type": "text",
              "analyzer": "ik_smart",
              "fields": {
                  "pinyin": {
                      "type": "text",
                      "store": False,
                      "term_vector": "with_positions_offsets",
                      "analyzer": "my_pinyin_analyzer",
                      "boost": 10
                  },
                  "initials": {
                      "type": "text",
                      "store": False,
                      "analyzer": "my_initials_analyzer"
                  },
                  "keyword": {
                      "type": "keyword",
                      "ignore_above": 256
                  }
              }
          },
        "21": {
                "type": "text",
                "analyzer": "ik_smart",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "store": False,
                        "term_vector": "with_positions_offsets",
                        "analyzer": "my_pinyin_analyzer",
                        "boost": 10
                    },
                    "initials": {
                        "type": "text",
                        "store": False,
                        "analyzer": "my_initials_analyzer"
                    },
                    "stop_synonym": {
                        "type": "text",
                        "store": False,
                        "analyzer": "my_stop_synonym_analyzer"
                    },
					"keyword": {
						"type": "keyword",
						"ignore_above": 256
					}
                }
            },
			"22": {
                "type": "text",
                "analyzer": "ik_smart",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "store": False,
                        "term_vector": "with_positions_offsets",
                        "analyzer": "my_pinyin_analyzer",
                        "boost": 10
                    },
                    "initials": {
                        "type": "text",
                        "store": False,
                        "analyzer": "my_initials_analyzer"
                    },
					"keyword": {
						"type": "keyword",
						"ignore_above": 256
					}
                }
            },
			"23": {
                "type": "text",
                "analyzer": "ik_smart",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "store": False,
                        "term_vector": "with_positions_offsets",
                        "analyzer": "my_pinyin_analyzer",
                        "boost": 10
                    },
                    "initials": {
                        "type": "text",
                        "store": False,
                        "analyzer": "my_initials_analyzer"
                    },
					"keyword": {
						"type": "keyword",
						"ignore_above": 256
					}
                }
            },
			"24": {
                "type": "text",
                "analyzer": "ik_smart",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "store": False,
                        "term_vector": "with_positions_offsets",
                        "analyzer": "my_pinyin_analyzer",
                        "boost": 10
                    },
                    "initials": {
                        "type": "text",
                        "store": False,
                        "analyzer": "my_initials_analyzer"
                    },
					"keyword": {
						"type": "keyword",
						"ignore_above": 256
					}
                }
            },
			"25": {
                "type": "text",
                "analyzer": "ik_smart",
                "fields": {
                    "pinyin": {
                        "type": "text",
                        "store": False,
                        "term_vector": "with_positions_offsets",
                        "analyzer": "my_pinyin_analyzer",
                        "boost": 10
                    },
                    "initials": {
                        "type": "text",
                        "store": False,
                        "analyzer": "my_initials_analyzer"
                    },
					"keyword": {
						"type": "keyword",
						"ignore_above": 256
					}
                }
            }
      }
    }
}
# 创建索引，传入以上定义的映射
if not es.indices.exists(index=jijinlicai_index_name):
    es.indices.create(index=jijinlicai_index_name, body=ssyq_jijinlicai)