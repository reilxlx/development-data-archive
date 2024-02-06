# 01 下载各个组件
## elasticsearch下载地址
https://www.elastic.co/downloads/past-releases/elasticsearch-7-17-3
## 可视化软件kibana下载地址
https://artifacts.elastic.co/downloads/kibana/kibana-7.17.3-linux-x86_64.tar.gz
## 中文处理
elasticsearch-analysis-ik-7.17.3.zip

# 02 创建非root用户
``` bash
sudo useradd -m app  
sudo passwd app  
sudo usermod -s /bin/bash app
``` 

使用 root 权限执行以下命令，改变指定abc用户目录及其内所有子文件的所属主与所属组</br>
``` bash
chown abc:abc -R /home/abc
```

# 03 修改系统配置，满足ES的必须条件
修改limits配置文件进行系统配置</br>
``` bash
vim /etc/security/limits.conf
在limits.conf配置文件的最下面新增以下4行代码
nofile 65536
memlock unlimited
esuser hard nofile 65536
esuser soft nofile 65536
``` 

# 04 修改与使用ES相关的系统配置，再启动ES
修改相关的系统配置</br>
sysctl -w vm.max_map_count=262114
关于报错：[2]: max virtual memory areas vm.max_map_count [65530] is too low, increase to at least [262144]
解决办法：
``` bash
vim /etc/sysctl.conf
添加：
vm.max_map_count=262144
保存后，执行：
sysctl -p
重新启动，成功。
```

vim config/elasticsearch.yml
在当前配置文件的最下面添加以下配置</br>
``` bash
cluster.name: elasticsearch
node.name: es-node0
http.port = 9200
network.host: 0.0.0.0
cluster.initial_master_node: ["es-node0"]
```
启动elasticsearch</br>
``` bash
./bin/elasticsearch
``` 

# 05 判断es是否启动成功
浏览器输入 http://ip:9200，返回
``````json
{
  "name" : "es-node1",
  "cluster_name" : "my-elasticsearch",
  "cluster_uuid" : "AEHRJfFiTmmSrwTIomdSbg",
  "version" : {
    "number" : "7.13.2",
    "build_flavor" : "default",
    "build_type" : "tar",
    "build_hash" : "4d960a0733be83dd2543ca018aa4ddc42e956800",
    "build_date" : "2021-06-10T21:01:55.251515791Z",
    "build_snapshot" : false,
    "lucene_version" : "8.8.2",
    "minimum_wire_compatibility_version" : "6.8.0",
    "minimum_index_compatibility_version" : "6.0.0-beta1"
  },
  "tagline" : "You Know, for Search"
}
``````

# 05 elasticsearch.yml配置说明
![aaa](./yml.JPG)

# 06 修改es内存参数
在elasticsearch根目录下的config文件夹中的jvm.options文件，修改两个参数，修改之后重启即可。
最大内存
-Xms4g
最小内存
-Xmx4g

# 07 启停脚本
``` bash
#!/bin/bash

# 定义Elasticsearch的安装路径
ES_HOME=/path/to/elasticsearch

# 检查是否提供了启动或停止的参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {start|stop}"
    exit 1
fi

# 根据提供的参数执行相应的操作
case $1 in
    start)
        echo "Starting Elasticsearch..."
        $ES_HOME/bin/elasticsearch -d # '-d' 参数表示以守护进程的方式运行
        ;;
    stop)
        echo "Stopping Elasticsearch..."
        # Elasticsearch没有提供内置的停止命令，通常需要使用pkill或kill命令
        # 这里使用的是pkill来匹配进程名称
        pkill -f "org.elasticsearch.bootstrap.Elasticsearch"
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac

```

# 09 安装中文分词插件 安装拼音插件
根据安装的es版本进行下载https://github.com/medcl/elasticsearch-analysis-ik/releases

解压zip包unzip elasticsearch-analysis-ik-7.13.2.zip -d elasticsearch-7.13.2/plugins/ik

./bin/elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-pinyin/releases/download/v7.17.3/elasticsearch-analysis-pinyin-7.17.3.zip
重启es

# 10 安装图形化界面kibana
修改文件<kibana_home>/config/kibana.yml
``````bash
server.port: 5601
server.host: "0.0.0.0"
elasticsearch.hosts: ["http://localhost:9200"]
``````
启动服务
``````bash
nohup bin/kibana > /dev/null 2>&1 &
``````

# 11 获取所有索引的列表
获取所有索引的列表
使用_cat/indices API来获取所有索引的列表。这是一个简洁版的命令，返回较易读的格式：
``````bash
GET /_cat/indices?v
``````
这个命令会返回所有的索引及其基本信息，比如索引名称、健康状态、文档数量等。参数 v 是用来使输出结果更为可读。
获取索引的详细设置
要获取每个索引的详细设置，你可以使用 _settings 和 _mappings API来获得配置和映射信息。
获取所有索引的配置设置：
``````bash
GET /_all/_settings
``````
这个命令会返回所有索引的设置信息，包括分片数量、副本数、索引创建时间等。
获取所有索引的映射信息：
``````bash
GET /_all/_mapping
``````
这个命令会给你所有索引的字段和类型信息，也就是Elasticsearch里的映射信息。

# 12 elasticsearch-analysis-ik 与 analysis-pinyin 的区别
elasticsearch-analysis-ik：这是一个Elasticsearch的IK Analysis插件，它集成了Lucene的IK分析器，支持自定义词典。该插件主要用于支持中文分词。 </br>
analysis-pinyin：这是一个用于Elasticsearch的拼音分析插件，它用于中文字符与拼音之间的转换，并集成NLP工具。</br>

两者的区别：</br>
用途和功能：elasticsearch-analysis-ik 插件专注于提供中文分词功能，它允许进行细粒度的中文文本分析。而 analysis-pinyin 插件提供中文文本到拼音的转换，通常用于实现中文文本的拼音搜索。
底层技术/集成：elasticsearch-analysis-ik 插件集成了Lucene的IK分析器，这是一种用于中文语言处理的分词技术。analysis-pinyin 插件则可能集成了其他的NLP工具用于实现拼音转换的功能。</br>

两者的联系：</br>
中文处理：两个插件都是针对中文文本处理的场景设计的，一个提供分词服务，一个提供拼音转换。
Elasticsearch插件：他们都是Elasticsearch的插件，可以被用于在Elasticsearch中进行中文相关的文本分析。
搜索优化：这两个插件可被用于优化中文的搜索体验。例如，IK分析器能够提高中文全文搜索的准确性，而拼音插件能够让用户通过输入拼音来搜索中文内容。</br>

结论：</br>
这两个插件在提供中文搜索和分析能力上是互补的。elasticsearch-analysis-ik 主要负责有效地分词，以提供精确的中文搜索；analysis-pinyin 则允许用户通过拼音来搜索中文字符，这在用户输入法受限或搜索习惯上有拼音输入时非常实用。在实际的应用中，结合使用这两个插件，将极大地提升中文搜索的用户体验。


# 13 多节点安装ES服务，以三个节点为例
## 下载Elasticsearch tar.gz安装包
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.x.x-linux-x86_64.tar.gz

## 解压安装包
tar -zxvf elasticsearch-7.x.x-linux-x86_64.tar.gz

## 进入Elasticsearch目录
cd elasticsearch-7.x.x/

## 配置Elasticsearch集群
每个节点的Elasticsearch配置文件默认位于解压目录下的config/elasticsearch.yml。您需要根据每个节点的角色和网络设置编辑这个文件。</br>

Node 1配置
``````bash
cluster.name: my-cluster
node.name: node-1
network.host: 10.1.0.1
http.port: 9200
discovery.seed_hosts: ["10.1.0.1", "10.1.0.2", "10.1.0.3"]
cluster.initial_master_nodes: ["node-1", "node-2", "node-3"]
``````
Node 2配置
``````bash
cluster.name: my-cluster
node.name: node-2
network.host: 10.1.0.2
http.port: 9200
discovery.seed_hosts: ["10.1.0.1", "10.1.0.2", "10.1.0.3"]
cluster.initial_master_nodes: ["node-1", "node-2", "node-3"]
``````
Node 3配置
``````bash
cluster.name: my-cluster
node.name: node-3
network.host: 10.1.0.3
http.port: 9200
discovery.seed_hosts: ["10.1.0.1", "10.1.0.2", "10.1.0.3"]
cluster.initial_master_nodes: ["node-1", "node-2", "node-3"]
``````
保存并关闭elasticsearch.yml。

## 启动Elasticsearch集群
在每个节点，通过以下命令启动Elasticsearch:
``````bash
./bin/elasticsearch -d
或
nohup ./bin/elasticsearch > start.log 2>&1 &
``````

## 验证集群状态
``````bash
curl -X GET "10.1.0.1:9200/_cluster/health?pretty"
``````
这个命令应该会返回集群的健康状态信息，其中会显示集群的状态是green、yellow还是red，以及集群的节点信息。




安装步骤参考链接</br>
https://blog.csdn.net/zx_1305769448/article/details/129427815</br>
https://blog.csdn.net/m0_50287279/article/details/131819482</br>
https://blog.csdn.net/u010080562/article/details/123843540</br>
https://blog.csdn.net/m0_67403272/article/details/126660382</br>