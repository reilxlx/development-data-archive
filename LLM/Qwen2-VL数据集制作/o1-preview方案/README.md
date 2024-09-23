1. **在每个 GPU 上加载模型并启动 API 服务：**
   - 使用 GPU 4、5 和 6，分别在端口 8124、8125 和 8126 上运行模型服务。
   - 每个模型服务都独立运行，加载完整的模型，使用指定的 GPU。

2. **实现 Python 版的 F5 功能（负载均衡）：**
   - 在端口 8123 上创建一个主服务，作为负载均衡器。
   - 该服务将收到的请求按照轮询（Round-Robin）方式分发到后台的三个模型服务上。

**运行模型服务：**
在 GPU 4 上的端口 8124：

```bash
python model_server.py --gpu 4 --port 8124
```

在 GPU 5 上的端口 8125：

```bash
python model_server.py --gpu 5 --port 8125
```

在 GPU 6 上的端口 8126：

```bash
python model_server.py --gpu 6 --port 8126
```

---

**2. 负载均衡器代码 (`load_balancer.py`)：**
**运行负载均衡器：**

```bash
python load_balancer.py
```

---

**使用说明：**

- **发送请求到主服务：** 现在，您可以将请求发送到 `http://localhost:8123/generate`，负载均衡器会自动将请求分发到后台的模型服务上。
  
- **请求示例：**

```python
import requests

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

response = requests.post("http://localhost:8123/generate", json={"messages": messages})
print(response.json())
```

---

- **负载均衡器 (`load_balancer.py`)：**
  - **后端服务器列表：** 包含所有模型服务的地址。
  - **轮询算法：** 使用全局变量 `current_server` 以及异步锁 `lock`，确保请求按顺序分发到不同的服务器。
  - **请求转发：** 接收到请求后，使用 `httpx.AsyncClient` 将请求转发到选定的后端模型服务。


通过上述步骤，在指定的 GPU 上运行模型服务，并使用 Python 实现类似 F5 的负载均衡功能，将请求均匀地分发到各个模型服务上。