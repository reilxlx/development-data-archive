from fastapi import FastAPI, Request
import httpx
import uvicorn
import asyncio
import logging

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("load_balancer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 后端模型服务列表
backend_servers = [
    "http://localhost:8124/generate",
    "http://localhost:8125/generate",
    "http://localhost:8126/generate",
]

current_server = 0
lock = asyncio.Lock()

@app.post("/generate")
async def generate(request: Request):
    global current_server
    async with lock:
        server_url = backend_servers[current_server]
        logger.info(f"Forwarding request to backend server: {server_url}")
        current_server = (current_server + 1) % len(backend_servers)

    # 获取请求体
    body = await request.json()
    logger.info("Received a new request.")

    # 将请求转发到后端模型服务
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(server_url, json=body, timeout=300)
            logger.info("Received response from backend server.")
            return response.json()
    except Exception as e:
        logger.error(f"Error while forwarding request: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting load balancer on port 8123")
    uvicorn.run(app, host="0.0.0.0", port=8123)
