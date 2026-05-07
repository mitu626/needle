# Needle

<img src="logo.svg" alt="needle logo" width="320"/>

轻量、锋利的 LLM 推理引擎。从零实现 Continuous Batching 与 PagedAttention，聚焦核心推理链路。

**核心特性**：Continuous Batching · PagedAttention · FlashInfer 加速 · 张量并行 · OpenAI 兼容 API

## Quick Start

### 安装

```bash
pip install torch transformers safetensors fastapi uvicorn pydantic pyzmq msgpack
pip install flashinfer  # 可选，attention 加速
```

### 启动服务

```python
# serve.py
from needle.launch import launch

launch(
    model_path="/path/to/model",
    host="0.0.0.0",
    port=8000,
    tp_size=1,
    dtype="bfloat16",
)
```

```bash
python serve.py
```

### 调用

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"needle","messages":[{"role":"user","content":"你好"}],"max_tokens":100}'

# 流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"needle","messages":[{"role":"user","content":"你好"}],"max_tokens":100,"stream":true}'
```

## 目录结构

```
needle/
├── backend/     # 调度：Scheduler、BlockAllocator、LLMEngine
├── model/       # 模型：Qwen2、LLaMA、ModelRunner、Sampler
├── layers/      # 算子：Attention、RMSNorm、RoPE、Linear
├── serving/     # 服务：FastAPI、OpenAI 协议、BackendClient
├── distributed/ # 通信：ZMQ transport
└── launch.py    # 启动入口
```

## License

Apache-2.0