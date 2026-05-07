# Needle 后续工作清单

## 已完成

- [x] 模型层：Qwen2 / LLaMA forward（RMSNorm、RoPE、GQA、SwiGLU）
- [x] KV Cache 管理：PagedAttention 块分配 / 回收 / swap in/out
- [x] FlashInfer 集成（prefill 用 single_prefill_with_kv_cache；decode 用 BatchDecodeWithPagedKVCacheWrapper，group_size 非 2^n 自动 fallback）
- [x] Continuous Batching 调度（waiting / running / swapped 三队列）
- [x] 权重加载（safetensors / pytorch bin，lm_head weight tying）
- [x] Sampler（greedy / top-k / top-p / temperature）
- [x] ZMQ 双工传输（API Server ↔ Backend Process）
- [x] FastAPI 服务（OpenAI 兼容 /v1/chat/completions，non-stream & SSE stream）
- [x] Qwen2-0.5B 验证：prefill logits 与 HuggingFace 完全一致，greedy decode 20 token 精确匹配
- [x] CLI 启动参数支持：`needle/args.py` + `needle/__main__.py`，`python -m needle --model ... --port ...`，与 miniSGL 风格一致（双别名参数、`dtype=auto`、路径校验）
- [x] Library 模式本地调用：`needle/llm.py` 暴露 `needle.LLM`，支持 `generate()` / `batch_generate()` / `stream()`，TP=1 时直接在调用进程内运行
- [x] 通用 debug 脚本：`tools/debug.py`，支持 `--model/--model-path`、`--prompt`、`--max-tokens`、`--validate`，自动运行六步检查（配置→构建→加载→prefill→decode→HF对比）
- [x] Bug 修复：Prefill / Decode 分批 forward（混合 batch 拆两次 forward）
- [x] Bug 修复：EOS token 填入 stop_token_ids（`BackendProc.__init__` 加载 tokenizer）
- [x] Bug 修复：double-free 风险（`scheduler.free_sequence` 改用 try/except 安全移除）
- [x] Bug 修复：KV swap 实际执行（`runner.execute()` 开头调用 `_swap_kv_blocks`）
- [x] Bug 修复：CPU KV Cache 分配（`init_kv_cache` 同步分配 `cpu_kv_caches`，`pin_memory=True`）
- [x] Bug 修复：block_table padding 改用 `dummy_block_id = num_gpu_blocks - 1`，消除脏数据
- [x] Bug 修复：流式输出中文乱码（`_stream_chat` 改为 token buffer + 增量解码策略）

---

## 待办（按优先级）

### 1. FlashInfer prefill 不支持多轮 / 续写（context_lens > 0）
**文件**: `needle/layers/attention.py:108-114`
**问题**: `_prefill_flashinfer` 仅传 `q/k/v`，不读 kv_cache 中已有的 prefix KV，多轮对话续写时结果错误。
**修复方向**: 改用 flashinfer `single_prefill_with_kv_cache` 并传入 paged kv 参数；或当 `context_lens > 0` 时 fallback 到 `_prefill_pytorch`。

### 2. /v1/completions 端点（非 chat）
**文件**: `needle/serving/server.py`
**现状**: 仅有 `/v1/chat/completions`，缺少原始文本补全接口，部分场景（batch eval、embedding pipeline）依赖此接口。
**TODO**: 新增 `/v1/completions`，接受 `prompt: str`，跳过 chat template，直接 encode → generate → decode。

### 3. 请求超时 / 取消传播
**现状**: HTTP 层 disconnect 检测只在 streaming 路径，non-stream 请求客户端断开后 backend 仍在推理浪费资源。
**修复方向**: non-stream handler 里用 `asyncio.wait_for` + `http_req.is_disconnected()` 轮询，断开时调用 `client.abort(uid)`。

### 4. 进程间大数据传输支持 SHM
**现状**: API Server ↔ Backend Process 之间通过 ZMQ + msgpack 传递消息，长上下文 / 图片 token 等大负载时序列化 + socket 拷贝会成为瓶颈。
**参考**: vLLM `shm_object_storage`——发送方将对象 pickle 写入具名共享内存块，消息里只传 shm 名称和大小，接收方直接 mmap 读取，零拷贝。
**TODO**:
- 在 `needle/distributed/` 下实现 `ShmObjectStorage`：
  - `put(obj) -> shm_key`：pickle 序列化后写入 `multiprocessing.shared_memory.SharedMemory`
  - `get(shm_key) -> obj`：按 key attach shm，反序列化后 unlink；引用计数或 TTL 自动回收
- `UserMsg` 扩展：当 `input_ids` 超过阈值（如 1 KB）时改走 SHM 路径，ZMQ 只传 `ShmRef(key, size)`
- 单元测试：多进程读写正确性；进程异常退出后 shm 能被清理

### 5. 张量并行实测验证
**文件**: `needle/layers/linear.py`, `needle/model/runner.py`
**现状**: ColumnParallelLinear / RowParallelLinear 骨架已有，未在多卡环境实际跑通过。
**TODO**: 2 卡 TP 启动，验证与单卡输出一致；检查 NCCL AllReduce 时序和权重分片正确性。

### 6. 支持更多模型（LLaMA 3、Mistral、Gemma）
**现状**: `build_model` 只有 `qwen2` / `llama` 分支。
**TODO**: 补充各模型的 config key 映射和结构差异（RoPE scaling、sliding window attention 等）。

### 7. 补充单元测试
- `tests/unit/test_block_allocator.py`：swap / ref_count / OOM 路径
- `tests/unit/test_scheduler.py`：preemption、swap in/out 决策逻辑
- `tests/unit/test_paged_attention.py`：多 block 跨界读写正确性

### 8. 补充集成测试
- `tests/integration/test_llama_inference.py`：LLaMA 端到端 token 匹配
- `tests/integration/test_concurrent_requests.py`：多并发请求结果正确性

### 9. 吞吐基准测试
- `benchmarks/bench_throughput.py`：tokens/s，对比 vLLM / SGLang 基线

### 10. 日志 & 指标完善
- Prometheus metrics（`utils/metrics.py` 已有骨架）接入实际 step 计时
- 每 step 记录 batch_size、prefill_tokens、decode_tokens、KV cache 使用率
