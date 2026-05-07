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

---

## 待办（按优先级）

### 1. CLI 启动参数支持
**现状**: 启动服务需要手写 Python 脚本，所有参数硬编码。
**TODO**:
- 实现 `needle/cli.py`，用 `argparse` 或 `click` 封装 `launch()` 全部参数
- 注册 `pyproject.toml` entry_point，支持：
  ```
  needle serve --model /path/to/model --port 8000 --tp-size 1 --dtype bfloat16
  ```
- 关键参数：`--model`、`--host`、`--port`、`--tp-size`、`--dtype`、`--gpu-memory-utilization`、`--max-tokens`、`--page-size`、`--model-name`
- 参数校验与友好报错（模型路径不存在、端口被占用等提前检测）

### 2. Library 模式本地调用
**现状**: 框架只能作为独立服务启动，不支持在脚本里直接 import 使用。
**TODO**:
- 暴露顶层 `needle.LLM` 入口类，屏蔽 Engine / Runner / Scheduler 细节：
  ```python
  from needle import LLM
  llm = LLM("/path/to/model")
  print(llm.generate("你好"))
  for chunk in llm.stream("你好"):
      print(chunk, end="", flush=True)
  ```
- `LLMEngine` 补充同步接口：`generate(prompt)` / `batch_generate(prompts)` / `stream(prompt)`
- TP=1 时直接在调用进程内运行推理，不启动 ZMQ / FastAPI 子进程

### 3. 通用 debug 脚本
**现状**: `debug_qwen2.py` / `validate_qwen2.py` 写死了 Qwen2 路径，换模型要改代码。
**TODO**: 实现 `tools/debug.py`：
```
python tools/debug.py --model /path/to/model [--prompt "你好"] [--max-tokens 20] [--validate]
```
六个步骤自动运行：
1. 配置解析 — 打印 ModelConfig 关键字段
2. 模型构建 — 统计参数量，检查权重文件完整性
3. 权重加载 — 打印 missing / unexpected keys 摘要
4. prefill 检查 — 验证 logits shape，检测 NaN / Inf
5. greedy decode — 运行 N 步并打印解码文本
6. （`--validate`）与 HuggingFace 参考对比，报告 max abs diff 和 token 匹配率

适配任意 `model_type`（qwen2 / llama / ...），通过 `ModelConfig.model_type` 自动选择。

### 4. Prefill / Decode 分批 forward
**文件**: `needle/model/runner.py:109`, `needle/backend/backend.py:161-165`
**问题**: 当前 `is_prefill` 判断为"有任意 prefill seq 则整批走 prefill 路径"，导致同 step 里的 decode seq 被错误地走 varlen prefill attention。
**修复方向**: `ModelRunner.execute` 拆成两次 forward——先 prefill batch，再 decode batch，分别拼装输入张量并合并采样结果。

### 5. EOS token 未传入 stop_token_ids
**文件**: `needle/backend/backend.py:135-140`
**问题**: `_handle_user_msg` 构造 `SamplingParams` 时 `stop_token_ids=[]`，模型只靠 `max_new_tokens` 强截断，不会在 EOS 处自然停止。
**修复方向**: `BackendProc.__init__` 里加载 tokenizer，取 `eos_token_id` 填入默认 `stop_token_ids`。

### 6. 已完成序列的内存清理不健壮
**文件**: `needle/backend/backend.py:178-184`
**问题**: `_step` 在 `finished=True` 时调用 `free_sequence`，后者内部 `running.remove(seq)` 若 seq 已不在列表中会抛异常，存在 double-free 风险。
**修复方向**: `free_sequence` 改用 `discard` 语义；或在 `_step` 结束后统一清理，不在迭代中修改列表。

### 7. KV swap 未实际执行
**文件**: `needle/model/runner.py:82-104`
**问题**: `SchedulerOutput` 携带 `swap_in_map` / `swap_out_map`，但 `ModelRunner.execute` 未调用 `swap_kv_blocks`，抢占逻辑形同虚设。
**修复方向**: `execute` 开头按 swap map 调用 `swap_kv_blocks(gpu_cache, cpu_cache, mapping_tensor)`。

### 8. CPU KV Cache 未分配
**文件**: `needle/model/runner.py:65-78`
**问题**: `init_kv_cache` 只分配 GPU tensor，CPU swap buffer 缺失，swap out 时会 crash。
**修复方向**: 同步分配 `cpu_kv_caches`（`device="cpu"`，pin_memory=True 以加速 DMA）。

### 9. block_table padding 用 block_id=0 存在数据污染
**文件**: `needle/model/runner.py:141`
**问题**: `padded = blocks + [0] * (max_blocks - len(blocks))` 填充用物理 block 0，若 block 0 存有真实 KV 数据则 attention 读到脏数据。
**修复方向**: 预留最后一个物理块作"dummy block"专供 padding，或保证 `_decode_pytorch` / flashinfer 只读 `context_lens` 以内的 slot。

### 10. FlashInfer prefill 不支持 context_lens > 0（多轮 / 续写场景）
**文件**: `needle/layers/attention.py:108-114`
**问题**: `_prefill_flashinfer` 仅传 `q/k/v`，不读 kv_cache 中已有的 prefix KV，多轮对话续写时结果错误。
**修复方向**: 改用 flashinfer `single_prefill_with_kv_cache` 并传入 paged kv 参数；或当 `context_lens > 0` 时 fallback 到 `_prefill_pytorch`。

### 11. 流式输出中文乱码
**文件**: `needle/serving/server.py:184`
**问题**: `_stream_chat` 逐 token 调用 `decode([token])`，中文字符跨多个 token 时会输出乱码（UTF-8 截断）。
**修复方向**: 维护 per-request token buffer，用类似 `transformers.TextStreamer` 的增量解码策略，积累到可完整解码时再 flush。

### 12. /v1/completions 端点（非 chat）
**文件**: `needle/serving/server.py`
**现状**: 仅有 `/v1/chat/completions`，缺少原始文本补全接口，部分场景（batch eval、embedding pipeline）依赖此接口。

### 13. 请求超时 / 取消传播
**现状**: HTTP 层 disconnect 检测只在 streaming 路径，non-stream 请求客户端断开后 backend 仍在推理浪费资源。
**修复方向**: non-stream handler 里用 `asyncio.wait_for` + `http_req.is_disconnected()` 轮询，断开时调用 `client.abort(uid)`。

### 14. 进程间大数据传输支持 SHM
**现状**: API Server ↔ Backend Process 之间通过 ZMQ + msgpack 传递消息，input_ids / output token 等小负载没有问题，但未来批量传输大型输入序列（长上下文、图片 token 等）时序列化 + socket 拷贝会成为瓶颈。
**参考**: vLLM `shm_object_storage`（`vllm/worker/shm_object_storage.py`）——发送方将对象 pickle 写入具名共享内存块，消息里只传 shm 名称和大小，接收方直接 mmap 读取，零拷贝。
**TODO**:
- 在 `needle/distributed/` 下实现 `ShmObjectStorage`：
  - `put(obj) -> shm_key`：pickle 序列化后写入 `multiprocessing.shared_memory.SharedMemory`，返回 key
  - `get(shm_key) -> obj`：按 key attach shm，反序列化后 unlink
  - 引用计数或 TTL 自动回收，防止 shm 泄漏
- `UserMsg` 扩展：当 `input_ids` 超过阈值（如 1 KB）时改走 SHM 路径，ZMQ 只传 `ShmRef(key, size)`
- `DetokenizeMsg` 批量回包同理（大 batch 输出时 output_ids 较大）
- 单元测试：多进程读写正确性；进程异常退出后 shm 能被清理

### 15. 张量并行实测验证
**文件**: `needle/layers/linear.py`, `needle/model/runner.py`
**现状**: ColumnParallelLinear / RowParallelLinear 骨架已有，未在多卡环境实际跑通过。
**TODO**: 2 卡 TP 启动，验证与单卡输出一致；检查 NCCL AllReduce 时序和权重分片正确性。

### 15. 支持更多模型（LLaMA 3、Mistral、Gemma）
**现状**: `build_model` 只有 `qwen2` / `llama` 分支。
**TODO**: 补充各模型的 config key 映射和结构差异（RoPE scaling、sliding window attention 等）。

### 16. 补充单元测试
- `tests/unit/test_block_allocator.py`：swap / ref_count / OOM 路径
- `tests/unit/test_scheduler.py`：preemption、swap in/out 决策逻辑
- `tests/unit/test_paged_attention.py`：多 block 跨界读写正确性

### 17. 补充集成测试
- `tests/integration/test_llama_inference.py`：LLaMA 端到端 token 匹配
- `tests/integration/test_concurrent_requests.py`：多并发请求结果正确性

### 18. 吞吐基准测试
- `benchmarks/bench_throughput.py`：tokens/s，对比 vLLM / SGLang 基线

### 19. 日志 & 指标完善
- Prometheus metrics（`utils/metrics.py` 已有骨架）接入实际 step 计时
- 每 step 记录 batch_size、prefill_tokens、decode_tokens、KV cache 使用率
