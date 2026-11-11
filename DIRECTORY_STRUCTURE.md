# TokenSpeed 目录结构说明

TokenSpeed 是面向 Agentic 负载的高性能 LLM 推理引擎。整体架构分三层：

- **控制面（Control plane）**：Rust（`tokenspeed-engine/`）+ C++ 调度器（`tokenspeed-scheduler/`），负责请求生命周期、KV cache 归属、调度循环。
- **执行面（Data plane）**：Python（`python/tokenspeed/`），负责模型前向、采样、通信等。
- **Kernel 层**：`tokenspeed-kernel/` 作为统一的 kernel 包边界，通过注册表分发到 NVIDIA / AMD / NPU 等具体实现（`tokenspeed-kernel-amd/`、`tokenspeed-kernel-npu/`、`tokenspeed-mla/`）。

```text
tokenspeed-annotated/
├── python/                 # 主包 tokenspeed：Python 执行面 + CLI
├── tokenspeed-engine/      # Rust 控制面（ts-host 等 crates）
├── tokenspeed-scheduler/   # C++ 调度器（FSM + KV cache 管理）
├── tokenspeed-kernel/      # 统一 kernel 库：公共 API、注册/选择、插件
├── tokenspeed-kernel-amd/  # AMD GPU（Gluon）专用 kernel
├── tokenspeed-kernel-npu/  # 昇腾 NPU 专用算子
├── tokenspeed-mla/         # Blackwell 上的 MLA 注意力 kernel（CuTe DSL）
├── test/                   # 运行时测试、CI 任务定义、基准测试
├── docs/                   # 用户文档、设计文档、RFC
├── docker/                 # NVIDIA / AMD / Release 镜像 Dockerfile
├── assets/                 # README 用的 banner 与性能图
└── .skills/                # 给 AI agent 用的操作手册（kernel 优化、Triton 二分等）
```

根目录文件：`README.md`（项目介绍）、`AGENTS.md` / `CLAUDE.md`（贡献与 agent 规范）、`REVIEW.md`（评审规范）、`CONTRIBUTING.md`、`GOVERNANCE.md`、`SECURITY.md`、`CODE_OF_CONDUCT.md`、`ACKNOWLEDGEMENTS.md`、`LICENSE`。

---

## 1. `python/` —— 主包 `tokenspeed`（执行面）

| 路径 | 作用 |
| --- | --- |
| `tokenspeed/__init__.py`、`version.py`、`env.py`、`_logging.py` | 包入口、版本号、环境变量定义、日志配置 |
| `tokenspeed/cli/` | 命令行 `ts`：`serve_smg.py`（经 SMG 网关启动服务）、`reload_forward.py`（`ts reload-forward` 热重载前向代码）、`trace_merge.py`（合并 trace）、参数拆分/进程管理/日志前缀等辅助 |
| `tokenspeed/runtime/` | 运行时核心，见下表 |

### 1.1 `python/tokenspeed/runtime/`

| 子目录 | 作用 |
| --- | --- |
| `engine/` | 每个 rank 的数据面引导：`rust_host.py`（被 Rust `ts-host` 调用以构建数据面与调度器）、`io_struct.py` / `zmq_wire.py`（与 Rust `ts-wire` 字节兼容的 IPC 结构与 msgpack 编解码）、logprobs、profiler、显存占用控制、GC hook |
| `entrypoints/` | `engine.py`（Python 侧引擎入口）、`wire_driver.py`（通过 msgpack wire 驱动引擎，供测试/工具使用） |
| `execution/` | 前向执行路径：`model_runner.py` / `model_executor.py`（模型执行器）、`forward_step.py`、`forward_batch_info.py`（批元数据）、CUDA Graph（`breakable_cuda_graph.py`、`cudagraph_memory.py`）、`input_buffer.py`、`forward_reload.py`（前向热重载）、`slot_state.py`（按请求槽位的状态导出，用于回撤快照）、NaN 守护等 |
| `execution/drafter/` | 投机解码的 draft 模型：EAGLE、MTP、DFlash/DFlash2、DSpark、树状投机（`tree.py`）、投机采样 |
| `layers/` | 模型通用层：线性层、RMSNorm、RoPE、激活、Embedding、LogitsProcessor、MLP 等 |
| `layers/attention/` | 注意力实现：后端注册（`backends/`、`registry.py`）、KV cache 管理（`kv_cache/`）、页表、MLA / DSA / QSA / 线性注意力、DCP（上下文并行）等 |
| `layers/moe/` | MoE 层：top-k 路由、专家权重加载与 schema、block 结构 |
| `layers/dense/`、`layers/quantization/` | 稠密线性层的各量化方案（FP8、MXFP4、NVFP4、W8A8、compressed-tensors、ModelOpt 混合精度等） |
| `models/` | 具体模型实现（DeepSeek V3/V3.2/V4.1、Kimi、GLM、Qwen、MiniMax、Inkling、Nemotron 等）及其 NextN/MTP 头；`models/base/` 是通用 CausalLM / Decoder Layer / Transformer 骨架 |
| `configs/` | 模型配置类（各模型的 HF config 适配）、`model_config.py`、`load_config.py`、`device_config.py`、数值配置 |
| `model_loader/` | 权重加载（safetensors 读取、分片、权重工具） |
| `sampling/` | 采样参数与批采样信息、logits 布局、树验证；`backends/` 下是 greedy、FlashInfer、Triton、Sonic、Ascend 等采样后端 |
| `grammar/` | 结构化输出 / 约束解码（对接 Rust `ts-grammar` 的 xgrammar） |
| `cache/` | 多级 KV cache：`l2/`（Host 内存层，含逐层加载）、`l3/`（外部存储层，如 Mooncake store）、`transfer/`（设备↔Host 拷贝通道、布局、归属），以及 embedding cache 与 host storage |
| `pd/` | Prefill/Decode 分离（PD disaggregation）：拓扑、cache 协议、host peer、KV 事件 |
| `epd/` | Encode/Prefill/Decode 三段分离：多模态编码 worker、编码结果接收池等 |
| `distributed/` | 并行通信：进程组管理、TP/DP/PP 映射、`comm_backend/`（NCCL、HCCL、TRT-LLM allreduce、Iris、Triton RS/AG 等）、DP 采样通信、NCCL 挂起 |
| `moe/` | 专家并行负载均衡（EPLB）：专家位置、重平衡、`eplb_algorithms/` |
| `multimodal/` | 多模态输入：编码器批处理与 CUDA Graph、M-RoPE、特征传输（共享内存） |
| `metrics/` | 指标采集、注册、上报与插件转发 |
| `plugins/` | 运行时插件注册表 |
| `utils/` | 通用工具：`server_args.py`（服务参数定义）、CUDA stream、NVTX、JIT 编译检查、启动计时、网络/进程工具等 |

---

## 2. `tokenspeed-engine/` —— Rust 控制面

每个引擎 rank 是一个 `ts-host` 进程：Rust 主线程跑调度循环，内嵌 CPython 在 forward 线程跑数据面（见 `docs/rfcs/0002-rust-control-plane.md`）。

| crate（`crates/`） | 作用 |
| --- | --- |
| `ts-host` | rank 可执行文件：嵌入解释器、调用 `engine/rust_host.py`、运行主循环 |
| `ts-control` | 单 rank 控制面：请求入口检查、请求状态与输出处理、调度循环与 in-flight 队列、cache 操作确认、语法、专家重平衡（`eplb/`）、KV 事件、多 rank 每轮协商 |
| `ts-scheduler` | 把 C++ 调度器及其 nanobind 模块编译进二进制（`cxx/` facade） |
| `ts-wire` | 引擎 IPC schema 与 msgpack 编解码（`tests/fixtures/` 保证与 Python 端字节一致） |
| `ts-client` | 前端到引擎的客户端：握手、attention-DP 路由、输出流、abort、控制操作 |
| `ts-py` | Python 绑定 `tokenspeed_engine._native`：供测试/工具使用的客户端、L3 执行器、PD bootstrap server |
| `ts-collective` | rank 间基于 TCP mesh 的 CPU 集合通信（broadcast / all-gather / all-reduce） |
| `ts-grammar` | 基于 xgrammar C++ 库的约束解码 |
| `ts-metrics` | Prometheus 多进程指标文件与插件事件流 |
| `ts-mooncake` | 运行时加载 Mooncake 传输引擎与 store 客户端的 C API |
| `ts-l3` | L3 存储层的 Host 侧执行器：对象命名、Host 池、有界队列与 worker、后端健康检查 |
| `ts-pd` | PD 传输对端：P/D 握手、room 状态、传输 worker、`epd/` 与 `layout/` |

其他：`python/tokenspeed_engine/`（Python 包壳，提供 `host_binary()` 等）、`Cargo.toml`、`pyproject.toml`、`setup.py`。

---

## 3. `tokenspeed-scheduler/` —— C++ 调度器

以有限状态机（FSM）描述请求生命周期，用类型系统在编译期保证 KV 资源安全复用（设计见 `docs/design/scheduler.md`、`cache-concepts.md`）。

| 路径 | 作用 |
| --- | --- |
| `csrc/scheduler/` | 调度器主体：`scheduler.cpp`、请求对象 `request.*`、容量模型、执行计划 `execution_plan.h`、KV cache 事件；`operations/`（forward / cache / storage / prefill 分块 / 组需求等调度操作）；`outside_events/`（来自执行面的外部事件：forward、cache、PD） |
| `csrc/fsm/` | 状态机定义：forward、PD、storage 的状态与事件 |
| `csrc/cache/core/` | cache 基础类型：block pool、block table、`CacheBlockRef`（RAII 块引用）、cache 配置 |
| `csrc/cache/prefix/` | 前缀匹配：prefix index / hasher / matcher、checkpoint tree、boundary tree、tail table |
| `csrc/cache/coordinator/` | cache 协调：准入、驱逐候选与驱逐计划、组几何、host 段、影子占用 |
| `csrc/cache/allocator/` | cache group 分配器 |
| `csrc/cache/tier/` | 多级存储：storage manager、`StorageLease`、transfer manager |
| `csrc/resource/allocator/` | 请求池槽位分配（`ReqPoolIndex`） |
| `csrc/core/` | token 容器 |
| `bindings/` | nanobind Python 模块 `python_module.cpp` |
| `python/tokenspeed_scheduler/` | Python 包装 |
| `tests/` | C++ 单测（`cpp/`）与源码检查脚本 |

---

## 4. `tokenspeed-kernel/` —— 统一 kernel 库

运行时唯一的 kernel 包边界：提供平台无关的公共 API，通过注册表（`registry.py`、`selection.py`）按硬件/特性选择具体实现，支持插件扩展其他硬件。

| 路径 | 作用 |
| --- | --- |
| `python/tokenspeed_kernel/registry.py`、`selection.py`、`signature.py` | kernel 注册（`register_kernel`）、选择（`select_kernel`）与签名 |
| `platform.py`、`_triton.py`、`_optional.py` | 平台探测、tokenspeed-triton 统一导入点、可选依赖 |
| `ops/` | 按算子族组织（`<family>/<solution>`）：`attention/`（mha、mla、dsa、gdn、kda、mamba2、msa、qsa、tree、prologue 等）、`gemm/`、`moe/`、`layernorm/`、`quantization/`、`sampling/`、`communication/`、`kvcache/`、`embedding/`、`activation/`、`residual/`、`conv/`、`copy/`、`transform/`、`metadata/`、`ple/`、`model/` |
| `thirdparty/` | 直接引入的第三方代码（FlashInfer、TRT-LLM blockwise、CuTe DSL、vLLM 等），再由 `ops/` 注册 |
| `plugins/` | 多硬件插件机制及 CLI |
| `compile_monitor/` | 运行时 JIT 编译监控（检测 constexpr 反复重编译） |
| `numerics/` | 数值正确性校验工具（容差、比较、参考实现） |
| `benchmark/` | kernel 基准测试框架（harness、生成器、报告、Proton profiling） |
| `hot_reload.py`、`warmup.py`、`workspace.py`、`weights.py`、`profiling.py` | 热重载、预热、workspace 管理、权重处理、profiling |
| `test/` | 测试：通用测试在根下，厂商相关在 `nvidia/`、`amd/`、`npu/` |
| `benchmarks/` | 独立的基准脚本（如 AMD） |

## 5. `tokenspeed-kernel-amd/` —— AMD GPU kernel

基于 Triton Gluon 的 AMD 高性能 kernel，不依赖 `tokenspeed-kernel`（由后者注册调用）。

- `python/tokenspeed_kernel_amd/ops/gfx950/`、`ops/gfx1250/`：按架构 → 算子族（attention 的 mha/mla/dsa/kda、moe、gemm、quantization、sampling 等）组织。
- `_intrinsics/`：LLVM IR 形式的底层内建函数（算术、同步、wave 指令调度）。
- `assets/`：性能图。

## 6. `tokenspeed-kernel-npu/` —— 昇腾 NPU 算子

- `python/tokenspeed_kernel_npu/ops/`：Ascend 专用算子（paged MHA、MLA、DSA、RMSNorm、RoPE、MoE、W4A4、采样等）。
- `test/`：NPU 测试。运行时仍通过 `tokenspeed-kernel` 调用。

## 7. `tokenspeed-mla/` —— MLA 注意力 kernel

面向 SM100/SM103/SM107（Blackwell）的 MLA kernel：CuTe DSL 实现的 prefill（变长 FMHA）与 decode（FP16/BF16/FP8，split-KV），以及 Triton 实现的 K/V 打包 + FP8 量化。

- `python/tokenspeed_mla/`：kernel 源码；`tests/`：正确性测试；`scripts/`：benchmark 与发布检查；`assets/`：性能图。

---

## 8. `test/` —— 测试与 CI

| 路径 | 作用 |
| --- | --- |
| `runtime/` | 执行面单元/集成测试（cache、distributed、execution、layers、models、moe、sampling 等），含 `test_slot_state.py` 等不变量守护测试 |
| `cli/` | CLI 测试（参数拆分、serve、reload-forward、trace merge 等） |
| 根下 `test_*.py` | L2/L3 cache、host storage、传输布局、异步日志等测试；`runners.py`、`test_utils.py` 为公共工具 |
| `ci/` | CI 任务声明（YAML）：`ut/`（单测）、`eval/`（精度评测）、`perf/`（性能）、`npu/`、`deepswe/`（Agent 编码评测），以及 `run_slurm.sh` |
| `ci_system/` | CI 基础设施脚本：流水线、依赖安装、引擎 wheel 构建、GPU 状态清理、kernel 基准 CI 等 |
| `agentic_benchmark/` | 各模型的 Agentic 负载基准（Kimi、GLM、Qwen、MiniMax、Inkling 等） |
| `long_context_benchmark/`、`random_benchmark/`、`single_request_benchmark/` | 长上下文、随机负载、单请求解码基准 |

## 9. `docs/` —— 文档（VitePress 站点）

| 路径 | 作用 |
| --- | --- |
| `design/` | 各子系统的设计不变量（event loop、scheduler、cache 概念、prefill delay、unified path、attention prologue、tracing、tree speculation、monokernel、async L3、数值等）——修改对应代码前必读 |
| `rfcs/` | 设计提案（Rust 控制面、前向热重载） |
| `guides/` | 使用指南（快速开始、启动、前向热重载、Hopper/CUDA 12.9、采样校准、发布流程等） |
| `serving/` | 部署相关：PD 分离、并行策略、权重显存 |
| `configuration/` | 服务参数说明与兼容参数 |
| `recipes/` | 各模型的启动配方 |
| `.vitepress/`、`package.json`、`bun.lock`、`index.md` | 文档站点配置与首页 |

## 10. 其他

- `docker/`：`Dockerfile.nvidia`、`Dockerfile.amd`、`Dockerfile.release` 镜像构建文件。
- `assets/`：`banner/`（项目横幅）、`perf/`（性能对比图）。
- `.skills/`：面向 AI agent 的技能文档（AMD kernel 优化、Triton 版本二分定位）。
