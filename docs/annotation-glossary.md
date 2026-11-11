# 执行面源码注释术语表

本表统一 `python/tokenspeed/` 源码注释中的中文用词。注释里首次出现某个术语时写成“中文（英文原词）”，之后只用中文。读者已经熟悉的通用术语（Transformer、GQA、RoPE、RMSNorm、列并行、行并行、CUDA Graph 等）不再收录。

“首次出现”指按 `ANNOTATION_PLAN.md` 的阅读顺序第一次需要理解它的文件。

## 1. 进程、线程与一轮调度

| 中文（英文） | 含义 | 首次出现 |
| --- | --- | --- |
| 控制面（control plane） | 由 Rust（`tokenspeed-engine/`）和 C++ 调度器（`tokenspeed-scheduler/`）组成，负责接收请求、调度、管理 KV cache 的归属、输出结果。不碰 GPU | `engine/rust_host.py` |
| 执行面 / 数据面（data plane） | `python/tokenspeed/` 中的 Python 代码，负责一切 GPU 工作：模型前向、采样、KV cache 拷贝。两个词同义，注释统一用“执行面” | `engine/rust_host.py` |
| rank 进程（`ts-host`） | 每个 GPU 一个进程。主线程（控制线程）跑 Rust 调度循环；forward 线程通过嵌入的 CPython 执行执行面的 Python 代码 | `engine/rust_host.py` |
| 一轮（round） | 调度循环的一次迭代：收请求 → 调度器产出计划 → 执行面执行 → 回收结果。一轮最多一次模型前向 | `engine/rust_host.py` |
| 计划（plan / execution plan） | 调度器每轮产出的对象，包含本轮的前向操作（forward op）和 cache 操作 | `engine/rust_host.py` |
| 前向操作（forward op） | 计划中描述一次模型前向的部分：哪些请求（`request_ids`）、各算多少 token（`input_lengths`）、其中几个是 extend（`num_extends()`）、各自的 block table | `execution/model_executor.py` |
| cache 操作（cache op） | 计划中 KV 页的搬运：页拷贝、页清零、写回 Host（write-back）、从 Host 读回（load-back）、快照与恢复等 | `execution/device.py` |
| 提交（commit） | 控制面读取某次前向的结果（采样出的 token 等）并据此推进请求状态 | `engine/rust_host.py` |
| 在途深度（in-flight depth） | 已经下发到 GPU、但结果还没被提交的前向个数上限。0 表示同步执行；1 表示下一轮的计划和这一轮的 GPU 计算重叠 | `engine/rust_host.py` |
| 重叠调度（overlap schedule） | 在途深度 ≥ 1 时的工作方式：CPU 在 GPU 计算本轮时准备下一轮；下一轮 decode 的输入 token 直接从 GPU 上取，不回到 CPU | `execution/model_executor.py` |

## 2. 一次前向

| 中文（英文） | 含义 | 首次出现 |
| --- | --- | --- |
| 前向上下文（`ForwardContext`） | 描述一次前向的参数对象（模式、各类计数、DP 布局，以及指向注意力后端和 KV 池的引用），作为参数显式传给每一层；本身不存张量 | `models/qwen3.py` |
| extend | 为一个请求追加若干 token 并写入 KV cache 的计算，涵盖首次 prefill、命中前缀缓存后的剩余部分、chunked prefill 的后续块。注释中“prefill”泛指 extend | `execution/forward_batch_info.py` |
| decode | 每个请求只算 1 个 token（投机解码时为若干个待验证 token）的计算 | `execution/forward_batch_info.py` |
| 前向模式（`ForwardMode`） | EXTEND（全是 extend）、DECODE（全是 decode）、MIXED（同一次前向里既有 extend 行又有 decode 行）、IDLE（本 rank 没有请求，只为配合其他 rank 的集合通信而空跑） | `execution/forward_batch_info.py` |
| 混合批（mixed batch） | MIXED 模式的一次前向。行顺序固定为 extend 行在前、decode 行在后 | `execution/output_layout.py` |
| 输出布局（`ForwardOutputLayout`） | 一次前向中哪些请求、哪些行产生输出：完成 prefill 的请求每个 1 行，decode 请求每个固定若干行 | `execution/output_layout.py` |
| 空跑前向（idle forward） | attention DP 下，某个 rank 本轮没有请求，但其他 rank 有，它也要跑一次零行的前向，以便参与集合通信 | `models/base/decoder_layer.py` |
| 请求槽位（request pool slot / request pool index） | 每个在跑的请求在执行面占用的一个固定编号，有效范围 1..max_requests；0 号不分配给请求，max_requests+1 号留给 padding 行。按槽位存放跨轮的 GPU 状态（下一轮输入 token、停止条件、递归状态等） | `execution/runtime_states.py` |
| 槽位状态（slot state） | 以请求槽位为下标、存在 cache group 之外的 GPU 状态。抢占时需要随 KV 页一起快照（`SlotStateExporter`） | `execution/slot_state.py` |
| 固定输入缓冲区（input buffers） | 预先分配、地址不变的输入张量（`input_ids`、`positions`、`seq_lens` 等），每轮把数据写进去，供 eager 和 CUDA Graph 共用 | `execution/input_buffer.py` |

## 3. KV cache

| 中文（英文） | 含义 | 首次出现 |
| --- | --- | --- |
| block | 调度器管理 cache 的通用单位：block table 中的一格，对应一个 `CacheBlock`。不论里面存的是 KV 还是递归状态都叫 block | `layers/attention/backends/cache_metadata.py` |
| page | 存放 KV 行、供分页注意力 kernel 读取的 block。page 一定是 block，递归状态的 block 不是 page | `layers/attention/backends/paged/router.py` |
| block table | 调度器下发的“请求 → block 编号”表，是在调度器与执行面之间传递的通用容器 | `layers/attention/backends/cache_metadata.py` |
| page table | 只用于分页 KV 注意力：kernel 读取的“请求的逻辑页 → 物理页”表。由 block table 换算而来 | `layers/attention/backends/paged/router.py` |
| page_size / block_granularity | 一个 block（page）覆盖多少个 token，以逻辑 token 计 | `layers/attention/backends/paged/router.py` |
| kernel 页大小（kernel_page_size） | 某个注意力 kernel 实现要求的页跨度（如 64、128）。属于 kernel 的属性，可能与调度器的 block 粒度不同，由 router 负责换算 | `layers/attention/kernel_page_sizes.py` |
| 槽位地址（slot） | 某个 token 的 KV 在 KV 池里的行号：`页号 × 页大小 + 页内偏移` | `layers/attention/backends/paged/write_locations.py` |
| cache group | 一组用同一种方式存放、同一种粒度管理的 cache。按注意力类型分三种：全历史（full_history，所有 token 常驻）、滑动窗口（sliding_window，只保留最近若干 token）、状态（state，递归/卷积状态，每个请求固定两格） | `layers/attention/backends/paged/router.py` |
| cache group 路由（`CacheGroupRouter`） | 分页 KV 模型实际使用的注意力后端：每个 cache group 一个叶子后端，负责把调度器的 block 换算成 kernel 的 page | `layers/attention/backends/paged/router.py` |
| 叶子后端（leaf backend） | 挂在 router 下、真正调用注意力 kernel 的后端，例如 `MHAAttnBackend` | `layers/attention/backends/paged/mha.py` |
| KV 池（`CachePool`） | 注意力层读写 KV 的视图对象，本身不持有显存 | `layers/attention/kv_cache/base.py` |
| arena（`CacheArena`） | 一个模型唯一的一块 cache 显存分配；所有 cache group 和所有 KV 池视图都建在它上面 | `layers/attention/kv_cache/arena.py` |
| LCM block | 物理分配单位：多个 cache group 的 `CacheBlock` 按各自尺寸的最小公倍数打包进同一个父块 | `layers/attention/kv_cache/recipes/plan.py` |
| recipe | 某一类模型声明“有哪些 cache group、每组多大、怎么打包”的规划代码，启动时据此确定 KV 容量 | `layers/attention/kv_cache/recipes/base.py` |
| 前缀粒度（prefix_granularity） | 前缀缓存按多少 token 一段计算哈希 | `engine/scheduler_utils.py` |
| 页清零（page zeroing） | 新分配、且不会被整页覆盖的页先清零，避免读到上一个使用者留下的数据 | `execution/device.py` |
| 抢占（retraction） | KV 容量不够时，调度器把某个请求的页和槽位状态逐字节快照到 Host、释放 GPU 页；之后恢复（restore）到新页上继续，不重新计算 | `execution/slot_state.py` |
| L2 / L3 缓存 | Host 内存上的前缀缓存层（L2）和外部存储层（L3，如 Mooncake），用于扩大前缀缓存容量 | `cache/l2/executor.py` |

## 4. 注意力层

| 中文（英文） | 含义 | 首次出现 |
| --- | --- | --- |
| 注意力后端（attention backend） | 负责一层注意力具体计算的对象：准备 kernel 元数据（`init_forward_metadata`）、分派到 extend 或 decode 的计算 | `layers/attention/backends/base.py` |
| prologue | 投影之后、核心注意力之前的统一入口：QK-Norm、RoPE、量化、把 K/V 写入 KV cache | `layers/paged_attention.py` |
| 核心注意力（core attention） | prologue 之后、`o_proj` 之前的 softmax 注意力计算本身 | `layers/paged_attention.py` |
| window_left | 滑动窗口注意力中，当前 token 之前还能看到多少个 token。等于 HF 配置中的 `sliding_window - 1` | `models/qwen3.py` |
| kernel 注册表（kernel registry） | `tokenspeed-kernel` 包中按硬件和特性选择 kernel 实现的机制；执行面只调用其族级 API，不直接 import 具体实现 | `layers/attention/backends/paged/mha.py` |

## 5. CUDA Graph

| 中文（英文） | 含义 | 首次出现 |
| --- | --- | --- |
| 可断开的 CUDA Graph（breakable CUDA graph） | 捕获时在标记为断点的方法处把图切成多段，断点内的代码每次回放都以 eager 方式执行，其余部分按图回放 | `execution/breakable_cuda_graph.py` |
| 断点（`@break_point` / eager break） | 标记一个“按序列长度变化”的方法（注意力、线性混合器等），让它在图中以 eager 方式执行 | `layers/paged_attention.py` |
| 档位（batch size bucket） | 预先捕获图的 batch size 列表；实际 batch 补齐到不小于它的最近档位 | `execution/forward_step.py` |
| 原地刷新元数据（refresh in place） | eager 与图共用同一套元数据缓冲区，每轮原地写入新值，不重新分配 | `layers/attention/backends/paged/mha.py` |

## 6. 并行

| 中文（英文） | 含义 | 首次出现 |
| --- | --- | --- |
| 并行映射（`Mapping`） | 一个 rank 在各种并行维度中的位置和通信组。注意力（`mapping.attn`）、稠密 MLP（`mapping.dense`）、MoE（`mapping.moe`）、lm_head（`mapping.lm_head`）可以有不同的并行方式 | `models/qwen3.py` |
| 层内通信管理（`CommManager`） | 每层一个，集中负责层与层之间的 all-reduce / reduce-scatter / all-gather 以及与 RMSNorm 的融合；线性层自身不做这些通信 | `models/base/decoder_layer.py` |
| RSAG 布局 | 层输出不做 all-reduce，而是 reduce-scatter，使每个 rank 只持有一部分 token 行；下一处需要完整行时再 all-gather | `models/base/decoder_layer.py` |
| attention DP | 注意力部分按请求切分到多个数据并行组，每组独立调度；MLP/MoE 部分仍可跨组并行 | `distributed/mapping.py` |
| EP（expert parallel） | MoE 的专家分布到多个 rank | `layers/moe/expert.py` |
| PP（pipeline parallel） | 层按顺序切成多段，分布到多个 rank | `distributed/pp_stage.py` |
| QCP（query context parallel） | 一次 prefill 的 query 行按请求分段分到注意力 TP 组的各个 rank | `execution/query_shard.py` |
| DCP（decode context parallel） | KV 历史按 token 分布到多个 rank，decode 时汇总 | `layers/attention/dcp/` |

## 7. 特性

| 中文（英文） | 含义 | 首次出现 |
| --- | --- | --- |
| 草稿器（drafter） | 投机解码中负责生成候选 token 的组件 | `execution/drafter/base.py` |
| 验证（verify） | 目标模型一次前向检查草稿 token、决定接受几个 | `execution/drafter/base.py` |
| 接受长度（accept length） | 一次验证中被接受的 token 数 | `execution/model_executor.py` |
| 层输入捕获（aux hidden capture） | 目标模型前向时保存若干层的输入 hidden，供 EAGLE3 / DFlash 草稿模型使用 | `models/base/transformer_model.py` |
| PD 分离（prefill/decode disaggregation） | prefill 和 decode 在不同的引擎实例上运行，KV cache 通过网络传输 | `pd/host_peer.py` |
| EPD 分离 | 在 PD 分离基础上，再把多模态编码器单独部署 | `epd/encode_host.py` |
| monokernel | decode 时每个解码层只启动一次持久化 kernel | `models/monokernel.py` |
