# 执行面（Python）源码注释计划

## 0. 目标与读者

- **注释对象**：执行面 `python/tokenspeed/runtime/`（约 21 万行），以及它直接调用、读懂主线必需的少量 `tokenspeed-kernel` 入口。
- **读者**：懂 Transformer、MHA、GQA 的初级推理工程师。默认读者知道 Q/K/V、softmax attention、KV cache 的基本概念，但**不熟悉** PagedAttention、连续批处理、CUDA Graph、张量并行、投机解码等推理系统技术。
- **目的**：读者按本计划的顺序读完带注释的源码后，能说清楚一个请求在执行面里**从一轮调度计划到采样出下一个 token** 的完整路径，以及每一步为什么这样设计。
- **不做的事**：只加注释，**不改任何代码逻辑**；也不注释控制面（Rust `tokenspeed-engine/`、C++ `tokenspeed-scheduler/`）的内部实现，只在边界处说明它们给执行面传了什么。

---

## 1. 注释格式规范（对齐 nano-vllm-annotated）

nano-vllm-annotated 的风格是：**中文注释直接写在源码里**，在文件头、类头、关键函数前放大段说明，并配 ASCII 图和带具体数字的例子；行内注释解释“是什么”和“为什么”。本项目沿用这一风格。

### 1.1 四个层级

| 层级 | 位置 | 写什么 |
| --- | --- | --- |
| **文件头** | `import` 之后、第一个类/函数之前 | ① `Shape记号`（本文件出现的维度字母）；② `基础概念`（读本文件前需要知道、但读者可能不会的知识）；③ 数据流图；④ 在整条主线中的位置（“上游是谁调用我，下游我调用谁”） |
| **类头** | `class` 定义上方 | 职责一句话；生命周期/状态图或决策流程图（用 `┌─┐│└┘▼` 画框） |
| **函数/方法** | `def` 上方或函数体开头 | 输入输出的形状与含义；分步骤说明；**带具体数字的走查例子** |
| **行内** | 代码行上方或行尾 | 字段含义、为什么这样做、边界情况（如 padding、`-1` 槽位） |

### 1.2 统一约定

- 注释一律用 `#`，**不改原有英文 docstring**（避免影响 `help()`、文档生成，也便于和上游 diff）。中文注释写在 docstring 下方。
- **Shape 记号**全项目统一，定义在第一个注释的文件（`models/qwen3.py`）中，其他文件引用即可：

  ```text
  # Shape记号
  #   T：本次前向的 token 总数（prefill 为所有请求拼接后的长度；decode 为 batch 大小 × 每请求 token 数）
  #   B：本次前向的请求数（bs）
  #   E：本次前向中做 prefill/extend 的请求数（num_extends），其余 B-E 个请求做 decode
  #   H：hidden_size
  #   Hq / Hkv：当前 rank 上 Q / KV 的头数（已除以 TP）
  #   D：head_dim
  #   P：page_size，一个 KV 页存多少个 token
  #   NP：KV 页总数
  #   L：层数
  #   V：词表大小
  ```

- **贯穿全文的例子**（所有走查都用同一组数字，读者不用反复换算）：
  - 模型：Qwen3-8B，`H=4096`、`L=36`、Q 头 32、KV 头 8（GQA，每 4 个 Q 头共享 1 个 KV 头）、`D=128`、`intermediate=12288`、`V=151936`。
  - 单卡（TP=1）；`P=64`（示例取值，实际由 cache 计划决定）。
  - 一轮混合批：请求 A 新到达，prompt 300 token；请求 B 已在 decode，当前长度 1000。于是 `B=2`、`E=1`、`T=301`。
- **略读标记**：大文件只注主线。主线以外的分支在入口处加 `# [略读] 投机解码/PD/DP 专用，见第 N 阶段`，读者第一遍可以跳过。
- **引用设计文档**：设计意图已写在 `docs/design/*.md` 的，注释只写一两句结论，加 `# 详见 docs/design/xxx.md「章节名」`，不整段翻译。
- **术语**：首次出现时给出“中文（英文原词）”，之后统一用中文。术语表放在 `docs/annotation-glossary.md`（第 0 阶段产出），所有注释保持一致。
- 每个阶段一个 commit；提交前运行 `pre-commit run --all-files`，确保 black/isort 不改动代码。

### 1.3 样例（第 1 阶段 `Qwen3Attention.forward` 的目标效果）

```python
    # Qwen3 一层 attention 的前向（以 Qwen3-8B、TP=1、T=301 为例）
    #   hidden_states [T, H] = [301, 4096]
    #        │ qkv_proj：一次 GEMM 同时算出 Q、K、V（三个权重在加载时拼成一个矩阵）
    #        ▼
    #   qkv [301, 4096 + 1024 + 1024]
    #        │ split
    #        ├── q [301, 4096] = 32 个 Q 头 × 128
    #        ├── k [301, 1024] =  8 个 KV 头 × 128   ← GQA：KV 头数是 Q 头的 1/4
    #        └── v [301, 1024]
    #        ▼
    #   self.attn（PagedAttention）：q/k 做 RMSNorm + RoPE，把 k/v 写入 KV cache，再算注意力
    #        ▼
    #   attn_output [301, 32, 128] -> reshape [301, 4096]
    #        │ o_proj（行并行；TP>1 时各卡结果是部分和，reduce 在层外做）
    #        ▼
    #   output [301, 4096]
    def forward(self, positions, hidden_states, ctx):
        qkv, _ = self.qkv_proj(hidden_states)
        # q_size = Hq*D = 4096，kv_size = Hkv*D = 1024
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        ...
```

---

## 2. 主线选择：先走最简单的一条路

TokenSpeed 支持很多模型和特性，但它们都是**同一条执行路径上的参数**（见 `AGENTS.md`「Design principles」）。所以先固定一条最简单的配置，把它走通，再逐个打开开关：

| 维度 | 第一遍选择 | 之后再看 |
| --- | --- | --- |
| 模型 | `models/qwen3.py`（稠密、GQA、QK-Norm） | `llama.py`（对照）、`qwen3_moe.py`（MoE）、`deepseek_v3.py`（MLA） |
| 并行 | 单卡（TP=1，无 DP/PP/EP） | TP → attention DP → EP |
| 注意力后端 | `backends/paged/mha.py` | trtllm、MLA、DSA、线性注意力/状态后端 |
| 采样 | `sampling/backends/greedy.py` | flashinfer、triton |
| 图 | 先 eager | CUDA Graph（第 7 阶段） |
| 投机解码 / PD 分离 / 多级 cache | 关闭 | 第 10 阶段选读 |

### 2.1 主线全景图（写进 `engine/rust_host.py` 文件头，并在每个阶段的文件头里标出“你在这里”）

```text
Rust 控制面（ts-host 主线程）：调度器产出本轮计划 plan（哪些请求、各算多少 token、KV 页表）
    │  通过嵌入的 CPython 调用（forward 线程）
    ▼
HostDataPlane.execute(plan)                         engine/rust_host.py        第 6 阶段
    ▼
DeviceHandle.execute(plan, planned, cache_ops)      execution/device.py        第 6 阶段
    │  先执行 cache 操作（页拷贝/清零/换入换出），再提交前向
    ▼
ModelExecutor.execute_forward_op(forward_op, ...)   execution/model_executor.py 第 4 阶段
    │  ① CacheBatchMetadata：把本批的页表打包上 GPU
    │  ② InputBuffers.fill_input_buffers：input_ids / positions / seq_lens 写入固定缓冲区
    │  ③ attn_backend.init_forward_metadata：注意力 kernel 需要的元数据
    ▼
ModelExecutor._forward_step                         execution/model_executor.py 第 4 阶段
    │  ├─ eager：ModelRunner.forward
    │  └─ 图：ForwardStepRunner（CUDA Graph 回放）  execution/forward_step.py   第 7 阶段
    ▼
Qwen3ForCausalLM.forward → BaseTransformerModel → BaseDecoderLayer × L   第 1 阶段
    │   embed → [RMSNorm → Attention → RMSNorm → MLP] × L → norm
    │   各层零件：Linear / RMSNorm / RoPE / SiluAndMul / Embedding        第 2 阶段
    │   Attention：PagedAttention → gqa_prologue（写 KV）→ MHAAttnBackend 第 3 阶段
    ▼
LogitsProcessor：取每个请求最后一个位置的 hidden，乘 lm_head 得到 logits   第 5 阶段
    ▼
SamplingBackend.sample：得到 next token                                    第 5 阶段
    ▼
RuntimeStates 更新（下一轮的输入 token、停止条件）→ complete() 交回 Rust   第 4/6 阶段
```

---

## 3. 分阶段计划（由浅入深）

每个阶段列出：要注释的文件、注释深度、重点讲什么、读完能回答的问题。注释深度分三档：

- **全注**：每个类、函数都注释，适合主线核心且不太长的文件。
- **主线注**：只注主线用到的类/函数，其余在入口加 `[略读]`。
- **头注**：只写文件头和类头，说明“这是什么、什么时候用”。

### 第 0 阶段：准备（不改源码）

| 产出 | 内容 |
| --- | --- |
| `docs/annotation-glossary.md` | 术语表：extend/prefill/decode/mixed、page/block、page_table/block_table、slot、request pool index、cache group、arena、prologue、eager break、in-flight depth、overlap、TP/DP/EP/PP、retraction……每条给出中文、英文、一句话解释、首次出现的文件 |
| 本计划第 2.1 节的全景图 | 定稿后作为后续注释的锚点 |

依据：`docs/design/cache-concepts.md`「block vs. page」「page_table vs. block_table」、`docs/design/event-loop.md`「The process」「Anatomy of a round」。

### 第 1 阶段：模型骨架（读者最熟悉的部分）

先读模型代码：它最接近读者已有的 Transformer 知识，而且能先建立“一次前向输入什么、输出什么”的直觉。

| 文件 | 行数 | 深度 | 重点 |
| --- | --- | --- | --- |
| `models/qwen3.py` | 255 | 全注 | 定义全项目的 Shape 记号；GQA 头数计算（`num_kv_heads = max(1, total/tp)`）；`qkv.split`；`get_stacked_params_mapping` 为什么能把 q/k/v 三个权重合成一个 `qkv_proj` |
| `models/base/transformer_model.py` | 195 | 全注 | embed → 层循环 → final norm；`residual` 为什么作为单独张量在层间传递（融合 add+norm） |
| `models/base/decoder_layer.py` | 250 | 全注 | pre-norm 结构；`resolve_attn/resolve_mlp` 模板方法：子类只换零件；norm 和通信都经 `comm_manager` 完成 |
| `distributed/comm_manager.py` | — | 头注 + `input_reduce_norm` / `post_attn_reduce_norm` / `post_mlp_fused` 主线注 | 单卡时它只是 “add residual + RMSNorm”；多卡时在同一位置插入 all-reduce 并与 norm 融合（细节留到第 9 阶段） |
| `models/base/causal_lm.py` | 692 | 主线注 | `forward` 和 `exit_logits`；`load_weights` 的大致流程；权重热更新部分 `[略读]` |
| `models/llama.py` | 220 | 头注 | 和 Qwen3 的差异（没有 QK-Norm），帮助读者确认“模型文件只是拼零件” |
| `models/registry.py` | — | 头注 | 架构名 → 模型类（`EntryClass`）是怎么注册的 |

读完能回答：Qwen3 一层的数据流和各张量形状？`positions` 从哪里来？为什么模型的 `forward` 里看不到 KV cache？

### 第 2 阶段：基础层零件

| 文件 | 行数 | 深度 | 重点 |
| --- | --- | --- | --- |
| `layers/layernorm.py` | 759 | 主线注（`RMSNorm`、`FusedRMSNorm`） | RMSNorm 公式；fused add+norm 如何省一次显存读写；allreduce+norm 融合 `[略读]` |
| `layers/activation.py` | 225 | 主线注（`SiluAndMul`） | gate/up 拼接后一次激活 |
| `layers/mlp.py` | 203 | 全注 | `GatedMLP`：gate_up_proj（列并行）→ SiluAndMul → down_proj（行并行） |
| `layers/rotary_embedding.py` | 1533 | 主线注（`RotaryEmbedding`、`get_rope`） | RoPE 原理、cos/sin 缓存表、neox 与 gptj 两种旋转；YaRN/Llama3/MRoPE 等变体只写头注 |
| `distributed/mapping.py` | — | 头注 | 先建立“attention 和 MLP 可以有不同 TP 组”的概念，为线性层做铺垫 |
| `layers/linear.py` | 1747 | 主线注（`LinearBase`、`ColumnParallelLinear`、`MergedColumnParallelLinear`、`QKVParallelLinear`、`RowParallelLinear`） | 列并行/行并行的切分图（各卡拿到哪一块权重、输出是完整结果还是部分和）；`weight_loader` 怎么把 checkpoint 中的分片放到正确位置；DP 线性层 `[略读]` |
| `layers/parameter.py` | 475 | 头注 | 带 `weight_loader` 的参数类 |
| `layers/vocab_parallel_embedding.py` | 734 | 主线注 | 词表切分、越界 token 的 mask、`ParallelLMHead` |

读完能回答：TP=2 时 `QKVParallelLinear` 每张卡算出哪些头？`RowParallelLinear` 的输出为什么要 all-reduce？RoPE 是在哪一步作用到 q/k 上的？

### 第 3 阶段：注意力与 KV cache（核心难点）

这是执行面和普通 Transformer 实现差别最大的地方。按“上层接口 → 写 cache → 读 cache 算注意力 → cache 存在哪”的顺序读。

| 文件 | 行数 | 深度 | 重点 |
| --- | --- | --- | --- |
| `layers/paged_attention.py` | 696 | 主线注（`PagedAttention.forward/prologue/attend`） | 一层注意力分两步：prologue（QK-Norm、RoPE、量化、写 KV）+ attend（核心注意力）；`@break_point` 是什么（先讲“这里在 CUDA Graph 中会断开改为 eager 执行”，细节留到第 7 阶段）；MLA 分支 `[略读]` |
| `tokenspeed-kernel/.../ops/attention/prologue/__init__.py` | — | 头注 | `gqa_prologue` 的输入输出约定；为什么写 cache 统一放在 prologue |
| `layers/attention/backends/base.py` | 902 | 主线注（`AttentionBackend`） | 后端的统一接口：`init_forward_metadata` → `forward` 分派到 `forward_extend` / `forward_decode` |
| `layers/attention/backends/paged/base.py` | 598 | 主线注 | 分页后端的公共逻辑；树投机部分 `[略读]` |
| `layers/attention/backends/paged/mha.py` | 806 | 全注（树相关除外） | `MHAExtendMetadata` / `MHADecodeMetadata` 各字段；用贯穿例子走查：A 的 300 个 token 怎么做 extend，B 的 1 个 token 怎么通过页表读出 1000 个历史 KV |
| `layers/attention/backends/paged/write_locations.py` | 320 | 全注 | 每个新 token 的 KV 写到哪个槽位：`slot = page_id * P + offset` |
| `layers/attention/page_table.py`、`backends/cache_metadata.py` | 90 / 213 | 全注 | 调度器给的 block table 怎么变成 kernel 用的 page table |
| `layers/attention/kv_cache/base.py` | 428 | 主线注（`CachePool`） | KV 池的抽象；cache group 概念 |
| `layers/attention/kv_cache/mha.py` | 356 | 全注 | `MHATokenToKVPool` 的显存布局图（每一维的含义，用贯穿例子画出 A、B 的 KV 落在哪些页） |
| `layers/attention/kv_cache/arena.py`、`factory.py` | 643 / 360 | 头注 | 显存怎么一次性分配、多个 cache group 怎么共享 |
| `layers/attention/registry.py` | 1604 | 头注 + `create_attn_components` 主线注 | 启动时如何按模型和硬件选出后端和 KV 池 |

配套阅读：`docs/design/attention-prologue.md`「One entry per attention shape」「Who writes the KV cache」；`docs/design/cache-concepts.md`「Two worlds」「block vs. page」。

读完能回答：A 的第 299 个 token 的 K 写到了哪个物理位置？decode 时 B 的 query 怎么找到它的 1000 个历史 KV？GQA 的“4 个 Q 头共享 1 个 KV 头”在 cache 布局和 kernel 调用里体现在哪里？

### 第 4 阶段：一次前向是怎么组织起来的

| 文件 | 行数 | 深度 | 重点 |
| --- | --- | --- | --- |
| `execution/forward_batch_info.py` | 145 | 全注 | `ForwardMode`：EXTEND / DECODE / MIXED / IDLE 的含义；positions 的计算 kernel |
| `execution/output_layout.py` | 103 | 全注 | 混合批中 prefill 行在前、decode 行在后的布局 |
| `execution/context.py` | 240 | 全注 | `ForwardContext` 为什么“不存张量”；各字段的生命周期 |
| `execution/input_buffer.py` | 799 | 主线注（`fill_input_buffers`、`load_round_inputs`） | 为什么用预分配的固定缓冲区（为 CUDA Graph 做准备）；贯穿例子中 `input_ids_buf` 前 301 个位置分别放什么 |
| `execution/runtime_states.py` | 619 | 主线注 | 按 request pool slot 存放的长期状态：`future_input_map`（下一轮的输入 token）、停止条件 |
| `execution/request_slots.py`、`slot_state.py` | 42 / 262 | 头注 | slot 是什么；retraction 时为什么要导出 slot 状态 |
| `execution/model_runner.py` | 765 | 主线注（`__init__`、`load_model`、`forward`） | 持有模型、调用 `model.forward`；权重更新 `[略读]` |
| `execution/model_executor.py` | 3138 | 主线注：`execute_forward_op`、`_forward_step`、`_run_target_forward`、`_run_sampling`、`_update_runtime_state` | 分段写流程图；**overlap 调度**（本轮 CPU 准备与上一轮 GPU 计算重叠，`settle()` 等上一轮结果）单独画时序图；drafter/PP/DP/grammar 分支 `[略读]` |

读完能回答：一轮里 GPU 上依次发生了什么？为什么下一轮 decode 的输入 token 不需要先拷回 CPU？

### 第 5 阶段：从 hidden states 到下一个 token

| 文件 | 行数 | 深度 | 重点 |
| --- | --- | --- | --- |
| `layers/logits_processor.py` | 1653 | 主线注（`LogitsMetadata`、`LogitsProcessor.forward` 主路径） | 只对每个请求最后一个位置算 logits（`gather_ids`）；词表并行下的 all-gather；softcap、input logprobs `[略读]` |
| `layers/logits_processor_output.py` | 78 | 全注 | 输出结构 |
| `sampling/sampling_params.py` | 276 | 全注 | temperature / top-p / top-k / min-p / 惩罚项 |
| `sampling/sampling_batch_info.py` | 139 | 全注 | 把一批请求的采样参数打包成张量 |
| `sampling/backends/base.py` | 804 | 主线注 | 采样后端接口；投机验证接口 `[略读]` |
| `sampling/backends/greedy.py` | 307 | 主线注 | argmax 采样（最简单的实现） |
| `sampling/backends/flashinfer.py` | 1029 | 头注 + `sample` 主线注 | 带温度和 top-p/top-k 的批量采样 |
| `sampling/registry.py` | 154 | 头注 | 采样后端选择 |

读完能回答：A 的 301 个 hidden 里只有哪一行进入 lm_head？top-p 在 GPU 上是怎么批量实现的？

### 第 6 阶段：执行面入口——一轮完整的调用链

到这里读者已经懂“一次前向”，再往上看它是被谁、以什么节奏调用的。

| 文件 | 行数 | 深度 | 重点 |
| --- | --- | --- | --- |
| `engine/rust_host.py` | 1674 | 主线注（`HostDataPlane.admit/execute/complete/release`、`HostBootstrap`） | 文件头放第 2.1 节全景图；Rust 控制面线程与 Python forward 线程的分工；`_Dispatched` 和 in-flight 队列 |
| `execution/device.py` | 2047 | 主线注（`DeviceHandle.execute`、`_launch_forward`、`build_components`） | 一个 plan 里 cache 操作和前向的提交顺序为什么不能乱（引用 event-loop.md「Anatomy of a round」第 4 步，只写结论） |
| `execution/factory.py` | 333 | 全注 | 启动时组装 ModelRunner、后端、KV 池、采样器 |
| `engine/io_struct.py`、`engine/zmq_wire.py` | 614 / 406 | 头注 | 请求/输出的数据结构与 msgpack 线格式 |
| `entrypoints/engine.py` | 466 | 主线注 | 启动流程：拉起每个 rank 的 `ts-host` 进程 |
| `cli/serve_smg.py` | 1120 | 头注 | `ts serve` 命令做了什么 |

配套阅读：`docs/design/event-loop.md`「The process」「Principle 1」「Anatomy of a round」。

读完能回答：从 `ts serve` 到第一个 token 返回，执行面经过了哪些函数？为什么调度循环本身不碰 GPU？

### 第 7 阶段：性能——CUDA Graph 与统一 decode 路径

| 文件 | 行数 | 深度 | 重点 |
| --- | --- | --- | --- |
| `execution/breakable_cuda_graph.py` | 643 | 全注 | 可断开的 CUDA Graph：图中遇到 `@break_point` 就切回 eager，执行后再继续回放；为什么核心注意力要放在断点里 |
| `execution/forward_step.py` | 1337 | 主线注（`ForwardStepRunner.capture/_capture_one/__call__`、`get_batch_sizes_to_capture`） | 按 batch size 档位捕获；bs=15 时 padding 到 16；padding 行怎么处理（读 seq_len 1、写到 slot 0，见 `write_locations.py`），为什么不会污染真实请求的 KV |
| `execution/cudagraph_memory.py`、`graph_ptr_guard.py` | 357 / 187 | 头注 | 图内存池；输入指针不能变 |
| `execution/prefill_graph.py` | 1778 | 头注 | prefill 也能用图（选读） |
| `paged/mha.py` 中的 `refresh_decode_metadata` / `publish_decode_metadata` | — | 补注 | “原地刷新元数据”契约：eager 和图走同一套元数据 |

配套阅读：`docs/design/unified_path.md`（只读开头的契约部分）。

### 第 8 阶段：权重加载、模型配置与量化

| 文件 | 行数 | 深度 | 重点 |
| --- | --- | --- | --- |
| `configs/model_config.py` | 1282 | 主线注 | HF config → 内部配置；head_dim、KV 头数、dtype 的推导 |
| `model_loader/loader.py`、`weight_utils.py` | 870 / 1207 | 主线注 | safetensors 迭代读取 → `load_weights` → 各参数的 `weight_loader` |
| `layers/quantization/base_config.py` | — | 全注 | `QuantizationConfig` / `QuantizeMethodBase` 接口：线性层怎么把计算委托给量化方法 |
| `layers/quantization/fp8.py`、`layers/dense/fp8.py` | — | 主线注 | 以 FP8 为例讲 per-tensor / per-block scale |
| 其他量化方案（mxfp4、nvfp4、w8a8……） | — | 头注 | 一句话说明格式与适用硬件 |

### 第 9 阶段：并行与 MoE

| 文件 | 深度 | 重点 |
| --- | --- | --- |
| `distributed/mapping.py`（回到全注） | 主线注 | attention TP、attention DP、dense TP、MoE EP 各自的分组 |
| `distributed/process_group_manager.py`、`comm_ops.py`、`comm_backend/` | 主线注 / 头注 | all-reduce、all-gather、reduce-scatter 在模型里的调用点；各通信后端的定位 |
| `models/qwen3_moe.py` | 全注 | 在 Qwen3 上把 MLP 换成 MoE |
| `layers/moe/topk.py`、`block.py`、`expert.py` | 主线注 | 路由 top-k → 分发 → 专家 GEMM → 合并；logical/physical expert 映射 |
| `moe/`（EPLB） | 头注 | 专家负载均衡的动机 |
| `execution/model_executor.py` 的 `execute_idle_forward` | 补注 | attention DP 下没有请求的 rank 为什么也要跑一次 idle 前向 |

### 第 10 阶段：进阶专题（选读，按兴趣挑）

| 专题 | 入口文件 | 配套设计文档 |
| --- | --- | --- |
| 投机解码 | `execution/drafter/base.py` → `mtp.py` → `eagle.py` → `tree.py`；`sampling/tree_verify.py` | `tree-speculation.md` |
| MLA（DeepSeek 系列） | `models/deepseek_v3.py`、`layers/attention/backends/paged/mla.py`、`PagedAttention.latent_prologue` | `attention-prologue.md`「MLA shapes」 |
| 稀疏注意力 / 线性注意力 | `backends/paged/dsa.py`、`backends/state/`、`backends/hybrid/` | `cache-concepts.md`「Recurrent state」 |
| 多级 KV cache | `cache/l2/executor.py`、`cache/l3/`、`cache/transfer/` | `async-l3-cache.md` |
| PD / EPD 分离 | `pd/`、`epd/` | `docs/serving/disaggregation.md` |
| 抢占与恢复（retraction） | `execution/slot_state.py`、`runtime_states.py` | `scheduler.md` |
| 结构化输出 | `grammar/` | — |
| 多模态 | `multimodal/`、`models/qwen3_vision.py` | — |
| 指标与 tracing | `metrics/`、`engine/profiler.py` | `tracing.md` |

专题阶段一般只写**文件头 + 类头 + 一张流程图**，不逐行注释。

---

## 4. 工作量与节奏

| 阶段 | 主要文件数 | 预计注释行数 | 建议 commit |
| --- | --- | --- | --- |
| 0 | 1（术语表） | 200 | `docs: annotation glossary` |
| 1 | 6 | 400 | 一个 commit |
| 2 | 8 | 700 | 一个 commit |
| 3 | 11 | 1200 | 按“上层接口 / 后端 / KV 池”拆 2–3 个 commit |
| 4 | 9 | 900 | 一个 commit；`model_executor.py` 可以单独一个 |
| 5 | 8 | 500 | 一个 commit |
| 6 | 6 | 600 | 一个 commit |
| 7 | 5 | 500 | 一个 commit |
| 8 | 5+ | 400 | 一个 commit |
| 9 | 6+ | 500 | 一个 commit |
| 10 | 按专题 | 每个专题 100–200 | 每个专题一个 commit |

每个阶段完成后：

1. 运行 `pre-commit run --all-files`，确认只有注释变化（`git diff --stat` 中不应出现逻辑改动）。
2. 运行一次相关单测（例如第 3 阶段跑 `test/runtime/layers/` 中注意力相关的测试），确保注释没有破坏缩进或字符串。
3. 在本文件末尾的进度表中打勾。

## 5. 进度

- [ ] 第 0 阶段：术语表
- [ ] 第 1 阶段：模型骨架
- [ ] 第 2 阶段：基础层零件
- [ ] 第 3 阶段：注意力与 KV cache
- [ ] 第 4 阶段：一次前向的组织
- [ ] 第 5 阶段：logits 与采样
- [ ] 第 6 阶段：执行面入口
- [ ] 第 7 阶段：CUDA Graph
- [ ] 第 8 阶段：权重加载与量化
- [ ] 第 9 阶段：并行与 MoE
- [ ] 第 10 阶段：进阶专题
