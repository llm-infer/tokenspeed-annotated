# 执行面（Python）源码注释计划

## 0. 目标与读者

- **注释范围**：执行面 `python/tokenspeed/` 下的**全部 487 个 Python 文件**（约 21 万行），一个不漏。完整清单和每个文件的阶段、等级见[附录 A](#附录-a全量文件清单)。
- **详略原则**：**越核心、越复杂的地方注释越详细**。核心程度看它离“请求 → 前向 → 采样”主线有多近；复杂程度看它的状态、并发、形状变换和隐含约定有多少。两者共同决定每个文件的注释等级（见 1.4）。
- **读者**：懂 Transformer、MHA、GQA，**已读完 nano-vllm-annotated** 的初级推理工程师。详细画像见 0.1，每一处注释的详略都以它为准（见 1.5）。
- **目的**：读者按本计划的顺序读完带注释的源码后，能说清楚一个请求在执行面里**从一轮调度计划到采样出下一个 token** 的完整路径、每一步为什么这样设计，并能在此基础上读懂任意一个特性或模型的实现。
- **边界**：只加注释，**不改任何代码逻辑**。控制面（Rust `tokenspeed-engine/`、C++ `tokenspeed-scheduler/`）和 kernel 包（`tokenspeed-kernel/` 等）不在本计划范围内，只在执行面与它们的接口处说明双方交换了什么。

### 0.1 读者画像

**背景**：初级推理工程师，熟悉 Transformer、MHA、GQA，已经逐文件读完 nano-vllm-annotated（一个约 1200 行、单机、离线推理的 Qwen3 引擎的中文注释版）。会读 PyTorch 代码和简单的 Triton kernel，习惯“形状记号 + 数据流图 + 具体数字走查”的注释方式。没有读过 vLLM、SGLang、TensorRT-LLM 等生产引擎的源码，也没有接触过 Rust/C++ 推理系统。

**已经掌握的（注释里一句话点明即可，不重新推导）**：

| 方面 | 具体内容 |
| --- | --- |
| 模型结构 | Qwen3 稠密模型：pre-norm、QK-Norm、GQA、SwiGLU MLP、tie embedding |
| 基础层 | RMSNorm 及“add residual + norm”融合；SiluAndMul；RoPE 原理、`cos_sin_cache` 查表、前后两半配对旋转；Embedding 查表与 lm_head 点积 |
| 张量并行 | 列并行（切输出维）、行并行（切输入维 + all_reduce）、两者配对不需中间通信；QKV 合并与按头切分；gate/up 合并；词表并行 Embedding（越界 mask + all_reduce）；lm_head 只取每条序列最后一个 token、gather 到 rank 0；`packed_modules_mapping` + `weight_loader` 式的权重加载 |
| 调度 | waiting / running 两个队列；prefill 优先、**同一步不混合** prefill 与 decode；chunked prefill；`max_num_seqs` / `max_num_batched_tokens` 预算；KV 不够时抢占最晚加入的请求、释放其 block、**之后重新 prefill** |
| 分页 KV cache | block、block_table（用 -1 补齐成矩阵）、空闲块队列、ref_count、按 block 内容哈希的前缀缓存；`slot = block_id * block_size + offset` 的 `slot_mapping`；KV cache 是**一个**预分配的大张量 `[2, L, NB, BS, Hkv, D]`；Triton `store_kvcache` 按 slot 写入 |
| 注意力计算 | 变长拼接、不做 padding、`cu_seqlens_q/k`、`max_seqlen`；prefill 用 `flash_attn_varlen_func`，decode 用 `flash_attn_with_kvcache` + `block_table` + `context_lens`；positions 的含义 |
| 前向组织 | `prepare_prefill` / `prepare_decode` 在 CPU 上逐条拼字段再拷上 GPU；元数据通过全局 `Context` 传给注意力层 |
| 性能 | decode 按 batch size 档位捕获**整张** CUDA Graph，回放时把输入拷进固定缓冲区、不足的档位 padding（padding 槽位写 -1 跳过）；graph pool 共享；torch.compile 融合小算子；启动时 warmup 测峰值显存、剩余全部分给 KV |
| 采样 | 温度缩放 + 指数分布噪声（Gumbel-max）一次 argmax 完成随机采样 |
| 多卡 | 每张卡一个进程，rank 0 通过共享内存 + Event 广播指令，NCCL 做 all_reduce/gather |
| 工具基础 | 张量 stride 与视图、pinned memory + `non_blocking` 拷贝、Triton 的 `program_id` / `tl.load` / `tl.store` / `constexpr` |

**还不了解的（必须按等级要求详细解释：动机 → 机制 → 例子）**：

- **控制面与执行面分离**：调度在 C++、请求和网络在 Rust；每个 rank 一个 `ts-host` 进程，主线程跑调度循环，forward 线程通过嵌入的 CPython 执行 Python；执行面拿到的是一份已经算好的 plan，而不是自己调度。
- **混合批**（一次前向里同时有 prefill 行和 decode 行）及其输出布局。
- **overlap 调度 / in-flight depth**：CPU 准备下一轮与 GPU 执行当前轮重叠；下一轮的输入 token 留在 GPU 上，不回 CPU。
- **request pool slot**（按请求槽位存放的长期 GPU 状态），以及 **retraction**：抢占时把页和 slot 状态快照到 Host、之后原样恢复，而不是重新 prefill。
- **cache group / arena / recipe**：KV cache 不是一个大张量，而是一块 arena 上的多个 cache group（全历史、滑动窗口、递归状态），调度器的 block 与 kernel 的 page 是不同粒度，由 router 换算。
- **attention prologue**（QK-Norm + RoPE + 量化 + 写 KV 统一入口）、**可断开的 CUDA Graph**（核心注意力在图里断开改为 eager）、prefill 也能捕获图、eager 与图共用一套“原地刷新”的元数据。
- **kernel 注册与选择**：执行面不直接 import 具体 kernel，而是通过 `tokenspeed-kernel` 的注册表按硬件和特性选择。
- 更完整的采样（top-p / top-k / min-p、惩罚项、logprobs、词表并行下的分布式 argmax）。
- **多 stream / event 同步**、CUDA host function 节点、forward 线程上的隐式同步为什么是 bug。
- attention DP、EP、PP、查询/解码上下文并行，通信后端的选择，all-reduce 与 norm 的融合。
- MoE、MLA、稀疏注意力、线性注意力与递归状态、投机解码、L2/L3 多级缓存、PD/EPD 分离、多模态、结构化输出、各种量化格式——**全部是新概念**。
- 本仓库的工程约定：行为参数必须显式传入、RAII 式的资源归属、`SlotStateExporter` 等。

**容易带进来的既有认知（注释里要主动纠正）**：读者会默认“prefill 和 decode 不在同一步”“抢占后重新 prefill”“block 就是 page”“KV cache 是一个大张量”“CUDA Graph 捕获整个前向”“调度在 Python 里”“元数据放在全局变量里”“padding 槽位写 -1”。TokenSpeed 在这些点上的做法都不同，遇到时要用 `# 注意：` 开头写清楚 TokenSpeed 实际怎么做、为什么。

---

## 1. 注释格式规范（参考 nano-vllm-annotated 的风格）

nano-vllm-annotated 的风格是：**中文注释直接写在源码里**，在文件头、类头、关键函数前放大段说明，并配 ASCII 图和带具体数字的例子；行内注释解释“是什么”和“为什么”。本项目沿用这一风格。

### 1.1 四个位置

| 位置 | 放在哪里 | 写什么 |
| --- | --- | --- |
| **文件头** | `import` 之后、第一个类/函数之前 | ① `Shape记号`（本文件出现的维度字母）；② `基础概念`（读本文件前需要知道、但读者可能不会的知识）；③ 数据流图；④ 在整条主线中的位置（“上游是谁调用我，下游我调用谁”） |
| **类头** | `class` 定义上方 | 职责一句话；生命周期/状态图或决策流程图（用 `┌─┐│└┘▼` 画框）；关键字段一览 |
| **函数/方法** | `def` 上方或函数体开头 | 输入输出的形状与含义；分步骤说明；**带具体数字的走查例子** |
| **行内** | 代码行上方或行尾 | 字段含义、为什么这样做、边界情况（如 padding 行、空 batch） |

### 1.2 统一约定

- 注释一律用 `#`，**不改原有英文 docstring**（避免影响 `help()`、文档生成，也便于和上游 diff）。中文注释写在 docstring 下方。
- **英文注释逐条翻译**：源码中原有的英文注释，都在它下面加上中文翻译，以 `# 译：` 开头，和讲解性的注释区分开。
  - 范围：`#` 注释（包括写在代码行尾的）和 docstring（模块、类、函数的）都要翻译。
  - 位置：`#` 注释的翻译紧跟在原注释块下面；行尾注释的翻译放在该行代码的下一行；docstring 的翻译放在 docstring 结束之后（模块和类的 docstring 之后 black 会强制空一行，译文放在这个空行之后）。
  - 要求：忠实翻译原意，不增不减；原文中的标识符、参数名、命令行选项保持英文。需要补充的解释另起一段讲解性注释，不混进译文。docstring 中的 `Args:` / `Returns:` / `Raises:` 等小节照样逐项翻译。
  - 不翻译：文件开头的许可证声明、工具指令（如 `# noqa`、`# type: ignore`、`# fmt: off`）、被注释掉的代码。
  - 与等级无关：B、C 级文件中的英文注释同样要全部翻译。
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

  特性专用的记号（如 MoE 的专家数 `X`、top-k 的 `K`，投机解码的草稿长度 `S`）在对应阶段第一个文件里补充定义。
- **贯穿全文的例子**（所有走查都用同一组数字，读者不用反复换算）：
  - 模型：Qwen3-8B，`H=4096`、`L=36`、Q 头 32、KV 头 8（GQA，每 4 个 Q 头共享 1 个 KV 头）、`D=128`、`intermediate=12288`、`V=151936`。
  - 单卡（TP=1）；`P=64`（示例取值，实际由 cache 计划决定）。
  - 一轮混合批：请求 A 新到达，prompt 300 token；请求 B 已在 decode，当前长度 1000。于是 `B=2`、`E=1`、`T=301`。
  - 特性阶段在此基础上扩展：并行阶段用 TP=2；MoE 阶段换成 Qwen3-30B-A3B（128 个专家、top-8）；投机解码阶段给 B 加 3 个草稿 token。
- **引用设计文档**：设计意图已写在 `docs/design/*.md` 的，注释写清结论，再加 `# 详见 docs/design/xxx.md「章节名」`。S/A 级文件需要把与该文件相关的设计要点用中文讲透，不能只给链接；B/C 级文件可以只写一句结论加链接。
- **术语**：首次出现时给出“中文（英文原词）”，之后统一用中文。术语表放在 `docs/annotation-glossary.md`（第 0 阶段产出），所有注释保持一致。
- **跨阶段内容**：一个文件属于某一阶段，但其中可能夹着后续阶段的特性分支（例如 `model_executor.py` 里的投机解码分支）。首次注释时，在这些分支入口写一行 `# 投机解码分支，第 11 阶段详解`，并按 B 级写清它做什么；到对应阶段再**回访**，补到该阶段要求的深度。每个阶段的“回访”列表列出了这些文件。
- 每个阶段一个或多个 commit；提交前运行 `pre-commit run --all-files`，确保 black/isort 不改动代码。

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

### 1.4 注释等级：越核心越复杂，越详细

每个文件按“核心程度 × 复杂程度”定一个等级：

| | 复杂（状态多、并发、形状变换多、隐含约定多） | 一般 | 简单（转发、枚举、配置） |
| --- | --- | --- | --- |
| **主线核心**（单卡 Qwen3 一次推理必经） | **S** | **A** | **B** |
| **重要特性**（大模型生产部署常开：MoE、投机解码、MLA、多级缓存、PD 分离……） | **A**（少数关键文件 S） | **B** | **C** |
| **外围**（特定模型/硬件、工具、可观测性） | **B** | **B** | **C** |

各等级的具体要求：

| 等级 | 名称 | 文件头 | 类 | 函数 | 行内 | 走查例子 | 注释量参考（注释行 : 代码行） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **S** | 逐行精注 | 记号 + 基础概念 + 数据流图 + 在主线中的位置，必要时附时序图 | 每个类都有职责、状态/流程图、字段表 | **每个**函数：输入输出形状、分步骤、为什么 | 几乎每个非平凡语句（索引计算、切片、stream/event、缓冲区复用、padding 处理）都有注释 | 每个主要函数至少一个，用贯穿例子的具体数字 | ≈ 1 : 1 |
| **A** | 详注 | 基础概念 + 数据流图 + 位置 | 每个类都有职责和关键字段 | 每个函数：输入输出、步骤要点 | 关键分支、非直观的计算、边界情况 | 核心函数至少一个 | ≈ 1 : 2 |
| **B** | 标准注 | 一段话：这个文件解决什么问题、何时被用到、和哪些文件配合 | 每个类一句到一段 | 每个函数 1–3 行：做什么、谁调用 | 只在复杂分支处 | 可选 | ≈ 1 : 5 |
| **C** | 简注 | 一到两句：文件作用 | 每个类/函数一句（`__init__.py` 说明它导出了什么、为什么在这里导出） | — | — | — | 少量 |

补充规则：

- 等级针对文件的**主体**。S/A 文件中属于后续阶段特性的分支，首次注释至少达到 B，回访时补到该特性阶段的要求。
- 超长文件（如 `model_executor.py` 3138 行、`kimi_k3.py` 5623 行、`server_args.py` 5138 行）先写一张**文件导航**放在文件头：按代码顺序列出各段落/类的行号范围、作用和对应阶段，再逐段注释。
- 同一类实现有多个变体时（例如多个采样后端、多个注意力后端、多个量化格式），**挑一个代表做高等级**，其余在文件头写“与代表实现的差异”，避免重复讲同样的事。

### 1.5 准则：每次加注释都按读者画像决定详略

**每写一段注释前，都要对照 0.1 的读者画像，判断读者能否读懂，哪些地方简单带过，哪些地方必须详细解释。** 等级（1.4）决定一个文件的注释总量，画像决定这些注释花在哪里：已掌握的内容少写，把篇幅留给新内容。

写之前逐项自检：

1. **这里用到的概念，读者已经掌握吗？**（画像“已经掌握”表）
   - 是：一句话点明，用读者熟悉的说法，不重新推导。例如 `RowParallelLinear` 只写“行并行，输出是部分和，需要 all_reduce”，不再画切分示意图。
   - 否：按等级要求完整解释——先讲**为什么需要**，再讲**怎么做**，最后用贯穿例子的数字走一遍。
2. **这里和读者的既有认知冲突吗？**（画像“容易带进来的既有认知”）冲突时用 `# 注意：` 开头，写清 TokenSpeed 实际怎么做、为什么这样做。
3. **同一个文件里，已知部分和新内容混在一起时**，篇幅向新内容倾斜。例如 `layers/linear.py` 是 S 级，但列并行/行并行本身一句带过，详细篇幅给量化方法委托、各种 `weight_loader` 分片情形、DP 线性层。
4. **依赖后续阶段的内容吗？**依赖时写一句“它做什么”，再指向对应阶段，不在这里展开。
5. **只靠“本文件 + 之前的阶段 + 读者画像”能读懂吗？**写完后以画像中的读者身份通读一遍；出现没解释过的术语、没交代来源的张量、没说明原因的特殊处理，就补上。

一般可以简单带过的：Python 语法和标准库、PyTorch 基础 API、画像“已经掌握”表中的概念、日志、断言、参数校验、纯转发函数。

一般需要详细解释的：本仓库特有的设计和其中的隐含约定、跨线程/跨 stream 的时序和同步、索引与形状变换、缓冲区复用、padding 和边界情况、为了性能做的取舍（为什么不用更直观的写法）、画像中“还不了解”的全部内容。

画像只用来判断详略。注释正文**不提 nano-vllm、不做对比**，直接讲 TokenSpeed 本身。

英文注释的翻译（`# 译：`，见 1.2）不受详略判断影响：不管内容对读者是否已知，原有英文注释都要完整翻译；详略只针对新增的讲解性注释。

### 1.6 规模估计

| 等级 | 文件数 | 代码行数 | 预计注释行数 |
| --- | --- | --- | --- |
| S | 34 | 约 2.9 万 | 约 2.9 万 |
| A | 117 | 约 7.8 万 | 约 3.9 万 |
| B | 285 | 约 10.1 万 | 约 2.0 万 |
| C | 51 | 约 0.3 万 | 约 0.1 万 |
| 合计 | 487 | 约 21.1 万 | 约 8.9 万 |

各阶段的文件数与等级分布见第 4 节。

---

## 2. 主线选择：先走最简单的一条路

TokenSpeed 支持很多模型和特性，但它们都是**同一条执行路径上的参数**（见 `AGENTS.md`「Design principles」）。所以先固定一条最简单的配置，把它走通（第 1–7 阶段），再逐个打开开关（第 8–19 阶段）：

| 维度 | 主线选择 | 之后在哪个阶段打开 |
| --- | --- | --- |
| 模型 | `models/qwen3.py`（稠密、GQA、QK-Norm） | MoE 第 10 阶段；MLA 第 12 阶段；线性注意力第 13 阶段；其他模型第 18 阶段 |
| 并行 | 单卡（TP=1，无 DP/PP/EP） | 第 9 阶段 |
| 注意力后端 | `CacheGroupRouter` + `MHAAttnBackend` | 第 12–14 阶段 |
| 采样 | `greedy.py`，再到 `flashinfer.py` | 其他后端在第 5 阶段同步覆盖 |
| 图 | 先 eager | 第 7 阶段 |
| 投机解码 / PD 分离 / 多级 cache / 多模态 / 结构化输出 | 关闭 | 第 11、15、14、16、17 阶段 |

### 2.1 主线全景图（写进 `engine/rust_host.py` 文件头，并在每个阶段的文件头里标出“你在这里”）

```text
Rust 控制面（ts-host 主线程）：调度器产出本轮计划 plan（哪些请求、各算多少 token、KV 页表）
    │  通过嵌入的 CPython 调用（forward 线程）
    ▼
HostDataPlane.execute(plan)                         engine/rust_host.py         第 6 阶段
    ▼
DeviceHandle.execute(plan, planned, cache_ops)      execution/device.py         第 6 阶段
    │  先执行 cache 操作（页拷贝/清零/换入换出），再提交前向
    ▼
ModelExecutor.execute_forward_op(forward_op, ...)   execution/model_executor.py 第 4 阶段
    │  ① CacheBatchMetadata：把本批的 block table 打包上 GPU
    │  ② InputBuffers.fill_input_buffers：input_ids / positions / seq_lens 写入固定缓冲区
    │  ③ attn_backend（CacheGroupRouter）.init_forward_metadata：block → page，准备 kernel 元数据
    ▼
ModelExecutor._forward_step                         execution/model_executor.py 第 4 阶段
    │  ├─ eager：ModelRunner.forward
    │  └─ 图：ForwardStepRunner（CUDA Graph 回放）  execution/forward_step.py    第 7 阶段
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

每个阶段列出：重点文件及要讲清的内容（S/A 级）、回访的文件、读完能回答的问题。**每个阶段的全部文件（含 B/C 级）见附录 A 对应小节**，B/C 级文件按 1.4 的标准注释，不在正文逐一展开。

### 第 0 阶段：准备（不改源码）

| 产出 | 内容 |
| --- | --- |
| `docs/annotation-glossary.md` | 术语表：extend/prefill/decode/mixed、page/block、page_table/block_table、slot、request pool index、cache group、arena、recipe、prologue、eager break、in-flight depth、overlap、TP/DP/EP/PP、retraction、drafter/verify……每条给出中文、英文、一句话解释、首次出现的文件 |
| 本计划第 2.1 节的全景图 | 定稿后作为后续注释的锚点 |

依据：`docs/design/cache-concepts.md`「block vs. page」「page_table vs. block_table」、`docs/design/event-loop.md`「The process」「Anatomy of a round」。

### 第 1 阶段：模型骨架（9 个文件）

先读模型代码：它最接近读者已有的 Transformer 知识，而且能先建立“一次前向输入什么、输出什么”的直觉。

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `models/qwen3.py` | S | 定义全项目的 Shape 记号。模型结构、QKV 合并读者已熟悉，简述；重点放在：`forward` 显式接收 `ctx`（不用全局变量）；`PagedAttention` 接收 `rotary_emb` 和 `qk_norm`，QK-Norm 与 RoPE 不在模型里做，而是交给注意力的 prologue；滑动窗口层的处理；`mapping.attn` 与 `mapping.dense` 两套并行参数 |
| `models/base/transformer_model.py` | S | embed → 层循环 → final norm、residual 与 norm 融合都已熟悉，简述；重点放在模型基类如何把通用流程固定下来、子类只覆盖零件，以及层输入捕获（给投机解码用，第 11 阶段回访） |
| `models/base/decoder_layer.py` | S | pre-norm 结构；`resolve_attn/resolve_mlp` 模板方法：子类只换零件；norm 和通信都经 `comm_manager` 完成 |
| `models/base/causal_lm.py` | A | `forward`、`exit_logits`、`load_weights` 流程；权重热更新部分 B 级，第 8 阶段回访 |
| `models/llama.py`、`models/registry.py`、`configs/qwen3_config.py` | B | 和 Qwen3 的差异；架构名 → 模型类（`EntryClass`）的注册；HF config 字段 |

读完能回答：Qwen3 一层的数据流和各张量形状？`positions` 从哪里来？为什么模型的 `forward` 里看不到 KV cache？

### 第 2 阶段：基础层零件（12 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `layers/linear.py` | S | 列并行/行并行、QKV 与 gate/up 合并读者已熟悉，一句带过；详细篇幅给：计算如何委托给量化方法（`quant_method`），各种 `weight_loader` 分片情形（KV 头数小于 TP 时的复制、带 scale 的量化参数、fused 参数的偏移），`reduce_results=False` 时 all-reduce 去了哪里；DP 线性层先 B 级，第 9 阶段回访 |
| `layers/rotary_embedding.py` | S | RoPE 原理和 cos/sin 缓存表读者已熟悉，简述；重点放在：neox 与 gptj 两种配对方式的区别，`as_rotary` 怎样把 RoPE 交给注意力 prologue，YaRN / Llama3 / LongRoPE / DeepSeek / 动态 NTK 各变体为什么需要、缩放公式是什么（用具体数字算一遍）；MRoPE 第 16 阶段回访 |
| `distributed/comm_manager.py` | S | 新概念：层内通信不写在线性层里，而是集中在这里。单卡时它只是 “add residual + RMSNorm”；多卡时在同一位置插入 all-reduce / reduce-scatter / all-gather 并与 norm 融合；各 `*_comm` 方法分别在层的哪个位置被调用 |
| `layers/layernorm.py` | A | RMSNorm 和 fused add+norm 已熟悉，简述；重点放在 allreduce+norm 融合、`FusedRMSNorm`、Gemma 式 norm（权重加 1）、sandwich norm |
| `layers/mlp.py`、`layers/activation.py` | A | GatedMLP 结构已熟悉，简述；重点放在 `fusions` 可选融合、带截断（`swiglu_limit`）的 SwiGLU 变体、`SwigluOAI` |
| `layers/vocab_parallel_embedding.py` | A | 词表切分与越界 mask 已熟悉，简述；重点放在词表 padding 到对齐大小、分片索引 `VocabParallelEmbeddingShardIndices`、embedding 的量化方法 |
| `layers/parameter.py` | A | 带 `weight_loader` 的参数类，与 `linear.py` 的权重加载配合 |
| `distributed/mapping.py` | A | attention / dense / MoE / lm_head 各自的并行分组，为什么可以不同（第 9 阶段回访补充 DP、EP、QCP） |

读完能回答：TP=2 时 `QKVParallelLinear` 每张卡算出哪些头？`RowParallelLinear` 的输出为什么要 all-reduce，这个 all-reduce 实际在哪里执行？RoPE 是在哪一步作用到 q/k 上的？

### 第 3 阶段：注意力与 KV cache 主线（25 个文件）

这是执行面和普通 Transformer 实现差别最大的地方。按“上层接口 → 写 cache → block 到 page 的映射 → 读 cache 算注意力 → cache 存在哪”的顺序读。

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `layers/paged_attention.py` | S | 一层注意力分两步：prologue（QK-Norm、RoPE、量化、写 KV）+ attend（核心注意力）；`@break_point` 是什么（先讲“这里在 CUDA Graph 中会断开改为 eager 执行”，细节在第 7 阶段）；MLA 的 `latent_prologue` 先 B 级，第 12 阶段回访 |
| `layers/attention/backends/base.py` | S | 后端统一接口：`init_forward_metadata` → `forward` 分派到 `forward_extend` / `forward_decode`；`CachePoolBinding` |
| `layers/attention/backends/paged/router.py` | S | `CacheGroupRouter`：所有分页 KV 模型实际使用的后端，“调度器 block 变成 kernel page 的唯一位置”；每个 cache group 一个叶子后端；以贯穿例子画出 A、B 的 block table → page table |
| `layers/attention/backends/paged/base.py` | S | 分页叶子后端的公共逻辑；树投机部分第 11 阶段回访 |
| `layers/attention/backends/paged/mha.py` | S | `MHAExtendMetadata` / `MHADecodeMetadata` 各字段；用贯穿例子走查混合批：A 的 300 个 token 做 extend、B 的 1 个 token 通过页表读出 1000 个历史 KV，**两者在同一次前向里**；kernel 通过 `tokenspeed_kernel` 的 attention API 调用（内部按注册表选择实现）；decode 元数据“原地刷新”（第 7 阶段回访） |
| `layers/attention/backends/paged/write_locations.py` | S | `slot = page_id * P + offset` 读者已熟悉，一句带过；重点放在：写位置由 GPU kernel 直接从 `seq_lens` 和页表算出（不在 CPU 上拼）、多个 cache group 各有自己的 kernel 页大小、每请求写多个 token（投机验证）时的布局；`# 注意：` padding 行读 seq_len 1、写到 slot 0，而不是写 -1 跳过 |
| `layers/attention/backends/cache_metadata.py` | S | `CacheBatchMetadata`：调度器下发的每组 block table 如何校验、打包、上 GPU |
| `layers/attention/kv_cache/base.py`、`kv_cache/mha.py` | S | `CachePool` 抽象；`MHATokenToKVPool` 的显存布局图（每一维的含义，用贯穿例子画出 A、B 的 KV 落在哪些页） |
| `layers/attention/backends/paged/group_tables.py` | A | 多个 cache group 的 block → page 映射表如何堆叠 |
| `layers/attention/kv_cache/arena.py`、`factory.py` | A | 显存怎么一次性分配、多个 cache group 怎么共享一块 arena（容量规划在第 14 阶段） |
| `layers/attention/registry.py` | A | 启动时如何按模型和硬件选出后端和 KV 池（`create_attn_components`） |
| `layers/attention/configs/base.py`、`configs/mha.py` | A | 注意力配置对象 |
| `layers/attention/backends/paged/trtllm.py` | A | 另一种 MHA 后端（TRT-LLM kernel），写清与 `mha.py` 的差异 |
| `layers/attention/page_table.py`、`chunk.py` | A | page table 结构；按块处理长 prefill |
| `engine/scheduler_utils.py` | A | 执行面从 C++ 调度器读到的东西：cache 几何、cache group、plan 中的 cache 操作、block table |

配套阅读：`docs/design/attention-prologue.md`「One entry per attention shape」「Who writes the KV cache」；`docs/design/cache-concepts.md`「Two worlds」「block vs. page」「page_table vs. block_table」。

读完能回答：A 的第 299 个 token 的 K 写到了哪个物理位置？decode 时 B 的 query 怎么找到它的 1000 个历史 KV？GQA 的“4 个 Q 头共享 1 个 KV 头”在 cache 布局和 kernel 调用里体现在哪里？

### 第 4 阶段：一次前向的组织（15 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `execution/model_executor.py` | S | 先写文件导航。主线逐段注：`execute_forward_op`、`_forward_step`、`_run_target_forward`、`_run_sampling`、`_update_runtime_state`；**overlap 调度**（本轮 CPU 准备与上一轮 GPU 计算重叠，`settle()` 等待上一轮结果）单独画时序图；drafter / PP / DP / grammar / 多模态分支先 B 级，在第 9、11、16、17 阶段回访 |
| `execution/input_buffer.py` | S | 固定缓冲区的动机读者已熟悉，简述；重点放在：输入分两步填写（`fill_input_buffers` 先填本批自身确定的部分，`load_round_inputs` 等上一轮结果出来后再填 decode 的输入 token）、pinned 批量拷贝、贯穿例子中 `input_ids_buf` 前 301 个位置分别放什么、哪些值来自上一轮的 GPU 输出 |
| `execution/runtime_states.py` | S | 按 request pool slot 存放的长期状态：`future_input_map`（下一轮的输入 token）、停止条件；投机解码相关状态第 11 阶段回访 |
| `execution/context.py` | S | `# 注意：` 元数据不放在全局变量里，`ForwardContext` 作为参数显式传入每一层；它为什么“不存张量”（张量放在固定缓冲区和后端里）；各字段的生命周期 |
| `execution/forward_batch_info.py` | S | `ForwardMode`：EXTEND / DECODE / MIXED / IDLE 的含义；positions 的计算 kernel |
| `execution/output_layout.py` | S | 混合批中 prefill 行在前、decode 行在后的布局 |
| `execution/model_runner.py` | A | 持有模型、调用 `model.forward`；权重更新部分第 8 阶段回访 |
| `execution/slot_state.py`、`request_slots.py` | A | slot 是什么；retraction（抢占）时为什么要把 slot 上的状态随页一起导出（`SlotStateExporter`） |
| `execution/types.py`、`cache_loc_kernel.py` | A | 执行结果类型；计算 cache 位置、更新页表的 Triton kernel |

读完能回答：一轮里 GPU 上依次发生了什么？为什么下一轮 decode 的输入 token 不需要先拷回 CPU？

### 第 5 阶段：logits 与采样（22 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `layers/logits_processor.py` | S | 只对每个请求最后一个位置算 logits（`gather_ids`）；词表并行下的 all-gather 与分布式 argmax；softcap；input logprobs（prompt 部分的 logprob） |
| `sampling/backends/base.py` | S | 采样后端接口；普通采样与投机验证两条入口（验证部分第 11 阶段回访） |
| `sampling/backends/greedy.py` | S | argmax 采样，作为采样后端的代表实现 |
| `sampling/backends/flashinfer.py`、`triton.py` | A | 带温度和 top-p/top-k/min-p 的批量采样；与 greedy 的差异 |
| `sampling/sampling_params.py`、`sampling_batch_info.py`、`logits_layout.py` | A | 温度已熟悉，一句带过；重点放在 top-p / top-k / min-p、各种惩罚项、停止条件的含义，参数如何打包成批张量，logits 的行布局 |
| `layers/logits_processor_output.py` | A | 输出结构 |
| 其余采样后端（sonic、pooled、`*_full`、ascend） | B | 各自与代表实现的差异和适用场景 |

读完能回答：A 的 301 个 hidden 里只有哪一行进入 lm_head？top-p 在 GPU 上是怎么批量实现的？

### 第 6 阶段：执行面入口与启动（23 个文件）

到这里读者已经懂“一次前向”，再往上看它是被谁、以什么节奏调用的，以及进程是怎么起来的。

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `engine/rust_host.py` | S | 文件头放第 2.1 节全景图；Rust 控制线程与 Python forward 线程的分工；`HostDataPlane.admit/execute/complete/release`；`_Dispatched` 和 in-flight 队列；`HostBootstrap` 启动流程 |
| `execution/device.py` | S | `DeviceHandle.execute`：一个 plan 里 cache 操作与前向的提交顺序为什么不能乱（结合 event-loop.md「Anatomy of a round」第 4 步讲透）；`build_components` 组装全部组件；权重更新、内存释放/恢复 |
| `execution/factory.py`、`distributed_initializer.py` | A | 启动时组装 ModelRunner、后端、KV 池、采样器；分布式环境初始化 |
| `engine/io_struct.py` | A | 请求/输出的数据结构（与 Rust `ts-wire` 字节兼容） |
| `entrypoints/engine.py` | A | 启动流程：拉起每个 rank 的 `ts-host` 进程 |
| `cli/*`、`engine/zmq_wire.py`、`rank_setup.py` 等 | B/C | `ts serve` 的参数拆分与子进程管理；msgpack 线格式；rank 初始化 |

配套阅读：`docs/design/event-loop.md`「The process」「Principle 1」「Anatomy of a round」。

读完能回答：从 `ts serve` 到第一个 token 返回，执行面经过了哪些函数？为什么调度循环本身不碰 GPU？

### 第 7 阶段：CUDA Graph 与流水线性能（11 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `execution/breakable_cuda_graph.py` | S | `# 注意：` 不是整张图：图中遇到 `@break_point` 就切回 eager，执行后再继续回放下一段；为什么核心注意力要放在断点里（形状随序列长度变化）；断点前后张量如何交接；padding 行的清理 |
| `execution/forward_step.py` | S | 按 batch size 档位捕获、补齐到最近档位读者已熟悉，简述；重点放在：捕获的是“可断开”的图（多段 + eager 断点）、同一档位按变体（采样方式等）分别捕获、捕获前如何准备占位元数据、回放时元数据如何原地刷新、padding 行为什么不会污染真实请求的 KV、warmup 流程 |
| `execution/prefill_graph.py` | A | prefill 也能用图：按 `(token_bucket, exact_bs)` 捕获 |
| `execution/cudagraph_memory.py` | A | 启动时用一次试捕获估算图内存池大小 |
| `graph_ptr_guard.py`、`forward_reload.py`、`utils/cuda_stream.py`、`device_stream.py`、`host_sync.py`、`jit_compile_check.py`、`cli/reload_forward.py` | B | 图输入指针不能变；前向代码热重载；多 stream 管理；forward 线程上的隐式同步检测；JIT 重编译检测 |

回访：`paged/mha.py` 的 `refresh_decode_metadata` / `publish_decode_metadata`（“原地刷新元数据”契约：eager 和图走同一套元数据）。配套阅读：`docs/design/unified_path.md`。

### 第 8 阶段：配置、权重加载与量化（46 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `configs/model_config.py` | A | HF config → 内部配置；head_dim、KV 头数、dtype、上下文长度的推导 |
| `model_loader/loader.py`、`weight_utils.py` | A | safetensors 迭代读取 → `load_weights` → 各参数的 `weight_loader`；分片与预取 |
| `layers/quantization/base_config.py` | A | `QuantizationConfig` / `QuantizeMethodBase` 接口：线性层怎么把计算委托给量化方法 |
| `layers/quantization/fp8.py`、`layers/dense/fp8.py`、`layers/dense/unquant.py` | A | 以 FP8 为代表讲 per-tensor / per-channel / per-block scale；未量化路径作为对照 |
| 其余量化格式（mxfp4、nvfp4、w8a8、modelopt、compressed-tensors……） | B | 每种格式的数据布局、scale 粒度、适用硬件，与 FP8 的差异 |
| `utils/server_args.py` | B | 先写文件导航：按功能分组列出全部参数；每组写清它影响执行面哪些组件 |
| 权重更新（`execution/weight_*`、`model_update.py`、`models/base/weight_update.py`） | B | 在线更新权重的流程 |

回访：`models/base/causal_lm.py` 和 `execution/model_runner.py` 的权重更新部分。

### 第 9 阶段：并行与通信（21 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `distributed/comm_ops.py` | A | all-reduce、all-gather、reduce-scatter 在模型里的调用点 |
| `distributed/process_group_manager.py` | A | 进程组的创建与查找 |
| `distributed/comm_backend/base.py`、`auto.py`、`nccl.py` | A | 通信后端接口；auto 后端按调用选择策略（节点内低延迟后端 vs NCCL） |
| `distributed/pp_stage.py` | A | 流水线并行：stage 之间传递的状态 |
| 其余通信后端、DP 采样通信、NCCL 挂起 | B | 各自定位 |

回访：`distributed/mapping.py`（补 DP、EP）、`comm_manager.py`（多卡路径逐个画通信图）、`layers/linear.py`（DP 线性层）、`model_executor.py`（`execute_idle_forward`：attention DP 下没有请求的 rank 为什么也要跑一次前向；PP 分支）。

### 第 10 阶段：MoE 与专家负载均衡（32 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `layers/moe/topk.py` | S | 路由：softmax/sigmoid 打分 → grouped top-k → 归一化；logical expert 到 physical expert 的映射；专家负载记录 |
| `layers/moe/expert.py` | S | `MoELayer`：分发 → 专家 GEMM → 合并的完整数据流，用 Qwen3-30B-A3B（128 专家、top-8）走查每一步的形状；EP 下 token 怎么跨卡 |
| `layers/moe/block.py`、`models/qwen3_moe.py` | A | 在 Qwen3 上把 MLP 换成 MoE |
| `layers/moe/loader.py`、`weights/loaders.py` | A | 专家权重加载与按 EP 切分 |
| `layers/moe/latent.py` | A | Kimi 系列的 latent MoE |
| `moe/expert_location.py` | A | 冗余专家与物理槽位：placement 是什么 |
| 其余（各量化格式的专家权重、EPLB 算法、重平衡） | B | 各自作用 |

### 第 11 阶段：投机解码（50 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `execution/drafter/base.py` | S | drafter 接口与一轮的时序：目标模型验证 → 接受 → 草稿模型生成下一轮候选 |
| `execution/drafter/eagle.py` | S | EAGLE：用目标模型的 hidden 作为草稿输入；草稿的 extend 图 |
| `execution/drafter/speculative_sampling.py` | S | 拒绝采样验证的数学原理与实现，为什么输出分布与目标模型一致 |
| `execution/drafter/mtp.py`、`tree.py`、`dflash.py`、`eagle_harness/drafter.py` | A | MTP；树状草稿；DFlash 块草稿；EAGLE-4 动态树 |
| `execution/tree_spec.py`、`sampling/tree_verify.py`、`backends/paged/tree_verify.py` | A | 树的位置与 KV 槽位；树验证的注意力 |
| `models/llama_eagle3.py`、`models/deepseek_nextn.py` | A | 草稿模型的代表实现 |
| 其余（各模型的 `*_nextn`、`*_dspark`、eagle_harness 其他文件……） | B | 与代表实现的差异 |

回访：`model_executor.py` 中 drafter 分支、`runtime_states.py` 的草稿状态、`sampling/backends/base.py` 的验证入口、`paged/base.py` 的树部分。配套阅读：`docs/design/tree-speculation.md`。

### 第 12 阶段：MLA、稀疏注意力与上下文并行（39 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `layers/attention/backends/paged/mla.py` | S | MLA 原理（KV 压缩到 latent、吸收矩阵）；prefill 与 decode 两种计算方式；cache 里存什么 |
| `models/deepseek_v3.py` | A | MLA + MoE 的代表模型 |
| `paged/trtllm_mla.py`、`tokenspeed_mla.py`、`flashmla.py` | A | 三种 MLA kernel 后端的差异 |
| `configs/mla.py`、`kv_cache/mla.py` | A | MLA 的配置与 cache 布局 |
| `paged/dsa.py`、`configs/dsa.py`、`kv_cache/dsa.py`、`kpool.py` | A | DeepSeek 稀疏注意力：indexer 选 top-k 历史 token |
| `execution/query_shard.py`、`dcp/cache.py`、`dcp/placement.py` | A | 查询上下文并行（QCP）与解码上下文并行（DCP） |
| 其余（MSA、QSA、Ascend 实现、DCP 辅助） | B | 各自作用 |

回访：`paged_attention.py` 的 `latent_prologue`。配套阅读：`docs/design/attention-prologue.md`「MLA shapes」、`docs/design/deepseek-v41-context-parallelism.md`。

### 第 13 阶段：线性注意力与混合模型（27 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `backends/state/checkpoint.py` | A | 递归状态为什么每个请求两个槽位（输入检查点与输出） |
| `backends/state/mamba.py`、`kda.py`、`gdn.py` | A | Mamba / KDA / GDN 的状态更新与 chunked prefill |
| `backends/hybrid/linear.py`、`kv_cache/hybrid_mha.py` | A | 全注意力层与线性注意力层交替的模型如何共用一个后端和 cache |
| `configs/linear_attn.py`、`layers/conv.py` | A | 线性注意力配置；短卷积 |
| `models/qwen3_5.py` | A | 混合模型的代表 |
| 其余 | B | — |

配套阅读：`docs/design/cache-concepts.md`「Recurrent state: two slots, natural checkpoints」、`docs/design/kda-prefill-subgraphs.md`。

### 第 14 阶段：KV cache 规划、卸载与多级缓存（38 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `kv_cache/recipes/plan.py`、`spec.py`、`base.py`、`ordinary.py` | A | 启动时 KV 容量如何规划：cache group spec、LCM 共享 arena 的整数几何；MHA/MLA/DSA/MSA 共用的 ordinary recipe |
| `kv_cache/recipes/scheduler_bridge.py` | A | recipe 与 C++ 调度器之间的契约 |
| `kv_cache/virtual_blocks.py` | A | 调度器虚拟 block ID → 本 rank 本地页 |
| `kv_cache/offload.py` | A | KV 卸载：Host 上的权威副本与 GPU 计算缓冲 |
| `cache/l2/executor.py`、`cache/transfer/lanes.py`、`layout.py` | A | L2 前缀缓存与抢占快照两个 Host 层；GPU↔Host 拷贝通道与字节布局 |
| 其余（各模型的 recipe、L3 配置与命名、embedding cache……） | B | — |

回访：`execution/device.py` 中 cache 操作的提交顺序、`slot_state.py`。配套阅读：`docs/design/cache-concepts.md`、`docs/design/async-l3-cache.md`、`docs/design/scheduler.md`。

### 第 15 阶段：PD / EPD 分离（10 个文件）

全部 A 级：`pd/cache_protocol.py`（P/D 两端约定的 cache 传输契约）、`pd/host_peer.py`（执行面在 PD 中的一半，传输本身在 Rust `ts-pd`）、`pd/factory.py`、`topology.py`、`kv_events.py`、`utils.py`；`epd/encode_host.py`（编码节点）、`prefill_receive.py`（prefill 节点接收 embedding）、`recv_pool.py`、`encode_worker.py`。

回访：`execution/device.py` 的远程 prefill 拉取与落地。配套阅读：`docs/serving/disaggregation.md`。

### 第 16 阶段：多模态（27 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `multimodal/embedder.py` | A | 把编码器输出拼接进文本 embedding 的三步 |
| `multimodal/mrope.py` | A | 多维 RoPE 的位置计算 |
| `execution/multimodal_runtime.py` | A | M-RoPE 位置覆盖、编码器 CUDA Graph |
| `layers/attention/mm_encoder_attention.py`、`models/qwen3_vision.py` | A | 视觉编码器注意力；代表视觉塔 |
| 其余视觉/音频模型与传输工具 | B | — |

回访：`rotary_embedding.py` 的 `MRotaryEmbedding`、`model_executor.py` 的多模态分支。

### 第 17 阶段：结构化输出（6 个文件）

`grammar/capturable_grammar.py`（A：用 host 函数节点把 grammar 填充放进 CUDA Graph，与前向重叠）、`grammar/grammar_host.py`（A：每次前向的 grammar 工作，不持有 GIL）；其余 B/C。回访 `model_executor.py` 的 grammar 分支。

### 第 18 阶段：模型族（51 个文件）

| 文件 | 等级 | 重点 |
| --- | --- | --- |
| `models/kimi_k3.py` | A | 先写文件导航；项目旗舰模型，讲清它组合了哪些前面阶段的部件 |
| `models/deepseek_v41.py`、`backends/specific/deepseek_v41.py` | A | DeepSeek V4.1 及其专用注意力后端 |
| `models/gpt_oss.py` | A | 注意力 sink、滑动窗口与 MoE 的组合 |
| `backends/specific/inkling.py` | A | Inkling 专用注意力后端 |
| `models/monokernel.py` | A | decode monokernel：每层一次持久化 kernel 启动（配套 `docs/design/monokernel.md`） |
| 其余模型、模型配置与专用层 | B | 每个模型文件头写：架构特点、与已讲过的代表模型的差异、用到的特性阶段 |

### 第 19 阶段：可观测性、插件与通用工具（23 个文件）

全部 B/C：指标采集与上报（`metrics/`）、插件注册（`plugins/`）、profiler、日志、trace 合并、通用工具（`utils/` 其余文件）。配套阅读：`docs/design/tracing.md`、`docs/metrics-plugins.md`。

---

## 4. 工作量与节奏

各阶段的文件数、行数与等级分布（由附录 A 统计）：

| 阶段 | 主题 | 文件数 | 行数 | S | A | B | C |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 模型骨架 | 9 | 2066 | 3 | 1 | 3 | 2 |
| 2 | 基础层零件 | 12 | 7473 | 3 | 6 | 1 | 2 |
| 3 | 注意力与 KV cache（主线） | 25 | 11823 | 9 | 10 | 4 | 2 |
| 4 | 一次前向的组织 | 15 | 7339 | 6 | 5 | 3 | 1 |
| 5 | logits 与采样 | 22 | 9523 | 3 | 6 | 12 | 1 |
| 6 | 执行面入口与启动 | 23 | 9070 | 2 | 4 | 11 | 6 |
| 7 | CUDA Graph 与流水线性能 | 11 | 5099 | 2 | 2 | 7 | 0 |
| 8 | 配置、权重加载与量化 | 46 | 17878 | 0 | 7 | 30 | 9 |
| 9 | 并行与通信 | 21 | 6392 | 0 | 6 | 13 | 2 |
| 10 | MoE 与专家负载均衡 | 32 | 8149 | 2 | 6 | 19 | 5 |
| 11 | 投机解码 | 50 | 19583 | 3 | 9 | 37 | 1 |
| 12 | MLA、稀疏注意力与上下文并行 | 39 | 19765 | 1 | 13 | 23 | 2 |
| 13 | 线性注意力与混合模型 | 27 | 11171 | 0 | 9 | 15 | 3 |
| 14 | KV cache 规划、卸载与多级缓存 | 38 | 12447 | 0 | 10 | 23 | 5 |
| 15 | PD / EPD 分离 | 10 | 2215 | 0 | 10 | 0 | 0 |
| 16 | 多模态 | 27 | 10212 | 0 | 5 | 18 | 4 |
| 17 | 结构化输出 | 6 | 1813 | 0 | 2 | 3 | 1 |
| 18 | 模型族 | 51 | 42575 | 0 | 6 | 44 | 1 |
| 19 | 可观测性、插件与通用工具 | 23 | 5947 | 0 | 0 | 19 | 4 |
| 合计 | | 487 | 210540 | 34 | 117 | 285 | 51 |

每个阶段完成后：

1. 运行 `pre-commit run --all-files`，确认只有注释变化（`git diff` 中不应出现逻辑改动）。
2. 运行一次相关单测（例如第 3 阶段跑 `test/runtime/layers/` 中注意力相关的测试），确保注释没有破坏缩进或字符串。
3. 以读者画像的身份通读本阶段新增的注释（1.5 第 5 条），补上读不懂的地方。
4. 在第 5 节的进度表中打勾，并在附录 A 中把已完成的文件标记出来。

建议节奏：第 1–7 阶段（主线，共 117 个文件、约 5.2 万行，其中 S 级 28 个）按顺序完成，不并行；它们定下的记号、术语和贯穿例子是后续阶段的基础。第 8 阶段之后的各阶段相互独立，可以按兴趣或需要调整顺序，但每个阶段开始前要先完成它列出的“回访”。

## 5. 进度

- [x] 第 0 阶段：术语表
- [x] 第 1 阶段：模型骨架
- [x] 第 2 阶段：基础层零件
- [ ] 第 3 阶段：注意力与 KV cache 主线
- [ ] 第 4 阶段：一次前向的组织
- [ ] 第 5 阶段：logits 与采样
- [ ] 第 6 阶段：执行面入口与启动
- [ ] 第 7 阶段：CUDA Graph 与流水线性能
- [ ] 第 8 阶段：配置、权重加载与量化
- [ ] 第 9 阶段：并行与通信
- [ ] 第 10 阶段：MoE 与专家负载均衡
- [ ] 第 11 阶段：投机解码
- [ ] 第 12 阶段：MLA、稀疏注意力与上下文并行
- [ ] 第 13 阶段：线性注意力与混合模型
- [ ] 第 14 阶段：KV cache 规划、卸载与多级缓存
- [ ] 第 15 阶段：PD / EPD 分离
- [ ] 第 16 阶段：多模态
- [ ] 第 17 阶段：结构化输出
- [ ] 第 18 阶段：模型族
- [ ] 第 19 阶段：可观测性、插件与通用工具

---

## 附录 A：全量文件清单

`python/tokenspeed/` 下全部 487 个 `.py` 文件，按阶段分组，组内按等级（S → A → B → C）排序。路径相对于 `python/tokenspeed/`。新增或删除文件时同步更新本表。等级后标 ✅ 表示该文件已完成注释（含英文注释的翻译）。

### 第 1 阶段：模型骨架（9 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/models/base/decoder_layer.py` | 250 | S ✅ |
| `runtime/models/base/transformer_model.py` | 195 | S ✅ |
| `runtime/models/qwen3.py` | 255 | S ✅ |
| `runtime/models/base/causal_lm.py` | 692 | A ✅ |
| `runtime/configs/qwen3_config.py` | 246 | B ✅ |
| `runtime/models/llama.py` | 220 | B ✅ |
| `runtime/models/registry.py` | 128 | B ✅ |
| `runtime/models/base/__init__.py` | 33 | C ✅ |
| `runtime/models/utils.py` | 47 | C ✅ |

### 第 2 阶段：基础层零件（12 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/distributed/comm_manager.py` | 826 | S ✅ |
| `runtime/layers/linear.py` | 1747 | S ✅ |
| `runtime/layers/rotary_embedding.py` | 1533 | S ✅ |
| `runtime/distributed/mapping.py` | 827 | A ✅ |
| `runtime/layers/activation.py` | 225 | A ✅ |
| `runtime/layers/layernorm.py` | 759 | A ✅ |
| `runtime/layers/mlp.py` | 203 | A ✅ |
| `runtime/layers/parameter.py` | 475 | A ✅ |
| `runtime/layers/vocab_parallel_embedding.py` | 734 | A ✅ |
| `runtime/layers/segmented_rmsnorm.py` | 76 | B ✅ |
| `runtime/layers/common.py` | 35 | C ✅ |
| `runtime/layers/utils.py` | 33 | C ✅ |

### 第 3 阶段：注意力与 KV cache（主线）（25 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/layers/attention/backends/base.py` | 902 | S |
| `runtime/layers/attention/backends/cache_metadata.py` | 213 | S |
| `runtime/layers/attention/backends/paged/base.py` | 598 | S |
| `runtime/layers/attention/backends/paged/mha.py` | 806 | S |
| `runtime/layers/attention/backends/paged/router.py` | 1234 | S |
| `runtime/layers/attention/backends/paged/write_locations.py` | 320 | S |
| `runtime/layers/attention/kv_cache/base.py` | 428 | S |
| `runtime/layers/attention/kv_cache/mha.py` | 356 | S |
| `runtime/layers/paged_attention.py` | 696 | S |
| `runtime/engine/scheduler_utils.py` | 1084 | A |
| `runtime/layers/attention/backends/paged/group_tables.py` | 566 | A |
| `runtime/layers/attention/backends/paged/trtllm.py` | 618 | A |
| `runtime/layers/attention/chunk.py` | 306 | A |
| `runtime/layers/attention/configs/base.py` | 394 | A |
| `runtime/layers/attention/configs/mha.py` | 108 | A |
| `runtime/layers/attention/kv_cache/arena.py` | 643 | A |
| `runtime/layers/attention/kv_cache/factory.py` | 360 | A |
| `runtime/layers/attention/page_table.py` | 90 | A |
| `runtime/layers/attention/registry.py` | 1604 | A |
| `runtime/layers/attention/backends/paged/cache_group_geometry.py` | 122 | B |
| `runtime/layers/attention/backends/support.py` | 146 | B |
| `runtime/layers/attention/kernel_page_sizes.py` | 55 | B |
| `runtime/layers/attention/utils.py` | 104 | B |
| `runtime/layers/attention/backends/__init__.py` | 51 | C |
| `runtime/layers/attention/backends/paged/__init__.py` | 19 | C |

### 第 4 阶段：一次前向的组织（15 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/execution/context.py` | 240 | S |
| `runtime/execution/forward_batch_info.py` | 145 | S |
| `runtime/execution/input_buffer.py` | 799 | S |
| `runtime/execution/model_executor.py` | 3138 | S |
| `runtime/execution/output_layout.py` | 103 | S |
| `runtime/execution/runtime_states.py` | 619 | S |
| `runtime/execution/cache_loc_kernel.py` | 304 | A |
| `runtime/execution/model_runner.py` | 765 | A |
| `runtime/execution/request_slots.py` | 42 | A |
| `runtime/execution/slot_state.py` | 262 | A |
| `runtime/execution/types.py` | 372 | A |
| `runtime/execution/nan_guard.py` | 216 | B |
| `runtime/execution/request_token_history.py` | 64 | B |
| `runtime/execution/workspace.py` | 181 | B |
| `runtime/execution/memory_delta.py` | 89 | C |

### 第 5 阶段：logits 与采样（22 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/layers/logits_processor.py` | 1653 | S |
| `runtime/sampling/backends/base.py` | 804 | S |
| `runtime/sampling/backends/greedy.py` | 307 | S |
| `runtime/layers/logits_processor_output.py` | 78 | A |
| `runtime/sampling/backends/flashinfer.py` | 1029 | A |
| `runtime/sampling/backends/triton.py` | 1136 | A |
| `runtime/sampling/logits_layout.py` | 119 | A |
| `runtime/sampling/sampling_batch_info.py` | 139 | A |
| `runtime/sampling/sampling_params.py` | 276 | A |
| `runtime/layers/sampled_vocab_shard.py` | 151 | B |
| `runtime/layers/vocab_parallel_topk.py` | 323 | B |
| `runtime/sampling/backends/ascend_full.py` | 378 | B |
| `runtime/sampling/backends/flashinfer_full.py` | 550 | B |
| `runtime/sampling/backends/full_base.py` | 168 | B |
| `runtime/sampling/backends/pooled.py` | 257 | B |
| `runtime/sampling/backends/sonic.py` | 975 | B |
| `runtime/sampling/backends/triton_full.py` | 591 | B |
| `runtime/sampling/custom_logit_processor.py` | 58 | B |
| `runtime/sampling/diagnostics.py` | 216 | B |
| `runtime/sampling/registry.py` | 154 | B |
| `runtime/sampling/utils.py` | 139 | B |
| `runtime/sampling/backends/__init__.py` | 22 | C |

### 第 6 阶段：执行面入口与启动（23 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/engine/rust_host.py` | 1674 | S |
| `runtime/execution/device.py` | 2047 | S |
| `runtime/engine/io_struct.py` | 614 | A |
| `runtime/entrypoints/engine.py` | 466 | A |
| `runtime/execution/distributed_initializer.py` | 326 | A |
| `runtime/execution/factory.py` | 333 | A |
| `cli/__main__.py` | 138 | B |
| `cli/_argsplit.py` | 286 | B |
| `cli/_proc.py` | 193 | B |
| `cli/serve_smg.py` | 1120 | B |
| `env.py` | 351 | B |
| `runtime/engine/gc_hooks.py` | 109 | B |
| `runtime/engine/logprobs.py` | 68 | B |
| `runtime/engine/memory_occupation.py` | 104 | B |
| `runtime/engine/rank_setup.py` | 307 | B |
| `runtime/engine/zmq_wire.py` | 406 | B |
| `runtime/entrypoints/wire_driver.py` | 283 | B |
| `__init__.py` | 25 | C |
| `cli/__init__.py` | 25 | C |
| `cli/_logo.py` | 100 | C |
| `cli/_logprefix.py` | 53 | C |
| `runtime/engine/__init__.py` | 21 | C |
| `version.py` | 21 | C |

### 第 7 阶段：CUDA Graph 与流水线性能（11 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/execution/breakable_cuda_graph.py` | 643 | S |
| `runtime/execution/forward_step.py` | 1337 | S |
| `runtime/execution/cudagraph_memory.py` | 357 | A |
| `runtime/execution/prefill_graph.py` | 1778 | A |
| `cli/reload_forward.py` | 105 | B |
| `runtime/execution/forward_reload.py` | 268 | B |
| `runtime/execution/graph_ptr_guard.py` | 187 | B |
| `runtime/utils/cuda_stream.py` | 104 | B |
| `runtime/utils/device_stream.py` | 135 | B |
| `runtime/utils/host_sync.py` | 113 | B |
| `runtime/utils/jit_compile_check.py` | 72 | B |

### 第 8 阶段：配置、权重加载与量化（46 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/configs/model_config.py` | 1282 | A |
| `runtime/layers/dense/fp8.py` | 413 | A |
| `runtime/layers/dense/unquant.py` | 110 | A |
| `runtime/layers/quantization/base_config.py` | 263 | A |
| `runtime/layers/quantization/fp8.py` | 165 | A |
| `runtime/model_loader/loader.py` | 870 | A |
| `runtime/model_loader/weight_utils.py` | 1207 | A |
| `runtime/configs/load_config.py` | 105 | B |
| `runtime/configs/model_profile.py` | 172 | B |
| `runtime/configs/numerics.py` | 234 | B |
| `runtime/configs/utils.py` | 802 | B |
| `runtime/execution/model_update.py` | 182 | B |
| `runtime/execution/weight_loader.py` | 138 | B |
| `runtime/execution/weight_memory.py` | 269 | B |
| `runtime/execution/weight_update_group.py` | 88 | B |
| `runtime/layers/dense/fp8_weight_only.py` | 106 | B |
| `runtime/layers/dense/mxfp4.py` | 184 | B |
| `runtime/layers/dense/nvfp4.py` | 320 | B |
| `runtime/layers/dense/w8a8_fp8.py` | 149 | B |
| `runtime/layers/quantization/compressed_tensors/compressed_tensors.py` | 708 | B |
| `runtime/layers/quantization/compressed_tensors/gptq_marlin_moe.py` | 55 | B |
| `runtime/layers/quantization/compressed_tensors/scalar_type.py` | 371 | B |
| `runtime/layers/quantization/compressed_tensors/schemes/compressed_tensors_scheme.py` | 77 | B |
| `runtime/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` | 401 | B |
| `runtime/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` | 81 | B |
| `runtime/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py` | 339 | B |
| `runtime/layers/quantization/fp8_weight_only.py` | 95 | B |
| `runtime/layers/quantization/modelopt_mixed.py` | 577 | B |
| `runtime/layers/quantization/mxfp4.py` | 405 | B |
| `runtime/layers/quantization/mxfp8_nvfp4.py` | 143 | B |
| `runtime/layers/quantization/nvfp4.py` | 124 | B |
| `runtime/layers/quantization/utils.py` | 472 | B |
| `runtime/layers/quantization/w8a8_fp8.py` | 76 | B |
| `runtime/model_loader/utils.py` | 74 | B |
| `runtime/models/base/weight_update.py` | 153 | B |
| `runtime/utils/hf_transformers_utils.py` | 936 | B |
| `runtime/utils/server_args.py` | 5138 | B |
| `runtime/configs/__init__.py` | 91 | C |
| `runtime/configs/device_config.py` | 38 | C |
| `runtime/hf_hub_retry.py` | 118 | C |
| `runtime/layers/dense/__init__.py` | 36 | C |
| `runtime/layers/quantization/__init__.py` | 60 | C |
| `runtime/layers/quantization/compressed_tensors/__init__.py` | 19 | C |
| `runtime/layers/quantization/compressed_tensors/schemes/__init__.py` | 41 | C |
| `runtime/model_loader/__init__.py` | 49 | C |
| `runtime/utils/torch_memory_saver_adapter.py` | 142 | C |

### 第 9 阶段：并行与通信（21 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/distributed/comm_backend/auto.py` | 795 | A |
| `runtime/distributed/comm_backend/base.py` | 371 | A |
| `runtime/distributed/comm_backend/nccl.py` | 336 | A |
| `runtime/distributed/comm_ops.py` | 946 | A |
| `runtime/distributed/pp_stage.py` | 242 | A |
| `runtime/distributed/process_group_manager.py` | 233 | A |
| `runtime/distributed/comm_backend/emulated.py` | 110 | B |
| `runtime/distributed/comm_backend/hccl.py` | 214 | B |
| `runtime/distributed/comm_backend/iris.py` | 408 | B |
| `runtime/distributed/comm_backend/projection.py` | 753 | B |
| `runtime/distributed/comm_backend/registry.py` | 60 | B |
| `runtime/distributed/comm_backend/triton_rsag.py` | 168 | B |
| `runtime/distributed/comm_backend/trtllm_allreduce.py` | 324 | B |
| `runtime/distributed/device_communicators/pynccl.py` | 299 | B |
| `runtime/distributed/dp_sampling_comm.py` | 441 | B |
| `runtime/distributed/dp_sampling_swap.py` | 77 | B |
| `runtime/distributed/nccl_suspend.py` | 102 | B |
| `runtime/distributed/utils.py` | 180 | B |
| `runtime/sampling/dp_sampling_config.py` | 268 | B |
| `runtime/distributed/__init__.py` | 26 | C |
| `runtime/distributed/comm_backend/__init__.py` | 39 | C |

### 第 10 阶段：MoE 与专家负载均衡（32 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/layers/moe/expert.py` | 841 | S |
| `runtime/layers/moe/topk.py` | 846 | S |
| `runtime/layers/moe/block.py` | 140 | A |
| `runtime/layers/moe/latent.py` | 834 | A |
| `runtime/layers/moe/loader.py` | 684 | A |
| `runtime/layers/moe/weights/loaders.py` | 468 | A |
| `runtime/models/qwen3_moe.py` | 117 | A |
| `runtime/moe/expert_location.py` | 1103 | A |
| `runtime/layers/moe/down_columns.py` | 135 | B |
| `runtime/layers/moe/schema.py` | 53 | B |
| `runtime/layers/moe/types.py` | 55 | B |
| `runtime/layers/moe/utils.py` | 239 | B |
| `runtime/layers/moe/weights/fp8.py` | 84 | B |
| `runtime/layers/moe/weights/fp8_channel.py` | 96 | B |
| `runtime/layers/moe/weights/int4.py` | 169 | B |
| `runtime/layers/moe/weights/int8.py` | 124 | B |
| `runtime/layers/moe/weights/mxfp4.py` | 140 | B |
| `runtime/layers/moe/weights/mxint4.py` | 124 | B |
| `runtime/layers/moe/weights/nvfp4.py` | 125 | B |
| `runtime/layers/moe/weights/unquant.py` | 86 | B |
| `runtime/layers/shared_expert_tp.py` | 182 | B |
| `runtime/moe/dispatch_algorithm.py` | 52 | B |
| `runtime/moe/eplb_algorithms/deepseek.py` | 265 | B |
| `runtime/moe/expert_load_rows.py` | 167 | B |
| `runtime/moe/expert_location_updater.py` | 307 | B |
| `runtime/moe/expert_rebalance.py` | 257 | B |
| `runtime/moe/placement_maps.py` | 133 | B |
| `runtime/configs/qwen3_moe_config.py` | 56 | C |
| `runtime/layers/moe/__init__.py` | 38 | C |
| `runtime/layers/moe/weights/__init__.py` | 143 | C |
| `runtime/moe/__init__.py` | 21 | C |
| `runtime/moe/eplb_algorithms/__init__.py` | 65 | C |

### 第 11 阶段：投机解码（50 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/execution/drafter/base.py` | 576 | S |
| `runtime/execution/drafter/eagle.py` | 1305 | S |
| `runtime/execution/drafter/speculative_sampling.py` | 336 | S |
| `runtime/execution/drafter/dflash.py` | 1450 | A |
| `runtime/execution/drafter/eagle_harness/drafter.py` | 1047 | A |
| `runtime/execution/drafter/mtp.py` | 694 | A |
| `runtime/execution/drafter/tree.py` | 169 | A |
| `runtime/execution/tree_spec.py` | 171 | A |
| `runtime/layers/attention/backends/paged/tree_verify.py` | 135 | A |
| `runtime/models/deepseek_nextn.py` | 453 | A |
| `runtime/models/llama_eagle3.py` | 750 | A |
| `runtime/sampling/tree_verify.py` | 105 | A |
| `runtime/execution/accept_simulation.py` | 135 | B |
| `runtime/execution/drafter/__init__.py` | 268 | B |
| `runtime/execution/drafter/_dflash_fused_kv.py` | 412 | B |
| `runtime/execution/drafter/deepseek_v41_dspark.py` | 299 | B |
| `runtime/execution/drafter/dflash2.py` | 266 | B |
| `runtime/execution/drafter/dspark.py` | 172 | B |
| `runtime/execution/drafter/dummy.py` | 131 | B |
| `runtime/execution/drafter/eagle_harness/config.py` | 409 | B |
| `runtime/execution/drafter/eagle_harness/cost.py` | 440 | B |
| `runtime/execution/drafter/eagle_harness/device_chain.py` | 110 | B |
| `runtime/execution/drafter/eagle_harness/device_policy.py` | 881 | B |
| `runtime/execution/drafter/eagle_harness/lanes.py` | 112 | B |
| `runtime/execution/drafter/eagle_harness/measure_run.py` | 479 | B |
| `runtime/execution/drafter/eagle_harness/measurement.py` | 320 | B |
| `runtime/execution/drafter/eagle_harness/round_stats.py` | 81 | B |
| `runtime/execution/drafter/eagle_harness/tree_verify.py` | 137 | B |
| `runtime/execution/dspark_context.py` | 153 | B |
| `runtime/models/beam_dflash2.py` | 103 | B |
| `runtime/models/deepseek_v32_nextn.py` | 37 | B |
| `runtime/models/deepseek_v41_dspark.py` | 593 | B |
| `runtime/models/deepseek_v41_dspark_heads.py` | 153 | B |
| `runtime/models/dflash.py` | 749 | B |
| `runtime/models/dflash2.py` | 697 | B |
| `runtime/models/dots3_note_nextn.py` | 198 | B |
| `runtime/models/dspark.py` | 131 | B |
| `runtime/models/glm53_flash_nextn.py` | 294 | B |
| `runtime/models/glm_moe_dsa_nextn.py` | 432 | B |
| `runtime/models/inkling_nextn.py` | 384 | B |
| `runtime/models/kimi_k3_dspark.py` | 857 | B |
| `runtime/models/kimi_k3_nextn.py` | 576 | B |
| `runtime/models/llama_eagle4.py` | 752 | B |
| `runtime/models/minimax_m3_dspark.py` | 204 | B |
| `runtime/models/nemotron_h_nextn.py` | 255 | B |
| `runtime/models/qwen3_5_nextn.py` | 413 | B |
| `runtime/models/qwen4_exp_nextn.py` | 449 | B |
| `runtime/models/target_capture.py` | 49 | B |
| `runtime/utils/spec_block_geometry.py` | 192 | B |
| `runtime/execution/drafter/eagle_harness/__init__.py` | 69 | C |

### 第 12 阶段：MLA、稀疏注意力与上下文并行（39 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/layers/attention/backends/paged/mla.py` | 776 | S |
| `runtime/execution/query_shard.py` | 488 | A |
| `runtime/layers/attention/backends/paged/dsa.py` | 2007 | A |
| `runtime/layers/attention/backends/paged/flashmla.py` | 986 | A |
| `runtime/layers/attention/backends/paged/tokenspeed_mla.py` | 959 | A |
| `runtime/layers/attention/backends/paged/trtllm_mla.py` | 644 | A |
| `runtime/layers/attention/configs/dsa.py` | 232 | A |
| `runtime/layers/attention/configs/mla.py` | 125 | A |
| `runtime/layers/attention/dcp/cache.py` | 1218 | A |
| `runtime/layers/attention/dcp/placement.py` | 333 | A |
| `runtime/layers/attention/kpool.py` | 616 | A |
| `runtime/layers/attention/kv_cache/dsa.py` | 308 | A |
| `runtime/layers/attention/kv_cache/mla.py` | 239 | A |
| `runtime/models/deepseek_v3.py` | 2706 | A |
| `runtime/layers/attention/backends/paged/ascend_dsa.py` | 971 | B |
| `runtime/layers/attention/backends/paged/ascend_mla.py` | 440 | B |
| `runtime/layers/attention/backends/paged/msa.py` | 592 | B |
| `runtime/layers/attention/backends/paged/qsa.py` | 330 | B |
| `runtime/layers/attention/backends/specific/qsa_indexer.py` | 352 | B |
| `runtime/layers/attention/configs/msa.py` | 147 | B |
| `runtime/layers/attention/dcp/ascend.py` | 827 | B |
| `runtime/layers/attention/dcp/comm.py` | 266 | B |
| `runtime/layers/attention/dcp/indexer.py` | 570 | B |
| `runtime/layers/attention/dcp/metadata.py` | 287 | B |
| `runtime/layers/attention/deepseek_v41_geometry.py` | 254 | B |
| `runtime/layers/attention/deepseek_v41_kvp.py` | 129 | B |
| `runtime/layers/attention/dsa/ascend.py` | 814 | B |
| `runtime/layers/attention/dsa/utils.py` | 108 | B |
| `runtime/layers/attention/gating.py` | 59 | B |
| `runtime/layers/attention/kv_cache/msa.py` | 65 | B |
| `runtime/layers/attention/mla/ascend.py` | 401 | B |
| `runtime/layers/attention/prefill_head_exchange.py` | 97 | B |
| `runtime/layers/attention/qsa/bf16_scratch.py` | 373 | B |
| `runtime/layers/attention/qsa/indexer.py` | 500 | B |
| `runtime/layers/attention/qsa/metadata.py` | 169 | B |
| `runtime/layers/attention/qsa/verify_state.py` | 218 | B |
| `runtime/layers/attention/query_partition.py` | 94 | B |
| `runtime/layers/attention/dcp/__init__.py` | 22 | C |
| `runtime/layers/attention/qsa/__init__.py` | 43 | C |

### 第 13 阶段：线性注意力与混合模型（27 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/layers/attention/backends/hybrid/linear.py` | 372 | A |
| `runtime/layers/attention/backends/state/checkpoint.py` | 274 | A |
| `runtime/layers/attention/backends/state/gdn.py` | 191 | A |
| `runtime/layers/attention/backends/state/kda.py` | 1226 | A |
| `runtime/layers/attention/backends/state/mamba.py` | 2586 | A |
| `runtime/layers/attention/configs/linear_attn.py` | 192 | A |
| `runtime/layers/attention/kv_cache/hybrid_mha.py` | 120 | A |
| `runtime/layers/conv.py` | 312 | A |
| `runtime/models/qwen3_5.py` | 1558 | A |
| `runtime/configs/qwen3_5_config.py` | 151 | B |
| `runtime/configs/qwen3_5_text_base_config.py` | 312 | B |
| `runtime/layers/attention/backends/state/mamba2.py` | 355 | B |
| `runtime/layers/attention/backends/state/prefill_capacity.py` | 594 | B |
| `runtime/layers/attention/backends/state/utils.py` | 45 | B |
| `runtime/layers/attention/kv_cache/hybrid_glm53_flash.py` | 179 | B |
| `runtime/layers/attention/kv_cache/hybrid_inkling.py` | 63 | B |
| `runtime/layers/attention/kv_cache/hybrid_kda.py` | 199 | B |
| `runtime/layers/attention/kv_cache/recipes/kda.py` | 127 | B |
| `runtime/layers/attention/kv_cache/recipes/mamba2.py` | 127 | B |
| `runtime/layers/attention/kv_cache/recipes/qwen35.py` | 380 | B |
| `runtime/layers/attention/linear/gdn.py` | 93 | B |
| `runtime/layers/attention/linear/layernorm_gated.py` | 117 | B |
| `runtime/models/nemotron_h.py` | 886 | B |
| `runtime/models/qwen3_5_moe.py` | 589 | B |
| `runtime/configs/nemotron_h_config.py` | 85 | C |
| `runtime/layers/attention/backends/hybrid/__init__.py` | 19 | C |
| `runtime/layers/attention/backends/state/__init__.py` | 19 | C |

### 第 14 阶段：KV cache 规划、卸载与多级缓存（38 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/cache/l2/executor.py` | 1276 | A |
| `runtime/cache/transfer/lanes.py` | 562 | A |
| `runtime/cache/transfer/layout.py` | 375 | A |
| `runtime/layers/attention/kv_cache/offload.py` | 354 | A |
| `runtime/layers/attention/kv_cache/recipes/base.py` | 614 | A |
| `runtime/layers/attention/kv_cache/recipes/ordinary.py` | 451 | A |
| `runtime/layers/attention/kv_cache/recipes/plan.py` | 965 | A |
| `runtime/layers/attention/kv_cache/recipes/scheduler_bridge.py` | 190 | A |
| `runtime/layers/attention/kv_cache/recipes/spec.py` | 703 | A |
| `runtime/layers/attention/kv_cache/virtual_blocks.py` | 187 | A |
| `runtime/cache/embedding_cache.py` | 224 | B |
| `runtime/cache/host_storage.py` | 291 | B |
| `runtime/cache/l2/layerwise_load.py` | 118 | B |
| `runtime/cache/l2/sizing.py` | 213 | B |
| `runtime/cache/l3/config.py` | 369 | B |
| `runtime/cache/l3/factory.py` | 333 | B |
| `runtime/cache/l3/identity.py` | 973 | B |
| `runtime/cache/transfer/ops.py` | 128 | B |
| `runtime/cache/transfer/ownership.py` | 238 | B |
| `runtime/layers/attention/backends/paged/offload_adapter.py` | 174 | B |
| `runtime/layers/attention/kv_cache/offload_config.py` | 129 | B |
| `runtime/layers/attention/kv_cache/recipes/cache_runtime.py` | 176 | B |
| `runtime/layers/attention/kv_cache/recipes/capacity.py` | 62 | B |
| `runtime/layers/attention/kv_cache/recipes/deepseek_v41.py` | 432 | B |
| `runtime/layers/attention/kv_cache/recipes/dots3_note.py` | 168 | B |
| `runtime/layers/attention/kv_cache/recipes/glm53_flash.py` | 483 | B |
| `runtime/layers/attention/kv_cache/recipes/inkling.py` | 354 | B |
| `runtime/layers/attention/kv_cache/recipes/kimi_k3.py` | 405 | B |
| `runtime/layers/attention/kv_cache/recipes/ownership.py` | 149 | B |
| `runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` | 495 | B |
| `runtime/layers/attention/kv_cache/recipes/setup.py` | 216 | B |
| `runtime/layers/attention/kv_cache/recipes/storage.py` | 274 | B |
| `runtime/layers/attention/kv_cache/recipes/transfer.py` | 240 | B |
| `runtime/cache/__init__.py` | 26 | C |
| `runtime/cache/l2/__init__.py` | 21 | C |
| `runtime/cache/l3/__init__.py` | 43 | C |
| `runtime/cache/transfer/__init__.py` | 21 | C |
| `runtime/layers/attention/kv_cache/recipes/__init__.py` | 15 | C |

### 第 15 阶段：PD / EPD 分离（10 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/epd/encode_host.py` | 627 | A |
| `runtime/epd/encode_worker.py` | 52 | A |
| `runtime/epd/prefill_receive.py` | 449 | A |
| `runtime/epd/recv_pool.py` | 77 | A |
| `runtime/pd/cache_protocol.py` | 195 | A |
| `runtime/pd/factory.py` | 126 | A |
| `runtime/pd/host_peer.py` | 400 | A |
| `runtime/pd/kv_events.py` | 127 | A |
| `runtime/pd/topology.py` | 98 | A |
| `runtime/pd/utils.py` | 64 | A |

### 第 16 阶段：多模态（27 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/execution/multimodal_runtime.py` | 326 | A |
| `runtime/layers/attention/mm_encoder_attention.py` | 572 | A |
| `runtime/models/qwen3_vision.py` | 710 | A |
| `runtime/multimodal/embedder.py` | 813 | A |
| `runtime/multimodal/mrope.py` | 481 | A |
| `runtime/execution/history_embedding.py` | 55 | B |
| `runtime/models/deepseek_v41_vision.py` | 389 | B |
| `runtime/models/moonvit.py` | 1009 | B |
| `runtime/models/muse_spark_audio.py` | 210 | B |
| `runtime/models/muse_spark_vision.py` | 622 | B |
| `runtime/models/nemotron_omni.py` | 437 | B |
| `runtime/models/parakeet.py` | 241 | B |
| `runtime/models/qwen3_asr.py` | 237 | B |
| `runtime/models/qwen3_audio.py` | 573 | B |
| `runtime/models/qwen3_omni.py` | 406 | B |
| `runtime/models/radio.py` | 377 | B |
| `runtime/multimodal/encoder_batching.py` | 94 | B |
| `runtime/multimodal/encoder_cudagraph.py` | 641 | B |
| `runtime/multimodal/encoder_feature_transport.py` | 453 | B |
| `runtime/multimodal/hash.py` | 84 | B |
| `runtime/multimodal/inputs.py` | 377 | B |
| `runtime/multimodal/mrope_facts.py` | 85 | B |
| `runtime/multimodal/shm_transport.py` | 360 | B |
| `runtime/configs/nemotron_omni_config.py` | 371 | C |
| `runtime/configs/qwen3_asr_config.py` | 204 | C |
| `runtime/configs/qwen3_vision_config.py` | 59 | C |
| `runtime/multimodal/__init__.py` | 26 | C |

### 第 17 阶段：结构化输出（6 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/grammar/capturable_grammar.py` | 518 | A |
| `runtime/grammar/grammar_host.py` | 698 | A |
| `runtime/grammar/base_grammar_backend.py` | 54 | B |
| `runtime/grammar/host_grammar.py` | 187 | B |
| `runtime/grammar/reasoning_structural_tag.py` | 335 | B |
| `runtime/grammar/__init__.py` | 21 | C |

### 第 18 阶段：模型族（51 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `runtime/layers/attention/backends/specific/deepseek_v41.py` | 3108 | A |
| `runtime/layers/attention/backends/specific/inkling.py` | 1133 | A |
| `runtime/models/deepseek_v41.py` | 3678 | A |
| `runtime/models/gpt_oss.py` | 917 | A |
| `runtime/models/kimi_k3.py` | 5623 | A |
| `runtime/models/monokernel.py` | 167 | A |
| `runtime/configs/beam_config.py` | 348 | B |
| `runtime/configs/deepseek_v41_config.py` | 131 | B |
| `runtime/configs/dots3_note_config.py` | 66 | B |
| `runtime/configs/glm53_flash_config.py` | 388 | B |
| `runtime/configs/inkling_config.py` | 551 | B |
| `runtime/configs/kimi_k25_config.py` | 200 | B |
| `runtime/configs/kimi_k2_config.py` | 37 | B |
| `runtime/configs/kimi_k3_config.py` | 491 | B |
| `runtime/configs/kimi_k3_dspark_config.py` | 327 | B |
| `runtime/configs/minimax_m3_config.py` | 229 | B |
| `runtime/configs/muse_spark_config.py` | 162 | B |
| `runtime/configs/qwen4_exp_config.py` | 211 | B |
| `runtime/layers/attention/backends/dots3_note.py` | 237 | B |
| `runtime/layers/attention/backends/specific/qwen4_exp.py` | 362 | B |
| `runtime/layers/attention/backends/specific/qwen4_exp_ple.py` | 478 | B |
| `runtime/layers/attention/configs/deepseek_v41.py` | 226 | B |
| `runtime/layers/attention/configs/dots3_note.py` | 89 | B |
| `runtime/layers/attention/kv_cache/deepseek_v41.py` | 143 | B |
| `runtime/layers/attention/kv_cache/dots3_note.py` | 116 | B |
| `runtime/layers/attention/kv_cache/qwen4_exp.py` | 77 | B |
| `runtime/layers/hyperconnection.py` | 355 | B |
| `runtime/layers/ple_lookup.py` | 478 | B |
| `runtime/layers/qwen4_exp_ple.py` | 1233 | B |
| `runtime/models/beam.py` | 1836 | B |
| `runtime/models/beam_quant.py` | 402 | B |
| `runtime/models/beam_sandwich_fusion.py` | 266 | B |
| `runtime/models/deepseek_v32.py` | 42 | B |
| `runtime/models/deepseek_v41_engram.py` | 960 | B |
| `runtime/models/deepseek_v41_moe.py` | 589 | B |
| `runtime/models/dots3_note.py` | 651 | B |
| `runtime/models/extensible.py` | 188 | B |
| `runtime/models/glm5.py` | 1960 | B |
| `runtime/models/glm53_flash.py` | 1878 | B |
| `runtime/models/inkling.py` | 2096 | B |
| `runtime/models/kimi_k25.py` | 311 | B |
| `runtime/models/kimi_k31.py` | 921 | B |
| `runtime/models/kimi_k3_comm.py` | 690 | B |
| `runtime/models/kimi_k3_deepep.py` | 266 | B |
| `runtime/models/kimi_k3_projection.py` | 327 | B |
| `runtime/models/longcat_flash.py` | 1191 | B |
| `runtime/models/minimax_m3.py` | 1598 | B |
| `runtime/models/muse_spark.py` | 3255 | B |
| `runtime/models/qwen4_exp.py` | 815 | B |
| `runtime/models/qwen4_exp_monokernel.py` | 753 | B |
| `runtime/layers/attention/backends/specific/__init__.py` | 19 | C |

### 第 19 阶段：可观测性、插件与通用工具（23 个文件）

| 文件 | 行数 | 等级 |
| --- | --- | --- |
| `_logging.py` | 226 | B |
| `cli/trace_merge.py` | 466 | B |
| `runtime/engine/profiler.py` | 503 | B |
| `runtime/metrics/collector.py` | 699 | B |
| `runtime/metrics/config.py` | 114 | B |
| `runtime/metrics/events.py` | 73 | B |
| `runtime/metrics/registry.py` | 160 | B |
| `runtime/metrics/relay.py` | 156 | B |
| `runtime/metrics/reporter.py` | 248 | B |
| `runtime/metrics/service.py` | 232 | B |
| `runtime/plugins/registry.py` | 519 | B |
| `runtime/utils/common.py` | 799 | B |
| `runtime/utils/env.py` | 405 | B |
| `runtime/utils/flashinfer_config.py` | 29 | B |
| `runtime/utils/launcher.py` | 384 | B |
| `runtime/utils/network.py` | 141 | B |
| `runtime/utils/process.py` | 220 | B |
| `runtime/utils/tensor.py` | 106 | B |
| `runtime/utils/triton.py` | 24 | B |
| `runtime/plugins/__init__.py` | 179 | C |
| `runtime/utils/__init__.py` | 21 | C |
| `runtime/utils/nvtx.py` | 121 | C |
| `runtime/utils/startup_timing.py` | 122 | C |
