# 10 从 trace 定位瓶颈：决定改哪一层

## 1. 从服务问题出发

“decode 慢”可能是排队久、KV 压力导致反复重算、expert 负载偏斜、通信链路、padded shape 太大、CPU 提交间隙，也可能是一个 kernel 的带宽利用率低。只看最长 GEMM 名字无法判断下一步。

本环境没有 GPU trace；本章数值全部为 **[教学]**。真实 scheduler 记录只提供状态和 token 分配，不提供设备耗时。CPU perf 时间也不能代替 NVIDIA kernel 时间。

## 2. 把因果链画出来

~~~mermaid
flowchart LR
  A["到达 / waiting"] --> B["schedule / allocate"]
  B --> C["pack / H2D / metadata"]
  C --> D["Attention projections / cache / indexer"]
  D --> E["TP reduction"]
  E --> F["MoE dispatch / experts / combine"]
  F --> G["logits / sampling / D2H"]
  G --> H["output update / 下一轮"]
  K["KV 容量 / prefix hit / preemption"] --> B
  P["rank layout / padding / backend"] --> D
  P --> F
~~~

先定位服务端指标对应图上的哪一段，再讨论 kernel 参数；跨 stream 时关键路径是依赖图中的最长链，不是所有 kernel 时间简单求和。

## 3. 一套可复现分析流程

### 3.1 固定工作负载

记录模型 revision、vLLM commit、runner 版本、GPU/拓扑、quant 与 KV dtype、Attention/MoE/all2all backend、TP/DP/EP/PP/PCP/DCP、SP/DBO/graph、prefix cache、prompt/output 长度分布。每轮记录 R、q_i、T、T_pad、seq_lens、prefix hit、抢占和 expert M_e 分布。

相同 T 可以来自一个长 prefill 或很多 decode。相同 R 也可以有不同 T。相同输出 token 数可能经历不同重算量，比较必须用有效工作量与服务指标共同限定。

### 3.2 分开 warmup、capture 与 steady state

首轮 JIT、权重转换、图捕获、cache 分配不能混入 steady state。prefill gather workspace 可按上限预留；虚拟 warmup metadata=None 分支不能当成真实模型计算。[vllm/models/deepseek_v4/nvidia/flashmla.py · DeepseekV4FlashMLAAttention.forward_mqa · L87–L172](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/flashmla.py#L87-L172) 的 dummy 分支会预留 workspace 后直接填零，就是例子。

### 3.3 沿请求与 rank 找关键路径

从 request 到 schedule，再找到对应 iteration/rank 的 forward / sample；将 CPU 范围关联到 GPU kernel，检查是否有跨轮 asynchronous overlap。比较最慢参与 rank，而不只比较 rank0。MoE combine 的等待常来自其他 rank 的 expert 负载或输入迟到。

### 3.4 分类时要有指标证据

| 观察 | 需要确认 | 可能的优化层 |
| --- | --- | --- |
| GEMM 长、tensor core 高、M 足够 | FLOPs、tile、padding、精度、workspace | kernel / quant / 投影结构 |
| cache gather/indexer 长 | 实际读 byte、L、K、cache record、局部性、HBM counters | kernel / backend / cache 策略 |
| dispatch/combine 长 | payload、拓扑、expert 分布、到达偏差、传输格式 | all2all backend / EP 布局 / EPLB |
| kernel 间大空白 | CPU launch、metadata、event 依赖、GC/编译、设备同步 | runner / graph / 调度 |
| preemption 与重复 prefill 多 | cache group 容量、prefix 命中、长请求占用 | KV/batch/admission |
| 总 T 太小、expert M 多为1 | 请求并发、DP 数、SP padding、TopK、服务 SLO | batch / 并行配置 |
| 重叠区双方明显变慢 | SM、HBM、L2、通信资源竞争 | streams / DBO thresholds / kernel 资源 |

某通信 kernel 时间很长，可能在等 peer 而不是持续传输。要将链路实际 byte/rate、前驱完成时间与同步一起看。GPU utilisation 高不等于 tensor core 有效计算效率高。

### 3.5 用一次受控改变建立因果

每次只变一个主要条件，例如 TP8->TP2/DP4、不同 token budget、shared overlap 开关、indexer dtype。保留正确性/精度检查，比较 TTFT、TPOT 分位数、吞吐、cache/显存峰值与抢占。优化吞吐可能损害长尾延迟；以用户的服务目标判断收益。

## 4. 通信 roofline 的口径

数据 dtype、逻辑 payload、单 rank send/recv、全组 send 和算法都记录。理论时间可用每 rank 关键链路传输字节 / 可达到带宽 + 相应阶段延迟作为粗估；不得用“全组 send 总量 / 单条 NVLink 带宽”预测实际时间，因为不同链路可并发。

若 FP8 dispatch 数据减少，scales/metadata 未减，端到端不一定减半。若 collectives 被 kernel 融合，trace 上无独立 NCCL bar 仍可能有 NVLink 读写。第5–7章公式限定 ring，仅为学习基线。

## 5. 教学 trace 区间与优化判断

有一轮 decode：CPU prepare 2 ms；GPU Attention 4 ms；EP通信 3 ms；expert GEMM 2 ms；sample/output 1 ms，初始依赖全串行，总12 ms。将 expert GEMM 优化25%只省0.5 ms，总变11.5 ms。把独立的2 ms shared branch 与3 ms通信完全重叠，最多从两者之和减到3 ms；必须确认原12 ms是否已经包含 shared，不能在同一个基线里重复减时间。

若真实 CPU prepare 不在 GPU 的关键路径上（下一轮提前提交），省下2 ms也未必改变TPOT。若8卡trace中rank7晚3 ms导致其他rank等待，那么优化rank0的expert可能无效。先画生产者/消费者边，再用 Amdahl 比例解释收益。

## 6. 进阶主题：与当前主线的连接

| 主题 | 改变哪条边 | 已有源码/文档入口 | 下一阶段只研究一个问题 |
| --- | --- | --- | --- |
| PP，pipeline parallel | layer 按阶段切；传 hidden/intermediate | [vllm/distributed/parallel_state.py · initialize_model_parallel · L2056–L2363](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/distributed/parallel_state.py#L2056-L2363)；[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4Model.forward · L1424–L1526](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1424-L1526) | microbatch 能否填满 pipeline，source sharing 可否跨切点 |
| PCP，prefill context parallel | 长 prefill token/上下文跨 rank | [vllm/distributed/parallel_state.py · initialize_model_parallel · L2056–L2363](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/distributed/parallel_state.py#L2056-L2363)；[vllm/v1/worker/block_table.py · ComputeSlotMappingKernel.kernel · L320–L376](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/block_table.py#L320-L376) | token 与 KV interleave、额外通信如何改变 Attention |
| DCP，decode context parallel | 历史 KV / Attention 归约跨 TP 子组 | [vllm/distributed/parallel_state.py · initialize_model_parallel · L2056–L2363](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/distributed/parallel_state.py#L2056-L2363)；[vllm/v1/attention/backends/mla/compressor_utils.py · CompressedSlotMappingKernel.kernel · L45–L106](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/attention/backends/mla/compressor_utils.py#L45-L106) | 长上下文读取与归约的收益，具体 backend 支持条件 |
| PD 分离，prefill/decode disaggregation | 两套执行资源之间转移 KV | [本地功能文档](../../features/disagg_prefill.md) | KV 传输代价能否换来 SLO 隔离 |
| KV offload | GPU cache 与 host/远端间搬页 | [功能文档](../../features/kv_offloading_usage.md) | 命中/预取如何隐藏搬移及避免阻塞 |
| speculative decoding | 一个 decode 请求每轮验证多个草稿 token | [vllm/v1/core/sched/scheduler.py · Scheduler.schedule · L580–L1556](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py#L580-L1556)；[文档](../../features/speculative_decoding/README.md) | 接受率如何改变有效 T 与拒绝回滚 |
| DSpark / MTP | DeepSeek 专用 draft 或多 token 预测 | [vllm/models/deepseek_v4/nvidia/dspark.py · DSparkDeepseekV4ForCausalLM · L349–L602](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/dspark.py#L349-L602)；[vllm/models/deepseek_v4/nvidia/mtp.py · DeepSeekV4MTP · L307–L576](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/mtp.py#L307-L576) | target/draft 的 hidden/KV 生命周期如何配合 |

表中机制在仓库中有入口，不等于所有机制能与本教程配置组合。V4.1 source-sharing 的 PP、bounded replay 的 runner/SP/PCP/DBO 限制尤其需要逐项核对。下一步最好选一个机制，用相同“例子->图->调用链->shape->通信->验证”方法深入，不一次开启全部功能。

## 7. 练习

按教学12 ms串行区间算 GEMM优化后的端到端加速；说明如果prepare与GPU已重叠，为什么不能再简单减2 ms。给一个“长通信其实主要等待peer”的可证伪假设，以及需要从trace提取的两条证据。

答案见 [答案第10节](answers.md#10)；继续看 [误区](misconceptions.md) 与 [资料/未验证项](sources_and_limits.md)。
