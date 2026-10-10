# 07 八卡请求全链路：TP=2、DP=4、EP=8

## 1. 选择理由与假设

本章是 **[代码 + 教学推导]**，没有多卡实测。采用一台 8 卡 NVIDIA SM90 级机器、PP=PCP=DCP=1，Attention TP=2、DP=4、EP 开启，普通 FusedMoE 与 allgather_reducescatter，V1 runner，无视觉/投机/DBO，E_physical 能均匀分配，H/h/g/I 与选定量化后端满足对齐约束。

选这组配置是因为能同时看到 Attention 的请求分配、TP heads、SP token 分片和跨 DP 的 expert dispatch。普通 BF16/FP8 权重接口的静态路径可追踪；真实 V4 checkpoint 是否适配指定 expert kernel、能否装入八卡、实际峰值 workspace，都仍要验证。它不是某个具体模型的已实测部署建议。

对照 TP=8、DP=1、EP 开、非 native-Mega 路径：EP group 仍是 0–7，但该模型 SP 自动条件不成立，Attention 与 MoE token 分片策略不同。若 EP 关，expert TP=8、每 rank 保存全部 experts 的权重切片。改成 native MegaMoE 又会改变 SP 与通信执行，不能只改配置标题后复用本章的所有表。

## 2. rank 图与三层决策

~~~mermaid
flowchart TD
  B["请求入口 / DP 路由"] --> S0["scheduler0 -> ranks 0,1"]
  B --> S1["scheduler1 -> ranks 2,3"]
  B --> S2["scheduler2 -> ranks 4,5"]
  B --> S3["scheduler3 -> ranks 6,7"]
  S0 --> A0["本副本 Attention / KV"]
  S1 --> A1["本副本 Attention / KV"]
  S2 --> A2["本副本 Attention / KV"]
  S3 --> A3["本副本 Attention / KV"]
  A0 --> M["EP 0..7: dispatch / 本地 expert / combine"]
  A1 --> M
  A2 --> M
  A3 --> M
  M --> O["回到原 DP/SP owner -> logits / sampling"]
~~~

| 启动时确定 | 每轮 CPU 调度/准备决定 | GPU runtime 决定/执行 |
| --- | --- | --- |
| rank groups、权重切分、backend、cache pool、图捕获尺寸 | 请求集、q_i、T、positions、block IDs、metadata、padding/graph 档位 | stream 顺序、event 等待、kernel launch、collective 内部算法及设备任务推进 |

集体通信必须让参与 rank 按兼容顺序调用；DP 副本本轮没有用户 token 时可能仍需 dummy/padded work 参与 MoE 协作，不能自行跳过整个 layer。

## 3. 一轮 prefill：每个 DP 有三个两-token prompt

各 DP 的 R=3、q=[2,2,2]、T=6。每个副本使用自己的 IDs、block_table，但同 TP group 两个 rank 的请求/positions 一致。positions=[0,1,0,1,0,1]，query_start_loc=[0,2,4,6]。SP 下每 rank 先拿3行；四 DP 总有效 T=24，EP group 的8个源分片各3行。

| 阶段 / 入口 | rank group | 每 rank 输入 -> 输出 | KV | 通信/依赖 |
| --- | --- | --- | --- | --- |
| [vllm/v1/core/sched/scheduler.py · Scheduler.schedule · L580–L1556](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py#L580-L1556) | 每 DP 的 CPU scheduler | 请求状态 -> q_i/block IDs | 分配逻辑 blocks | 不算 GPU Attention |
| [vllm/v1/executor/multiproc_executor.py · MultiprocExecutor.execute_model · L341–L352](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/executor/multiproc_executor.py#L341-L352) -> [vllm/v1/worker/gpu_worker.py · Worker.execute_model · L1267–L1325](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_worker.py#L1267-L1325) | 本 DP 的 TP=2 worker | 同一 SchedulerOutput -> runner | 传递页表更新 | 控制消息/RPC，不是模型 collective |
| [vllm/v1/worker/gpu_model_runner.py · GPUModelRunner._prepare_inputs · L1943–L2270](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py#L1943-L2270) / [vllm/v1/worker/gpu_model_runner.py · GPUModelRunner.execute_model · L4108–L4465](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py#L4108-L4465) | 本 DP 每 worker | IDs[6]、positions[6]、metadata | 生成 group slot mapping | H2D、设置 forward_context |
| [vllm/model_executor/layers/vocab_parallel_embedding.py · VocabParallelEmbedding.forward · L518–L563](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/vocab_parallel_embedding.py#L518-L563) | TP [2j,2j+1] | vocab 分片查表 -> [6,H] | 无 | TP sum lookup partial |
| [vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4Model.forward · L1424–L1526](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1424-L1526) / [vllm/models/common/ops/sequence_parallel.py · sp_shard · L42–L50](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/common/ops/sequence_parallel.py#L42-L50) | 同 TP | [6,H] -> [3,H] | 无 | token 行切分本身无需网络 |
| [vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4DecoderLayer.forward · L1188–L1282](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1188-L1282) / [vllm/models/common/ops/sequence_parallel.py · sp_all_gather · L23–L27](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/common/ops/sequence_parallel.py#L23-L27) | 同 TP | [3,H] -> [6,H] | 无 | AG 是 Attention 输入依赖 |
| [vllm/models/deepseek_v4/attention.py · DeepseekV4Attention.forward · L471–L501](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L471-L501) | 同 TP，各半 heads | [6,H] -> Q[6,h/2,d] -> partial[6,H] | SWA 写6行；压缩边界写记录；index K / partial states | replicated 投影无通信；本轮 kernel 读各自历史 |
| [vllm/models/common/ops/sequence_parallel.py · sp_reduce_scatter · L30–L39](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/common/ops/sequence_parallel.py#L30-L39) | 同 TP | partial[6,H] -> 完整行片[3,H] | 无 | RS 归约 O projection partial |
| [vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4MoE.forward · L896–L992](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L896-L992) / [vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py · MoEPrepareAndFinalizeNaiveDPEPModular.prepare · L136–L208](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py#L136-L208) | EP 0..7 | 本源[3,H] -> 汇总[24,H] | 不传 Attention KV | AGv activation + routing tensors |
| expert GEMMs / [vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py · MoEPrepareAndFinalizeNaiveDPEPModular.finalize · L210–L245](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py#L210-L245) | 每 EP rank 的 experts | 各 M_e -> 本地贡献[24,H] -> [3,H] | 无 | RSv 归并贡献回 owner |
| shared branch / mHC | 本 TP token 片 | [3,H] -> [3,H]，residual=[3,hc_mult,H] | 无 | SP shared 权重复制，无 shared AR |
| final [vllm/models/common/ops/sequence_parallel.py · sp_all_gather · L23–L27](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/common/ops/sequence_parallel.py#L23-L27) / norm | 同 TP | residual 行片 -> [6,H] | 无 | 最后收集全部行 |
| [vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4ForCausalLM.compute_logits · L1865–L1870](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1865-L1870) / [vllm/v1/worker/gpu_model_runner.py · GPUModelRunner.sample_tokens · L4483–L4738](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py#L4483-L4738) | 本 DP 的输出路径 | 取每请求末行[3,H] -> logits[3,V] -> IDs | 采样 token 还没有 KV | vocab Gather/专用候选通信，sample 后 D2H |
| [vllm/v1/core/sched/scheduler.py · Scheduler.update_from_output · L2025–L2423](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py#L2025-L2423) | 原 DP scheduler | 输出 -> token ID/停止/释放 | 归还结束请求引用 | 下一轮可推进 |

压缩比 c=4 时首次 seq_len=2 没有完整压缩记录，但仍有 SWA 与 compressor partial state 写入。长上下文 prefill 后续 chunk 则读历史压缩 cache，不能把“prefill”理解成永远无历史。主线稀疏 kernel 的 Q heads 可能补到64，表里 Q 先列语义 shape，真实缓冲见第4章。

## 4. 下一轮 decode：三个请求各推进一个 token

prefill 各采样一个输出 ID 后，各 DP known=3、computed=2。本轮 q=[1,1,1]，T=3，positions=[2,2,2]，seq_lens=[3,3,3]，query_start_loc=[0,1,2,3]。每个请求读取自己的合法历史并写 position 2。

SP 的 T 不可被2整除：补为4行，各 rank 拿2行，其中一行是 padding。Attention 前 AG 得4行后按 positions 长度截到3；O partial 的 RS 又补到4。MoE 输入每源2行，因此 EP AG/RS buffer 总行数为16，其中有效为12。**T_pad 不是请求多生成了 token**。

| 量 | prefill | decode | 主要性能变化 |
| --- | --- | --- | --- |
| 本 DP T / SP padded T | 6 / 6 | 3 / 4 | 小 M，更容易被 padding/launch 放大 |
| 本 rank MoE source 行数 | 3 | 2（可能含 padding） | 有效 expert M 更小 |
| EP 汇总行数 | 24 | 16，12 有效 | 容量/通信看 padded 口径 |
| Attention 可见历史 | 首次短 prompt | 至 position2 | 长上下文时历史读取成为关键 |
| logits 行数 | 3 | 3 | 通常与采样请求数而非 T 相等 |

具体 graph/DP 协调还可能增添 padding，所以上述数目是模型 SP 的最小行数推导，不是保证 kernel 最终 launch shape。记录 T、T_pad、valid mask 与 DP metadata 才能解释 trace。

## 5. 精确通信口径练习

教学 H=4096、BF16、EP p=8，只计算 AG/RS 的 activation data，均匀 ring：

- prefill S_EP=24*4096*2=196,608 B=0.196608 MB=0.1875 MiB；
- 每 rank 每次 AG 或 RS send=recv=172,032 B=0.172032 MB=0.1640625 MiB；
- 每次全组 send=1,376,256 B=1.376256 MB=1.3125 MiB；全组 recv 相同；
- decode 按16 padded 行，S_EP=131,072 B=0.125 MiB；每 rank send=recv=114,688 B=0.109375 MiB；全组 send=917,504 B=0.875 MiB。

AG+RS 将“每次”数乘2，全组 send+recv 再乘2。路由 IDs、FP32 weights、量化 scales、padding extras 另计。实际 All-Gatherv/Reduce-Scatterv 未必执行 ring，本节只是指定 ring 的学习计算，不能用这些数反推 runtime 算法。

## 6. 调用链验证与练习

打开 [源码定位表](source_index.md)，按上表从 EngineCore.step 走到 Attention insert 和 MoE finalize，再沿 output 回到 scheduler。每条边分别写“直接 Python 调用”“RPC 到 worker”“torch op 到 CUDA kernel”或“collective”；避免把控制路径通信和 activation 通信混为一谈。

手算 decode 的 SP padding 及 EP 源行数；解释一个来自 DP3 的 token 为何能由 rank0 的 expert 处理，却不能直接读取 DP0 的 Attention KV。答案见 [答案第07节](answers.md#07)；下一章：[Stream 与重叠](08_streams.md)。
