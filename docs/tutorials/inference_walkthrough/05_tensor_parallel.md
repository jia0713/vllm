# 05 TP：一个 layer 不能只说“权重平分”

## 1. 两卡先从部分和理解

Tensor Parallel（张量并行，TP）让多个 rank 合作执行同一个请求。ColumnParallelLinear 切输出通道，RowParallelLinear 切归约输入维度。后者每卡得到的是相同输出坐标的部分和，必须相加，才能进入下一层的非线性。

[教学] p=2，y=x0*W0+x1*W1。rank0 只能得到第一项，rank1 只能得到第二项；仅拼接两项不等于 y。All-Reduce 对部分和求和并让各 rank 持有完整结果；All-Gather 则拼接不同坐标的分片；Reduce-Scatter 先相加再分发不同 token 行。

## 2. V4 Attention 的 rank 数据流

~~~mermaid
flowchart LR
  X["复制 x: T x H"] --> R0["rank0: Q heads 0..h/2-1"]
  X --> R1["rank1: Q heads h/2..h-1"]
  K["复制共享 KV / indexer"] --> R0
  K --> R1
  R0 --> O0["wo_a 本地组 -> wo_b 部分和: T x H"]
  R1 --> O1["wo_a 本地组 -> wo_b 部分和: T x H"]
  O0 --> AR["TP All-Reduce"]
  O1 --> AR
  AR --> Y["每 rank 完整 T x H"]
~~~

## 3. 权重、activation 与 KV 的分布

[代码] [vllm/models/deepseek_v4/attention.py · DeepseekV4Attention.__init__ · L195–L469](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L195-L469) 定义全部 Attention 线性层；[vllm/model_executor/layers/linear.py · RowParallelLinear.forward · L1762–L1788](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/linear.py#L1762-L1788) 展示最终 tensor_model_parallel_all_reduce；[vllm/models/deepseek_v4/nvidia/ops/o_proj.py · deep_gemm_fp8_o_proj · L35–L103](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/ops/o_proj.py#L35-L103) 是 V4 分组 O 投影的实际调用。

| 对象 | TP=2 时每 rank | TP=8 时每 rank | 通信 |
| --- | --- | --- | --- |
| input x | [T,H] 复制 | [T,H] 复制 | 由前一层保证一致 |
| fused_wqa_wkv | [r_q+d,H] 完整；disable_tp=True | 同左 | 无 |
| qr / shared kv | [T,r_q] / [T,d] 复制 | 同左 | 无 |
| wq_b | [h*d/2,r_q] | [h*d/8,r_q] | 默认不 gather heads |
| Attention Q | [T,h/2,d] | [T,h/8,d] | heads 各自算 |
| SWA / compressed / indexer cache | 相同请求历史副本 | 相同请求历史副本 | 普通 TP 无 cache 分片 |
| indexer 投影 | 完整 h_idx,d_idx,r_q | 完整 | 重复计算与存储 |
| wo_a | [g/2,r_o,h*d/g] | [g/8,r_o,h*d/g] | 分组投影，无直接 collective |
| wo_b | [H,g*r_o/2] | [H,g*r_o/8] | [T,H] 部分和 All-Reduce |
| embedding | [V_pad/2,H] | [V_pad/8,H] | mask local lookup，sum 得完整 hidden |
| LM head | [V_pad/2,H] | [V_pad/8,H] | vocab logits Gather / All-Gather，取决于平台/采样路径 |

需同时满足 h%p=0、g%p=0 以及权重量化对齐等约束。V_pad 是 padded vocabulary，不一定等于 V。[vllm/model_executor/layers/vocab_parallel_embedding.py · VocabParallelEmbedding.forward · L518–L563](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/vocab_parallel_embedding.py#L518-L563) 做 local token mask 和 reduction；[vllm/model_executor/layers/logits_processor.py · LogitsProcessor._gather_logits · L122–L134](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/logits_processor.py#L122-L134) 在平台策略下选 gather / all-gather。普通 NVIDIA 主线不应盲目写成“所有 rank 都 All-Gather 完整 logits”；某些 greedy/top-k 专用路径只通信候选值/ID，[vllm/model_executor/layers/logits_processor.py · LogitsProcessor._get_logits · L180–L199](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/logits_processor.py#L180-L199) 也可以跳过全量 gather。

增加 TP 可减少某些权重和 Q heads，却不减少 replicated 投影、indexer、shared KV，还增加通信次数。FlashMLA 会把本地 Q heads 补到 64/128，[vllm/models/deepseek_v4/nvidia/flashmla.py · DeepseekV4FlashMLAAttention.get_padded_num_q_heads · L78–L85](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/flashmla.py#L78-L85)；当 h/p 远小于 64，实际 Attention kernel 的 heads 工作量不会继续线性下降。

## 4. Sequence Parallel 改变哪条边

Sequence Parallel（序列并行，SP）在这条模型路径里切的是 packed token 行，不是独立新增的 GPU 维度，也不是 DCP 的历史 KV 切分。

[vllm/models/deepseek_v4/nvidia/model.py · _use_sequence_parallel · L1045–L1056](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1045-L1056) 仅在 PP=1、EP 开、TP>1 且 MegaMoE 或 DP>1 时自动启用该模型的 SP。不是一般的编译 sequence-parallel flag 一开就能保证整条路径如此。[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4Model.forward · L1424–L1526](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1424-L1526) 首先 sp_shard hidden/input_ids；[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4DecoderLayer.__init__ · L1060–L1186](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1060-L1186) 关闭 wo_b 的 All-Reduce；[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4DecoderLayer.forward · L1188–L1282](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1188-L1282) 每层做：

~~~text
局部 [ceil(T/p),H]
  -> SP All-Gather
  -> 完整 [T,H] 的 Attention，heads 按 TP 切分
  -> wo_b 输出完整 token 的部分和
  -> SP Reduce-Scatter
  -> 局部 [ceil(T/p),H] 的 MoE 与 residual
~~~

[vllm/models/common/ops/sequence_parallel.py · sp_shard · L42–L50](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/common/ops/sequence_parallel.py#L42-L50) 对 T 不能整除 p 的情况补零；[vllm/models/common/ops/sequence_parallel.py · sp_all_gather · L23–L27](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/common/ops/sequence_parallel.py#L23-L27) 和 [vllm/models/common/ops/sequence_parallel.py · sp_reduce_scatter · L30–L39](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/common/ops/sequence_parallel.py#L30-L39) 调 TP collective（可有 custom 实现）。输出行被 scatter 后，Attention 的 KV 依然复制：下一层 Attention 前重新 all-gather token 行，不能把 SP 当成 cache 容量除以 p。

shared experts 在 SP 下使用复制权重，[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4MLP.__init__ · L133–L183](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L133-L183) 的 disable_tp=is_sequence_parallel；因此每卡计算不同 token，无需为 shared expert 单独相加。无 SP 时 shared MLP 用 TP 权重切分，归约位置需看 MoERunner 是否已经合并 routed 输出，见第 6 章。

## 5. 通信量：先统一单位与算法

从线性层的tensor_model_parallel_all_reduce继续往下，可到 [vllm/distributed/parallel_state.py · GroupCoordinator.all_reduce · L724–L745](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/distributed/parallel_state.py#L724-L745)，再到 [vllm/distributed/device_communicators/cuda_communicator.py · CudaCommunicator.all_reduce · L362–L435](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/distributed/device_communicators/cuda_communicator.py#L362-L435)。CUDA communicator会按输入与配置选择FlashInfer、自定义all-reduce、symmetric memory、PyNCCL或torch.distributed回退。模型只规定“这个partial必须求和”，并没有在RowParallelLinear里固定ring算法。trace应按实际设备通信实现判断。

以下都是 **[教学] ring 算法**，不是此环境捕获的 NCCL trace；custom all-reduce、tree、NVLS 等算法字节数可能不同。令 S=一份完整输入/输出 tensor 的 byte 数。BF16 每元素 2 B；FP8 数据为 1 B，但 scale/metadata 另加。

| collective | 逻辑 payload | 单 rank send | 单 rank recv | 全组 send 总量 |
| --- | --- | --- | --- | --- |
| All-Reduce | S（每 rank 的完整 partial tensor） | 2(p-1)S/p | 相同 | 2(p-1)S |
| All-Gather | S（最终完整 tensor），本地 S/p | (p-1)S/p | 相同 | (p-1)S |
| Reduce-Scatter | S（每 rank 的完整 partial tensor），结果 S/p | (p-1)S/p | 相同 | (p-1)S |

全组 recv 总量等于全组 send，send+recv 是两者之和，不能叫它唯一的“网络流量”再与 send 指标混用。MB=10^6 B，MiB=2^20 B。

[教学] T=1024,H=4096,BF16，S=8,388,608 B=8.388608 MB=8 MiB：

| p | All-Reduce 每 rank send / recv | 全组 send | AG 或 RS 每 rank send / recv |
| --- | --- | --- | --- |
| 2 | 8 MiB / 8 MiB | 16 MiB | 4 MiB / 4 MiB |
| 8 | 14 MiB / 14 MiB | 112 MiB | 7 MiB / 7 MiB |

对同一 S，AG+RS 的理论 ring 字节总量等于一次 AR；SP 的优势还在后续 activation、residual、MoE token 的分片，不是字节总量必然减半。小 decode 的通信常由延迟主导，不能用大 tensor 的带宽公式预测 TPOT。

## 6. 练习

沿用第 4 章的教学维度，p=2，列出每 rank 的 wq_b、wo_a、wo_b shape，解释 wo_b 为什么不可只 concat。再用上表手算 p=8 的 BF16 send+recv 全组总量；将元素改成 FP8 时先不计 scales，明确得到的是 data-only 数量。最后解释 TP=8、DP=1、非 Mega FusedMoE、EP 开启时是否自动启用 SP。

答案见 [答案第 05 节](answers.md#05)；下一章：[DP/EP/MoE](06_moe_parallel.md)。
