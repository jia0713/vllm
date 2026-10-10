# 02 Scheduler：请求数如何变成算子的 T

## 1. 从两个 decode 加一个 prefill 开始

continuous batching（持续批处理）意味着每轮重新选择可推进的请求，完成请求退出，新请求加入，不等整个 batch 一起结束。它不是“一次把所有请求的完整序列送给 GPU”。

[教学] A、B 已经完成两 token 的 prompt 并各采样一个 token；P 有 10 token prompt 刚到。令 max_num_seqs=3，max_num_batched_tokens=6，max_model_len=64。A/B 的当前已知数都是 3，computed 都是 2；P 是 known=10，computed=0。

| 轮次 | 调度前 A/B computed | P computed | scheduled A/B/P | packed positions | 输出 |
| --- | --- | --- | --- | --- | --- |
| 2 | 2/2 | 0 | 1/1/4 | A:2；B:2；P:0–3 | A/B 各 1，P 无 |
| 3 | 3/3 | 4 | 1/1/4 | A:3；B:3；P:4–7 | A/B 各 1，P 无 |
| 4 | 4/4 | 8 | 1/1/2 | A:4；B:4；P:8–9 | 三者各 1 |

这三轮的请求数都是 3，但 T 分别是 6、6、4。如果 graph 把 T=4 补齐到某个捕获尺寸，kernel 实际看到的是 T_pad；padding 行不应写入真实 KV slot。后端可能重排请求，表中顺序只是便于演示。

## 2. 请求状态与数据流

在线请求先由 [vllm/v1/engine/async_llm.py · AsyncLLM.add_request · L382–L546](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/async_llm.py#L382-L546) 将原始prompt异步tokenize/预处理成EngineCoreRequest，并注册输出收集器；[vllm/v1/engine/core_client.py · AsyncMPClient.add_request_async · L1296–L1299](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core_client.py#L1296-L1299) 把ADD控制消息送到独立EngineCore进程；[vllm/v1/engine/core.py · EngineCore.add_request · L503–L547](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core.py#L503-L547) 最终调用scheduler.add_request。若用户已经提供token IDs或EngineCoreRequest，预处理分支不同。离线/in-process client可以省掉跨进程消息，所以不要把所有generate调用都当成HTTP RPC。

传递的是请求ID、token/参数与控制状态；embedding之后的GPU activation通信才是TP/EP层讨论的collective。API入口和worker之间还有engine控制链，源码中的“通信”要先区分传什么对象。

~~~mermaid
flowchart LR
  N["add_request"] --> W["waiting"]
  W --> C["prefix lookup / admission"]
  C --> S["schedule: token budget + allocate_slots"]
  S --> R["running / SchedulerOutput"]
  R --> G["runner: pack inputs + metadata"]
  G --> O["ModelRunnerOutput"]
  O --> U["update_from_output"]
  U --> R
  U --> F["finished: free request"]
  S --> X["KV 不足: preempt"]
  X --> W
~~~

[代码] 调度入口 [vllm/v1/core/sched/scheduler.py · Scheduler.add_request · L2615–L2646](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py#L2615-L2646)、[vllm/v1/core/sched/scheduler.py · Scheduler.schedule · L580–L1556](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py#L580-L1556)；执行循环 [vllm/v1/engine/core.py · EngineCore.step · L647–L676](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core.py#L647-L676)；回传更新 [vllm/v1/core/sched/scheduler.py · Scheduler.update_from_output · L2025–L2423](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py#L2025-L2423)；释放 [vllm/v1/core/sched/scheduler.py · Scheduler._free_request · L2713–L2743](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py#L2713-L2743)。

## 3. 先读这几行，再理解约束

[vllm/v1/core/sched/scheduler.py · Scheduler.schedule · L580–L1556](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py#L580-L1556) 的 running 部分包含：

~~~python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
~~~

普通同步、无投机时，就是 known-computed。num_tokens_with_spec 包括草稿 token；num_output_placeholders 用于尚未回传的输出占位，异步路径不能直接套我们的简单算式。

随后用剩余 token budget、input budget、long_prefill_token_threshold、max_model_len 等截断，并调用 allocate_slots。waiting 请求先做 prefix lookup 与准入判断，逻辑不完全等同 running 请求。代码注释明确没有两个互斥的“prefill phase / decode phase”；调度器要让 computed 追上已知前沿，Attention metadata builder 再按 query length 等条件选择 kernel 路径。

[vllm/v1/core/sched/scheduler.py · Scheduler._update_after_schedule · L1616–L1648](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py#L1616-L1648) 在 schedule 返回前增加 computed 和 in-flight 计数。这是乐观推进，不证明 GPU 已完成写 KV；投机拒绝、抢占或加载失败可回退。因此观察 trace 时必须注明状态抓取发生在 schedule 前、schedule 后还是 update 后。

## 4. 三个限制与四种 batch 口径

| 名称 | 控制什么 | 不控制什么 |
| --- | --- | --- |
| max_num_seqs | 可调度请求数上限；当前版本还需注意 active-seq 等策略 | 不是总 query token 数 |
| max_num_batched_tokens | 单轮可处理的 token 行预算 | 不是单请求最长上下文 |
| max_model_len | 单请求合法序列长度上限 | 不是 GPU 总 KV 容量 |
| R / 请求 batch size | 本轮活跃请求数 | 与 GEMM M 不总相等 |
| T=sum(q_i) | 本轮实际 token 总数，常是投影 GEMM 的 M | 与历史总 token 数不相等 |
| q_i | 每请求本轮 query length | 与该请求累计 seq_len 不相等 |
| T_pad | CUDA graph / SP / DP 约束的实际行数 | 包括不能计作有效 token 的 padding |

## 5. 混合 batch 的边界在哪里

A/B/P 第二轮 packed IDs 是 [A_y0, B_y0, P_0, P_1, P_2, P_3]，hidden 为 [6,H]。query_start_loc=[0,1,2,6] 给出请求边界；positions=[2,2,0,1,2,3]；seq_lens=[3,3,4]；每个 KV group 的 block_table 保存每个请求独立的物理页。

[vllm/v1/worker/gpu_model_runner.py · GPUModelRunner._prepare_inputs · L1943–L2270](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py#L1943-L2270) 用 computed+局部 offset 构造 positions，并以 cumulative token 数构造 query_start_loc；[vllm/v1/attention/backends/mla/sparse_swa.py · DeepseekSparseSWAMetadataBuilder.build · L575–L783](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/attention/backends/mla/sparse_swa.py#L575-L783) 区分 prefill/decode metadata。边界由这些 metadata 保持，Attention 不能跨请求读 cache。hidden 打包不是把三个请求拼成一条可相互 Attention 的长序列。

通信、graph 或 backend 重排也会改变 kernel 观察到的行序。分析输出必须沿 req_id_to_index 和 logits_indices 映射回请求，不能按 dictionary 顺序猜测输出归属。

## 6. 容量不足与性能决策

[vllm/v1/core/sched/scheduler.py · Scheduler._preempt_request · L1571–L1614](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py#L1571-L1614) 释放请求占有的 blocks，把状态改为 PREEMPTED，computed 重置为 0，放回 waiting；已生成的 token IDs 仍可用于重算。prefix caching 可能让重新进入时恢复部分前沿，所以“重置为 0”不代表最终一定全重算。异步路径有 stale output 与延迟释放保护，不能在仍有读者时复用页。

增大 token budget 会让 P 的 GEMM 更有效，却可能延长 A/B 的单轮等待；减小 prefill chunk 可以改善 decode 的服务延迟，却增加 launch/metadata 开销。max_num_seqs 已经够大但 KV 不够时，继续增加它可能只增加抢占。先看 TTFT、TPOT、排队时间和 preemption，再决定是否改 kernel。

## 7. 可复现练习

运行 README 的 scheduler_lab.py。它执行真实 Scheduler，但将计算结果替换成合成 token 7；validation/scheduler_mixed.json 是本次 CPU 执行记录。先不看输出，推导上表以及 P 的 query_start_loc。再把预算改成 4，解释 P 每轮只能获得几个 token。

另外观察脚本的 pressure 模式：找出 computed 回退的请求，区分“无 KV 准入等待”和“已运行请求被抢占”。prefix 模式则检查结束请求的 block 可以被命中复用。答案见 [答案第 02 节](answers.md#02)；下一章：[KV 生命周期](03_kv_cache.md)。
