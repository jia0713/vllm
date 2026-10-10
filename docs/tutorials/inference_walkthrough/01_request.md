# 01 一个请求：从 prompt 到下一个 token

## 1. 问题与具体例子

先把 tokenizer 暂时当成“字符串到整数”的函数。短 prompt 的教学 token IDs 为 [11, 22, 33, 44]，不是 DeepSeek tokenizer 的真实编码。模型的任务是根据已知前缀预测下一 token 的分布。

**prefill**（预填充）处理已知 prompt：四个 token 的输入都已知，可以并行生成四行 hidden states。因果 Attention 仍限制第 j 行只能看它前面的合法位置，所以“同时计算”不意味着未来 token 可见。**decode**（生成阶段）通常每轮处理一个新 token：上一轮采样得到它之前，无法知道要对哪个 token 做 embedding。

kernel 工程师在 prefill 看到 M=4 的投影 GEMM，在单请求普通 decode 中看到 M=1；同一套权重的 kernel 效率会截然不同。完整 generate 是调度、forward、采样、停止判断组成的循环，而一次 forward 只是推进一次计算前沿。

## 2. 数据流

~~~mermaid
flowchart LR
  P["已知 IDs: 4 行"] --> E["Embedding: 4 x H"]
  E --> L["多层 Attention / MoE"]
  L --> X["Hidden: 4 x H"]
  X --> Z["末行 logits: 1 x V"]
  Z --> S["采样 token y0"]
  S --> D["下一轮输入 y0: 1 行"]
  D --> K["读取历史 KV，再写 y0 的 KV"]
  K --> S1["采样 y1"]
~~~

四行输出都可以计算 logits，但常规生成只需要末行预测第一个输出 token。prompt logprobs、投机验证可能要求更多行。不要把 logits 的行数固定等同于 T。

## 3. 真实调用入口

[代码] [vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4ForCausalLM.forward · L1878–L1888](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1878-L1888) 调用文本模型；[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4Model.forward · L1424–L1526](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1424-L1526) 完成 embedding、decoder layers、mHC head 和 final norm；[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4ForCausalLM.compute_logits · L1865–L1870](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1865-L1870) 进入 logits processor。mHC 是多路 residual 的混合机制，后续第 4 章说明；这里仍把送进 Attention/MoE 的工作张量记为 [T,H]。

整轮循环在 [vllm/v1/engine/core.py · EngineCore.step · L647–L676](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core.py#L647-L676)：先 schedule，再 executor.execute_model，必要时 sample_tokens，然后 scheduler.update_from_output。V1 runner 的 [vllm/v1/worker/gpu_model_runner.py · GPUModelRunner.execute_model · L4108–L4465](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py#L4108-L4465) 与 [vllm/v1/worker/gpu_model_runner.py · GPUModelRunner.sample_tokens · L4483–L4738](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py#L4483-L4738) 是分开的入口，不能因为 forward 返回了 hidden 就认为采样结束。

## 4. 输入、输出与 KV

| 时刻 | 本轮 input_ids / positions | 本轮 hidden | 已计算 KV | 本轮输出 |
| --- | --- | --- | --- | --- |
| 首轮 | [11,22,33,44] / [0,1,2,3] | [4,H] | 写位置 0–3 | 采样 y0 |
| 下一轮 | [y0] / [4] | [1,H] | 读 0–3，写 4 | 采样 y1 |
| 再下一轮 | [y1] / [5] | [1,H] | 读合法历史，写 5 | 采样 y2 |

**已采样的 token 不一定已有 KV**。刚生成 y0 时它尚未作为输入跑过网络，它的 KV 要到下一轮才产生。每层都有自己的 KV；在 V4.1 的压缩分支有跨层 source 共享，但不是所有层所有 cache 都共享。

本轮投影产生的 kv 是暂存 activation，持续存在的历史是 cache 中的记录。V4 的 [vllm/models/deepseek_v4/attention.py · DeepseekV4Attention._fused_qnorm_rope_kv_insert · L759–L847](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L759-L847) 明确说明 Attention 通过 SWA cache 读取 kv，本轮 kv 参数不是一份可直接代表全部历史的矩阵。

## 5. 首次、分块、前缀命中

[教学] 四 token prompt 被切成 2+2：

- 第一次 positions=[0,1]，只写这些位置，尚不生成有效输出；
- 第二次 positions=[2,3]，新 Q 读取第一个 chunk 的历史 cache，并写新 KV；最后一行才预测 y0；
- 第三次输入 y0。

[代码] runner 用 discard_request_mask 丢弃未完成 prefill 的采样结果，定位 [vllm/v1/worker/gpu_model_runner.py · GPUModelRunner._prepare_inputs · L1943–L2270](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py#L1943-L2270)。调度端用 is_prefill_chunk 表达前沿还未追上已知 token，定位 [vllm/v1/core/sched/scheduler.py · Scheduler._update_after_schedule · L1616–L1648](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/sched/scheduler.py#L1616-L1648)。

若复用前缀，只重算没有命中的后缀。缓存保存历史 K/V，不保存可直接替代最终 logits 的全部层末行 hidden，所以完整 prompt 命中也常需重算最后一行。[vllm/v1/core/kv_cache_manager.py · KVCacheManager.get_computed_blocks · L264–L321](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/kv_cache_manager.py#L264-L321) 将可复用上限限制为 num_tokens-1；按完整 block 查找的基线可能让实际重算多于一行。异构 cache、bounded replay 与远程 KV 还有额外约束，不把命中数简单当成 prompt 长度。

## 6. 计算与性能

一次线性投影 [T,H]×[H,N] 约 2THN 浮点运算；对单请求 T=1，权重读取很难被足够多行摊薄。prefill 增大 T 通常改善 GEMM 利用率，但 Attention 的历史访问还由上下文决定，不能仅靠 GEMM 的 M 判断整轮成本。

TTFT 是到第一个输出 token 的延迟；包含等待和多轮 chunked prefill。TPOT 是输出 token 之间的时间；包含后续调度、cache 读取、通信和采样。某个 GEMM 快 20% 不会直接让 TTFT/TPOT 都快 20%。

## 7. 练习

已知 prompt 长 4，第一轮只计算 2 token。第二轮补完 prompt 并采样 y0。写出三轮 positions、num_computed_tokens、已知 token 数，指出第三轮开始时哪个 token 有 ID 却还没有 KV。再解释为什么 prefix cache 不能直接给出最终 logits。

参考答案见 [答案第 01 节](answers.md#01)；下一章：[调度](02_scheduler.md)。
