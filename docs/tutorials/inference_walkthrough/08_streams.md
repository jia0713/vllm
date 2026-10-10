# 08 Stream、事件与通信计算重叠

## 1. 先区分提交时间和执行时间

CPU 调用 CUDA op 多数是在 stream 上排队，并不等待执行完。CPU trace 里相邻函数返回只证明提交顺序，不能证明 GPU kernel 已完成。一个 stream 上普通 kernel 按顺序执行；不同 stream 可同时推进，但必须显式建立读写依赖，还会竞争 SM、寄存器、HBM、L2 与通信链路。

[教学] Q 投影耗时 a，compressor 耗时 b，随后 Attention 要读两者。串行耗时约 a+b；理想并行约 max(a,b)，还需加同步与资源竞争。把两个独立 compute-bound GEMM 放不同 stream 上，可能没有任何收益。

## 2. V4 的真实 fan-out / join

~~~mermaid
flowchart LR
  X["x 已就绪"] --> E["main 记录 start event"]
  E --> Q["main: Q 投影 / SWA insert"]
  E --> I["aux0: indexer preparation"]
  E --> C["aux1: compressor"]
  Q --> J["main 等待 aux done events"]
  I --> J
  C --> J
  J --> S["score / top-k -> sparse Attention"]
~~~

[vllm/utils/multi_stream_utils.py · execute_in_parallel · L66–L141](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/utils/multi_stream_utils.py#L66-L141) 的 execute_in_parallel 先 record start，再在 aux stream 上 wait；每条 aux 链结尾 record done；main stream 等 done 后才消费结果。event.wait 是往目标 stream 插入依赖，不是要求 CPU 在函数处忙等。

[vllm/models/deepseek_v4/attention.py · DeepseekV4Attention._run_parallel_input_projections · L672–L731](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L672-L731) 可把 fused_wqa_wkv 放 main，其余 compressor/indexer 投影放 aux0–2，且受 VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD 控制。[vllm/models/deepseek_v4/attention.py · DeepseekV4Attention._prepare_and_attn · L568–L645](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L568-L645) 又让 Q/SWA、indexer preparation 和主 compressor 分支重叠。这里 aux stream 可能复用，但同一阶段的 join 保证前一批工作结束；不能无视源码 event 直接把“有三条 stream”推成“三倍性能”。

[vllm/utils/multi_stream_utils.py · maybe_execute_in_parallel · L16–L63](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/utils/multi_stream_utils.py#L16-L63) 检测 BreakableCUDAGraphCapture.is_active 时会退回顺序路径；execute_in_parallel 的 enable、stream 列表也控制退化行为。图捕获模式、T 档位、平台、配置不同，重叠并不必然相同。

## 3. event、collective 和 barrier 的区别

| 机制 | 保证的关系 | 不保证什么 |
| --- | --- | --- |
| stream 内普通顺序 | 后面的 kernel 消费前面结果 | 不保证另一个 stream 已完成 |
| event record + wait | wait stream 的后续任务等待特定生产点 | 不自动等所有 rank，也不自动等 CPU |
| CPU synchronize / event.synchronize | CPU 等待设备或事件完成 | 不应被误当成免费操作 |
| collective | 参与 rank 执行兼容交换/归约，结果遵循流依赖 | 不要求所有 GPU 同一时刻进入/退出，不等同全局设备 barrier |
| distributed barrier | 按实现等待通信组成员到达 | 不自动使所有无关 CUDA stream 的工作都结束 |

All-Reduce 在某个层边界有真实数据依赖：下一非线性要用完整结果。但其他 stream 的独立工作、其他 microbatch 的 Attention，可能仍可继续。看到 EP communication 不能推导“全局所有计算必须停止”。

## 4. shared expert overlap：当前代码有哪些条件

[vllm/model_executor/layers/fused_moe/runner/shared_experts.py · SharedExperts._determine_shared_experts_order · L101–L121](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/runner/shared_experts.py#L101-L121) 根据 backend 能力、CUDA-like 平台、aux stream、token threshold、EPLB 安全性和 sequence-sharded shared layer 等决定执行方式。可能是 NO_OVERLAP、MULTI_STREAM_OVERLAPPED 或 modular kernel 内部安排的重叠。

[vllm/model_executor/layers/fused_moe/runner/shared_experts.py · SharedExperts.maybe_forward_async · L123–L142](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/runner/shared_experts.py#L123-L142) 在 aux 上计算 shared branch；[vllm/model_executor/layers/fused_moe/runner/shared_experts.py · SharedExperts.wait · L144–L147](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/runner/shared_experts.py#L144-L147) 让 main 等到结果可用。[vllm/model_executor/layers/fused_moe/runner/moe_runner.py · MoERunner.forward · L688–L797](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L688-L797) 在 routed expert 路径中调用该机制，最终加 shared/routed 输出。此路径有代码支持，但第7章的 SP shared 是否触发 aux overlap，要检查 _disable_shared_experts_overlap 的条件；不能将 generic capability 自动当成选定配置已生效。

V4 native MegaMoE 的分支 [vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4MoE.forward · L896–L992](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L896-L992) 如果 shared 没 fused，会在 routed 返回后顺序调用 shared；它与上述 MoERunner aux 路径不同。第三方 DeepGEMM 的 [官方固定版 README](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/README.md) 描述 MegaMoE 在一个 mega-kernel 内融合 dispatch、两次 GEMM、SwiGLU 和 combine。**多 stream 的 shared overlap**与**单 kernel 内 NVLink/计算重叠**是两种机制。

## 5. DBO / microbatch 不代表每种后端都支持

DBO 是 dual batch overlap：将一轮划成 microbatch，让一个 microbatch 的通信与另一个的计算交错。[vllm/v1/worker/ubatching.py · UBatchContext · L20–L145](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/ubatching.py#L20-L145) 的 yield_and_switch_from_compute_to_comm、反向切换与 events 是实际调度机制；MoE prepare/finalize 的 async/recv hook 能力决定可在哪些位置让出。

~~~mermaid
flowchart LR
  A0["microbatch0 计算"] --> C0["microbatch0 通信"]
  A1["microbatch1 计算"] --> C1["microbatch1 通信"]
  C0 --> A0N["microbatch0 后续计算"]
  C1 --> A1N["microbatch1 后续计算"]
~~~

图中 C0 与 A1 可有重叠空间，但仅是依赖图上的可能性。backend、图模式、DP 对齐、threshold 与 kernel 资源决定实际收益。主线关闭 DBO；不把它描述成此环境已测试的 V4 全路径特性。V4.1 decoder bounded replay 明确排除 use_ubatching / SP / PCP 等组合，见第9章。

## 6. 如何从 GPU trace 验证重叠

需要真实设备采集后才能给时间结论：

1. 标注 rank、stream、iteration、microbatch、有效 T / padded T、kernel 参数；先排除不同轮次错误对齐。
2. 查看 GPU kernel 时间区间是否相交，不能只看 CPU 调用区间相交。
3. 找到 event wait、NCCL/通信 kernel 及生产/消费链。空白可能是等待，也可能是 CPU 未及时提交。
4. 比较打开/关闭功能的相同负载端到端时间，记录 warmup、capture 与 steady state。
5. 测 SM/HBM/链路指标，判断并行时双方是否变慢，最终关键路径是否缩短。

截图可证明“展示的时间窗内某两个 GPU 活动区间交叠”，不能证明所有 rank 同样交叠、通信全被隐藏、数值正确或吞吐提升。必须保留原 trace、配置与测量脚本。

## 7. 练习

教学时间窗：main Q=[0,5]、aux compress=[1,7]、Attention=[7,11]。计算 Q 与 compress 的重叠、总关键路径；与顺序 5+6+4=15 比较。若并行资源竞争把两支结束拖到9与10，是否仍应只报告“重叠5单位时间”作为收益？答案见 [答案第08节](answers.md#08)；下一章：[两版对照](09_v4_vs_v41.md)。
