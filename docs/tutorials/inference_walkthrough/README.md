# 从 GPU 算子到推理系统：vLLM 与 DeepSeek V4 / V4.1

这套课程面向熟悉 CUDA、GEMM、FlashAttention、Triton / TileLang 的工程师。目标是沿着一次请求，解释调度怎样产生 kernel shape，哪些数据属于哪个 rank，KV 如何跨轮存活，以及性能优化应该落在哪一层。

## 固定版本与证据边界

核查日期：2026-10-10（用户时区 Asia/Shanghai）。仓库 /workspace/vllm；remote 为 https://github.com/jia0713/vllm.git；branch 为 work；固定源码 commit 为 **10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf**。开始时工作区干净。教程仅新增本目录，没有修改框架实现或发布 PR。适用约定来自仓库根 AGENTS.md；docs 下未找到更细的 AGENTS.md。

教程自身可以在该源码提交之后新增 Git 提交。校验脚本允许当前 HEAD 是固定源码提交的后继，并逐文件核对引用源码的 SHA256；引用涉及的源码发生变化时仍会报错。

这是当前 fork 的实现快照，不把它自动等同于 upstream release 或模型官方设计。两版实现均已定位：[vllm/model_executor/models/registry.py · 模型注册项 · L95–L95](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/models/registry.py#L95-L95)；平台入口 [vllm/models/deepseek_v4/__init__.py · 平台导出分支 · L17–L17](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/__init__.py#L17-L17)、[vllm/models/deepseek_v41/__init__.py · 平台导出分支 · L9–L9](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v41/__init__.py#L9-L9)；配置 [vllm/transformers_utils/configs/deepseek_v4.py · DeepseekV4Config.__init__ · L11–L52](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/transformers_utils/configs/deepseek_v4.py#L11-L52)、[vllm/transformers_utils/configs/deepseek_v41.py · DeepseekV41Config.__init__ · L20–L62](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/transformers_utils/configs/deepseek_v41.py#L20-L62)。V4.1 的公开注册名是 DeepseekV41ForCausalLM，NVIDIA 文本子模型类名是 DeepseekV41LLMForCausalLM，Attention 内部仍有 DeepseekV4 类名；不能凭类名判断版本。

全文使用三种证据标签：

- **[代码]**：已阅读本地固定提交；链接提供路径、符号和版本行号。静态分支存在不代表已在 GPU 上运行。
- **[官方]**：实际读取并固定版本的外部资料。此环境核实了 DeepGEMM README 的 057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7 版本，见 [资料与限制](sources_and_limits.md)。
- **[教学]**：手算、合成 token、简化调度或假设维度；不能作为真实 checkpoint 的规格或性能结果。

没有得到 V4/V4.1 官方模型卡的固定快照，因而不填入猜测的 hidden size、层数、参数量。代码中的示例注释也不作为 checkpoint 配置。正文以实际配置字段的符号表达维度，数值练习另行标为教学值。

## 当前能运行什么

| 项目 | 核查结果 | 对教程的影响 |
| --- | --- | --- |
| CPU | x86_64，3 个可见逻辑 CPU | 可运行真实 scheduler、源码检查和手算验证 |
| Python / PyTorch | Python 版本见 validation/environment.json；torch 2.13.0+cpu | 没有 CUDA runtime 实测 |
| 其他依赖 | transformers 5.19.0、triton 3.8.0+cpu、vllm 0.1.dev22801+g10cc2f6ae.cpu、pre-commit 4.6.2 | 使用现有 .venv，经 uv 运行 |
| GPU | torch.cuda.is_available() 为 False，device_count=0；无 /dev/nvidia*，无 nvidia-smi | 单卡及 8 卡路径只做静态推导 |
| 模型权重 | /workspace 中未发现 V4/V4.1 safetensors；常见模型目录不存在；部分外部缓存不可读 | 不能断言机器其他位置绝对没有权重；不运行真实 DeepSeek forward |
| 已有 trace | scratch/continuous_batching/trace.json 是合成输出的 scheduler 日志；没有找到 nsys-rep | 不使用它宣称 GPU 耗时、带宽或重叠 |

## 主线约束

先学习 V4 的文本路径：**NVIDIA SM90 级硬件的静态分析、V1 GPUModelRunner、FlashMLA sparse DSV4、FusedMoE 路径**。明确设置 VLLM_USE_V2_MODEL_RUNNER=0，关闭 speculative decoding、异步调度、DBO、PCP/DCP、PP 和视觉输入。这样一轮对应一次清楚的 token 前沿推进。没有模型权重，不声称某个真实 checkpoint 已在该组合上部署成功。

MoE 先按普通 BF16 / FP8 的 FusedMoE 接口解释；具体量化方法是否允许某个专家 kernel 必须再由 checkpoint 与后端检查决定。第 7 章选 TP=2、DP=4、EP 开启、allgather_reducescatter 作为可由代码逐项推导的 8 卡布局，采用非 native-Mega 路径，不承诺特定量化 checkpoint 的 kernel 兼容性或容量。第 8、9 章才引入 MegaMoE、SM100 及 V2 的条件分支。

## 每天 1–2 小时的阅读路线

每章先读“问题与例子”，画出自己的图，再看源码，最后做练习。复杂章节拆成两个学习日：第一天推导 shape，第二天沿代码核对通信与 cache。

| 顺序 | 一次学习的核心问题 | 建议时间 |
| --- | --- | --- |
| [01 单请求](01_request.md) | 一个 forward 为什么不是完整 generate？ | 1 天 |
| [02 调度](02_scheduler.md) | 多个请求怎样变成 T 行输入？ | 1–2 天 |
| [03 KV 生命周期](03_kv_cache.md) | 一个位置怎样变成物理 slot？ | 2 天 |
| [04 Attention](04_attention.md) | 新 Q 如何访问旧 KV 和压缩索引？ | 2 天 |
| [05 TP](05_tensor_parallel.md) | 切分后为什么必须 collective？ | 2 天 |
| [06 DP / EP / MoE](06_moe_parallel.md) | token 和 expert 如何相遇？ | 2 天 |
| [07 八卡全链路](07_eight_gpu.md) | 一轮数据怎样走完所有 rank？ | 2 天 |
| [08 Stream 与重叠](08_streams.md) | 代码上的并行是否真的隐藏耗时？ | 1–2 天 |
| [09 两版对照](09_v4_vs_v41.md) | 跨层复用改变了哪些工作量？ | 2 天 |
| [10 trace 与进阶](10_trace.md) | 应该优化 kernel、batch 还是系统配置？ | 2 天 |

共约 17–19 个学习日；每天不必把一章全部读完。通用符号：R=本轮请求数，T=实际计算 token 总数，T_pad=graph/SP 对齐后的行数，H=hidden_size，V=vocab_size，h=num_attention_heads，d=head_dim，r_q=q_lora_rank，r_o=o_lora_rank，g=o_groups，I=moe_intermediate_size，E=n_routed_experts，k=num_experts_per_tok，p=Attention TP size。历史长度 L_i 与本轮 query length q_i 分开记。

## 可复现验证

在仓库根运行（不下载模型、不启动 GPU）：

~~~bash
UV_CACHE_DIR=/workspace/work/uv-cache uv run --no-sync .venv/bin/python docs/tutorials/inference_walkthrough/scripts/shape_lab.py
UV_CACHE_DIR=/workspace/work/uv-cache uv run --no-sync .venv/bin/python docs/tutorials/inference_walkthrough/scripts/scheduler_lab.py --out /workspace/work/inference_walkthrough/scheduler
UV_CACHE_DIR=/workspace/work/uv-cache uv run --no-sync .venv/bin/python docs/tutorials/inference_walkthrough/scripts/verify_tutorial.py
~~~

第一条是教学公式检查；第二条调用真实 Scheduler / KVCacheManager 并注入合成 ModelRunnerOutput，使用本地微型 OPT 配置仅避免模型下载，**不执行 OPT 或 DeepSeek 的模型计算**；第三条检查固定源码引用、局部链接、章节与练习完整性。实际结果见 [验证记录](validation/REPORT.md)。

[练习答案](answers.md) · [常见误区](misconceptions.md) · [资料与限制](sources_and_limits.md) · [源码定位表](source_index.md)
