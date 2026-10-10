# 资料来源、证据等级与未验证事项

## 本地固定版本

主来源为 /workspace/vllm，commit **10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf**。引用链接固定到该fork提交；离线阅读可用source_index.md中的路径/符号/行号在本地定位。source_refs.json记录每条引用的源码SHA256和声明行，用脚本检查引用没有指向其他版本或不存在的符号。

已实际阅读系统调度、KV pool/manager、V1 runner/worker/executor、parallel groups、线性层/vocab、MoE prepare/finalize、Stream与ubatching，以及两版独立Attention、compressor、NVIDIA模型和FlashMLA实现。源码定位表覆盖正文关键入口；本次没有执行完整DeepSeek模型。

根AGENTS.md要求Python经uv与.venv运行，遵守88字符风格与仓库lint约定。本次利用已安装环境，不重新安装GPU依赖。仓库已有pre-commit与commit-msg hooks，但初始化检查环境所需的网络访问失败，完整lint未完成，详见validation/REPORT.md。无PR，贡献前的PR检查流程不适用于本次本地教程交付。

## 官方资料：已核实与仅有指针

已通过GitHub连接读取 [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/README.md)，固定commit **057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7**，提交日期2026-09-30；资料说明SM90/SM100要求、量化scale布局、grouped GEMM、indexer与MegaMoE单kernel内通信计算重叠。本教程不引用其benchmark数值为当前vLLM结果，也不声称本地安装的DeepGEMM就是该revision。

检索候选官方GitHub地址 deepseek-ai/DeepSeek-V4 与 deepseek-ai/DeepSeek-V4.1 得到404。这只说明这两个具体地址未读到，不证明官方模型/资料不存在。仓库含Hugging Face官方模型指针：

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro ：由V4 kernel源码注释提供；
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash ：由docs/features/engram.md提供。

这些指针**本次没有获取内容或固定revision**，不能拿来证实模型维度、层拓扑、设计或性能。环境HTTP策略没有列入huggingface.co，命令行代理连接也失败；没有尝试绕过网络策略。本地实现可读，因此继续完成所有不依赖模型卡的章节。

## 实验的三种边界

| 验证 | 做了什么 | 没有证明什么 |
| --- | --- | --- |
| 静态源码引用/调用链 | 确认路径、声明、版本、分支条件及必要调用点 | runtime一定选中该backend |
| 真实CPU Scheduler | 调用真实Scheduler/KV管理，合成采样输出，验证mixed/pressure/prefix | GPU cache内容、kernel正确性、真实模型精度/吞吐 |
| CPU教学计算 | slot/shape/rank/通信字节/trace算术 | GPU传输算法、实际耗时、真实checkpoint合法性 |

chapter引用的语义权重shape忽略packed storage、scales、padding和transpose的物理细节；第4章给出逐项区分方法。所有通信表指定BF16/FP8、data-only口径与ring假设；没有捕获NCCL实际算法。

## 后续需要真实设备与权重验证的问题

1. 固定V4/V4.1官方checkpoint config和权重revision，导出H/h/g/r_q/r_o/E/k/I/source IDs；核对整个模型是否满足选定TP/量化后端。
2. 在SM90或SM100加载模型，记录实际Attention/MoE/通信backend、runner代际、KV specs和真实page_size_bytes。
3. 跑prefill/decode及长上下文混合batch，验证output/精度；用真实trace测KV读取、indexer/gather、MoE负载、padding与launch。
4. 验证TP2/DP4/EP8的容量、拓扑和all2all行为；比较替代backend时同时记录metadata/scale字节。
5. 对V4.1 source sharing、candidate filtering、replay、Engram确认config与runtime条件，而不是使用源码注释里的示例层ID。
6. 独立验证shared overlap/DBO/MegaMoE的正确性与端到端收益；不以函数名或截图代替对照实验。
7. 若换ROCm、XPU、CPU或V2 runner，重新追分支；此NVIDIA/V1教程没有为这些平台做GPU验证。

## 阅读材料的用途

仓库内 [DP/EP部署文档](../../serving/expert_parallel_deployment.md)、[Engram](../../features/engram.md)、[KV offload](../../features/kv_offloading_usage.md)、[PD分离](../../features/disagg_prefill.md)、[speculative decoding](../../features/speculative_decoding/README.md)可用于后续扩展。它们说明功能入口，不替代固定模型的支持矩阵。第10章提供每个主题与主线的连接问题，避免一次展开所有机制。
