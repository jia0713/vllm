# 常见误区与定位方法

| 容易误读的说法 | 正确理解 | 回看 |
| --- | --- | --- |
| 采样出的token已经有KV | 下一轮作为输入forward才生成KV | 01 |
| computed增加代表GPU完成 | scheduler可乐观推进，in-flight/拒绝/抢占会修正 | 02 |
| batch size=GEMM M | R、T、q_i、T_pad、expert M_e不同 | 02、06 |
| max_model_len决定并发 | 并发还受各KV group、workspace、权重、准入限制 | 03 |
| free就会cudaFree/清零 | 池页引用归还，prefix内容可保留 | 03 |
| FP8 KV必为head_dim字节 | packed record含RoPE/scale/alignment | 03、09 |
| 所有DeepSeek MLA都缓存同一格式 | V4共享KV行与V2/V3公式不能混用 | 04 |
| indexer TopK选expert | Attention TopK选历史/压缩位置，MoE TopK选expert | 04、06 |
| 一个forward是一个fused kernel | 投影、cache、indexer、Attention、O各有调用；融合边界需实证 | 04 |
| TP所有权重/KV都除以p | replicated Q/KV投影、indexer和共享KV是反例 | 05 |
| logits总是All-Gather全vocab | Gather/All-Gather/专用候选路径取决于平台与采样 | 05 |
| DP×TP×EP是卡数 | EP可由DP/PCP/TP合并，expert TP随之变化 | 06 |
| shared expert也占TopK | shared对每token执行，routed与redundant语义不同 | 06 |
| EP必用稀疏All-to-All | AG/RS基线也能实现dispatch/combine | 06 |
| SP就是历史KV分片 | 此模型SP切本轮token行；DCP是另一机制 | 05、07 |
| 每rank自己的scheduler | TP同组合作一个DP副本的调度，DP副本才处理不同请求 | 07 |
| collective是所有GPU同时进出的全局barrier | 集体数据交换和流依赖，不禁止独立工作的推进 | 08 |
| 两个stream就有性能收益 | 还要看event、资源竞争、实际GPU区间和关键路径 | 08 |
| NCCL bar少了就没有通信 | 通信可进入custom/fused kernel | 08、09 |
| V4.1 c=1也是SWA-only | V4.1只有c=0是pure SWA | 09 |
| source-sharing复用全部layer状态 | compressed KV/index可复用，Q/SWA/mHC仍按层执行 | 09 |
| kernel快20%，系统必快20% | 服务关键路径与Amdahl比例决定收益 | 10 |

分析时始终附上模型/代码版本、实际backend、valid/padded shape、rank group与数据类型。表中纠正的是概念，不替代某个运行配置的逐项证据。
