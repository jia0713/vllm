# 06 DP、EP 与 MoE：让 token 找到 expert

## 1. 先把五个概念分开

Data Parallel（数据并行，DP）在推理里让 Attention 副本处理不同请求，通常每个副本有自己的 scheduler 和 KV。Expert Parallel（专家并行，EP）把不同 expert 的完整权重放到不同 rank，token 要去 expert 所在的卡计算再回来。MoE 是 mixture of experts：每个 token 只激活部分 routed experts，另可有 shared experts。

| 名称 | 切分对象 | 一轮中的意义 |
| --- | --- | --- |
| Attention TP | Attention heads、若干投影权重、vocab 权重 | 同一个请求在多个 rank 合作 |
| Attention DP | 请求 / batch / KV 所属副本 | 每副本有不同的 T 和历史 |
| EP | expert ID 与完整 expert 权重 | 多个请求副本的 token 共同利用 experts |
| expert 内 TP | 单 expert intermediate 维度等 | 每 rank 算 partial，输出要归约 |
| SP | Attention TP group 内 packed token 行 | 避免 MoE 重复处理同一 token |

DP 副本对 Attention 独立，并不代表它们的 MoE 没有通信。

## 2. 真实 rank group：EP 不是额外乘一次

[代码] [vllm/distributed/parallel_state.py · initialize_model_parallel · L2056–L2363](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/distributed/parallel_state.py#L2056-L2363) 的 rank 布局顺序为 ExternalDP×DP×PP×PCP×TP。本章令 ExternalDP=PP=PCP=1，global_rank=dp_rank*p+tp_rank。

TP=2、DP=4：

~~~mermaid
flowchart TD
  E["EP group: ranks 0..7"] --> D0["DP replica 0 / TP group [0,1]"]
  E --> D1["DP replica 1 / TP group [2,3]"]
  E --> D2["DP replica 2 / TP group [4,5]"]
  E --> D3["DP replica 3 / TP group [6,7]"]
  D0 --> P0["DP group tp=0: [0,2,4,6]"]
  D1 --> P1["DP group tp=1: [1,3,5,7]"]
~~~

[vllm/model_executor/layers/fused_moe/config.py · FusedMoEParallelConfig.make · L1108–L1236](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/config.py#L1108-L1236) 先将 Attention 的 TP×DP×PCP 合并。开启 EP 后 ep_size=该合并尺寸、expert tp_size=1；不开 EP 时 expert tp_size 反而可为这个合并尺寸。因此 2×4×EP8 不是 64 卡，这里总共只有 8 卡。DP group 的 [0,2,4,6] 与 [1,3,5,7] 是进程通信分组，不是另外 2 个 Attention 副本。

## 3. 四 token、八 expert、TopK=2 的完整路由

[教学] E=8、k=2、H=4、I=6，EP=4，每 rank 持有连续两个 experts：

| token / 原 owner | expert IDs | weights | dispatch 目的 rank |
| --- | --- | --- | --- |
| a / rank0 | 0,5 | 0.7,0.3 | 0,2 |
| b / rank1 | 2,5 | 0.4,0.6 | 1,2 |
| c / rank2 | 3,7 | 0.2,0.8 | 1,3 |
| d / rank3 | 0,6 | 0.5,0.5 | 0,3 |

每个 expert 的 M 分别为 [2,0,1,1,0,2,1,1]；assignment 总数=8，但各 rank 收到的不同 token 行、每个 expert 的 M、源请求数是三个不同口径。

~~~mermaid
flowchart LR
  T["源 rank 的 token"] --> G["gate / TopK IDs + weights"]
  G --> D["dispatch activation + routing metadata"]
  D --> W["按 expert 分组 / 对齐 M"]
  W --> L1["GEMM1: M_e x H -> M_e x 2I"]
  L1 --> A["SwiGLU: M_e x I"]
  A --> L2["GEMM2: M_e x I -> M_e x H"]
  L2 --> C["weighted sum / combine"]
  C --> O["原 owner 的 token 输出"]
~~~

a 的结果为 0.7*f0(a)+0.3*f5(a)，如果还有 shared branch，再加 shared(a)。本地 expert 不关心 token 来自哪个 DP 的 Attention；它只需要正确 activation 和路由权重，combine 保持 owner/row 对应关系。

## 4. 路由与权重的代码入口

[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4MoE.__init__ · L661–L794](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L661-L794) 读取 E、k、I、scoring_func、norm_topk_prob；GateLinear 的输出 [T,E] 是 FP32 logits。默认 scoring_func 为 sqrtsoftplus，不能把所有 DeepSeek 路由都讲成 V3 的 sigmoid / grouped top-k。

前 num_hash_layers 层可以由 input_ids 查 tid2eid，后续 learned routing 可使用 correction bias；[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4MoE.forward · L896–L992](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L896-L992) 要求 hash 路由具有 input_ids。教学表直接指定 expert IDs，目的是推导 dispatch，不模拟训练出的 gate。

[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4MoE._init_fused_moe_experts · L848–L894](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L848-L894) 将配置交给 FusedMoEFactory，[vllm/model_executor/layers/fused_moe/layer.py · FusedMoEFactory · L88–L441](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/layer.py#L88-L441) 构造 expert storage、router、quant method、prepare/finalize 和 MoERunner。[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4MoE._forward_fused_moe · L994–L1004](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L994-L1004) 中 router_logits=hidden_states 是 gate fusion 的输入接口，不能将它误认为 shape=[T,E] 的最终 logits 已由模型外层计算完成。

普通逻辑权重：

| 模式 | W_gate_up | W_down | 本地 expert 数 |
| --- | --- | --- | --- |
| EP 开、expert TP=1 | [E_local,2I,H] | [E_local,H,I] | physical E / EP size，取决于 placement |
| EP 关、expert TP=q | [E,2I/q,H] | [E,H,I/q] | E，各 rank 持所有 expert 的分片 |
| shared + SP | [2I_shared,H] | [H,I_shared] | shared 整体复制 |
| shared 无 SP | [2I_shared/p,H] | [H,I_shared/p] | 按 Attention TP 切 |

这是语义 shape；FP4 packed storage、zero padding、scale block 不在表内。[vllm/model_executor/layers/fused_moe/config.py · FusedMoEParallelConfig.make · L1108–L1236](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/config.py#L1108-L1236) 与 [vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4MLP.__init__ · L133–L183](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L133-L183) 是对应依据。

## 5. dispatch/combine 不只有一种算法

选择 **allgather_reducescatter**：[vllm/distributed/device_communicators/all2all.py · AgRsAll2AllManager.dispatch · L128–L161](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/distributed/device_communicators/all2all.py#L128-L161) All-Gatherv activation、topk weights、IDs 和可选 extras，各 rank 看见汇总 token，但只计算本地 expert 的贡献；[vllm/distributed/device_communicators/all2all.py · AgRsAll2AllManager.combine · L163–L173](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/distributed/device_communicators/all2all.py#L163-L173) Reduce-Scatterv 合并贡献，输出回到原有 token 分片。SP 时 [vllm/distributed/device_communicators/all2all.py · AgRsAll2AllManager._get_comm_group · L76–L86](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/distributed/device_communicators/all2all.py#L76-L86) 选择 EP group；非 SP 时可能选 DP/PCP group，随后还需处理 TP reduction。不能仅因为 EP group 存在，就断言每层调用同一种 all-to-all。

[vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py · MoEPrepareAndFinalizeNaiveDPEPModular.prepare · L136–L208](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py#L136-L208) 包含量化、dispatch；[vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py · MoEPrepareAndFinalizeNaiveDPEPModular.finalize · L210–L245](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py#L210-L245) 做 weighted reduction 与 combine_into_output。真正 sparse All-to-All 的 backend 则把 activation 只送到所需专家 rank，可能对同目的 rank 的多个专家去重。DeepEP / FlashInfer / native MegaMoE 的数据格式与同步不同，不能把它们的流量公式套给 AG/RS。

shared experts 每个 token 都计算，不占用 routed TopK 名额；冗余 experts 是 EPLB（expert parallel load balancing）的物理副本，logical E 与 physical E 不同。增加冗余副本不是增加模型新函数，也不是让每个 token 多选一次 expert。[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4MoE._init_fused_moe_experts · L848–L894](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L848-L894) 显式计算 physical=logical+redundant；负载表与映射更新必须保持一致。

无 SP 的 shared 可能是 TP partial：[vllm/model_executor/layers/fused_moe/runner/moe_runner.py · MoERunner._maybe_reduce_shared_expert_output · L429–L451](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L429-L451) 只在 routed 已归约、shared 未归约时单独 All-Reduce，否则可在合并结果后统一归约。不能固定给每个 shared MLP 加一次 AR，重复归约会改变数值。

## 6. 计算量与通信量

GEMM1 (M,N,K)=(M_e,2I,H)，GEMM2=(M_e,H,I)，两次约 6*M_e*H*I FLOPs（激活另计）；总 M_e=T*k，但 expert padding 增加实际算量。decode 中很多 expert M=0/1，grouped kernel 的效率与 load imbalance 很关键；prefill 的 M_e 通常更大。

对上表 BF16 activation、EP=4：源数据总量 S=T*H*2=32 B。均匀环形 AG 每 rank send=recv=24 B，全组 send=96 B；RS 同样 24 B/rank、96 B 全组。**仅 activation payload**，不计路由元数据；k=2 的 FP32 weights 与 int32 IDs 总量=4*2*(4+4)=64 B，若同样 AG，另加每 rank send=recv=48 B、全组 send=192 B。FP8 activation data-only 减半，但 scales、IDs 和 weights 不减半。

如果改为教学的按 assignment 发送的 sparse dispatch，上表远端 assignments=5，BF16 remote activation send 总量=5*H*2=40 B，remote combine 回传也为40 B；自本地处理的三条不计上网。每 rank remote send 分别为8/8/16/8 B，remote recv 为8/8/16/8 B；全组recv为40 B，dispatch的send+recv共80 B。这是指定路由、无目的端去重的逻辑边 payload，不是某个 NCCL/DeepEP 算法实测字节。

## 7. 练习

推导 rank2 上 expert5 的两次 GEMM M/N/K；计算全组有效 assignment 数及 padding 前工作量。将 Attention TP=2、DP=4、PCP=1 代入配置，分别给出 EP 开/关时 expert TP/EP size。再比较 AG/RS 与 sparse dispatch 的流量，解释为什么 k 不会乘到 AG 的 activation 总量上。

答案见 [答案第 06 节](answers.md#06)；下一章：[八卡全链路](07_eight_gpu.md)。
