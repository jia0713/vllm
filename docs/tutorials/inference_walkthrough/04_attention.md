# 04 单卡 V4 Attention：Q、窗口、压缩历史与索引

## 1. 先问“这个 Q 能读什么”

从一个 C4A layer 入手：本轮输入 x=[T,H]，每个 token 生成多头 Q；历史长程 KV 每 4 个 token 形成压缩记录，再用 indexer 选出重要记录；最近窗口走 SWA。indexer 的 TopK 是**历史位置/压缩记录索引**，不是 MoE 的 expert ID，也不是 vocabulary token ID。

[教学] 历史长度 20，c=4，所以有 5 条完整压缩记录。令窗口 W=4，index_topk=2，某个 query 选择压缩位置 [1,4]；它还要看最近合法 SWA 位置。Attention 访问的不是“原始 token 1 和 4”，而是这些位置对应的压缩记录。模型的压缩含学习到的权重及可能重叠，不把一条记录解释成简单平均值。

## 2. 数据流与依赖

~~~mermaid
flowchart TD
  X["x: T x H"] --> F["fused_wqa_wkv: T x (rq+d)"]
  F --> N["split + fused Q/KV RMSNorm"]
  N --> Q["wq_b: T x h x d"]
  Q --> R["Q norm / RoPE + SWA KV 插入"]
  X --> CS["主 compressor 投影 / partial states"]
  CS --> CC["压缩 Norm / RoPE / quant / cache insert"]
  X --> IS["indexer 投影 / 独立 compressor"]
  N --> IQ["index Q"]
  IS --> IK["index K cache"]
  IQ --> TK["score 历史 index K + top-k"]
  IK --> TK
  TK --> A["sparse MQA + SWA"]
  CC --> A
  R --> A
  A --> O["inverse RoPE / quant -> grouped wo_a -> wo_b"]
~~~

这张图只说明数据依赖。哪条支路实际并行、是否 fused、是否被 graph capture 改变，需要继续追实现，第 8 章展开。

## 3. 真实代码：从 outer forward 走到 kernel

[vllm/models/deepseek_v4/attention.py · DeepseekV4Attention.__init__ · L195–L469](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L195-L469) 从 config 读取 H、h、d、r_q、r_o、g、W 和各层 compress_ratios；先断言 h 可被 TP size 整除。[vllm/models/deepseek_v4/attention.py · DeepseekV4Attention.forward · L471–L501](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L471-L501) 的关键顺序是：

~~~python
qr_kv, kv_score, indexer_kv_score, indexer_weights = (
    self._run_parallel_input_projections(hidden_states)
)
qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
self._prepare_and_attn_fn(...)
return self._o_proj(attn_out, positions)
~~~

第一行并行准备多个投影结果；第二行才做 Q/KV RMSNorm；第三行构造 Q、写 cache、运行 indexer 与 Attention；最后是 output projection。这是多个真实调用，不能把整个 forward 描述成一个 fused kernel。

[vllm/models/deepseek_v4/attention.py · DeepseekV4Attention._prepare_and_attn · L568–L645](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L568-L645) 的 project_query_and_cache_kv 调用 wq_b 并 reshape，再进入 [vllm/models/deepseek_v4/attention.py · DeepseekV4Attention._fused_qnorm_rope_kv_insert · L759–L847](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L759-L847)。FlashMLA packed 路径用一个 CUDA op 同时完成 Q per-head RMSNorm/RoPE/padding 和 KV RoPE/quant/paged insert。它是**真正的横向融合**。[vllm/models/deepseek_v4/compressor.py · DeepseekCompressor.forward · L354–L499](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/compressor.py#L354-L499) 先保存 partial states，再用 fused compressor 完成 compress、Norm、RoPE、quant、cache write；两个步骤仍是独立 launch。

[vllm/models/deepseek_v4/attention.py · DeepseekV4Indexer.forward · L1069–L1138](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L1069-L1138) 准备 index Q、index K 与权重；[vllm/models/deepseek_v4/attention.py · DeepseekV4Attention._sparse_indexer_and_attn · L734–L757](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L734-L757) 调 indexer_op 产出 topk_indices_buffer，再调用 forward_mqa。indexer cache 与主 Attention cache 内容不同，不能互换。

## 4. 单卡 shape：先保留配置符号

| 张量 / 权重的逻辑 shape | 输入/输出 | 是否持久 |
| --- | --- | --- |
| fused_wqa_wkv weight=[r_q+d,H] | [T,H] -> [T,r_q+d] | 权重 |
| qr=[T,r_q]；kv=[T,d] | fused RMSNorm 后的两支 | 本轮 activation |
| wq_b weight=[h*d,r_q] | [T,r_q] -> [T,h,d] | 权重 |
| padded Q / Attention out=[T,h_pad,d] | FlashMLA h_pad=64 或 128 | 本轮，padding head 最后切掉 |
| main kv_score=[T,2*coff*d] | coff=2（c=4），否则 1 | 投影中间值为 FP32 |
| index Q=[T,h_idx,d_idx] | index wq_b 复制投影 | 本轮量化 activation 与 scale |
| index weights=[T,h_idx] | hidden -> weights_proj | 本轮 |
| top-k=[T,index_topk] | int32，填充无效项另有 length | 本轮共享 buffer |
| wo_a 的 grouped weight=[g,r_o,h*d/g] | 每组输入 [T,h*d/g] -> [T,r_o] | 权重 |
| wo_b weight=[H,g*r_o] | [T,g*r_o] -> [T,H] | 权重 |

逻辑 shape 不是低精度 packed 权重的 storage shape：FP4 会打包两元素，block-FP8/MX 还有 scales 和布局变换。优化 kernel 前要同时记录 semantic dimensions、strides、storage dtype、scale shape。[vllm/models/deepseek_v4/nvidia/ops/o_proj.py · deep_gemm_fp8_o_proj · L35–L103](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/ops/o_proj.py#L35-L103) 中 wo_a 标为 BMM，FP8 分支走 fp8_einsum，再 flatten 给 wo_b，不是普通 dense [h*d,g*r_o] 矩阵。

mHC 在 Attention/FFN 周边保存 [T,hc_mult,H] residual、[T,hc_mult,1] 的 post mix 和 [T,hc_mult,hc_mult] 的 residual mix，但通过 pre 函数归约成工作输入 [T,H]。[vllm/models/deepseek_v4/nvidia/model.py · DeepseekV4DecoderLayer.forward · L1188–L1282](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/model.py#L1188-L1282) 展示 pre、fused post-pre 以及 FFN。不能将残差放大倍数误乘到所有 Attention GEMM 的 H。

## 5. prefill / decode 的真实差异

[vllm/models/deepseek_v4/nvidia/flashmla.py · DeepseekV4FlashMLAAttention.forward_mqa · L87–L172](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/flashmla.py#L87-L172) 按 metadata 分出 prefill 与 decode tokens。prefill 在 [vllm/models/deepseek_v4/nvidia/flashmla.py · DeepseekV4FlashMLAAttention._forward_prefill · L266–L398](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/flashmla.py#L266-L398) 中将压缩/SWA cache 解量化 gather 到 BF16 workspace，组合索引后调用 flash_mla_sparse_fwd；decode 在 [vllm/models/deepseek_v4/nvidia/flashmla.py · DeepseekV4FlashMLAAttention._forward_decode · L174–L264](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/flashmla.py#L174-L264) 中通过 block table 把局部压缩索引转成物理全局索引，并调用 flash_mla_with_kvcache 读 packed cache。

C4A 使用 indexer 产出的稀疏索引；C128A 使用 metadata 构造的压缩历史索引；SWA-only 没有长程压缩 cache。不能说所有压缩比都先做 top-k 搜索。FlashMLA planner metadata 可在本轮同类型 layer 之间复用，但这不是 V4 的 KV 跨层共享，也不代表 indexer 分数结果可以跨层随便共享。

V1 使用较宽的 eager region；V2 可以把部分准备放进 captured graph。[vllm/models/deepseek_v4/attention.py · DeepseekV4Attention.__init__ · L195–L469](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L195-L469) 的 _prepare_and_attn_fn 条件解释这一点。模型 runner 的代际与 engine 的 V1 名称是不同维度，不能把“v1 engine”自动解释成旧 runner。

## 6. 工作量与优化位置

投影主要随 T 增长；SWA 的 Attention 访问随合法窗口大小增长；indexer 打分依赖可见压缩历史规模，约 T*h_idx*d_idx*(L/c) 的点积工作量；主 sparse Attention 约 T*h*d*(K+W)（忽略屏蔽、去重、padding、sink 与分块）。这是工作量量级，不是 kernel 精确 FLOPs 或实测时间。

当 L/c <= index_topk 时，代码可填顺序索引绕过无意义的 top-k 搜索；短上下文与长上下文不可混测。prefill gather 的 BF16 workspace/HBM traffic 可能主导成本，即使 cache 本体用 FP8。decode 的 padded heads 意味着 TP 继续增大时 kernel 工作量未必按 1/p 下降。共享 KV 还意味着 TP 不会给本 rank 的历史容量除以 p。

## 7. 练习

用教学 H=16、h=8、d=4、r_q=6、r_o=3、g=4、T=6 推导每个投影 shape。该尺寸用于 CPU 手算，**不满足真实 FlashMLA kernel 的所有约束**。再回答 c=4、L=20、K=2 时 TopK 的候选记录数、输出索引空间与 SWA 是否需要同一 cache。

答案见 [答案第 04 节](answers.md#04)；下一章：[TP](05_tensor_parallel.md)。
