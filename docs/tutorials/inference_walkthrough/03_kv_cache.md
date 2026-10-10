# 03 KV cache：从 GPU 池到一个物理 slot

## 1. 为什么需要两套管理

你熟悉 kernel 指针，但推理系统必须保证一个请求跨许多轮仍读到自己的历史。scheduler 管“哪个请求占哪些逻辑 block”；worker 管“这些 block 对应哪些 tensor view 和 slot”；kernel 按 metadata 读写。一个请求结束通常归还页，不是逐个调用 cudaFree。

[教学] block_size B=4，某请求 block_table=[7,2,9]。位置 p=0–3 在物理块 7，p=4–7 在块 2，p=8–11 在块 9。这个 B 仅用于手算，不是主线 backend 的合法部署参数。

## 2. 生命周期图

~~~mermaid
flowchart TD
  A["启动: cache spec + 可用内存"] --> P["分配 GPU cache tensors"]
  P --> V["按层绑定 views"]
  S["Scheduler BlockPool"] --> B["request -> logical blocks -> block IDs"]
  B --> W["worker block_table"]
  W --> M["positions -> slot_mapping"]
  V --> K["写入 / Attention 读取"]
  M --> K
  K --> R["下一轮保留 blocks"]
  R --> K
  R --> F["结束 / 抢占: refcount 减少"]
  F --> C["保留 hash 的可复用页"]
  C --> H["prefix hit: touch / refcount 增加"]
  C --> E["分配压力: evict 后覆盖"]
  H --> B
~~~

## 3. 启动分配与绑定：不要从 scheduler 找 cudaMalloc

[vllm/v1/engine/core.py · EngineCore._initialize_kv_caches · L256–L401](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/engine/core.py#L256-L401) 收集模型的 KV specs，结合 worker 的可用内存确定 GPU cache 配置；[vllm/v1/worker/gpu_worker.py · Worker.determine_available_memory · L599–L772](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_worker.py#L599-L772) 通过加载权重和 profiling 估算余量；[vllm/v1/worker/gpu_model_runner.py · GPUModelRunner.initialize_kv_cache · L7303–L7368](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py#L7303-L7368) 初始化后端、metadata builders、cache tensors。[vllm/v1/worker/gpu_model_runner.py · GPUModelRunner.initialize_kv_cache_tensors · L7227–L7274](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/gpu_model_runner.py#L7227-L7274) 调用通用 allocate_kv_cache_tensors / reshape_kv_cache_tensors，再 bind_kv_cache。每个 AttentionLayerBase 的 get_kv_cache_spec 描述持久状态；模型的 bind_kv_cache 接收对应 view。

可用 cache 内存不是显存减去参数量那么简单：还包括 runtime、workspace、图捕获、通信 buffer 等。多卡通常需采用各 worker 的可行容量限制；一个 rank 的最坏占用会限制全组。没有 GPU 的本任务没有执行该 profiling。

## 4. block / slot 的精确映射

在 CP=1 的普通页布局：

~~~text
logical_block = position // B
offset = position % B
physical_block = block_table[request, logical_block]
slot = physical_block * B + offset
~~~

| position | logical block | physical block | offset | slot |
| --- | --- | --- | --- | --- |
| 0 | 0 | 7 | 0 | 28 |
| 3 | 0 | 7 | 3 | 31 |
| 4 | 1 | 2 | 0 | 8 |
| 7 | 1 | 2 | 3 | 11 |
| 8 | 2 | 9 | 0 | 36 |

[vllm/v1/worker/block_table.py · ComputeSlotMappingKernel.kernel · L320–L376](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/worker/block_table.py#L320-L376) 就是实际 Triton 计算；它还支持 CP interleave 和 PAD_ID。block_table 的一行与 request 绑定，而不与 token ID 绑定。同样的 token ID 出现在不同位置或前缀，不代表可读同一页。

压缩 cache 的 p 要先变成 compressed position。[vllm/v1/attention/backends/mla/compressor_utils.py · CompressedSlotMappingKernel.kernel · L45–L106](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/attention/backends/mla/compressor_utils.py#L45-L106) 仅在 (p+1)%c==0 且原 slot 非 PAD 时写入，compressed_position=p//c，再走相应压缩页表。若 B 表示原始 token block，backend 压缩页的记录数可能是 B/c；公式中的 B 必须使用 metadata 实际提供的 compressed block_size，不能用 SWA 页尺寸替代。

[教学] c=4，压缩页每页 4 记录，table=[5,8]。p=3,7,11,15,19 分别生成 compressed_pos=0,1,2,3,4，slot=20,21,22,23,32。p=4 不生成完整压缩记录。中途 chunk 边界的部分状态由 compressor state cache 保留，不能直接丢弃。

## 5. V4 的持久状态不只有一个 K/V tensor

| 状态 | 内容与用途 | 代码入口 | TP=无 CP 时 |
| --- | --- | --- | --- |
| SWA cache | 最近窗口中的完整共享 KV 行，供局部 Attention | [vllm/v1/attention/backends/mla/sparse_swa.py · DeepseekV4SWACache.get_kv_cache_spec · L116–L140](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/attention/backends/mla/sparse_swa.py#L116-L140) | 每个 Attention TP rank 持有副本 |
| 压缩 MLA cache | c 个位置得到的语义 KV 记录，供历史稀疏读取 | [vllm/models/deepseek_v4/attention.py · DeepseekV4Attention.get_kv_cache_spec · L856–L879](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L856-L879) | num_kv_heads=1、max_tp_shards=1 |
| indexer K cache | 用于打分的低维历史 key，独立于 Attention KV | [vllm/models/deepseek_v4/attention.py · DeepseekV4IndexerCache.get_kv_cache_spec · L907–L920](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L907-L920) | indexer 投影复制，cache 复制 |
| compressor state cache | 尚未凑满窗口的 KV/score 等状态 | [vllm/models/deepseek_v4/compressor.py · CompressorStateCache.get_kv_cache_spec · L179–L192](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/compressor.py#L179-L192) | 单独 spec；不能当成输出 KV |
| 临时 top-k / gather workspace | 本轮索引与 BF16 gather 中间值 | [vllm/models/deepseek_v4/nvidia/flashmla.py · DeepseekV4FlashMLAAttention._forward_prefill · L266–L398](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/nvidia/flashmla.py#L266-L398) | 工作空间，不是 prefix cache 本体 |

SWA 是 sliding window attention（滑动窗口），描述“哪些历史可见”。页是否实际回收由 cache manager、hybrid 分组与 prefix 策略决定，不能只看 W 就断言显存严格为 W×记录宽度。MLA 是多头 latent attention 系列的存储/计算接口；此 V4 路径的 KV 投影输出是共享的一行 [T,d]，不能照搬 V2/V3 的 latent 加独立 RoPE 双张量公式。

[代码] V4 packed FP8 记录为 **584 B**：448 B NoPE + 128 B RoPE + 8 B scale，语义 head_size 仍为 512，[vllm/models/deepseek_v4/attention.py · DeepseekV4Attention.get_kv_cache_spec · L856–L879](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v4/attention.py#L856-L879)。这是布局契约，不是通过模型卡核实的全部模型维度。页需要 alignment=576；实际 page_size_bytes 包括 tokens_per_state、对齐和分组 padding，见 [vllm/v1/kv_cache_interface.py · MLAAttentionSpec · L686–L754](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/kv_cache_interface.py#L686-L754)。FP8 是数据格式，不意味着整个 KV 每元素都是 1 byte，RoPE 段与 scale 就是反例。

V4.1 只在 kv source 层拥有压缩 cache，consumer get_kv_cache_spec 返回 None，[vllm/models/deepseek_v41/attention.py · DeepseekV4Attention.get_kv_cache_spec · L1138–L1163](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/models/deepseek_v41/attention.py#L1138-L1163)；但每层 SWA 仍有自己的状态。SM100 条件下记录宽度 528 B，NVFP4 压缩分支 288 B，SM90 回退布局可为 584 B，详见第 9 章。

## 6. 复用、释放与容量

[vllm/v1/core/kv_cache_manager.py · KVCacheManager.allocate_slots · L371–L618](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/kv_cache_manager.py#L371-L618) 分配 slots，[vllm/v1/core/kv_cache_manager.py · KVCacheManager.free · L620–L629](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/kv_cache_manager.py#L620-L629) 归还请求引用，[vllm/v1/core/block_pool.py · BlockPool.free_blocks · L774–L805](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/block_pool.py#L774-L805) 将 refcount=0 的块加入可分配队列。prefix cache 的 hash 关联可继续存在；[vllm/v1/core/block_pool.py · BlockPool.get_new_blocks · L666–L700](https://github.com/jia0713/vllm/blob/10cc2f6ae2c9ba7cc5841ece95e27d0562aef1bf/vllm/v1/core/block_pool.py#L666-L700) 真正重新分配时可能淘汰旧 hash。因而 free 与 erase cached bytes 不同，cache hit 会 touch 已缓存页。

prefix key 需要 token 前缀、前一块 hash 和相关额外信息等保持一致，不能只比较最后一个 block 的 token IDs。重复的 full block 可共享引用；正在写的尾块不能随意多请求共写。当前代码还存在 partial-cache/COW 扩展，入门脚本验证的是 full-attention 的整块基线，不能由此声称 V4 所有异构 group 都可同样命中。

估算并发应按所有持久状态的 pages 求和，再加 workspace；不是“显存 / 每 token KV / max_model_len”一行公式就足够。c=128 减少长程记录，也带来 compressor partial states；SWA、indexer、source 层数都可能成为主要容量项。prefix 共享省的是重复页，不改变本轮 query 的所有计算。

## 7. 练习

手算上述普通 slot 和压缩 slot；说明 p=4 为什么是 PAD。再设两个请求共享块 7，refcount 从 2 降到 1，回答块 7 是否可覆盖。运行 shape_lab.py 的 slot 与 packed-page 检查，并对照真实 prefix/pressure scheduler 输出。答案见 [答案第 03 节](answers.md#03)；下一章：[Attention](04_attention.md)。
