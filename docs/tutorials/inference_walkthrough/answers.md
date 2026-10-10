# 练习参考答案

所有数值维度都是教学值；实际 scheduler 结果另见 validation 下的 JSON。符号模型 shape 不替代 checkpoint 配置，ring 流量不替代 NCCL trace。

<a id="01"></a>

## 01

| 轮次 | 输入 positions | schedule 后 computed | update 后 known |
| --- | --- | --- | --- |
| 1 | [0,1] | 2 | 4，未完成 prefill，无有效采样 |
| 2 | [2,3] | 4 | 5，包含 y0 |
| 3 | [4] | 5 | 6，包含新 y1 |

第三轮开始前 y0 有 ID，但没有 KV；第三轮为它写 KV 后采样 y1。prefix cache 是各层历史记录，不是末行最终 hidden/logits，且整块 lookup 将命中上限设为 known-1，完整 prefix 仍要留工作推进。

<a id="02"></a>

## 02

预算6的真实 scheduler lab 中第二至四轮为 {A:1,B:1,P:4}、{A:1,B:1,P:4}、{A:1,B:1,P:2}。P positions 分别是0–3、4–7、8–9。按教学顺序 query_start_loc 分别为 [0,1,2,6]、[0,1,2,6]、[0,1,2,4]。A/B 在第四轮计算 position4；A 的第四个输出使它结束，后续轮次不再占 budget。

预算4时只要 A/B 都活跃，P 剩余2行；A 结束后可分到更多行，不能机械地将整个10 token prompt 固定拆成5轮。真实 scheduler 的 admission、prefix、抢占和输出长度还会改变轮数。

pressure 模式中 before/remaining computed 的回退与 preempted ID 一起证明重置；仅 waiting 状态或 scheduled=0 不能证明抢占。prefix 模式 first 请求 prompt32、输出1，second 同前缀，复用16 token，second 从 position16 开始，复用第一物理页。这个实验使用整块 full-attention spec，没运行 Attention kernel。

<a id="03"></a>

## 03

普通 slots 为 [28,31,8,11,36]。压缩 slots 对 p=[3,4,7,11,15,19] 为 [20,-1,21,22,23,32]；-1 是教学 PAD。p=4 没满足 (p+1)%4=0，尚未组成新的完整压缩组，但其 partial state 应保留。

共享页7的 refcount=1 时不可覆盖，仍有一个请求需要读取。refcount=0 后可进入可分配队列，但 hash/内容仍可能被 prefix 命中；实际再分配时才淘汰旧关联。存在 in-flight readers 时还需 deferred-free 等保护。

一页内 n 条584 B记录、alignment576，内容是 n*584 B，页尺寸需要 round_up 到576，并可能有更高层 group padding。这与语义 head_size=512、FP8 名称都不是同一口径。

<a id="04"></a>

## 04

教学单卡：

| 项目 | shape |
| --- | --- |
| x | [6,16] |
| fused weight / output | [10,16] / [6,10] |
| qr / kv | [6,6] / [6,4] |
| wq_b weight / Q | [32,6] / [6,8,4] |
| grouped wo_a weight / z | [4,3,8] / [6,4,3] |
| wo_b weight / output | [16,12] / [6,16] |

真实 FlashMLA Q 缓冲可能补 head 数到64，这个教学配置不能直接运行该 GPU kernel。L=20、c=4 产生5条完整压缩记录，TopK=2选择该记录空间的两个索引；SWA读取最近窗口的完整行，cache内容与索引cache不同。

<a id="05"></a>

## 05

p=2时 wq_b=[16,6]，每rank语义Q=[6,4,4]，wo_a=[2,3,8]，wo_b=[16,6]，partial output=[6,16]。两个partial是同一输出坐标的加项，必须相加；concat会得到错误的输出维度/数值。

BF16、S=8 MiB、p=8、ring AR：每rank send=recv=14 MiB，全组send=112 MiB，recv=112 MiB，send+recv=224 MiB=234.881024 MB。FP8 data-only S=4 MiB，对应7/7 MiB、56/56 MiB、112 MiB总收发；真实FP8需要另加scale与metadata。

TP8/DP1、EP开、普通非Mega FusedMoE时，V4模型的自动SP条件不成立。如果切到Mega后端，条件可能成立，其他路径也随之变化。不要只根据EP开关判断SP。

<a id="06"></a>

## 06

expert5收到a、b，H=4,I=6：GEMM1 (M,N,K)=(2,12,4)，GEMM2=(2,4,6)，两次约288 FLOPs。全组8个有效 assignments，总两次GEMM工作量=6*8*4*6=1152 FLOPs，尚未计activation、路由、对齐padding、shared。

TP2/DP4/PCP1，EP开：expert TP=1、EP=8。EP关：expert TP=8、EP=1。Attention TP仍为2。EP不是第三个独立乘法维度。

AG的activation把每个源token收集一次，S=32 B；不是按每个expert assignment复制k次。sparse教学发送的5条远端assignment共40 B，与AG全组send96 B口径不同；metadata和实际算法也不同，不能直接凭这两个数字给部署结论。

<a id="07"></a>

## 07

每DP decode T=3，经SP补为4，两TP rank各2行；全EP8源共16行，其中12有效。Attention前AG后按positions长度截到3，O partial再补到4后RS。group[0,1]、[2,3]、[4,5]、[6,7]是四个TP组；DP组为[0,2,4,6]和[1,3,5,7]；EP组为0–7。

DP3的token可被rank0的expert处理，因为dispatch发送了对应activation，combine按源行回到DP3。DP0的KV是不同请求历史，不能提供DP3的Attention上下文；MoE没有“读取源请求历史KV”的语义。

prefill EP每次AG/RS每ranksend=recv172032 B，全组send1376256 B；decode分别114688 B和917504 B。AG+RS以及send+recv的倍数必须分开。此处为BF16均匀ring、activation-only，没有实测。

<a id="08"></a>

## 08

Q=[0,5]，compress=[1,7]，重叠区间[1,5]长4；join到7，Attention结束11。串行5+6+4=15，理想节约4，速度比15/11约1.364。

若并行两支结束于9和10，Attention结束14，节约仅1。仍可能有很长的重叠区间，但资源竞争使单支耗时增加，不能用交叠面积直接当端到端收益。应比较相同负载、正确性与服务指标。

<a id="09"></a>

## 09

layer3最近KV source=0，index source=2；它不是source，不分配自己的长程compressed cache，不重新index打分，仍计算自己的Q/SWA并使用最新结果。layer5最近KV/index source都为4；它同样是consumer。layer2虽重新打分，但可复用KV source0生成的index K cache。

教学内容容量598016 B与540672 B，差57344 B；没计SWA、index K、partial states、页alignment、group padding和workspace，所以不能推出真实模型总显存差。V4 c=1表示SWA-only，V4.1 c=1保留全长度的source长程cache；必须读版本对应分支。

V4.1 NVIDIA MoE构造函数显式传num_hash_layers=0，继承关系不能覆盖这个参数差异。hash可固定expert IDs，却仍需gate scores生成权重；V4 forward就有相应gate调用，所以不能只凭取消hash推断增加一整次gate GEMM。应比较实际router/后端的执行分支。

<a id="10"></a>

## 10

12 ms串行基线中expert2 ms优化25%变1.5 ms，总11.5 ms，加速12/11.5约1.0435，即总耗时下降约4.17%。已被GPU覆盖的CPU prepare不在关键路径时，缩短它不会再从TPOT直接扣2 ms。

假设“长通信由peer晚到造成”：检查同iteration其他rank的前驱kernel结束/collective提交时间，是否比本rank晚；再检查通信kernel的实际链路字节/活跃区间，是否长时间等待后才短暂传输。若peer都准时、链路持续接近饱和且流量大，则等待假设被削弱，应调查带宽/算法。跨rank时间需要可靠对齐，截图单rank不足以检验。
