# 按照 H2O / InfiniGen 逻辑改造通用 Transformer 的实现指南

这份文档不重复解释项目里已有的两份实现说明，而是把它们收束成一份“可迁移改造手册”：

- H2O 细节说明见 [h2o-implementation.md](./h2o-implementation.md)
- InfiniGen 细节说明见 [infinigen-implementation.md](./infinigen-implementation.md)
- InfiniGen 的离线准备流程见 [infinigen-setup-to-hello-world-flow.md](./infinigen-setup-to-hello-world-flow.md)

本文回答的是另一个问题：

如果你现在手里有一个标准 `Transformer` 架构，想按这个仓库中的 H2O 或 InfiniGen 思路改造它，应该改哪些模块、按什么顺序落地、哪些状态必须新增、哪些步骤必须放到离线阶段。

## 1. 先把两种改造思路分清楚

这个仓库里，H2O 和 InfiniGen 都在做“缩小 attention 的可见区域”，但实现层级完全不同。

### 1.1 H2O 的落点

H2O 的核心路径是：

- 入口脚本 [accuracy/lm_eval/run_lm_eval_harness.py](../accuracy/lm_eval/run_lm_eval_harness.py)
- attention 替换函数 [accuracy/lm_eval/utils_lm_eval/modify_opt.py](../accuracy/lm_eval/utils_lm_eval/modify_opt.py) 与 [accuracy/lm_eval/utils_lm_eval/modify_llama.py](../accuracy/lm_eval/utils_lm_eval/modify_llama.py)

它的特点是：

- 不改模型主干类的整体结构
- 模型加载完成后，直接替换中后层 `self_attn`
- 在真实 attention 分数上生成 `heavy + recent` mask
- 再把 mask 叠加回真实 attention

也就是说，H2O 更像“在现有 attention forward 内部加一段选择逻辑”。

### 1.2 InfiniGen 的落点

InfiniGen 的核心路径是：

- 入口脚本 [accuracy/lm_eval/run_lm_eval_harness.py](../accuracy/lm_eval/run_lm_eval_harness.py)
- 自定义模型实现 [accuracy/src/modeling_opt_ours.py](../accuracy/src/modeling_opt_ours.py) 与 [accuracy/src/modeling_llama_ours.py](../accuracy/src/modeling_llama_ours.py)
- 离线生成脚本 [accuracy/setup/gen_opt_model.py](../accuracy/setup/gen_opt_model.py)、[accuracy/setup/gen_llama_skewing_matrix.py](../accuracy/setup/gen_llama_skewing_matrix.py)、[accuracy/setup/gen_partial_weight.py](../accuracy/setup/gen_partial_weight.py)

它的特点是：

- 不只是替换 attention，而是直接切换成自定义模型实现
- attention forward 里新增 speculative attention 分支
- mask 不是来自真实 attention，而是来自“上一层隐藏状态推测出的代理 attention”
- 需要额外的离线产物：`partial_weight_q`，以及某些架构下的 `skewing_matrix`

也就是说，InfiniGen 更像“给 decoder 增加一条跨层预测通路，再用预测结果限制真实 attention”。

## 2. 改造通用 Transformer 前，你的基线实现至少要有这几个钩子

无论选 H2O 还是 InfiniGen，你的基线 Transformer 最少都要暴露出这些结构。这个仓库里最接近“通用最小实现”的参照是：

- 标准 attention: [speedup/uvm/selfattention.py](../speedup/uvm/selfattention.py)
- H2O 化 attention: [speedup/uvm/h2o_attention.py](../speedup/uvm/h2o_attention.py)
- 最小 transformer layer: [speedup/uvm/transformerlayer.py](../speedup/uvm/transformerlayer.py)

建议你先确认目标模型是否满足下面四个条件：

1. `self_attn.forward(...)` 内部可以直接拿到 `hidden_states`
2. `self_attn.forward(...)` 内部可以访问或更新 `past_key_value`
3. decoder 主循环能够在层与层之间传递额外状态
4. 每次请求结束后，能够逐层清理自定义缓存状态

如果这四点里缺任何一个，先补齐，再做 H2O / InfiniGen 改造。

## 3. 如果你要走 H2O 路线，建议按下面顺序改

H2O 适合“尽量少改模型主干，只改 attention 可见性”的场景。它是两种方案里更容易迁移到任意 decoder-only Transformer 的路线。

### 3.1 第一步：新增运行时超参数

至少要有：

- `heavy_ratio`
- `recent_ratio`
- 可选的 `protected_layers` 或“前几层不改”的规则

这个仓库里的 H2O 参数注入在 [accuracy/lm_eval/run_lm_eval_harness.py](../accuracy/lm_eval/run_lm_eval_harness.py) 中完成，真正读取这些参数的是 `modify_opt.py` / `modify_llama.py` 里的 attention 类。

对通用 Transformer 来说，这一步通常可以放在：

- `config`
- `self_attn.__init__`
- 或者模型构建后的模块级属性注入

### 3.2 第二步：在真实 attention 上生成 heavy mask

H2O 的关键不是直接裁 KV，而是先从真实 attention 中识别“重要 key”。

仓库里的核心函数是：

- [accuracy/lm_eval/utils_lm_eval/modify_opt.py](../accuracy/lm_eval/utils_lm_eval/modify_opt.py) 里的 `local_heavy_hitter_mask(...)`
- [accuracy/lm_eval/utils_lm_eval/modify_llama.py](../accuracy/lm_eval/utils_lm_eval/modify_llama.py) 里的 `local_heavy_hitter_mask(...)`

迁移到通用 Transformer 时，建议保持同一执行顺序：

1. 正常算出 `attn_weights = q @ k^T`
2. 对 `attn_weights` 先做一次 softmax，得到概率分布
3. 维护每个 key 的累计关注分数
4. 对每个 query 选择累计分数最高的 `heavy_budget` 个 key

其中：

- `heavy_budget = int(heavy_ratio * key_len)`
- 这个预算应当随着当前 `key_len` 动态变化

### 3.3 第三步：再叠加 recent mask

H2O 不是只保留 heavy hitter，它还强制保留最近窗口。

迁移时直接照着现有逻辑做即可：

1. 构造一个只覆盖最近 `recent_budget` 个历史位置的带状 mask
2. 和 heavy mask 做逻辑或
3. 再和 causal 下三角约束求交

其中：

- `recent_budget = int(recent_ratio * key_len)`

### 3.4 第四步：在 softmax 前屏蔽未保留位置

这个仓库的 accuracy 路径里，H2O 并没有真的把 KV cache 张量裁短，而是：

1. 保持 `key_states / value_states` 仍然是完整长度
2. 把未保留位置的 attention score 赋成极小值
3. 再执行正常 softmax

因此，如果你的目标只是先验证算法正确性，最稳妥的第一版也是：

- 不要先做物理裁剪
- 只做逻辑 mask

这一步最容易落地，也最容易验证精度。

### 3.5 第五步：如果你还要追求真实 cache 缩短，再补“物理裁剪版”

这个仓库的 `accuracy` 路径偏重精度验证，而 `speedup` 路径更接近工程化缓存管理。

对应的参考实现是：

- [speedup/uvm/h2o_attention.py](../speedup/uvm/h2o_attention.py)

它比 `accuracy` 路径多做了一步：

- 在 decode 过程中维护 `acc`
- 把被判定为不重要的位置从缓存集合里驱逐
- 用新的 token 覆盖被驱逐的位置

因此，如果你要把 H2O 迁到一个真正用于长序列推理的 Transformer 上，建议分两期：

1. 先做逻辑 mask 版，验证行为和精度
2. 再做物理 KV 裁剪版，验证时延和显存收益

## 4. 如果你要走 InfiniGen 路线，建议按下面顺序改

InfiniGen 更激进，也更依赖架构细节。它适合“你愿意改 decoder 主循环，并且接受增加离线准备步骤”的场景。

### 4.1 第一步：把在线逻辑和离线逻辑拆开

这是 InfiniGen 与 H2O 最大的工程差异。

H2O 基本只需要在线阶段。

InfiniGen 必须拆成两部分：

- 离线阶段：生成 `partial_weight_q`，有些架构还要生成 `skewing_matrix` 或重写权重
- 在线阶段：在 attention forward 中使用这些离线产物生成 speculative mask

你可以把这个仓库里的离线链理解为三段：

1. `gen_opt_model.py`
2. `gen_llama_skewing_matrix.py`
3. `gen_partial_weight.py`

### 4.2 第二步：先定义 attention 新增的运行时状态

迁移到通用 Transformer 时，attention 类里至少要新增这些字段：

- `previous_hidden_states`
- `current_hidden_states`
- `partial_weight_q`
- `alpha`
- `capacity`
- `budget`
- `eviction_policy`
- `density`

对应参考：

- [accuracy/src/modeling_opt_ours.py](../accuracy/src/modeling_opt_ours.py)
- [accuracy/src/modeling_llama_ours.py](../accuracy/src/modeling_llama_ours.py)

如果目标架构也有 RoPE 或类似位置编码变换，通常还要额外考虑：

- speculative query 是否也要走同样的位置编码
- key 在 speculative 分支里是否要做相同的变换

### 4.3 第三步：让每层 attention 保存自己的当前隐藏状态

InfiniGen 的预测信号不是来自当前层的真实 attention，而是来自上一层隐藏状态。

所以 attention forward 一开始要做一件事：

- `self.current_hidden_states = hidden_states.clone()`

这一点在 OPT 和 LLaMA 实现里都是一样的。

### 4.4 第四步：在 decoder 主循环里把上一层隐藏状态传给下一层

这一步是通用 Transformer 改造里最容易漏掉、但最关键的一步。

参考实现：

- OPT: [accuracy/src/modeling_opt_ours.py](../accuracy/src/modeling_opt_ours.py)
- LLaMA: [accuracy/src/modeling_llama_ours.py](../accuracy/src/modeling_llama_ours.py)

迁移时你要在 decoder layer loop 中补上类似逻辑：

1. 当前层 forward 结束
2. 取出当前层 `self_attn.current_hidden_states`
3. 写入下一层 `self_attn.previous_hidden_states`

如果目标模型是带偏置拼接的实现，像 OPT 版那样，还要在最后一维拼一个常数 1；如果目标模型是标准无偏置路径，通常不需要这一步。

### 4.5 第五步：实现 speculative attention 分支

这是 InfiniGen 的核心。

建议你把这条支路拆成两个部分理解。

#### 4.5.1 先得到一个更便宜的 query 代理

这个仓库里有两种方式：

- OPT 路线：`previous_hidden_states @ partial_weight_q`
- LLaMA 路线：先过 `q_proj` 重建 query，再做 RoPE，再做 skewing，再做通道筛选

因此，迁移时的关键判断是：

你的目标 Transformer 更接近哪一种？

- 如果 `q_proj / k_proj` 结构简单、没有特殊位置变换，优先走 OPT 式
- 如果有 RoPE、head 内部结构敏感，优先走 LLaMA 式

#### 4.5.2 再把代理 attention 变成真正的可执行 mask

这个仓库里通用的控制器是 `kv_cache_mask(...)`，它做的事情是：

1. 用 `alpha` 给每个 query 设阈值
2. 估计需要 fetch 的 key 数量
3. 用 `budget` 限制每步最多拉取多少 KV
4. 用 `capacity` 限制可驻留 cache 容量
5. 用 `eviction_policy` 决定超容量时踢谁

对应参考：

- [accuracy/src/modeling_opt_ours.py](../accuracy/src/modeling_opt_ours.py)
- [accuracy/src/modeling_llama_ours.py](../accuracy/src/modeling_llama_ours.py)

如果你更想拆成可复用控制器，而不是写死在 attention 类里，可以参考 `speedup/infinigen` 目录中的三个文件：

- [speedup/infinigen/infinigen/partial_weight_generation_controller.py](../speedup/infinigen/infinigen/partial_weight_generation_controller.py)
- [speedup/infinigen/infinigen/kv_selection_controller.py](../speedup/infinigen/infinigen/kv_selection_controller.py)
- [speedup/infinigen/infinigen/skewing_controller.py](../speedup/infinigen/infinigen/skewing_controller.py)

这三个控制器其实就是把 InfiniGen 拆成了三个独立问题：

- 怎样找重要通道
- 怎样预测重要 KV 索引
- 怎样把投影空间变得更适合做稀疏预测

### 4.6 第六步：把 speculative mask 叠加回真实 attention

InfiniGen 并不替代真实 attention，它只是在真实 attention 前加一道门。

标准顺序应该是：

1. 正常算真实 `attn_weights`
2. 如果 speculative 分支已就绪，就把 `attn_mask` 加上去
3. 再叠加正常 causal / padding mask
4. 再 softmax
5. 再和 `value_states` 聚合

也就是说，真实 `q @ k^T` 仍然要算，InfiniGen 只是提前决定“哪些 key 值得进入竞争”。

### 4.7 第七步：请求结束后一定要清状态

InfiniGen 的 mask 依赖跨层的 `previous_hidden_states`。如果不在请求边界清空，下一条样本会继承上一条样本的状态，结果一定错。

这个仓库在 harness 末尾显式做了重置。

迁移到通用 Transformer 时，至少要在请求结束后清空：

- `previous_hidden_states`
- 必要时还包括统计量、prefetch 状态、density 状态

## 5. 离线阶段到底要准备什么

如果你做的是 H2O，一般不需要额外离线产物。

如果你做的是 InfiniGen，至少要明确三类产物的来源。

### 5.1 `partial_weight_q`

对应脚本：

- [accuracy/setup/gen_partial_weight.py](../accuracy/setup/gen_partial_weight.py)

它的本质是：

- 先用一次 warmup / generate，让每层 attention 看到真实 query
- 根据 query 激活强弱，选出最重要的一部分通道
- 把这些通道对应的 query 权重抽出来，保存成每层一个 `partial_weight_q`

### 5.2 `skewing_matrix`

对应脚本：

- [accuracy/setup/gen_llama_skewing_matrix.py](../accuracy/setup/gen_llama_skewing_matrix.py)

它的本质是：

- 根据 query / key 的 SVD 结果构造一个 head 内变换矩阵
- 让少数列在变换后更“显眼”
- 这样后面的 partial channel 选择会更有效

如果目标 Transformer 没有 RoPE、没有明显的 head 内正交结构需求，这一步不一定必须保留。

### 5.3 改写后的投影权重

对应脚本：

- [accuracy/setup/gen_opt_model.py](../accuracy/setup/gen_opt_model.py)

它主要做的是：

- 把 query / key 的投影权重改写成更适合做稀疏预测的形式
- 对 OPT 路线，还把 bias 并入了矩阵乘法路径

所以，如果你的目标 Transformer 没有偏置拼接需求，或者不想动预训练权重语义，可以先不做这一步，先做“只依赖 partial weight 的简化版 InfiniGen”。

## 6. 迁移时最推荐的落地顺序

如果你的目标是“尽快把一个标准 Transformer 改成可运行版本”，建议不要一步到位。

### 路线 A：先做 H2O，再视情况升级

1. 先在 attention 内实现 `heavy + recent` 逻辑 mask
2. 验证 causal 正确性、输出稳定性、长序列可运行性
3. 再决定是否继续做物理 KV 裁剪

这条路线侵入性最低，最适合先做 proof of concept。

### 路线 B：先做简化版 InfiniGen，再补齐离线优化

1. 先实现 `previous_hidden_states -> speculative attention -> kv_cache_mask`
2. `partial_weight_q` 先用简单 top-k 通道或固定掩码替代
3. 跑通后再补 `skewing_matrix`、改写权重、容量控制优化

这条路线更适合你已经确定要做跨层预测式稀疏 attention，只是想先缩短调试路径。

## 7. 可以直接照搬的模块映射

把这个仓库映射到一个通用 Transformer，可以按下面理解：

| 需求 | H2O 参考 | InfiniGen 参考 | 你在目标模型里要改的位置 |
| --- | --- | --- | --- |
| 在线入口参数 | `run_lm_eval_harness.py` | `run_lm_eval_harness.py` | CLI / config / model wrapper |
| attention 改写 | `modify_opt.py` / `modify_llama.py` | `modeling_*_ours.py` | `self_attn.forward(...)` |
| 跨层状态传递 | 不需要 | `modeling_*_ours.py` 的 decoder loop | `decoder.layers` 主循环 |
| 离线权重生成 | 不需要 | `accuracy/setup/*.py` | 预处理脚本 |
| 逻辑 mask | `heavy + recent` | `kv_cache_mask(...)` | softmax 前 |
| 物理 KV 管理 | `speedup/uvm/h2o_attention.py` | `speedup/infinigen/*` | decode cache manager |
| 请求边界状态清理 | 可选 | 必需 | 推理循环末尾 |

## 8. 改造时最容易踩的坑

### 8.1 不要混淆“逻辑裁剪”和“物理裁剪”

这个仓库里：

- `accuracy` 下的 H2O 主要是逻辑裁剪
- `speedup` 下的 H2O 才更接近物理 KV 裁剪

如果你只做了 mask，却期待显存占用立刻大幅下降，结果会和预期不一致。

### 8.2 speculative 分支必须和真实 attention 的坐标系一致

如果真实 attention 用了：

- RoPE
- head 内重排
- bias 拼接
- 特殊的 q/k scaling

那 speculative query / key 也必须做同构处理。否则预测出来的 mask 会系统性偏移。

### 8.3 请求边界不清状态，结果会串样本

这一点对 InfiniGen 尤其重要。

### 8.4 先验证 prefill，再验证 decode

很多实现 prefill 能跑通，但 decode 阶段因为 `past_key_value` 更新、索引替换、容量限制而出错。建议把验证分成两段：

1. 全量 prefill 正确
2. 单步 decode 正确

最后再做长序列 decode。

## 9. 一个最实用的结论

如果你的目标只是“把某个现有 Transformer 快速改成这个项目类似的稀疏注意力版本”，最稳的优先级通常是：

1. 先做 H2O 风格的逻辑 mask 版
2. 再决定是否需要 H2O 的物理 cache 裁剪
3. 如果你明确需要“跨层预测下一层 KV 访问模式”，再升级到 InfiniGen

原因很简单：

- H2O 更容易迁移
- 调试面更小
- 不依赖离线产物
- 更适合先验证你的目标 Transformer 是否允许稀疏 attention 改造

而 InfiniGen 更适合在你已经确认：

- decoder 层间状态可控
- attention 结构足够稳定
- 愿意接受离线预处理成本

之后再做。

## 10. 建议你在目标代码里实际落下来的改造清单

最后把全文压缩成一份真正能执行的 checklist。

### H2O 版 checklist

1. 在 attention 类新增 `heavy_ratio`、`recent_ratio`
2. 在 `q @ k^T` 后、softmax 前插入 `heavy + recent` mask 逻辑
3. 前几层保留原始 attention，先只改中后层
4. 保持 `past_key_value` 逻辑不变，先做逻辑 mask 版
5. 验证后再决定是否加物理 cache 驱逐

### InfiniGen 版 checklist

1. attention 类新增 `previous_hidden_states`、`current_hidden_states`、`partial_weight_q`、`alpha`、`capacity`、`budget`
2. decoder 主循环增加跨层隐藏状态传递
3. attention forward 内新增 speculative attention 分支
4. 实现 `kv_cache_mask(...)`
5. 离线准备 `partial_weight_q`
6. 如架构需要，再补 `skewing_matrix` 与投影改写
7. 请求结束后清理跨层状态

如果你后面准备真的对某个具体 Transformer 文件动手，最建议优先对照这几个最小参考文件看：

- [speedup/uvm/selfattention.py](../speedup/uvm/selfattention.py)
- [speedup/uvm/h2o_attention.py](../speedup/uvm/h2o_attention.py)
- [accuracy/src/modeling_opt_ours.py](../accuracy/src/modeling_opt_ours.py)
- [accuracy/src/modeling_llama_ours.py](../accuracy/src/modeling_llama_ours.py)

这四个文件基本就覆盖了：

- 标准 attention
- H2O 化 attention
- InfiniGen 的无 RoPE / 偏置拼接版本
- InfiniGen 的 RoPE / skewing 版本

从这里开始迁移，路径最短。
