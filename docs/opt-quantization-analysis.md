# OPT 量化实现对比分析

本文对比以下两个文件，说明本项目如何在 Hugging Face 的 OPT 实现基础上注入量化逻辑：

- 上游参考实现: [`transformers/src/transformers/models/opt/modeling_opt.py`](../transformers/src/transformers/models/opt/modeling_opt.py)
- 项目量化实现: [`accuracy/src/modeling_opt_orig.py`](../accuracy/src/modeling_opt_orig.py)

补充说明一个容易混淆的事实：

- 从文件名看，`modeling_opt_orig.py` 像“原始版”。
- 但从代码内容和运行入口看，真正承载量化逻辑的是 `accuracy/src/modeling_opt_orig.py`。
- 运行脚本会在非 `--ours` 模式下，把 Hugging Face 的 `modeling_opt.py` 软链接到这个文件，再通过 `--enable_quant` 打开量化开关，见 [`run_lm_eval_harness.py#L58-L61`](../accuracy/lm_eval/run_lm_eval_harness.py#L58-L61) 和 [`run_lm_eval_harness.py#L76-L86`](../accuracy/lm_eval/run_lm_eval_harness.py#L76-L86)。

因此，本文采用下面的比较口径：

- “原始实现”指 Hugging Face 上游文件 `transformers/.../modeling_opt.py`
- “量化实现”指项目内的 `accuracy/src/modeling_opt_orig.py`

## 1. 背景介绍

### 1.1 OPT 模型结构简述

OPT 是标准的 decoder-only Transformer。其核心路径包括：

- token embedding 和 positional embedding
- 多层 `OPTDecoderLayer`
- 每层包含 `OPTAttention` 和 MLP (`fc1 -> activation -> fc2`)
- 自回归解码时通过 `past_key_values` 复用历史 K/V cache

对本次分析最关键的是 `OPTAttention`，因为 K/V cache 正是在这里生成、缓存和复用的。

### 1.2 为什么需要量化

在大语言模型推理中，量化通常有三个目的：

- 降低显存或内存占用，尤其是权重和 KV cache
- 降低内存带宽压力，提高长序列推理吞吐
- 为更低比特的部署内核做精度评估或行为模拟

对于 decoder-only 模型，长上下文推理时 KV cache 往往是主要的运行时开销之一，因此只量化 K/V 也是一个常见设计点。

### 1.3 本项目采用的量化类型

从 [`accuracy/src/modeling_opt_orig.py#L217-L244`](../accuracy/src/modeling_opt_orig.py#L217-L244) 可以看出，本项目这版 OPT 量化并不是传统的“把 `Linear` 替换成 `QuantLinear`”。

它的实际形式更接近：

- `K/V` 激活动态量化
- 量化对象是 attention 中间张量 `key_states` 和 `value_states`
- 量化时机发生在 forward 期间
- 粒度是按 `head_dim` 上的 `group_size=64` 分组
- 每组使用当前激活的 `min/max` 在线计算量化参数
- 量化后立刻反量化，再继续用浮点执行后续计算

所以更准确的分类是：

- 不是权重量化
- 不是静态量化
- 不是模块替换式 INT8 推理
- 更像 runtime fake-quant / QDQ simulation
- 目标更偏向“模拟低比特 KV 表示对精度的影响”，而不是直接获得真实 INT8 kernel 的吞吐收益

## 2. 文件对比总览

### 2.1 核心改动方向

相对于上游 [`modeling_opt.py`](../transformers/src/transformers/models/opt/modeling_opt.py)，项目版 [`modeling_opt_orig.py`](../accuracy/src/modeling_opt_orig.py) 的核心改动集中在 `OPTAttention`：

- 在构造函数中加入量化控制字段
- 在 attention forward 中对 `key_states/value_states` 插入量化-反量化逻辑
- 在 decoder 层和 decoder 容器中补充 `layer_idx` 传递

同时，文件里还存在一组与量化无直接关系的“版本漂移”差异：

- 保留旧版 attention mask 构造函数 `_make_causal_mask` / `_expand_mask`
- 保留旧版 `_prepare_decoder_attention_mask`
- 保留旧版 gradient checkpointing 写法
- `OPTDecoderLayer.forward` 少了一次 `fc2` 后 dropout

这些差异不应与量化逻辑混为一谈。

### 2.2 改动类型分类

| 类别 | 量化实现 | 上游实现 | 结论 |
| --- | --- | --- | --- |
| 权重量化 | 无 | 无 | 未实现 |
| 激活量化 | 有，且仅量化 K/V | 无 | 这是主要改动 |
| Forward 逻辑修改 | `OPTAttention.forward` 插入 QDQ | 无 | 量化入口 |
| 模块替换 | 无，仍使用 `nn.Linear` | `nn.Linear` | 未替换为 `QuantLinear` |
| 额外参数 / flag | `enable_quant`, `qbits`, `layer_idx` | 无 | 用于控制量化 |
| Attention 中间变量量化 | 仅 `key_states/value_states` | 无 | Q/attn_weights/MLP 未量化 |

### 2.3 关键结论先行

这套实现的本质是：

- 保持模型结构和参数加载方式不变
- 只在 attention 中对 K/V 激活做分组量化
- 量化后立即反量化，后续计算仍在浮点域中完成
- 因而它更像“低比特 KV cache 的行为模拟器”，不是完整的部署级量化推理实现

## 3. 函数级别详细对比

### 3.1 `OPTAttention.__init__`

#### 3.1.1 函数基本信息

- 所属类: `OPTAttention`
- 原始实现: [`modeling_opt.py#L94-L119`](../transformers/src/transformers/models/opt/modeling_opt.py#L94-L119)
- 量化实现: [`modeling_opt_orig.py#L125-L156`](../accuracy/src/modeling_opt_orig.py#L125-L156)
- 功能简介: 初始化注意力投影层及运行时状态

#### 3.1.2 修改内容对比

```diff
 class OPTAttention(nn.Module):
     def __init__(..., bias: bool = True,
+        layer_idx: int = 0,
     ):
         ...
         self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
         self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
+
+        self.layer_idx = layer_idx
+        self.enable_quant = False
+        self.qbits = 0
```

#### 3.1.3 修改点拆解

- 新增 `layer_idx`
- 新增运行时开关 `enable_quant`
- 新增量化位宽 `qbits`
- 四个投影层仍然是标准 `nn.Linear`

#### 3.1.4 修改原因

- `enable_quant` 允许同一模型结构在量化与非量化模式间切换
- `qbits` 用于在 forward 中计算 `qmax = 2^qbits - 1`
- `layer_idx` 为按层启用或调试量化预留了元信息

需要注意的是，`layer_idx` 在这个文件里并未直接参与量化计算。真正的逐层启用逻辑来自外部入口 [`run_lm_eval_harness.py#L76-L86`](../accuracy/lm_eval/run_lm_eval_harness.py#L76-L86)，脚本只对第 3 层及以后开启量化。

#### 3.1.5 对模型行为的影响

- 精度: 构造函数本身不影响精度
- 性能: 几乎无影响
- forward/backward: 只是增加状态字段，不改变计算图

### 3.2 `OPTAttention.forward`

#### 3.2.1 函数基本信息

- 所属类: `OPTAttention`
- 原始实现: [`modeling_opt.py#L124-L242`](../transformers/src/transformers/models/opt/modeling_opt.py#L124-L242)
- 量化实现: [`modeling_opt_orig.py#L161-L309`](../accuracy/src/modeling_opt_orig.py#L161-L309)
- 量化代码核心区间: [`modeling_opt_orig.py#L217-L244`](../accuracy/src/modeling_opt_orig.py#L217-L244)
- 功能简介: 生成 Q/K/V，处理 KV cache，计算注意力权重和输出

#### 3.2.2 修改内容对比

```diff
         proj_shape = (bsz * self.num_heads, -1, self.head_dim)
         query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
         key_states = key_states.view(*proj_shape)
         value_states = value_states.view(*proj_shape)
         src_len = key_states.size(1)
+
+        if self.enable_quant:
+            qmax = (2 ** self.qbits) - 1
+            group_size = 64
+            num_groups = int(self.head_dim / group_size)
+            for i in range(num_groups):
+                start = i * group_size
+                end = start + group_size
+
+                key_max = torch.max(key_states[:, :, start:end], dim=-1, keepdim=True)[0].repeat(...)
+                key_min = torch.min(key_states[:, :, start:end], dim=-1, keepdim=True)[0].repeat(...)
+                value_max = torch.max(value_states[:, :, start:end], dim=-1, keepdim=True)[0].repeat(...)
+                value_min = torch.min(value_states[:, :, start:end], dim=-1, keepdim=True)[0].repeat(...)
+
+                key_states[:, :, start:end] = torch.round(((key_states[:, :, start:end] - key_min) / key_delta) * qmax)
+                value_states[:, :, start:end] = torch.round(((value_states[:, :, start:end] - value_min) / value_delta) * qmax)
+
+                key_states[:, :, start:end] = (key_delta * key_states[:, :, start:end] / qmax) + key_min
+                value_states[:, :, start:end] = (value_delta * value_states[:, :, start:end] / qmax) + value_min

         attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
```

#### 3.2.3 修改点拆解

- 新增条件分支 `if self.enable_quant`
- 使用 `qbits` 计算离散整数范围 `qmax`
- 把 `head_dim` 按 `group_size=64` 分组
- 对每个组分别计算 `key/value` 的 `min/max`
- 用当前组的 `min/max` 做线性量化
- 量化后立刻反量化回浮点
- 只有 `key_states` 和 `value_states` 被处理
- `query_states`、`attn_weights`、`attn_output` 没有被量化

#### 3.2.4 修改原因

这是整份实现里最关键的量化注入点，原因有三：

- K/V 是自回归解码时最容易累积开销的状态，单独处理 K/V 成本更低
- 把量化插在投影之后、`QK^T` 之前，可以直接影响注意力得分与缓存复用
- 量化后立即反量化，不需要改写后续 `torch.bmm` 和 `out_proj` 路径，最容易在现有代码上实现

从公式看，它采用的是在线 min-max 仿射量化：

- `scale = (max - min) / qmax`
- `x_q = round((x - min) / scale)`
- `x_hat = x_q * scale + min`

这里没有显式保存 `zero_point`，但它在数学上是隐式存在的。由于量化区间是 `[0, qmax]`，这属于非对称量化。

#### 3.2.5 对模型行为的影响

- 精度:
  - K/V 会被压到离散量化格点上
  - 注意力分数 `QK^T` 直接受到量化误差影响
  - 因为量化作用于缓存路径，长序列生成时误差会逐步累积
- 性能:
  - 这段实现没有真正把数据存成 `int8`
  - 也没有调用低比特 GEMM kernel
  - 反而引入了额外的 `max/min/round` 和循环开销
  - 因此仅从这两个文件看，性能提升并不显著，甚至可能变慢
- forward/backward:
  - forward 会改变 K/V 数值分布
  - backward 并未实现 straight-through estimator
  - 如果训练态强行开启量化，`round` 会带来不可导或近零梯度问题
  - 结合运行入口中的 [`model.half().eval().cuda()`](../accuracy/lm_eval/run_lm_eval_harness.py#L129-L129) 与 [`torch.no_grad()`](../accuracy/lm_eval/run_lm_eval_harness.py#L139-L140)，这套实现明显是为推理评测准备的

#### 3.2.6 一个非常重要的实现含义

量化发生在 `past_key_value` 被组装之后，因此它不仅影响当前 step 的 K/V，也会影响后续复用的缓存值。

但由于反量化后缓存仍然以浮点张量形式存在，所以：

- 信息内容被低比特化了
- 存储类型却没有真的变成 `int8`

这再次说明它更像“低比特 KV cache 的数值仿真”，而不是“真实压缩存储格式”。

### 3.3 `OPTDecoderLayer.__init__`

#### 3.3.1 函数基本信息

- 所属类: `OPTDecoderLayer`
- 原始实现: [`modeling_opt.py#L246-L265`](../transformers/src/transformers/models/opt/modeling_opt.py#L246-L265)
- 量化实现: [`modeling_opt_orig.py#L314-L334`](../accuracy/src/modeling_opt_orig.py#L314-L334)
- 功能简介: 构造单层 decoder block

#### 3.3.2 修改内容对比

```diff
-class OPTDecoderLayer(nn.Module):
-    def __init__(self, config: OPTConfig):
+class OPTDecoderLayer(nn.Module):
+    def __init__(self, config: OPTConfig, layer_idx: int):
         ...
         self.self_attn = OPTAttention(
             ...,
+            layer_idx=layer_idx,
         )
```

#### 3.3.3 修改点拆解

- 构造函数增加 `layer_idx`
- `layer_idx` 向 `OPTAttention` 透传
- `fc1/fc2` 仍是普通 `nn.Linear`

#### 3.3.4 修改原因

它的主要作用是把层号带到 attention 模块里，方便外部工具或后续实验按层区分行为。对量化本身来说，这是一条“控制链路”，不是量化算子本体。

#### 3.3.5 对模型行为的影响

- 精度: 无直接影响
- 性能: 无直接影响
- forward/backward: 无直接影响

### 3.4 `OPTDecoderLayer.forward`

#### 3.4.1 函数基本信息

- 所属类: `OPTDecoderLayer`
- 原始实现: [`modeling_opt.py#L267-L342`](../transformers/src/transformers/models/opt/modeling_opt.py#L267-L342)
- 量化实现: [`modeling_opt_orig.py#L336-L409`](../accuracy/src/modeling_opt_orig.py#L336-L409)
- 功能简介: 执行单层 attention + MLP + residual + layer norm

#### 3.4.2 修改内容对比

```diff
         hidden_states = self.fc1(hidden_states)
         hidden_states = self.activation_fn(hidden_states)
         hidden_states = self.fc2(hidden_states)
-        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
```

#### 3.4.3 修改点拆解

- 量化版少了一次 `fc2` 后 dropout
- attention 调用接口本身没有为了量化额外加参数
- 真正的量化仍然发生在 `self.self_attn(...)` 内部

#### 3.4.4 修改原因

这不是量化所必需的改动，更像是该文件基于旧版上游代码分叉后保留下来的版本差异。

#### 3.4.5 对模型行为的影响

- 精度: 训练态下可能改变 MLP 输出分布和正则化强度
- 性能: 差异很小
- forward/backward: 与量化关系弱，更偏向实现版本差异

### 3.5 `OPTPreTrainedModel._set_gradient_checkpointing`

#### 3.5.1 函数基本信息

- 所属类: `OPTPreTrainedModel`
- 量化实现: [`modeling_opt_orig.py#L450-L452`](../accuracy/src/modeling_opt_orig.py#L450-L452)
- 上游现状: 当前参考文件中已无该函数
- 功能简介: 旧版 Hugging Face 风格的 gradient checkpointing 开关

#### 3.5.2 修改内容对比

```diff
+def _set_gradient_checkpointing(self, module, value=False):
+    if isinstance(module, (OPTDecoder)):
+        module.gradient_checkpointing = value
```

#### 3.5.3 修改点拆解

- 量化版保留了旧版接口
- 这与量化逻辑无直接耦合

#### 3.5.4 修改原因

主要是与旧版 `OPTDecoder.forward` 中的 checkpointing 写法保持一致，而不是为了量化。

#### 3.5.5 对模型行为的影响

- 精度: 无
- 性能: 只影响训练时显存-算力权衡
- forward/backward: 与量化无直接关系

### 3.6 `OPTDecoder.__init__`

#### 3.6.1 函数基本信息

- 所属类: `OPTDecoder`
- 原始实现: [`modeling_opt.py#L454-L489`](../transformers/src/transformers/models/opt/modeling_opt.py#L454-L489)
- 量化实现: [`modeling_opt_orig.py#L525-L560`](../accuracy/src/modeling_opt_orig.py#L525-L560)
- 功能简介: 构造 decoder 容器和所有层

#### 3.6.2 修改内容对比

```diff
-self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
+self.layers = nn.ModuleList([OPTDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
```

#### 3.6.3 修改点拆解

- 遍历层时显式传入层号
- 为每个 `OPTAttention` 补充 `layer_idx`

#### 3.6.4 修改原因

这是为了完成量化控制链路的层级透传。虽然当前文件里 `layer_idx` 未用于条件分支，但它为逐层分析、日志记录或后续差异化量化策略提供了基础。

#### 3.6.5 对模型行为的影响

- 精度: 无直接影响
- 性能: 无直接影响
- forward/backward: 无直接影响

### 3.7 `OPTDecoder._prepare_decoder_attention_mask`

#### 3.7.1 函数基本信息

- 所属类: `OPTDecoder`
- 量化实现: [`modeling_opt_orig.py#L568-L590`](../accuracy/src/modeling_opt_orig.py#L568-L590)
- 上游现状: 当前参考文件中已移除，改用 [`_prepare_4d_causal_attention_mask`](../transformers/src/transformers/modeling_attn_mask_utils.py)
- 功能简介: 生成 decoder 的 4D causal mask

#### 3.7.2 修改内容对比

```diff
+def _prepare_decoder_attention_mask(...):
+    if input_shape[-1] > 1:
+        combined_attention_mask = _make_causal_mask(...)
+    if attention_mask is not None:
+        expanded_attn_mask = _expand_mask(...)
+        combined_attention_mask = ...
```

#### 3.7.3 修改点拆解

- 量化版内联实现了 mask 逻辑
- 上游版把它抽到公共工具函数里
- 与量化无直接关系

#### 3.7.4 修改原因

这是代码基线版本不同导致的差异，不是量化所必须的改动。

#### 3.7.5 对模型行为的影响

- 精度: 理论上应等价
- 性能: 差别很小
- forward/backward: 无量化相关影响

### 3.8 `OPTDecoder.forward`

#### 3.8.1 函数基本信息

- 所属类: `OPTDecoder`
- 原始实现: [`modeling_opt.py#L497-L685`](../transformers/src/transformers/models/opt/modeling_opt.py#L497-L685)
- 量化实现: [`modeling_opt_orig.py#L592-L786`](../accuracy/src/modeling_opt_orig.py#L592-L786)
- 功能简介: decoder 主执行入口，负责 mask、位置编码、逐层调用和 cache 管理

#### 3.8.2 修改内容对比

```diff
-causal_attention_mask = _prepare_4d_causal_attention_mask(...)
+causal_attention_mask = self._prepare_decoder_attention_mask(...)

-layer_outputs = self._gradient_checkpointing_func(
-    decoder_layer.__call__, ...
-)
+def create_custom_forward(module):
+    def custom_forward(*inputs):
+        return module(*inputs, output_attentions, None)
+    return custom_forward
+layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(decoder_layer), ...)
```

#### 3.8.3 修改点拆解

- 使用旧版 mask 准备逻辑
- 使用旧版 checkpointing 包装方式
- 没有在这里直接插入量化算子
- 量化仍通过 `decoder_layer -> self_attn.forward` 传导

#### 3.8.4 修改原因

这说明量化实现并没有改 `OPTDecoder` 的主接口和主路径，而是把变化局部化在 attention 内部。这是一个很典型的“最小侵入式”设计：

- `OPTModel` / `OPTForCausalLM` API 不需要改
- 权重加载路径不需要改
- 调用方也不需要感知量化细节

#### 3.8.5 对模型行为的影响

- 精度: 由 attention 内部量化主导
- 性能: decoder 自身没有额外量化收益
- forward/backward: 公共接口保持兼容

## 4. 关键模块分析

### 4.1 `OPTAttention`

这是唯一真正发生量化行为的核心模块：

- 原始投影层仍是 `q_proj/k_proj/v_proj/out_proj = nn.Linear`
- 没有替换为 `QuantLinear`
- `Q` 没有量化
- `K/V` 在投影后、attention score 计算前被量化再反量化
- cache 复用路径会继续使用已被低比特化后的 K/V 数值

因此，`OPTAttention` 体现的是“中间激活量化”，不是“权重层量化”。

### 4.2 `OPTDecoderLayer`

`OPTDecoderLayer` 只承担两件与量化相关的事情：

- 把 `layer_idx` 传给 `OPTAttention`
- 调用已经被改写的 attention forward

除此之外，MLP 路径 `fc1/fc2` 没有量化逻辑，也没有 scale、clamp、fake-quant 节点。

### 4.3 `OPTModel` / `OPTForCausalLM`

- `OPTModel` 定义见 [`modeling_opt.py#L687-L751`](../transformers/src/transformers/models/opt/modeling_opt.py#L687-L751) 和 [`modeling_opt_orig.py#L788-L852`](../accuracy/src/modeling_opt_orig.py#L788-L852)
- `OPTForCausalLM` 定义见 [`modeling_opt.py#L753-L970`](../transformers/src/transformers/models/opt/modeling_opt.py#L753-L970) 和 [`modeling_opt_orig.py#L854-L1071`](../accuracy/src/modeling_opt_orig.py#L854-L1071)

这两个顶层模块没有直接增加量化逻辑。它们保持了原有 API 和输出格式，因此量化对外部调用方是透明的。

这也是本实现的优点之一：把变化尽量约束在内部 attention 模块，避免影响模型封装层。

### 4.4 所有 `Linear` 层相关改动

从两份文件对比可得出非常明确的结论：

- `q_proj/k_proj/v_proj/out_proj` 仍是 `nn.Linear`
- `fc1/fc2` 仍是 `nn.Linear`
- `project_in/project_out` 仍是 `nn.Linear`
- `lm_head` 仍是 `nn.Linear`
- 没有新增 `QuantLinear`
- 没有 packed weight
- 没有离线量化权重参数
- 没有 observer、calibration buffer、fake-quant module

所以本实现并没有“改 Linear 层本身”，而是“改 Linear 层输出后的 attention 中间张量”。

## 5. 量化实现机制总结

### 5.1 量化发生在哪里

量化只发生在：

- `OPTAttention.forward` 中的 `key_states`
- `OPTAttention.forward` 中的 `value_states`

不发生在：

- 权重
- `query_states`
- attention logits
- softmax 概率
- MLP 激活
- `lm_head`

### 5.2 量化粒度

这套实现最接近：

- 动态量化
- 激活动态量化
- 分组量化
- 非对称量化

更细地说，它的量化参数是按当前 batch/head/token/group 在线计算的。因为 `min/max` 是对 `key_states[:, :, start:end]` 和 `value_states[:, :, start:end]` 的最后一维求得，所以 scale 会随着：

- 当前样本
- 当前 head
- 当前时间步
- 当前 64 维 group

动态变化。

### 5.3 是否使用 fake quant / calibration

- 没有独立的 calibration 过程
- 没有 observer
- 没有静态预先统计的 scale/zero-point
- 但“量化后立刻反量化”的行为在语义上属于 QDQ / fake-quant 风格

### 5.4 `scale` 与 `zero-point` 如何计算和使用

代码虽然没有显式命名 `scale` 和 `zero_point`，但可以等价写成：

- `scale = (max - min) / qmax`
- `zero_point = -min / scale`
- `x_q = round(x / scale + zero_point)`
- `x_hat = (x_q - zero_point) * scale`

实现里用了更直接的 `min/max` 形式：

- 量化: `round(((x - min) / delta) * qmax)`
- 反量化: `(delta * x_q / qmax) + min`

其中 `delta = max - min`。

## 6. 关键设计权衡

### 6.1 精度 vs 性能

优点：

- 只处理 K/V，侵入性小
- 不改顶层 API，不影响权重加载
- 很适合快速评估低比特 KV 对生成精度的影响

代价：

- 没有真实整数存储和整数算子
- Python 循环逐组处理会增加额外开销
- 在这两个文件的范围内，看不到明显的真实推理加速路径

### 6.2 静态 vs 动态量化

本实现选择动态量化：

- 优点是实现简单，不需要 calibration
- 缺点是每个 forward 都要重新计算 `min/max`
- 对部署而言，动态计算量化参数会带来额外算子开销

### 6.3 推理 vs 训练

这套设计明显偏推理：

- 入口脚本在评测时使用 `eval` 与 `no_grad`
- 没有训练态量化感知机制
- 没有 straight-through estimator

因此它更适合：

- accuracy study
- ablation
- 低比特 KV cache 行为分析

而不适合直接作为量化训练框架。

## 7. 总结

相对于上游 `transformers/src/transformers/models/opt/modeling_opt.py`，本项目的 `accuracy/src/modeling_opt_orig.py` 通过一种非常局部的方式实现了 OPT 的量化改造：

- 不改模型总体结构
- 不改权重层类型
- 不改顶层模型 API
- 只在 `OPTAttention.forward` 内对 `K/V` 激活插入分组量化-反量化逻辑

最关键的修改点有三个：

- 在 `OPTAttention.__init__` 中加入 `enable_quant` 和 `qbits`
- 在 `OPTAttention.forward` 中对 `key_states/value_states` 执行基于 `min/max` 的 64 维分组动态量化
- 在 `OPTDecoderLayer` 与 `OPTDecoder` 中补充 `layer_idx` 透传，为逐层控制留出接口

它与原始实现的本质区别不是“把 OPT 改成了一个真正的 INT8 模型”，而是“在保持浮点执行框架不变的前提下，模拟了低比特 KV 表示对 attention 行为的影响”。

换句话说，这是一种以最小代码改动完成的 KV 激活量化实验实现，而不是完整的生产级量化部署实现。
