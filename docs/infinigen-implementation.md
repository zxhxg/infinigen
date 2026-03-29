# InfiniGen 在 `run_lm_eval_harness.py` 中的实现说明

本文档解释 `D:\infinigen\accuracy\lm_eval\run_lm_eval_harness.py` 是怎样把标准模型切换到 InfiniGen 实现的，并串起 `accuracy/src/modeling_opt_ours.py` 和 `accuracy/src/modeling_llama_ours.py` 中真正的改造逻辑。阅读目标是让读者理解三件事：

1. 入口脚本如何切换到 InfiniGen 版本模型。
2. `run_lm_eval_harness.py` 如何给每层 attention 注入 InfiniGen 所需参数。
3. InfiniGen attention 是如何利用上一层隐藏状态预测 KV 取用模式，并把预测结果变成真正生效的 attention mask。

## 1. 命令行入口在哪里

InfiniGen 的启动脚本是 [ours.sh#L1](../accuracy/lm_eval/ours.sh#L1) 到 [ours.sh#L43](../accuracy/lm_eval/ours.sh#L43)。

这个脚本接收的核心输入有：

- `model_path`: 本地模型路径
- `model`: 模型名
- `model_arch`: 架构名，如 `opt` 或 `llama`
- `partial_weight`: 层内稀疏参数比例
- `alpha`
- `capacity`
- `budget`
- 可选的 `no_skewing`

其中几个重要的外部文件路径由 shell 脚本拼出来：

- 部分权重目录 `weight_path`: [ours.sh#L15](../accuracy/lm_eval/ours.sh#L15) 到 [ours.sh#L19](../accuracy/lm_eval/ours.sh#L19)
- skewing matrix 路径 `skewing_path`: [ours.sh#L20](../accuracy/lm_eval/ours.sh#L20)

真正启动推理的命令在 [ours.sh#L22](../accuracy/lm_eval/ours.sh#L22) 到 [ours.sh#L34](../accuracy/lm_eval/ours.sh#L34)。

## 2. `run_lm_eval_harness.py` 如何识别 InfiniGen 模式

InfiniGen 相关参数定义在 [run_lm_eval_harness.py#L48](../accuracy/lm_eval/run_lm_eval_harness.py#L48) 到 [run_lm_eval_harness.py#L55](../accuracy/lm_eval/run_lm_eval_harness.py#L55)：

- `--ours`
- `--partial_weight_ratio`
- `--partial_weight_path`
- `--skewing_matrix_path`
- `--alpha`
- `--capacity`
- `--budget`

只要传了 `--ours`，脚本就会进入 InfiniGen 路径。

## 3. 为什么 InfiniGen 先要替换 `transformers` 里的模型实现

`run_lm_eval_harness.py` 最关键的切换点是 [run_lm_eval_harness.py#L58](../accuracy/lm_eval/run_lm_eval_harness.py#L58) 到 [run_lm_eval_harness.py#L61](../accuracy/lm_eval/run_lm_eval_harness.py#L61)。

如果 `args.ours` 为真，脚本会执行：

- [run_lm_eval_harness.py#L59](../accuracy/lm_eval/run_lm_eval_harness.py#L59)

也就是把 Hugging Face 模型源码链接到：

- OPT: [modeling_opt_ours.py](../accuracy/src/modeling_opt_ours.py)
- LLaMA: [modeling_llama_ours.py](../accuracy/src/modeling_llama_ours.py)

这里的关键认识是：

- H2O 是“模型对象加载出来以后再替换 attention”
- InfiniGen 是“模型类本身就换成了自定义实现”

换句话说，InfiniGen 改动得更深，它不是在标准 attention 外面包一层，而是直接把模型实现文件切到一套修改版。

## 4. `set_symlink(...)` 的相对路径实际指向哪里

`set_symlink(...)` 在 [run_lm_eval_harness.py#L8](../accuracy/lm_eval/run_lm_eval_harness.py#L8) 到 [run_lm_eval_harness.py#L24](../accuracy/lm_eval/run_lm_eval_harness.py#L24)。

从 `accuracy/lm_eval` 这个工作目录出发：

- `../src` 实际指向 [accuracy/src](../accuracy/src)
- `../transformers/src/transformers/models/...` 实际指向 [transformers/src/transformers/models](../transformers/src/transformers/models)

因此，运行这个脚本时，仓库内置的自定义 `modeling_*_ours.py` 会临时替换 Hugging Face 对应架构的实现文件。

## 5. 模型是怎样被加载的

模型加载代码在 [run_lm_eval_harness.py#L68](../accuracy/lm_eval/run_lm_eval_harness.py#L68) 到 [run_lm_eval_harness.py#L74](../accuracy/lm_eval/run_lm_eval_harness.py#L74)。

这里有一个很重要的细节：

- `config` 和 `tokenizer` 来自 `model_name`
- 真正的权重在 `args.model_path` 不为空时来自 `model_path`

对应代码：

- config: [run_lm_eval_harness.py#L69](../accuracy/lm_eval/run_lm_eval_harness.py#L69)
- tokenizer: [run_lm_eval_harness.py#L70](../accuracy/lm_eval/run_lm_eval_harness.py#L70)
- model load: [run_lm_eval_harness.py#L71](../accuracy/lm_eval/run_lm_eval_harness.py#L71) 到 [run_lm_eval_harness.py#L74](../accuracy/lm_eval/run_lm_eval_harness.py#L74)

`ours.sh` 总是传 `--model-path`，见 [ours.sh#L30](../accuracy/lm_eval/ours.sh#L30)。这说明 InfiniGen 使用的是本地准备好的模型目录，而不是直接拿官方模型名裸加载。

## 6. `run_lm_eval_harness.py` 对 InfiniGen 做了哪些参数注入

InfiniGen 的参数注入在 [run_lm_eval_harness.py#L109](../accuracy/lm_eval/run_lm_eval_harness.py#L109) 到 [run_lm_eval_harness.py#L127](../accuracy/lm_eval/run_lm_eval_harness.py#L127)。

这里做的不是替换类，而是把运行时控制参数写进每一层 attention 对象。

### 6.1 OPT 分支

OPT 的循环在 [run_lm_eval_harness.py#L110](../accuracy/lm_eval/run_lm_eval_harness.py#L110) 到 [run_lm_eval_harness.py#L116](../accuracy/lm_eval/run_lm_eval_harness.py#L116)。

每一层 `self_attn` 会被写入：

- `partial_weight_ratio`
- `partial_weight_q`
- `alpha`
- `capacity`
- `budget`

### 6.2 LLaMA 分支

LLaMA 的循环在 [run_lm_eval_harness.py#L117](../accuracy/lm_eval/run_lm_eval_harness.py#L117) 到 [run_lm_eval_harness.py#L127](../accuracy/lm_eval/run_lm_eval_harness.py#L127)。

除了与 OPT 相同的字段以外，还会额外加载整份 skewing matrix：

- 加载矩阵文件: [run_lm_eval_harness.py#L118](../accuracy/lm_eval/run_lm_eval_harness.py#L118) 到 [run_lm_eval_harness.py#L119](../accuracy/lm_eval/run_lm_eval_harness.py#L119)
- 写入每层 `self_attn.skewing_matrix`: [run_lm_eval_harness.py#L126](../accuracy/lm_eval/run_lm_eval_harness.py#L126) 到 [run_lm_eval_harness.py#L127](../accuracy/lm_eval/run_lm_eval_harness.py#L127)

这里可以看到 `run_lm_eval_harness.py` 的角色非常明确：

- 它不负责生成这些离线参数
- 它只负责把离线准备好的 `partial_weight_q` 和 `skewing_matrix` 注入模型

## 7. InfiniGen 的“真正实现”藏在 attention 类内部

理解 InfiniGen 的重点不在 harness，而在 `accuracy/src/modeling_*_ours.py`。

### 7.1 OPT attention 的入口

OPT 的自定义 attention 类是 [modeling_opt_ours.py#L123](../accuracy/src/modeling_opt_ours.py#L123) 到 [modeling_opt_ours.py#L340](../accuracy/src/modeling_opt_ours.py#L340) 的 `OPTAttention`。

### 7.2 LLaMA attention 的入口

LLaMA 的自定义 attention 类是 [modeling_llama_ours.py#L160](../accuracy/src/modeling_llama_ours.py#L160) 到 [modeling_llama_ours.py#L361](../accuracy/src/modeling_llama_ours.py#L361) 的 `LlamaAttention`。

两者共享同一条主思路：

1. 当前层保留自己的 `current_hidden_states`
2. 下一层拿到上一层传过来的 `previous_hidden_states`
3. 用这些上一层隐藏状态做一次“投机性” attention 估计
4. 根据这次估计生成一个 KV fetch mask
5. 再把这个 mask 叠加到真实 attention 上

因此，InfiniGen 的本质不是 H2O 那种“基于真实 attention 历史挑 key”，而是“用跨层传递的隐藏状态先预测哪些 key 值得取，再约束真实 attention”。

## 8. attention 对象里新增了哪些运行时状态

OPT 在构造函数里新增的字段见 [modeling_opt_ours.py#L157](../accuracy/src/modeling_opt_ours.py#L157) 到 [modeling_opt_ours.py#L168](../accuracy/src/modeling_opt_ours.py#L168)。

LLaMA 在构造函数里新增的字段见 [modeling_llama_ours.py#L185](../accuracy/src/modeling_llama_ours.py#L185) 到 [modeling_llama_ours.py#L197](../accuracy/src/modeling_llama_ours.py#L197)。

共同的重要字段包括：

- `previous_hidden_states`
- `current_hidden_states`
- `partial_weight_q`
- `alpha`
- `capacity`
- `budget`
- `eviction_policy`
- `density`

这些字段把 InfiniGen 的运行逻辑拆成两个阶段：

- 预测阶段：利用 `previous_hidden_states` 和 `partial_weight_q`
- 约束阶段：利用 `alpha / capacity / budget / eviction_policy`

## 9. `current_hidden_states` 是在哪里保存的

每次 forward 一开始，attention 都会先把当前输入 `hidden_states` 复制下来。

OPT:

- [modeling_opt_ours.py#L252](../accuracy/src/modeling_opt_ours.py#L252) 到 [modeling_opt_ours.py#L253](../accuracy/src/modeling_opt_ours.py#L253)

LLaMA:

- [modeling_llama_ours.py#L281](../accuracy/src/modeling_llama_ours.py#L281) 到 [modeling_llama_ours.py#L282](../accuracy/src/modeling_llama_ours.py#L282)

这一步很关键，因为 decoder 在跑完当前层以后，会把这一层保存下来的 `current_hidden_states` 传给下一层，作为 `previous_hidden_states`。

## 10. 层与层之间是如何传递 `previous_hidden_states` 的

这部分不在 attention 里，而在 decoder 主循环里。

### 10.1 OPT 的跨层传递

OPT decoder 在 [modeling_opt_ours.py#L793](../accuracy/src/modeling_opt_ours.py#L793) 到 [modeling_opt_ours.py#L801](../accuracy/src/modeling_opt_ours.py#L801) 把第 `idx` 层的当前隐藏状态传给第 `idx + 1` 层：

- 读取当前层保存的 `current_hidden_states`
- 在最后一维拼接一个常数 1
- 写入下一层的 `self_attn.previous_hidden_states`

这里拼接常数 1 的原因，要和 OPT attention 里 `q_proj / k_proj` 的改造一起看。

### 10.2 LLaMA 的跨层传递

LLaMA decoder 在 [modeling_llama_ours.py#L700](../accuracy/src/modeling_llama_ours.py#L700) 到 [modeling_llama_ours.py#L718](../accuracy/src/modeling_llama_ours.py#L718) 做同样的事情。

当前启用的是最简单的 `i -> i + 1` 传递策略：

- [modeling_llama_ours.py#L703](../accuracy/src/modeling_llama_ours.py#L703) 到 [modeling_llama_ours.py#L704](../accuracy/src/modeling_llama_ours.py#L704)

你会看到代码里还保留了 `i -> i + 2`、`i -> i + 3` 的注释版本，这说明作者实验过更远层间传递，但当前启用的是只传给下一层。

## 11. OPT 版 InfiniGen 为什么改了 `q_proj / k_proj` 的维度

OPT 的 `q_proj` 和 `k_proj` 定义在：

- [modeling_opt_ours.py#L149](../accuracy/src/modeling_opt_ours.py#L149) 到 [modeling_opt_ours.py#L152](../accuracy/src/modeling_opt_ours.py#L152)

与常规实现不同的是：

- `k_proj = nn.Linear(embed_dim, embed_dim + 1, bias=False)`
- `q_proj = nn.Linear(embed_dim, embed_dim + 1, bias=False)`

随后 forward 会构造 `new_attn_in`，即把当前 `hidden_states` 的最后一维再拼接一个全 1 通道：

- [modeling_opt_ours.py#L255](../accuracy/src/modeling_opt_ours.py#L255) 到 [modeling_opt_ours.py#L257](../accuracy/src/modeling_opt_ours.py#L257)

接着它用 `self.q_proj.weight.data` 和 `self.k_proj.weight.data` 直接去乘这个扩展后的输入：

- query: [modeling_opt_ours.py#L257](../accuracy/src/modeling_opt_ours.py#L257)
- key: [modeling_opt_ours.py#L265](../accuracy/src/modeling_opt_ours.py#L265)

这说明 OPT 版 InfiniGen 在实现上等价于把一个常数偏置通道并入了投影输入，再让下一层收到的 `previous_hidden_states` 也保持同样格式。

## 12. LLaMA 版 InfiniGen 的 attention 改造思路

LLaMA 没有像 OPT 那样扩展 `q_proj / k_proj` 的输入维度。它的投影仍然是标准形状：

- [modeling_llama_ours.py#L176](../accuracy/src/modeling_llama_ours.py#L176) 到 [modeling_llama_ours.py#L179](../accuracy/src/modeling_llama_ours.py#L179)

它的改造重点在“投机 query 的构造”上，而不是在投影层形状上。

## 13. InfiniGen 的投机性 attention 是怎么生成的

这是整套实现最核心的地方。

### 13.1 OPT 的投机 attention

OPT 的 speculative 分支在 [modeling_opt_ours.py#L276](../accuracy/src/modeling_opt_ours.py#L276) 到 [modeling_opt_ours.py#L284](../accuracy/src/modeling_opt_ours.py#L284)。

触发条件是：

- `previous_hidden_states is not None`
- `partial_weight_q is not None`

满足后它会：

1. 用上一层隐藏状态乘 `partial_weight_q` 得到一个压缩后的 query
   - [modeling_opt_ours.py#L277](../accuracy/src/modeling_opt_ours.py#L277) 到 [modeling_opt_ours.py#L279](../accuracy/src/modeling_opt_ours.py#L279)
2. 再和当前层 `key_states` 做乘法，得到一份“预测 attention”
   - [modeling_opt_ours.py#L280](../accuracy/src/modeling_opt_ours.py#L280)
3. 把这份预测 attention 送入 `kv_cache_mask(...)`
   - [modeling_opt_ours.py#L282](../accuracy/src/modeling_opt_ours.py#L282)

注意，这里用来生成 mask 的不是最终真实 attention，而是一个 cheaper 的代理 attention。

### 13.2 LLaMA 的投机 attention

LLaMA 的 speculative 分支在 [modeling_llama_ours.py#L305](../accuracy/src/modeling_llama_ours.py#L305) 到 [modeling_llama_ours.py#L317](../accuracy/src/modeling_llama_ours.py#L317)。

它比 OPT 多了几步：

1. 先用上一层隐藏状态和 `q_proj.weight^T` 重建 query
   - [modeling_llama_ours.py#L307](../accuracy/src/modeling_llama_ours.py#L307)
2. 给这份 query 应用 RoPE
   - [modeling_llama_ours.py#L308](../accuracy/src/modeling_llama_ours.py#L308)
3. 用 `skewing_matrix` 变换 query 和 key
   - query 变换: [modeling_llama_ours.py#L309](../accuracy/src/modeling_llama_ours.py#L309)
   - key 变换: [modeling_llama_ours.py#L313](../accuracy/src/modeling_llama_ours.py#L313)
4. 用 `partial_weight_q` 生成一个通道级 mask，只保留部分 query 维度
   - [modeling_llama_ours.py#L310](../accuracy/src/modeling_llama_ours.py#L310) 到 [modeling_llama_ours.py#L311](../accuracy/src/modeling_llama_ours.py#L311)
5. 用这份变换后的 query / key 算预测 attention
   - [modeling_llama_ours.py#L313](../accuracy/src/modeling_llama_ours.py#L313)
6. 把预测 attention 送进 `kv_cache_mask(...)`
   - [modeling_llama_ours.py#L315](../accuracy/src/modeling_llama_ours.py#L315)

因此，LLaMA 版 InfiniGen 比 OPT 版多了两层结构：

- `skewing_matrix`
- `partial_weight_q` 对 query 通道的显式稀疏门控

## 14. `kv_cache_mask(...)` 在做什么

无论 OPT 还是 LLaMA，真正把“预测 attention”变成“可执行 mask”的函数都是 `kv_cache_mask(...)`。

- OPT: [modeling_opt_ours.py#L173](../accuracy/src/modeling_opt_ours.py#L173) 到 [modeling_opt_ours.py#L235](../accuracy/src/modeling_opt_ours.py#L235)
- LLaMA: [modeling_llama_ours.py#L203](../accuracy/src/modeling_llama_ours.py#L203) 到 [modeling_llama_ours.py#L270](../accuracy/src/modeling_llama_ours.py#L270)

它不是 H2O 那种 heavy+recent 规则，而是一套“预测 fetch 数量 + 容量控制 + 驱逐策略”的机制。

### 14.1 `alpha` 的作用

函数先拿每个 query 的最大 attention 值，再用 `max - alpha` 当阈值：

- OPT: [modeling_opt_ours.py#L186](../accuracy/src/modeling_opt_ours.py#L186) 到 [modeling_opt_ours.py#L188](../accuracy/src/modeling_opt_ours.py#L188)
- LLaMA: [modeling_llama_ours.py#L220](../accuracy/src/modeling_llama_ours.py#L220) 到 [modeling_llama_ours.py#L222](../accuracy/src/modeling_llama_ours.py#L222)

含义是：

- 离当前 query 的峰值越近的 key，越可能被保留
- `alpha` 越小，阈值越高，保留越少
- `alpha` 越大，阈值越低，保留越多

### 14.2 `fetch_num` 的作用

代码会统计每个 query 下有多少 key 超过阈值，然后在 head 维度做平均：

- OPT: [modeling_opt_ours.py#L190](../accuracy/src/modeling_opt_ours.py#L190) 到 [modeling_opt_ours.py#L193](../accuracy/src/modeling_opt_ours.py#L193)
- LLaMA: [modeling_llama_ours.py#L222](../accuracy/src/modeling_llama_ours.py#L222) 到 [modeling_llama_ours.py#L227](../accuracy/src/modeling_llama_ours.py#L227)

这个平均动作很关键，因为实现要求每个 head 在同一个 query 上取用同样多的 KV。

### 14.3 `budget` 的作用

`fetch_num` 最终会被截断到 `fetch_max = int(src_len * budget)`：

- OPT: [modeling_opt_ours.py#L192](../accuracy/src/modeling_opt_ours.py#L192)
- LLaMA: [modeling_llama_ours.py#L226](../accuracy/src/modeling_llama_ours.py#L226)

这个量代表：

- 每一层、每个 query 最多允许预取多少比例的 KV

所以 `budget` 控制的是单步最大取用强度。

### 14.4 `capacity` 的作用

`store_max = int(src_len * capacity)` 决定最多在缓存中留多少比例的历史 key：

- OPT: [modeling_opt_ours.py#L195](../accuracy/src/modeling_opt_ours.py#L195)
- LLaMA: [modeling_llama_ours.py#L229](../accuracy/src/modeling_llama_ours.py#L229)

脚本还强制要求：

- `budget < capacity`

对应：

- OPT: [modeling_opt_ours.py#L177](../accuracy/src/modeling_opt_ours.py#L177)
- LLaMA: [modeling_llama_ours.py#L207](../accuracy/src/modeling_llama_ours.py#L207)

这意味着：

- 单次预取量不能超过缓存总容量

### 14.5 初始化和稳定期

在最初的 `fetch_max` 阶段，代码直接允许完整下三角访问：

- OPT: [modeling_opt_ours.py#L197](../accuracy/src/modeling_opt_ours.py#L197)
- LLaMA: [modeling_llama_ours.py#L231](../accuracy/src/modeling_llama_ours.py#L231)

在 `fetch_max` 到 `store_max` 之间，代码只按每个 query 的 top-k 预测结果去选 key：

- OPT: [modeling_opt_ours.py#L199](../accuracy/src/modeling_opt_ours.py#L199) 到 [modeling_opt_ours.py#L201](../accuracy/src/modeling_opt_ours.py#L201)
- LLaMA: [modeling_llama_ours.py#L233](../accuracy/src/modeling_llama_ours.py#L233) 到 [modeling_llama_ours.py#L235](../accuracy/src/modeling_llama_ours.py#L235)

### 14.6 容量满了之后怎样驱逐

超过 `store_max` 后，代码一边为当前 query 选 top-k key，一边在后续 query 的可选 key 中驱逐一部分历史位置：

- OPT: [modeling_opt_ours.py#L203](../accuracy/src/modeling_opt_ours.py#L203) 到 [modeling_opt_ours.py#L231](../accuracy/src/modeling_opt_ours.py#L231)
- LLaMA: [modeling_llama_ours.py#L237](../accuracy/src/modeling_llama_ours.py#L237) 到 [modeling_llama_ours.py#L265](../accuracy/src/modeling_llama_ours.py#L265)

实现里支持 3 种驱逐策略：

- `fifo`
- `lru`
- `counter`

默认值是 `counter`：

- OPT: [modeling_opt_ours.py#L166](../accuracy/src/modeling_opt_ours.py#L166)
- LLaMA: [modeling_llama_ours.py#L195](../accuracy/src/modeling_llama_ours.py#L195)

当前默认策略的逻辑是：

- 统计到当前位置为止，每个 key 被 fetch 过多少次
- 把最少被取用的 key 从未来 query 的候选集中剔掉

对应：

- OPT: [modeling_opt_ours.py#L224](../accuracy/src/modeling_opt_ours.py#L224) 到 [modeling_opt_ours.py#L228](../accuracy/src/modeling_opt_ours.py#L228)
- LLaMA: [modeling_llama_ours.py#L258](../accuracy/src/modeling_llama_ours.py#L258) 到 [modeling_llama_ours.py#L262](../accuracy/src/modeling_llama_ours.py#L262)

### 14.7 `density` 是什么

`kv_cache_mask(...)` 最后会统计一次保留密度：

- OPT: [modeling_opt_ours.py#L233](../accuracy/src/modeling_opt_ours.py#L233)
- LLaMA: [modeling_llama_ours.py#L267](../accuracy/src/modeling_llama_ours.py#L267)

然后把布尔保留矩阵变成可加到 attention 上的数值 mask：

- 保留位置写成 `0`
- 屏蔽位置写成 `-10000`

对应：

- OPT: [modeling_opt_ours.py#L234](../accuracy/src/modeling_opt_ours.py#L234)
- LLaMA: [modeling_llama_ours.py#L268](../accuracy/src/modeling_llama_ours.py#L268)

## 15. 预测出来的 mask 是怎样作用到真实 attention 上的

在生成完 `attn_mask` 后，真正的 attention 仍然是用标准 `query_states @ key_states^T` 算出来的。

OPT:

- 真正的 attention: [modeling_opt_ours.py#L286](../accuracy/src/modeling_opt_ours.py#L286) 到 [modeling_opt_ours.py#L287](../accuracy/src/modeling_opt_ours.py#L287)
- 叠加预测 mask: [modeling_opt_ours.py#L304](../accuracy/src/modeling_opt_ours.py#L304) 到 [modeling_opt_ours.py#L306](../accuracy/src/modeling_opt_ours.py#L306)

LLaMA:

- 真正的 attention: [modeling_llama_ours.py#L319](../accuracy/src/modeling_llama_ours.py#L319)
- 叠加预测 mask: [modeling_llama_ours.py#L321](../accuracy/src/modeling_llama_ours.py#L321) 到 [modeling_llama_ours.py#L324](../accuracy/src/modeling_llama_ours.py#L324)

然后再叠加常规 causal / padding `attention_mask`：

- OPT: [modeling_opt_ours.py#L295](../accuracy/src/modeling_opt_ours.py#L295) 到 [modeling_opt_ours.py#L302](../accuracy/src/modeling_opt_ours.py#L302)
- LLaMA: [modeling_llama_ours.py#L333](../accuracy/src/modeling_llama_ours.py#L333) 到 [modeling_llama_ours.py#L339](../accuracy/src/modeling_llama_ours.py#L339)

最后 attention 再正常做 softmax：

- OPT: [modeling_opt_ours.py#L308](../accuracy/src/modeling_opt_ours.py#L308) 到 [modeling_opt_ours.py#L310](../accuracy/src/modeling_opt_ours.py#L310)
- LLaMA: [modeling_llama_ours.py#L341](../accuracy/src/modeling_llama_ours.py#L341) 到 [modeling_llama_ours.py#L345](../accuracy/src/modeling_llama_ours.py#L345)

因此，InfiniGen 不是直接替代 attention 计算，而是用“预测出来的数值 mask”去约束真实 attention。

## 16. `partial_weight_q` 在这套实现里扮演什么角色

从 `run_lm_eval_harness.py` 看，`partial_weight_q` 是逐层从磁盘加载的：

- OPT: [run_lm_eval_harness.py#L113](../accuracy/lm_eval/run_lm_eval_harness.py#L113)
- LLaMA: [run_lm_eval_harness.py#L122](../accuracy/lm_eval/run_lm_eval_harness.py#L122)

在 OPT 里，它直接参与上一层隐藏状态到 speculative query 的线性变换：

- [modeling_opt_ours.py#L277](../accuracy/src/modeling_opt_ours.py#L277) 到 [modeling_opt_ours.py#L279](../accuracy/src/modeling_opt_ours.py#L279)

在 LLaMA 里，它更多地像一个 query 通道筛选器：

- [modeling_llama_ours.py#L310](../accuracy/src/modeling_llama_ours.py#L310) 到 [modeling_llama_ours.py#L311](../accuracy/src/modeling_llama_ours.py#L311)

所以两种架构虽然共用相同参数名，但用途并不完全相同：

- OPT 更像“投机 query 的直接投影权重”
- LLaMA 更像“投机 query 的通道门控模板”

## 17. `skewing_matrix` 为什么只在 LLaMA 分支出现

LLaMA 版在 harness 中会额外注入 `self_attn.skewing_matrix`：

- [run_lm_eval_harness.py#L126](../accuracy/lm_eval/run_lm_eval_harness.py#L126) 到 [run_lm_eval_harness.py#L127](../accuracy/lm_eval/run_lm_eval_harness.py#L127)

forward 中确实会使用这个字段：

- query 侧: [modeling_llama_ours.py#L309](../accuracy/src/modeling_llama_ours.py#L309)
- key 侧: [modeling_llama_ours.py#L313](../accuracy/src/modeling_llama_ours.py#L313)

实现上有一个细节值得注意：

- 构造函数里声明的是 `self.skewing_matrx = None`，见 [modeling_llama_ours.py#L191](../accuracy/src/modeling_llama_ours.py#L191)
- 但 forward 里读取的是 `self.skewing_matrix`

这说明运行时依赖 harness 在推理前动态写入正确的 `self.skewing_matrix` 字段。由于 [ours.sh#L31](../accuracy/lm_eval/ours.sh#L31) 总是传 `--skewing_matrix_path`，当前主路径是可以工作的。

## 18. 推理主循环如何统计 InfiniGen 的保留密度

`run_lm_eval_harness.py` 在每个请求前向结束后，如果开启 `ours`，就调用 `model.get_density()`：

- [run_lm_eval_harness.py#L145](../accuracy/lm_eval/run_lm_eval_harness.py#L145) 到 [run_lm_eval_harness.py#L147](../accuracy/lm_eval/run_lm_eval_harness.py#L147)

OPT 的 `get_density()` 定义在：

- [modeling_opt_ours.py#L934](../accuracy/src/modeling_opt_ours.py#L934) 到 [modeling_opt_ours.py#L939](../accuracy/src/modeling_opt_ours.py#L939)

LLaMA 的 `get_density()` 定义在：

- [modeling_llama_ours.py#L773](../accuracy/src/modeling_llama_ours.py#L773) 到 [modeling_llama_ours.py#L778](../accuracy/src/modeling_llama_ours.py#L778)

它们都是：

- 遍历所有层
- 收集每层 `self_attn.density`
- 再求平均

主循环结束后，harness 再把平均密度变成一个 `retain ratio` 并打印：

- [run_lm_eval_harness.py#L184](../accuracy/lm_eval/run_lm_eval_harness.py#L184) 到 [run_lm_eval_harness.py#L188](../accuracy/lm_eval/run_lm_eval_harness.py#L188)

这个指标不是 attention 计算本身的一部分，但它是评估 InfiniGen 稀疏程度的重要观测值。

## 19. 为什么每个请求结束后要清空 `previous_hidden_states`

在请求循环结尾，脚本会对每层 attention 执行：

- OPT: [run_lm_eval_harness.py#L177](../accuracy/lm_eval/run_lm_eval_harness.py#L177) 到 [run_lm_eval_harness.py#L179](../accuracy/lm_eval/run_lm_eval_harness.py#L179)
- LLaMA: [run_lm_eval_harness.py#L180](../accuracy/lm_eval/run_lm_eval_harness.py#L180) 到 [run_lm_eval_harness.py#L182](../accuracy/lm_eval/run_lm_eval_harness.py#L182)

也就是把 `previous_hidden_states = None`。

这是为了避免不同样本之间串状态。因为 InfiniGen 的 mask 依赖上一层传来的隐藏状态，如果不清空，下一条 request 会错误继承上一条 request 的跨层缓存。

## 20. InfiniGen 和 H2O 的实现差异可以怎样理解

两者都在做“裁掉一部分 attention 可见区域”，但思路完全不同。

H2O：

- 先算当前层完整 attention
- 根据历史累计关注和最近窗口构造规则型 mask
- mask 直接来自当前层 attention 分布

对应实现位置：

- [modify_opt.py#L190](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L190) 到 [modify_opt.py#L208](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L208)
- [modify_llama.py#L130](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L130) 到 [modify_llama.py#L147](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L147)

InfiniGen：

- 先用上一层隐藏状态构造 cheaper 的 speculative attention
- 再从 speculative attention 估计出 KV fetch mask
- 再把这个 mask 叠加到真实 attention 上

对应实现位置：

- OPT: [modeling_opt_ours.py#L276](../accuracy/src/modeling_opt_ours.py#L276) 到 [modeling_opt_ours.py#L306](../accuracy/src/modeling_opt_ours.py#L306)
- LLaMA: [modeling_llama_ours.py#L305](../accuracy/src/modeling_llama_ours.py#L305) 到 [modeling_llama_ours.py#L324](../accuracy/src/modeling_llama_ours.py#L324)

一句话概括：

- H2O 更像“基于真实 attention 历史统计做保留”
- InfiniGen 更像“基于跨层状态预测下一层需要什么 KV，再提前限制访问”

## 21. 建议按什么顺序读代码

如果要最快建立完整心智模型，建议按下面顺序点开代码：

1. [ours.sh#L22](../accuracy/lm_eval/ours.sh#L22) 到 [ours.sh#L34](../accuracy/lm_eval/ours.sh#L34)
2. [run_lm_eval_harness.py#L48](../accuracy/lm_eval/run_lm_eval_harness.py#L48) 到 [run_lm_eval_harness.py#L55](../accuracy/lm_eval/run_lm_eval_harness.py#L55)
3. [run_lm_eval_harness.py#L58](../accuracy/lm_eval/run_lm_eval_harness.py#L58) 到 [run_lm_eval_harness.py#L61](../accuracy/lm_eval/run_lm_eval_harness.py#L61)
4. [run_lm_eval_harness.py#L109](../accuracy/lm_eval/run_lm_eval_harness.py#L109) 到 [run_lm_eval_harness.py#L127](../accuracy/lm_eval/run_lm_eval_harness.py#L127)
5. OPT 读 [modeling_opt_ours.py#L173](../accuracy/src/modeling_opt_ours.py#L173) 到 [modeling_opt_ours.py#L235](../accuracy/src/modeling_opt_ours.py#L235)，再读 [modeling_opt_ours.py#L237](../accuracy/src/modeling_opt_ours.py#L237) 到 [modeling_opt_ours.py#L310](../accuracy/src/modeling_opt_ours.py#L310)
6. LLaMA 读 [modeling_llama_ours.py#L203](../accuracy/src/modeling_llama_ours.py#L203) 到 [modeling_llama_ours.py#L270](../accuracy/src/modeling_llama_ours.py#L270)，再读 [modeling_llama_ours.py#L272](../accuracy/src/modeling_llama_ours.py#L272) 到 [modeling_llama_ours.py#L345](../accuracy/src/modeling_llama_ours.py#L345)
7. 最后回到 [run_lm_eval_harness.py#L145](../accuracy/lm_eval/run_lm_eval_harness.py#L145) 到 [run_lm_eval_harness.py#L188](../accuracy/lm_eval/run_lm_eval_harness.py#L188) 看推理期间如何统计密度和清理状态

## 22. 一句话总结

`run_lm_eval_harness.py` 对 InfiniGen 的实现方式可以概括为：

- 先通过符号链接把 Hugging Face 模型类切换成自定义 `modeling_*_ours.py`
- 再把离线准备好的 `partial_weight_q`、`skewing_matrix` 和运行参数注入每一层 attention
- attention 在 forward 中利用上一层隐藏状态生成 speculative attention
- speculative attention 再通过 `alpha / budget / capacity / eviction_policy` 变成 KV fetch mask
- 最后把这个 mask 叠加到真实 attention 上，从而实现稀疏访问和密度统计

如果你要对比两种方案的设计哲学，建议把本文和 [h2o-implementation.md](./h2o-implementation.md) 对照着读。

