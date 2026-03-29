# H2O 在 `run_lm_eval_harness.py` 中的实现说明

本文档解释 `D:\infinigen\accuracy\lm_eval\run_lm_eval_harness.py` 是怎样把标准 Hugging Face Causal LM 改造成 H2O 模式的，并串起它依赖的 attention 改写代码。目标是让读者顺着文档和代码链接，完整理解这套实现的执行路径、关键数据结构和实现边界。

## 1. 这套实现的入口在哪里

H2O 的命令行入口是 [h2o.sh#L1](../accuracy/lm_eval/h2o.sh#L1) 到 [h2o.sh#L28](../accuracy/lm_eval/h2o.sh#L28)。

这个脚本做了两件事：

1. 调用 [run_lm_eval_harness.py#L26](../accuracy/lm_eval/run_lm_eval_harness.py#L26) 开始推理。
2. 调用 `evaluate_task_result.py` 对生成的 `jsonl` 结果做评测。

H2O 相关的参数在 shell 层只传了 3 个：

- `--enable_small_cache`
- `--heavy_ratio`
- `--recent_ratio`

对应命令见 [h2o.sh#L12](../accuracy/lm_eval/h2o.sh#L12) 到 [h2o.sh#L19](../accuracy/lm_eval/h2o.sh#L19)。

## 2. `run_lm_eval_harness.py` 如何识别 H2O 模式

脚本里的参数定义在 [run_lm_eval_harness.py#L33](../accuracy/lm_eval/run_lm_eval_harness.py#L33) 到 [run_lm_eval_harness.py#L56](../accuracy/lm_eval/run_lm_eval_harness.py#L56)。

其中 H2O 相关的是：

- [run_lm_eval_harness.py#L43](../accuracy/lm_eval/run_lm_eval_harness.py#L43) `--enable_small_cache`
- [run_lm_eval_harness.py#L45](../accuracy/lm_eval/run_lm_eval_harness.py#L45) `--heavy_ratio`
- [run_lm_eval_harness.py#L46](../accuracy/lm_eval/run_lm_eval_harness.py#L46) `--recent_ratio`

这三个参数决定脚本会不会进入 H2O 分支，以及 H2O 的两个保留预算比例是多少。

## 3. 运行前先做了什么模型切换

在真正加载模型之前，脚本先调用了 [run_lm_eval_harness.py#L8](../accuracy/lm_eval/run_lm_eval_harness.py#L8) 定义的 `set_symlink(...)`。

逻辑在 [run_lm_eval_harness.py#L58](../accuracy/lm_eval/run_lm_eval_harness.py#L58) 到 [run_lm_eval_harness.py#L61](../accuracy/lm_eval/run_lm_eval_harness.py#L61)：

- 如果是 `ours` 模式，链接到 `accuracy/src/modeling_{type}_ours.py`
- 否则链接到 `accuracy/src/modeling_{type}_orig.py`

H2O 不走 `ours` 分支，所以这里会把 `transformers/src/transformers/models/{model_type}/modeling_{model_type}.py` 指回仓库里的原始实现文件：

- OPT 原始实现: [modeling_opt_orig.py](../accuracy/src/modeling_opt_orig.py)
- LLaMA 原始实现: [modeling_llama_orig.py](../accuracy/src/modeling_llama_orig.py)

这里的关键点是：H2O 不是靠替换整个 Hugging Face 模型文件来实现的，而是在加载出原始模型对象以后，再把其中的 `self_attn` 模块替换掉。

## 4. 模型是怎么被加载出来的

模型加载逻辑在 [run_lm_eval_harness.py#L68](../accuracy/lm_eval/run_lm_eval_harness.py#L68) 到 [run_lm_eval_harness.py#L74](../accuracy/lm_eval/run_lm_eval_harness.py#L74)：

- `AutoConfig.from_pretrained(model_name)`
- `AutoTokenizer.from_pretrained(model_name, ...)`
- `AutoModelForCausalLM.from_pretrained(model_name, ...)`

H2O 模式下通常不会传 `--model-path`，所以这里直接从 `model_name` 加载标准权重。

## 5. H2O 分支的整体改造步骤

真正的 H2O 入口在 [run_lm_eval_harness.py#L88](../accuracy/lm_eval/run_lm_eval_harness.py#L88) 到 [run_lm_eval_harness.py#L107](../accuracy/lm_eval/run_lm_eval_harness.py#L107)。

这段代码可以拆成 5 步：

1. 导入各个模型架构对应的 H2O attention 替换函数。
2. 把 `heavy_ratio` 和 `recent_ratio` 写进 `config`。
3. 备份当前原始模型权重到 `../h2o_model/{base_name}.pt`。
4. 用 `convert_kvcache_*_heavy_recent(...)` 替换模型内部的 attention 类。
5. 把刚保存的原始权重重新加载回新模型结构中。

对应代码位置：

- 导入和路由表: [run_lm_eval_harness.py#L89](../accuracy/lm_eval/run_lm_eval_harness.py#L89) 到 [run_lm_eval_harness.py#L96](../accuracy/lm_eval/run_lm_eval_harness.py#L96)
- 写入比例参数: [run_lm_eval_harness.py#L97](../accuracy/lm_eval/run_lm_eval_harness.py#L97) 到 [run_lm_eval_harness.py#L99](../accuracy/lm_eval/run_lm_eval_harness.py#L99)
- 缓存原始权重: [run_lm_eval_harness.py#L100](../accuracy/lm_eval/run_lm_eval_harness.py#L100) 到 [run_lm_eval_harness.py#L104](../accuracy/lm_eval/run_lm_eval_harness.py#L104)
- 替换 attention 并重新加载权重: [run_lm_eval_harness.py#L105](../accuracy/lm_eval/run_lm_eval_harness.py#L105) 到 [run_lm_eval_harness.py#L107](../accuracy/lm_eval/run_lm_eval_harness.py#L107)

这里最重要的理解是：

- H2O 并没有重新训练参数。
- 它做的是“改 forward 逻辑，不改参数语义”。
- 所以可以先替换模块，再把旧权重灌回来。

## 6. attention 替换是在哪里发生的

H2O 支持的模型架构在路由表里写得很明确：

- LLaMA: [run_lm_eval_harness.py#L92](../accuracy/lm_eval/run_lm_eval_harness.py#L92) 到 [run_lm_eval_harness.py#L94](../accuracy/lm_eval/run_lm_eval_harness.py#L94)
- OPT: [run_lm_eval_harness.py#L93](../accuracy/lm_eval/run_lm_eval_harness.py#L93) 到 [run_lm_eval_harness.py#L95](../accuracy/lm_eval/run_lm_eval_harness.py#L95)
- GPT-NeoX: [run_lm_eval_harness.py#L90](../accuracy/lm_eval/run_lm_eval_harness.py#L90) 和 [run_lm_eval_harness.py#L95](../accuracy/lm_eval/run_lm_eval_harness.py#L95)

本文重点解释当前仓库里最常看的 OPT 和 LLaMA。

### 6.1 OPT 的替换方式

OPT 的替换函数在 [modify_opt.py#L262](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L262)。

它会遍历 `model.model.decoder.layers`，把前两层之外的 `self_attn` 替换成 `OPTAttention_Mask`：

- 遍历层: [modify_opt.py#L264](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L264)
- 跳过前两层: [modify_opt.py#L265](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L265)
- 新 attention 类: [modify_opt.py#L266](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L266) 到 [modify_opt.py#L273](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L273)

### 6.2 LLaMA 的替换方式

LLaMA 的替换函数在 [modify_llama.py#L170](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L170)。

逻辑与 OPT 基本一致：

- 遍历层: [modify_llama.py#L177](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L177)
- 跳过前两层: [modify_llama.py#L178](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L178)
- 新 attention 类: [modify_llama.py#L180](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L180)

这说明 H2O 的实现策略是：

- 前两层保留原生 attention
- 中后层改成稀疏选择式 attention

## 7. `OPTAttention_Mask` / `LlamaAttention_heavy_hitter` 在做什么

两个类本质上都遵循同一模式：

1. 正常算出 `q / k / v`
2. 正常算出 `attn_weights`
3. 根据 `heavy_ratio` 和 `recent_ratio` 生成一个布尔保留 mask
4. 把未保留位置强制赋成极小值
5. 再做 softmax 和 value 聚合

OPT 类定义在 [modify_opt.py#L77](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L77)，LLaMA 类定义在 [modify_llama.py#L56](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L56)。

## 8. heavy 和 recent 两个预算是怎么计算的

在 attention forward 中，预算由当前 `key_len` 动态计算：

- OPT: [modify_opt.py#L190](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L190) 到 [modify_opt.py#L192](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L192)
- LLaMA: [modify_llama.py#L130](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L130) 到 [modify_llama.py#L132](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L132)

公式是：

- `heavy_budget = int(heavy_ratio * key_len)`
- `recent_budget = int(recent_ratio * key_len)`

这意味着 H2O 不是用固定整数窗口，而是用相对比例控制每个 query 允许保留的 key 数量。

## 9. heavy mask 是如何构造出来的

heavy mask 的核心在 `local_heavy_hitter_mask(...)`：

- OPT: [modify_opt.py#L23](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L23) 到 [modify_opt.py#L59](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L59)
- LLaMA: [modify_llama.py#L24](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L24) 到 [modify_llama.py#L53](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L53)

它的实现可以按时间顺序理解。

### 9.1 先把原始 attention 分数转成概率

第一步先对完整 `attn_weights` 做 softmax：

- OPT: [modify_opt.py#L34](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L34)
- LLaMA: [modify_llama.py#L32](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L32)

原因是 heavy hitter 的判断不是基于未归一化分数，而是基于“每个 query 实际在关注谁”。

### 9.2 初始化累计关注分数

接着它用前 `heavy_budget` 个 query 的注意力分布，把每个 key 的累计关注分数 `accumulated_attention_score` 先建出来：

- OPT: [modify_opt.py#L36](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L36)
- LLaMA: [modify_llama.py#L34](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L34)

同时它会把“超过初始化窗口之外”的 key 先清零：

- OPT: [modify_opt.py#L37](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L37)
- LLaMA: [modify_llama.py#L35](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L35)

### 9.3 前 `heavy_budget` 个 query 默认全保留

为了让初始化阶段可用，代码直接把前 `heavy_budget x heavy_budget` 的左下角块设成 `True`：

- OPT: [modify_opt.py#L42](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L42) 到 [modify_opt.py#L43](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L43)
- LLaMA: [modify_llama.py#L38](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L38) 到 [modify_llama.py#L39](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L39)

这一步的含义是：最开始的若干 token 还没有足够历史去判断谁是 heavy hitter，所以先不裁。

### 9.4 后续 query 动态选重 token

从第 `heavy_budget` 个 query 开始，函数会对每个 query 做同样的操作：

1. 先看当前累计关注分数最高的 `heavy_budget - 1` 个 key。
2. 再强制保留当前 query 自己对应的位置。
3. 用这个结果作为当前 query 的 heavy mask。

关键代码：

- 当前 query 的 softmax: [modify_opt.py#L47](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L47), [modify_llama.py#L43](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L43)
- 取历史 top-k: [modify_opt.py#L48](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L48), [modify_llama.py#L44](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L44)
- 保留当前 token 自己: [modify_opt.py#L51](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L51), [modify_llama.py#L47](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L47)

### 9.5 累计分数是动态滚动更新的

当前 query 被处理完后，当前的注意力分布会被加回累计分数里：

- OPT: [modify_opt.py#L54](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L54)
- LLaMA: [modify_llama.py#L50](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L50)

随后再和本轮保留的 key 相乘，把本轮未保留的 key 清掉：

- OPT: [modify_opt.py#L55](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L55)
- LLaMA: [modify_llama.py#L51](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L51)

因此，heavy hitter 不是一组固定 token，而是一种“随上下文推进持续更新的重要 token 集”。

## 10. recent mask 是如何构造出来的

recent mask 的构造比 heavy 简单很多：

- OPT: [modify_opt.py#L200](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L200) 到 [modify_opt.py#L203](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L203)
- LLaMA: [modify_llama.py#L140](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L140) 到 [modify_llama.py#L142](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L142)

实现是：

1. 先创建一个全 `True` 矩阵。
2. 用 `torch.triu(..., diagonal=-recent_budget)` 留下一条靠近主对角线的带。

这个带在 causal 约束之后，表示“每个 query 最近 `recent_budget` 个历史 token 必须保留”。

它不是基于重要性排序，而是基于时间局部性做硬保留。

## 11. heavy 和 recent 是怎么合并的

heavy mask 和 recent mask 直接做逻辑或：

- OPT: [modify_opt.py#L203](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L203)
- LLaMA: [modify_llama.py#L142](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L142)

也就是只要一个 key 满足下面任意一种条件，就会被保留：

- 它是 heavy hitter
- 它处于 recent 窗口

## 12. 为什么还要再套一层 causal mask

合并完后，代码还会再做一次下三角裁剪：

- OPT: [modify_opt.py#L205](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L205) 到 [modify_opt.py#L206](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L206)
- LLaMA: [modify_llama.py#L144](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L144)

原因是 recent 那条带在构造时会包含 query 右侧的位置，必须再截成下三角，才能保证 decoder 仍然只看过去。

因此最终有效区域其实是：

- 在 causal 下三角里
- 且属于 `heavy OR recent`

## 13. 未保留的位置是怎么被“屏蔽”的

没有被保留的位置并不会从张量里删除，而是直接赋成极小值：

- OPT: [modify_opt.py#L208](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L208)
- LLaMA: [modify_llama.py#L147](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L147)

然后 attention 再正常做 softmax：

- OPT: [modify_opt.py#L211](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L211) 到 [modify_opt.py#L215](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L215)
- LLaMA: [modify_llama.py#L149](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L149) 到 [modify_llama.py#L150](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L150)

由于 softmax 对极小值会给出接近 0 的概率，这等价于“这些 key 在注意力里不可见”。

## 14. H2O 在这份 harness 里到底改变了什么

在请求处理主循环里，脚本仍然是直接把整段 `prompt` 编码后丢给 `model(input_ids)`：

- 读请求: [run_lm_eval_harness.py#L131](../accuracy/lm_eval/run_lm_eval_harness.py#L131) 到 [run_lm_eval_harness.py#L135](../accuracy/lm_eval/run_lm_eval_harness.py#L135)
- 前向推理: [run_lm_eval_harness.py#L140](../accuracy/lm_eval/run_lm_eval_harness.py#L140) 到 [run_lm_eval_harness.py#L145](../accuracy/lm_eval/run_lm_eval_harness.py#L145)
- 生成 logprob 结果: [run_lm_eval_harness.py#L149](../accuracy/lm_eval/run_lm_eval_harness.py#L149) 到 [run_lm_eval_harness.py#L172](../accuracy/lm_eval/run_lm_eval_harness.py#L172)

因此，H2O 在这份文件中的作用主要是：

- 改写 attention 的可见性模式
- 让模型在 logprob 评测时只关注 heavy + recent 的 key
- 观察这种稀疏注意力对任务精度的影响

## 15. 它有没有真的把 KV cache 物理裁短

从这套代码本身看，答案是“没有完全物理裁短”。

例如在 OPT 的 H2O attention 中，`past_key_value` 仍然保存的是完整的 `key_states` 和 `value_states`：

- [modify_opt.py#L144](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L144) 到 [modify_opt.py#L149](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L149)
- [modify_opt.py#L155](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L155) 到 [modify_opt.py#L163](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L163)

LLaMA 也是一样：

- [modify_llama.py#L107](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L107) 到 [modify_llama.py#L112](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L112)

也就是说，这里的 H2O 更准确地说是：

- 逻辑上裁剪 attention 可见性
- 数值上屏蔽大量 key
- 但不等于“在这份 harness 中已经实现了完整的物理 KV cache 压缩”

这也是理解这份评测代码时最容易混淆的一点。

## 16. 这套 H2O 实现的完整调用链

如果要按调用顺序梳理，建议按下面路径读代码：

1. [h2o.sh#L12](../accuracy/lm_eval/h2o.sh#L12) 到 [h2o.sh#L19](../accuracy/lm_eval/h2o.sh#L19)
2. [run_lm_eval_harness.py#L43](../accuracy/lm_eval/run_lm_eval_harness.py#L43) 到 [run_lm_eval_harness.py#L46](../accuracy/lm_eval/run_lm_eval_harness.py#L46)
3. [run_lm_eval_harness.py#L88](../accuracy/lm_eval/run_lm_eval_harness.py#L88) 到 [run_lm_eval_harness.py#L107](../accuracy/lm_eval/run_lm_eval_harness.py#L107)
4. [modify_opt.py#L262](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L262) 或 [modify_llama.py#L170](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L170)
5. [modify_opt.py#L116](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L116) 到 [modify_opt.py#L259](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L259) 或 [modify_llama.py#L85](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L85) 到 [modify_llama.py#L167](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L167)
6. `local_heavy_hitter_mask(...)` 的细节: [modify_opt.py#L23](../accuracy/lm_eval/utils_lm_eval/modify_opt.py#L23) 或 [modify_llama.py#L24](../accuracy/lm_eval/utils_lm_eval/modify_llama.py#L24)

## 17. 一句话总结

`run_lm_eval_harness.py` 对 H2O 的实现方式是：

- 先加载原始模型
- 再把中后层 attention 替换成 heavy+recent 版本
- 用历史累计注意力和最近窗口共同构造保留 mask
- 将未保留位置压成极小值
- 最终在 lm-eval 的 logprob 评测流程里测试这种稀疏注意力策略

如果读者已经理解本文，再继续看 [infinigen-implementation.md](./infinigen-implementation.md)，会更容易对比出 H2O 和 InfiniGen 在“谁负责产生 mask、mask 基于什么信号、mask 作用在何处”上的差异。

