# InfiniGen 从 Setup 到 Hello World 的完整执行流程

本文按照 [accuracy/README.md](../accuracy/README.md) 中的命令顺序，梳理 InfiniGen 在 `accuracy` 目录下从准备工作到运行 Hello World 示例的全过程。

目标是把下面三件事串成一条可以顺着跟代码走的主线：

1. `setup.sh` 到底生成了哪些文件，为什么要先做这一步。
2. `generate_task_data.py` 和 `ours.sh` 如何把评测任务转成模型推理，再转回 lm-eval 的分数。
3. 每个关键函数定义在哪个文件里，函数内部主要语句分别做什么。

本文只展开 Hello World 这条实际会走到的主路径，并单独标注 `setup.sh` 中与 LLaMA 或对照实验有关、但不在 OPT Hello World 里实际执行的旁路分支。

## 1. README 中两条主命令在做什么

README 给了两段关键命令。

### 1.1 Setup

在 [accuracy/README.md#L19](../accuracy/README.md#L19)：

```sh
cd setup
export LLAMA_PATH=/path/to/llama-2
bash setup.sh
```

这一段的作用是生成 InfiniGen 运行所需的“预处理产物”：

- skewed 的 OPT 模型目录，例如 `accuracy/setup/opt-model/opt-6.7b`
- LLaMA 的 skewing matrix，例如 `accuracy/setup/skewing_matrix/llama-2-7b.pt`
- 每层的 `partial_weight_q_x.pt` 文件，例如 `accuracy/setup/weights/opt-6.7b_0.2/partial_weight_q_0.pt`

Hello World 的 OPT 示例真正依赖的是前两类里的 OPT 相关产物：

- `../setup/opt-model/opt-6.7b`
- `../setup/weights/opt-6.7b_0.2`

### 1.2 Hello World

在 [accuracy/README.md#L26](../accuracy/README.md#L26)：

```sh
cd lm_eval
mkdir results
python -u generate_task_data.py --output-file results/openbookqa-5.jsonl --task-name openbookqa --num-fewshot 5
bash ours.sh openbookqa ../setup/opt-model/opt-6.7b facebook/opt-6.7b opt 5 0.2 4 1.0 0.2
```

这一段分成两步：

1. `generate_task_data.py` 用 lm-eval-harness 先把 `openbookqa` 任务转成 prompt 列表，写成输入 JSONL。
2. `ours.sh` 调 `run_lm_eval_harness.py` 做真实模型推理，再调 `evaluate_task_result.py` 把输出结果重新喂回 lm-eval 计算准确率。

所以完整主线其实是：

`README -> setup/setup.sh -> gen_opt_model.py -> gen_partial_weight.py -> lm_eval/generate_task_data.py -> lm_eval/ours.sh -> run_lm_eval_harness.py -> src/modeling_opt_ours.py -> evaluate_task_result.py`

## 2. 总体流程图

### 2.1 Setup 阶段

```text
README
  -> setup/setup.sh
     -> setup/utils.py:set_symlink
     -> setup/gen_opt_model.py
        -> src/modeling_opt_orig.py
     -> setup/gen_partial_weight.py
        -> src/modeling_opt_ours_setup.py
```

### 2.2 Hello World 阶段

```text
README
  -> lm_eval/generate_task_data.py
     -> lm_eval/tasks/eval_harness.py
     -> 外部 lm_eval.evaluator.evaluate

  -> lm_eval/ours.sh
     -> lm_eval/run_lm_eval_harness.py
        -> src/modeling_opt_ours.py
     -> lm_eval/evaluate_task_result.py
        -> lm_eval/tasks/eval_harness.py
        -> 外部 lm_eval.evaluator.evaluate
```

## 3. Setup 阶段详细流程

## 3.1 `setup.sh` 顶层脚本在做什么

文件：[`accuracy/setup/setup.sh`](../accuracy/setup/setup.sh)

### 3.1.1 第 1 到 10 行：先把原始 `transformers` 文件挪开

对应 [setup.sh#L1](../accuracy/setup/setup.sh#L1) 到 [setup.sh#L10](../accuracy/setup/setup.sh#L10)。

- `CWD=${PWD}`：记住当前目录，后面切回来。
- `cd ../transformers/src/transformers/models`：进入 Hugging Face `transformers` 的模型源码目录。
- `for model in llama opt; do ... done`：分别处理 `llama` 和 `opt`。
- `mv ${model}/modeling_${model}.py ${model}/modeling_${model}_orig.py`：把官方入口文件改名成 `_orig.py`，为后续软链接留出 `modeling_xxx.py` 这个入口名。
- `cd ${CWD}`：回到 `setup` 目录。

这一步的核心意义是：后面脚本不会直接 import `accuracy/src/*.py`，而是通过软链接把 `transformers` 中的 `modeling_opt.py` 指向项目自定义版本。这样 `AutoModelForCausalLM.from_pretrained(...)` 仍然走 Hugging Face 标准加载流程，但真正加载的是你们改过的实现。

### 3.1.2 第 12 到 18 行：生成 skewed OPT 模型

对应 [setup.sh#L12](../accuracy/setup/setup.sh#L12) 到 [setup.sh#L18](../accuracy/setup/setup.sh#L18)。

循环运行：

```sh
python gen_opt_model.py --model facebook/opt-${size} --output ./opt-model
```

对 `6.7b`、`13b`、`30b` 三个 OPT 模型都生成一个经过 skewing 处理后的模型目录。Hello World 只会用到 `accuracy/setup/opt-model/opt-6.7b`。

### 3.1.3 第 20 到 25 行：生成 LLaMA skewing matrix

对应 [setup.sh#L20](../accuracy/setup/setup.sh#L20) 到 [setup.sh#L25](../accuracy/setup/setup.sh#L25)。

这部分调用 `gen_llama_skewing_matrix.py`，只和 LLaMA 路线有关。Hello World 的 OPT 示例不会用到，但它属于 README 要求的完整 setup。

### 3.1.4 第 28 到 48 行：生成 partial weight

对应 [setup.sh#L28](../accuracy/setup/setup.sh#L28) 到 [setup.sh#L48](../accuracy/setup/setup.sh#L48)。

其中 OPT 路径这一段最关键：

```sh
python gen_partial_weight.py \
  --our_model_path "./opt-model/opt-${size}" \
  --model "facebook/opt-${size}" \
  --model_type "opt" \
  --partial_weight_ratio 0.2 \
  --output "./weights"
```

它会读取前面生成好的 skewed OPT 模型，用一次短生成过程，自动提取每层的 `partial_weight_q`，并保存成：

- `accuracy/setup/weights/opt-6.7b_0.2/partial_weight_q_0.pt`
- `accuracy/setup/weights/opt-6.7b_0.2/partial_weight_q_1.pt`
- ...

### 3.1.5 第 51 到 80 行：对照实验

对应 [setup.sh#L51](../accuracy/setup/setup.sh#L51) 到 [setup.sh#L80](../accuracy/setup/setup.sh#L80)。

这几段用于：

- 不做 skewing 的对照实验
- partial ratio sweep

它们不是 README Hello World 的必经路径，可以先不看。

## 3.2 `setup/utils.py:set_symlink`

文件：[`accuracy/setup/utils.py`](../accuracy/setup/utils.py)

函数定义：[`set_symlink`](../accuracy/setup/utils.py#L2)

这个函数是 setup 阶段最重要的“入口切换器”。它把：

- `../transformers/src/transformers/models/<model_type>/modeling_<model_type>.py`

替换成一个软链接，让它指向：

- `../src/<fname>`

例如：

- `set_symlink("opt", "modeling_opt_orig.py")`
- `set_symlink("opt", "modeling_opt_ours_setup.py")`

逐句解释，对应 [utils.py#L2](../accuracy/setup/utils.py#L2) 到 [utils.py#L18](../accuracy/setup/utils.py#L18)：

- `model_path = "../transformers/src/transformers/models/" + model_type`：拼出 Hugging Face 模型源码目录。
- `linker_path = os.path.realpath("../src/" + fname)`：得到项目内目标文件的绝对路径。
- `if not os.path.exists(linker_path): ...`：确认要链接过去的文件存在。
- `if not os.path.exists(model_path): ...`：确认目标模型目录存在。
- `curr_dir = os.getcwd()`：记住当前目录。
- `os.chdir(model_path)`：进入 Hugging Face 模型目录。
- `if os.path.exists(f'modeling_{model_type}.py'):`：如果当前入口文件存在，先删掉。
- `cmd = f"rm modeling_{model_type}.py"`：构造删除命令。
- `os.system(cmd)`：执行删除。
- `cmd = f"ln -s {linker_path} modeling_{model_type}.py"`：构造软链接命令。
- `os.system(cmd)`：创建软链接。
- `os.chdir(curr_dir)`：恢复原工作目录。

## 3.3 `gen_opt_model.py` 如何生成 skewed OPT 模型

文件：[`accuracy/setup/gen_opt_model.py`](../accuracy/setup/gen_opt_model.py)

### 3.3.1 入口函数列表

- `process_options()` 定义在 [gen_opt_model.py#L7](../accuracy/setup/gen_opt_model.py#L7)
- `main()` 定义在 [gen_opt_model.py#L17](../accuracy/setup/gen_opt_model.py#L17)
- 嵌套函数 `get_query()` / `get_key()` 定义在 `main()` 内部 [gen_opt_model.py#L35](../accuracy/setup/gen_opt_model.py#L35)

### 3.3.2 `process_options()` 逐句解释

对应 [gen_opt_model.py#L7](../accuracy/setup/gen_opt_model.py#L7) 到 [gen_opt_model.py#L15](../accuracy/setup/gen_opt_model.py#L15)。

- `parser = argparse.ArgumentParser(...)`：创建命令行参数解析器。
- `parser.add_argument("--model", default="facebook/opt-6.7b", ...)`：指定要处理的原始 OPT 模型。
- `parser.add_argument("--output", required=True, ...)`：指定保存新模型的输出目录。
- `parser.add_argument("--no_skewing", action='store_true', ...)`：如果设置，则跳过 skewing，直接保存未 skew 的版本。
- `return parser`：把解析器交给 `main()` 使用。

### 3.3.3 `main()` 的运行步骤

对应 [gen_opt_model.py#L17](../accuracy/setup/gen_opt_model.py#L17) 到 [gen_opt_model.py#L105](../accuracy/setup/gen_opt_model.py#L105)。

#### 第一步：切到 `modeling_opt_orig.py`

- `parser = process_options()`：调用本文件里的参数定义函数。
- `args = parser.parse_args()`：读取命令行参数。
- `set_symlink("opt", "modeling_opt_orig.py")`：调用 [`accuracy/setup/utils.py`](../accuracy/setup/utils.py) 里的 `set_symlink`，让后面的 `AutoModelForCausalLM.from_pretrained()` 实际加载 [accuracy/src/modeling_opt_orig.py](../accuracy/src/modeling_opt_orig.py)。

#### 第二步：加载模型与 tokenizer

- `model_name = os.path.basename(args.model)`：取模型名最后一段，例如 `opt-6.7b`。
- `config = AutoConfig.from_pretrained(args.model)`：从 Hugging Face 配置里读超参数。
- `tokenizer = AutoTokenizer.from_pretrained(args.model)`：加载 tokenizer。
- `model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda()`：加载模型到 GPU。
- `head_dim = model.model.decoder.layers[0].self_attn.head_dim`：取单头维度。
- `n_head = model.model.decoder.layers[0].self_attn.num_heads`：取注意力头数。

#### 第三步：注册 hook，抓取每层的 Q/K 输出

- `query_v = {}` / `key_v = {}`：用于缓存各层的 query/key 投影输出。
- `def get_query(name): ...`：返回一个 forward hook，执行时把输出保存到 `query_v[name]`。
- `def get_key(name): ...`：返回一个 forward hook，执行时把输出保存到 `key_v[name]`。
- `for i, layer in enumerate(model.model.decoder.layers):`：遍历所有 decoder layer。
- `layer.self_attn.q_proj.register_forward_hook(...)`：给每层 `q_proj` 挂 query hook。
- `layer.self_attn.k_proj.register_forward_hook(...)`：给每层 `k_proj` 挂 key hook。

这里的关键点是：`gen_opt_model.py` 并不是手工重新算 Q/K，而是利用模型正常前向传播时的真实激活值来构造 skewing 变换。

#### 第四步：用 `pg19_firstbook.txt` 跑一次真实生成

- `file_path = "./pg19_firstbook.txt"`：指定用于采样激活分布的文本。
- `with open(file_path, 'r') as file: prompt = file.read()`：读入整本书开头。
- `input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()[:, :2048]`：tokenization 后截前 2048 个 token。
- `generated_ids = model.generate(input_ids, max_new_tokens = 1, min_new_tokens = 1)`：做一次最小生成。

这一步的目的不是生成文本本身，而是让每层 `q_proj` 和 `k_proj` 真正执行一次，从而触发 hook，把 `query_v` / `key_v` 填满。

#### 第五步：按每层每个 head 生成新权重

对应 [gen_opt_model.py#L62](../accuracy/setup/gen_opt_model.py#L62) 到 [gen_opt_model.py#L99](../accuracy/setup/gen_opt_model.py#L99)。

- `for name in query_v:`：遍历所有层。
- `layer = int(name)`：字符串层号转整数。
- `query = query_v[name][0]`：取该层 query 输出。
- `query = query * (head_dim ** -0.5)`：对齐 attention 中的 scaling。
- `key = key_v[name][0]`：取该层 key 输出。

接着读取原始投影权重：

- `wq = ...q_proj.weight.data`
- `bq = ...q_proj.bias.data`
- `wk = ...k_proj.weight.data`
- `bk = ...k_proj.bias.data`

然后构造带偏置的一体化权重矩阵：

- `new_wq = torch.cat((wq.transpose(-1,-2), bq.unsqueeze(0)), dim = 0) * (head_dim**-0.5)`
- `new_wk = torch.cat((wk.transpose(-1,-2), bk.unsqueeze(0)), dim = 0)`

这一步的目的，是把线性层 `y = xW + b` 改写成扩维输入 `[x; 1] @ W'` 的形式，方便后续做统一的列变换。

如果没有 `--no_skewing`，就进入每个 head 的 skewing 过程：

- `uq, sq, vq = torch.svd(query[:, start:end].to(torch.float))`：对该 head 的 query 激活做 SVD。
- `uk, sk, vk = torch.svd(key[:, start:end].to(torch.float))`：对该 head 的 key 激活做 SVD。
- `s = sq * sk`：用 query/key 奇异值乘积衡量共同重要方向。
- `A = torch.zeros(head_dim, head_dim)...`：初始化变换矩阵。
- `_, ind = s.sort()`：得到重要性排序。
- `A = A.scatter(-1, ind.unsqueeze(0).repeat(r,1), vq)`：把 `vq` 的列按排序重新排到 `A` 中。
- `new_wq[:, start:end] = new_wq[:, start:end] @ A`：对 query 权重做列变换。
- `new_wk[:, start:end] = new_wk[:, start:end] @ A`：对 key 权重做同样列变换。

#### 第六步：把新权重写回模型并保存

- `model.model.decoder.layers[layer].self_attn.q_proj.weight.data = new_wq`：用新权重替换原 `q_proj`。
- `model.model.decoder.layers[layer].self_attn.k_proj.weight.data = new_wk`：用新权重替换原 `k_proj`。
- `save_dir = args.output + "/" + model_name`：确定模型输出目录。
- `os.system(f"mkdir -p {save_dir}")`：创建目录。
- `model.save_pretrained(save_dir)`：保存成 Hugging Face 格式模型目录。

### 3.3.4 `gen_opt_model.py` 运行时进入了哪个模型文件

因为 `set_symlink("opt", "modeling_opt_orig.py")` 已经执行，所以 `AutoModelForCausalLM.from_pretrained(...)` 最终会进入 [`accuracy/src/modeling_opt_orig.py`](../accuracy/src/modeling_opt_orig.py)。

这条调用链是：

- `OPTForCausalLM.forward` 定义在 [modeling_opt_orig.py#L886](../accuracy/src/modeling_opt_orig.py#L886)
- `OPTModel.forward` 定义在 [modeling_opt_orig.py#L811](../accuracy/src/modeling_opt_orig.py#L811)
- `OPTDecoder.forward` 定义在 [modeling_opt_orig.py#L592](../accuracy/src/modeling_opt_orig.py#L592)
- `OPTDecoderLayer.forward` 定义在 [modeling_opt_orig.py#L336](../accuracy/src/modeling_opt_orig.py#L336)
- `OPTAttention.forward` 定义在 [modeling_opt_orig.py#L161](../accuracy/src/modeling_opt_orig.py#L161)

### 3.3.5 `modeling_opt_orig.py` 中与 setup 最相关的函数

这个文件大部分代码是 Hugging Face OPT 的复制版。为了顺着 `gen_opt_model.py` 理解 setup，只需要重点看下面几处。

#### `OPTAttention.__init__`

定义在 [modeling_opt_orig.py#L125](../accuracy/src/modeling_opt_orig.py#L125)。

关键语句：

- `self.k_proj / self.v_proj / self.q_proj / self.out_proj = nn.Linear(...)`：定义 attention 的四个线性投影层。
- `self.enable_quant = False`：默认关闭量化。
- `self.qbits = 0`：量化 bit 数初始化为 0。

#### `OPTAttention.forward`

定义在 [modeling_opt_orig.py#L161](../accuracy/src/modeling_opt_orig.py#L161)。

第一块：生成 `query_states` / `key_states` / `value_states`

- `query_states = self.q_proj(hidden_states) * self.scaling`：计算 query 并乘缩放因子。
- `if past_key_value is not None: ...`：如果是增量解码，就把新的 K/V 与缓存拼接。
- `past_key_value = (key_states, value_states)`：把当前 K/V 缓存下来供下一步解码使用。

第二块：可选量化

- `if self.enable_quant:`：只有显式打开量化时才执行。
- `qmax = (2 ** self.qbits) - 1`：计算量化等级上界。
- 按 `group_size = 64` 分组。
- 对每组：
  - 求 `key_max/key_min` 和 `value_max/value_min`
  - 把实数映射到整数区间
  - 再反量化回浮点

第三块：标准 attention 计算

- `attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))`：计算 QK^T。
- `attn_weights + attention_mask`：加 causal/padding mask。
- `softmax`：转成注意力概率。
- `attn_output = torch.bmm(attn_probs, value_states)`：用注意力概率加权求和 V。
- `self.out_proj(attn_output)`：做输出投影。

#### `OPTDecoder.forward`

定义在 [modeling_opt_orig.py#L592](../accuracy/src/modeling_opt_orig.py#L592)。

它做的事很标准：

- 准备 token embedding 和位置编码
- 构建 causal mask
- 遍历每一层 decoder layer
- 逐层调用 `decoder_layer(...)`
- 聚合所有层输出

对 `gen_opt_model.py` 来说，最重要的是它会让每层的 `q_proj/k_proj` 真正跑起来，触发注册好的 hook。

## 3.4 `gen_llama_skewing_matrix.py` 的位置

文件：[`accuracy/setup/gen_llama_skewing_matrix.py`](../accuracy/setup/gen_llama_skewing_matrix.py)

这个脚本是 LLaMA 路线的对应物，不会出现在 OPT Hello World 主线里。它会：

- 用 `set_symlink("llama", "modeling_llama_orig.py")` 切换到 LLaMA 原始实现
- 跑一次文本生成
- 从每层 attention 中读出 rope 后的 query/key
- 构造每层每个 head 的 skewing matrix `A`
- 保存为 `skewing_matrix/*.pt`

## 3.5 `gen_partial_weight.py` 如何生成 `partial_weight_q`

文件：[`accuracy/setup/gen_partial_weight.py`](../accuracy/setup/gen_partial_weight.py)

### 3.5.1 入口函数列表

- `process_options()` 定义在 [gen_partial_weight.py#L7](../accuracy/setup/gen_partial_weight.py#L7)
- `main()` 定义在 [gen_partial_weight.py#L23](../accuracy/setup/gen_partial_weight.py#L23)

### 3.5.2 `process_options()` 逐句解释

对应 [gen_partial_weight.py#L7](../accuracy/setup/gen_partial_weight.py#L7) 到 [gen_partial_weight.py#L21](../accuracy/setup/gen_partial_weight.py#L21)。

- `--our_model_path`：指向前一步生成的 skewed 模型目录。
- `--skewing_matrix_path`：LLaMA 路线用的 skewing matrix。
- `--model`：原始模型名，用于 tokenizer 等。
- `--model_type`：区分 `opt` 或 `llama`。
- `--partial_weight_ratio`：设定保留多少比例的重要列。
- `--output`：保存 `partial_weight_q_*.pt` 的目录。

### 3.5.3 `main()` 的运行步骤

对应 [gen_partial_weight.py#L23](../accuracy/setup/gen_partial_weight.py#L23) 到 [gen_partial_weight.py#L73](../accuracy/setup/gen_partial_weight.py#L73)。

#### 第一步：切到 `modeling_opt_ours_setup.py`

- `fname = f"modeling_{args.model_type}_ours_setup.py"`：根据模型类型选择 setup 专用模型文件。
- `set_symlink(args.model_type, fname)`：把 `transformers` 入口切到：
  - OPT: `accuracy/src/modeling_opt_ours_setup.py`
  - LLaMA: `accuracy/src/modeling_llama_ours_setup.py`

这一步很关键：setup 阶段生成 `partial_weight_q` 用的是 `ours_setup` 版本，不是正式推理时的 `ours` 版本。

#### 第二步：加载模型

- `if args.our_model_path is not None:`：如果给了 skewed 模型目录，就加载那个目录。
- `else:`：否则加载原始模型名。

对 OPT Hello World 来说，走的是 `--our_model_path "./opt-model/opt-6.7b"`。

#### 第三步：可选加载 LLaMA 的 skewing matrix

对应 [gen_partial_weight.py#L37](../accuracy/setup/gen_partial_weight.py#L37) 到 [gen_partial_weight.py#L42](../accuracy/setup/gen_partial_weight.py#L42)。这段只影响 LLaMA；OPT 主线不会执行。

#### 第四步：准备 prompt 并设置 `partial_weight_ratio`

- `tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)`：加载 tokenizer。
- `prompt = [...]`：准备一段固定英文样例。
- `input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()`：转成 GPU tensor。
- 对每一层：
  - `layer.self_attn.partial_weight_ratio = float(args.partial_weight_ratio)`

这一步告诉 `ours_setup` 模型，在第一次前向时按多大比例保留列。

#### 第五步：用一次生成触发 `partial_weight_q` 自动构建

- `generated_ids = model.generate(input_ids, max_new_tokens = 1, min_new_tokens = 1)`

这一步会进入 `accuracy/src/modeling_opt_ours_setup.py`。这个文件里的 `OPTAttention.forward` 在第一次运行时，如果 `self.partial_weight_q is None`，会根据当前 query/key 激活自动生成 `partial_weight_q`。

#### 第六步：保存每层的 `partial_weight_q`

- `basepath = args.output + "/" + os.path.basename(os.path.normpath(args.model)) + "_%s"%(args.partial_weight_ratio)`：例如生成 `./weights/opt-6.7b_0.2`。
- `mkdir -p`：创建目录。
- 对 OPT 每层：
  - `partial_weight = model.model.decoder.layers[layer].self_attn.partial_weight_q`
  - `torch.save(partial_weight, ".../partial_weight_q_<layer>.pt")`

这就是 Hello World 推理时 `run_lm_eval_harness.py` 会重新加载的那批文件。

## 3.6 `modeling_opt_ours_setup.py` 在 setup 阶段做了什么

文件：[`accuracy/src/modeling_opt_ours_setup.py`](../accuracy/src/modeling_opt_ours_setup.py)

这个文件的职责不是正式推理，而是“在一次真实前向里自动构造 partial weight”。

### 3.6.1 `OPTAttention.__init__`

定义在 [modeling_opt_ours_setup.py#L126](../accuracy/src/modeling_opt_ours_setup.py#L126)。

关键变化：

- `self.k_proj = nn.Linear(embed_dim, embed_dim+1, bias=False)`
- `self.q_proj = nn.Linear(embed_dim, embed_dim+1, bias=False)`

这里把原来的 `embed_dim -> embed_dim` 变成了“通过扩维输入把 bias 合进权重矩阵”的形式，方便后面统一处理权重列。

另外新增：

- `self.partial_weight_q = None`
- `self.partial_weight_ratio = None`

### 3.6.2 `OPTAttention.forward`

定义在 [modeling_opt_ours_setup.py#L158](../accuracy/src/modeling_opt_ours_setup.py#L158)。

这个函数是 setup 阶段真正的核心。

第一块：把输入扩成 `[hidden_states; 1]`

- `new_attn_in = torch.cat((hidden_states, torch.ones(...)), dim = -1)`：给每个 token 的 hidden state 末尾拼一个常数 1。
- `query_states = torch.matmul(new_attn_in, self.q_proj.weight.data)`：用扩维后的输入直接乘以新格式的 `q_proj.weight`。

目的：把原来的“矩阵乘法 + bias”变成单次矩阵乘法。

第二块：计算 key/value

- 如果有 `past_key_value`，就走增量解码路径。
- 否则：
  - `key = torch.matmul(new_attn_in, self.k_proj.weight.data)`
  - `key_states = self._shape(key, -1, bsz)`
  - `value_states = self._shape(self.v_proj(hidden_states), -1, bsz)`

第三块：第一次前向时构建 `partial_weight_q`

对应 [modeling_opt_ours_setup.py#L218](../accuracy/src/modeling_opt_ours_setup.py#L218) 到 [modeling_opt_ours_setup.py#L230](../accuracy/src/modeling_opt_ours_setup.py#L230)。

- `if self.partial_weight_q is None:`：只在第一次计算。
- `wq = self.q_proj.weight.data.clone()`：复制 query 权重。
- `for head in range(...):`：按 attention head 分块处理。
- `weight_mask = torch.zeros_like(wq[:, start:end])`：初始化 mask。
- `torch.topk(torch.sum(torch.abs(query_states[:, :, start:end]), dim = -2), int(self.head_dim * self.partial_weight_ratio))`：找出该 head 中 query 激活绝对值和最大的若干列。
- 同样对 `key[:, :, start:end]` 再做一次 `topk`：把 key 也认为是重要性来源。
- `weight_mask = weight_mask.scatter(...)`：把这些列位置标为 1。
- `wq[:, start:end] = wq[:, start:end] * weight_mask`：其余列清零，仅保留重要列。
- `self.partial_weight_q = wq`：保存下来，供外部脚本导出。

这就是 `gen_partial_weight.py` 想要的产物。

### 3.6.3 `OPTDecoder.forward`

定义在 [modeling_opt_ours_setup.py#L597](../accuracy/src/modeling_opt_ours_setup.py#L597)。

它整体仍是标准 OPT decoder 流程：

- 构建 embedding
- 构建 causal mask
- 遍历所有 decoder layer
- 逐层调用 `decoder_layer(...)`

对 setup 来说，作用很直接：它负责把输入 prompt 顺序推过所有层，使每层 attention 都有机会生成自己的 `partial_weight_q`。

## 4. Hello World 阶段详细流程

## 4.1 `generate_task_data.py` 如何把任务变成输入 JSONL

文件：[`accuracy/lm_eval/generate_task_data.py`](../accuracy/lm_eval/generate_task_data.py)

### 4.1.1 文件总体作用

它不做真实推理，只做一件事：让 lm-eval-harness 按任务模板把题目展开成 prompt，然后把这些 prompt 写成 `results/openbookqa-5.jsonl`。

### 4.1.2 顶层语句逐句解释

对应 [generate_task_data.py#L8](../accuracy/lm_eval/generate_task_data.py#L8) 到 [generate_task_data.py#L73](../accuracy/lm_eval/generate_task_data.py#L73)。

- `parser.add_argument('--output-file', ...)`：指定输出 JSONL。
- `parser.add_argument('--task-name', ...)`：指定任务名，例如 `openbookqa`。
- `parser.add_argument('--num-fewshot', ...)`：指定 few-shot 数，例如 5。
- `seq = 1024`：设定评测序列长度。
- `total_batch = 1`：批大小固定为 1。
- `pe = 'fixed'`：表示不做 shrink。
- `with open(args.output_file, 'w') as f: pass`：先清空输出文件。

### 4.1.3 `DryRunner.eval()`

定义在 [generate_task_data.py#L28](../accuracy/lm_eval/generate_task_data.py#L28)。

它是假模型，作用是“只写请求，不算结果”。

- `with open(args.output_file, 'a') as f:`：以追加方式打开输出 JSONL。
- `for text in batch['text']:`：遍历当前 batch 中的 prompt 文本。
- 构造 `item = {...}`：把 prompt 包装成 OpenAI completion 风格的请求结构。
- `f.write(json.dumps(item) + '\n')`：一行一条请求写入 JSONL。
- `out = {'mask_loss': [1.0] * len(batch), 'each_correct': [True] * len(batch)}`：返回一个假的评测结果，只为了让 lm-eval 的流程继续跑完。

### 4.1.4 `EvalHarnessAdaptor`

`generate_task_data.py` 用的是 `from tasks import EvalHarnessAdaptor`，其定义在 [`accuracy/lm_eval/tasks/eval_harness.py`](../accuracy/lm_eval/tasks/eval_harness.py)。

### 4.1.5 `tasks/eval_harness.py` 中的关键函数

#### `process_init()`

定义在 [eval_harness.py#L17](../accuracy/lm_eval/tasks/eval_harness.py#L17)。

作用：

- 根据环境变量 `MODEL_NAME` 选择 tokenizer
- 设置 `pad_token` 或关闭 `bos`

在 `generate_task_data.py` 这条路径里，它主要负责保证 tokenizer 初始化正常。

#### `process_request(x, seq)`

定义在 [eval_harness.py#L33](../accuracy/lm_eval/tasks/eval_harness.py#L33)。

这个函数把 lm-eval 给出的 `(ctx, cont)` 转成项目自己的 batch 格式。

关键语句：

- `ctx, cont = x`：拆出上下文与续写目标。
- `ctx_text = ftfy.fix_text(ctx, normalization="NFKC")`：修正文本编码。
- `cont_text = ftfy.fix_text(cont, normalization="NFKC")`：同上。
- `all_text = ctx_text + cont_text`：拼成完整文本。
- `ctx_tokens = tokenizer(ctx_text, add_special_tokens=False)['input_ids']`：tokenize 上下文。
- `cont_tokens = tokenizer(cont_text, add_special_tokens=False)['input_ids']`：tokenize 待评估续写部分。
- `all_tokens = np.array(all_tokens)[-seq:]`：截成最多 `seq` 长度。
- `obs`：输入给模型看到的 token 序列。
- `target`：右移一位后的真值 token。
- `eval_mask`：标记哪些位置属于 continuation，需要计入评测。
- `text`：保存完整 prompt 文本。

#### `EvalHarnessAdaptor.loglikelihood()`

定义在 [eval_harness.py#L80](../accuracy/lm_eval/tasks/eval_harness.py#L80)。

这是把本地 runner 接到 lm-eval 的关键适配层。

主要步骤：

- `r = self.convert_requests(requests)`：把任务请求流转成 `process_request` 的输出。
- `zero_example = process_request(requests[0], self.seq)`：生成一个样板，用于补齐 batch。
- `for b in tqdm(sample_batch(...))`：按 batch 取数据。
- `if self.shrink: b = shrink_seq(...)`：可选压缩序列长度。
- `out = self.tpu.eval(b)`：调用外部 runner 的 `eval()`。
- `output.append((float(-loss), bool(correct)))`：转成 lm-eval 期望的 `(loglikelihood, correct)` 格式。

### 4.1.6 `tasks/util.py` 中的辅助函数

文件：[`accuracy/lm_eval/tasks/util.py`](../accuracy/lm_eval/tasks/util.py)

- `grouper()` 定义在 [util.py#L6](../accuracy/lm_eval/tasks/util.py#L6)：把样本按 `n` 个一组打包。
- `shrink_seq()` 定义在 [util.py#L13](../accuracy/lm_eval/tasks/util.py#L13)：如果后半段是 padding，就递归缩短序列长度。
- `sample_batch()` 定义在 [util.py#L34](../accuracy/lm_eval/tasks/util.py#L34)：把单样本字典堆成 batch 字典。

### 4.1.7 `lm_eval.evaluator.evaluate`

`generate_task_data.py` 和 `evaluate_task_result.py` 都调用了 `evaluator.evaluate(...)`。它来自外部 `lm_eval` 包，不在本仓库中定义。这里把它当作黑盒任务驱动器来理解即可：

- 它会取任务数据
- 调 `LM.loglikelihood`
- 聚合成每个 benchmark 的最终分数

在本项目里，`EvalHarnessAdaptor` 就是专门给它准备的桥接层。

## 4.2 `ours.sh` 如何把预处理产物接到推理脚本

文件：[`accuracy/lm_eval/ours.sh`](../accuracy/lm_eval/ours.sh)

### 4.2.1 顶层变量逐句解释

对应 [ours.sh#L4](../accuracy/lm_eval/ours.sh#L4) 到 [ours.sh#L20](../accuracy/lm_eval/ours.sh#L20)。

README 的命令：

```sh
bash ours.sh openbookqa ../setup/opt-model/opt-6.7b facebook/opt-6.7b opt 5 0.2 4 1.0 0.2
```

会被映射成：

- `task=openbookqa`
- `model_path=../setup/opt-model/opt-6.7b`
- `model=facebook/opt-6.7b`
- `model_arch=opt`
- `shots=5`
- `partial_weight=0.2`
- `alpha=4`
- `capacity=1.0`
- `budget=0.2`

后面几行路径拼接含义是：

- `base_name=$(basename "${model}")`：得到 `opt-6.7b`
- `weight_path="../setup/weights/${base_name}_${partial_weight}"`：得到 `../setup/weights/opt-6.7b_0.2`
- `skewing_path="../setup/skewing_matrix/${base_name}.pt"`：得到 `../setup/skewing_matrix/opt-6.7b.pt`

注意：OPT 路线实际上不使用 `skewing_path`，这里只是脚本统一保留了参数。

### 4.2.2 调 `run_lm_eval_harness.py`

对应 [ours.sh#L22](../accuracy/lm_eval/ours.sh#L22) 到 [ours.sh#L34](../accuracy/lm_eval/ours.sh#L34)。

这一步做真实推理，输出 `results/openbookqa-5-opt-6.7b-ours.jsonl`。

### 4.2.3 调 `evaluate_task_result.py`

对应 [ours.sh#L36](../accuracy/lm_eval/ours.sh#L36) 到 [ours.sh#L41](../accuracy/lm_eval/ours.sh#L41)。

这一步把上一步的推理 JSONL 转回 lm-eval 结果，并打印最终 benchmark 分数。

### 4.2.4 删除中间结果

- `rm results/${task}-${shots}-${base_name}-ours.jsonl`：只保留终端输出，不保留中间 result JSONL。

## 4.3 `run_lm_eval_harness.py` 如何做真实推理

文件：[`accuracy/lm_eval/run_lm_eval_harness.py`](../accuracy/lm_eval/run_lm_eval_harness.py)

### 4.3.1 本文件有哪些函数

- `set_symlink()` 定义在 [run_lm_eval_harness.py#L8](../accuracy/lm_eval/run_lm_eval_harness.py#L8)
- 顶层主逻辑在 `if __name__ == '__main__':` 中，从 [run_lm_eval_harness.py#L26](../accuracy/lm_eval/run_lm_eval_harness.py#L26) 开始

这个 `set_symlink()` 与 `setup/utils.py` 中的是同一思路，只是复制了一份到本文件里，便于 `lm_eval` 目录单独执行。

### 4.3.2 参数解析

对应 [run_lm_eval_harness.py#L28](../accuracy/lm_eval/run_lm_eval_harness.py#L28) 到 [run_lm_eval_harness.py#L56](../accuracy/lm_eval/run_lm_eval_harness.py#L56)。

分成三组：

- 通用参数：`input-path`、`output-path`、`model-name`、`model-path`、`model-type`
- H2O/量化对照实验参数
- InfiniGen 参数：`--ours`、`--partial_weight_ratio`、`--partial_weight_path`、`--alpha`、`--capacity`、`--budget`

### 4.3.3 切换到 `modeling_opt_ours.py`

对应 [run_lm_eval_harness.py#L58](../accuracy/lm_eval/run_lm_eval_harness.py#L58) 到 [run_lm_eval_harness.py#L61](../accuracy/lm_eval/run_lm_eval_harness.py#L61)。

因为 `ours.sh` 传了 `--ours`，所以这里会执行：

- `set_symlink(args.model_type, f"modeling_{args.model_type}_ours.py")`

对于 Hello World 的 OPT 路线，就是 `set_symlink("opt", "modeling_opt_ours.py")`。

这意味着后续 `AutoModelForCausalLM.from_pretrained(...)` 实际加载的是 [`accuracy/src/modeling_opt_ours.py`](../accuracy/src/modeling_opt_ours.py)。

### 4.3.4 加载模型与 tokenizer

对应 [run_lm_eval_harness.py#L64](../accuracy/lm_eval/run_lm_eval_harness.py#L64) 到 [run_lm_eval_harness.py#L74](../accuracy/lm_eval/run_lm_eval_harness.py#L74)。

- `config = AutoConfig.from_pretrained(model_name)`：读取 Hugging Face 配置。
- `tokenizer = AutoTokenizer.from_pretrained(model_name, ...)`：读取 tokenizer。
- `if args.model_path is None: ... else: ...`：优先从 `--model-path` 读取本地保存好的 skewed 模型目录。

Hello World 里这里会加载 `../setup/opt-model/opt-6.7b`。

### 4.3.5 把 setup 产物注入每一层 attention

对应 [run_lm_eval_harness.py#L109](../accuracy/lm_eval/run_lm_eval_harness.py#L109) 到 [run_lm_eval_harness.py#L127](../accuracy/lm_eval/run_lm_eval_harness.py#L127)。

对 OPT：

- `layer.self_attn.partial_weight_ratio = args.partial_weight_ratio`：注入保留比例。
- `layer.self_attn.partial_weight_q = torch.load(...partial_weight_q_<layer>.pt)`：把 setup 阶段生成的稀疏权重载入当前层。
- `layer.self_attn.alpha = args.alpha`：设置阈值参数。
- `layer.self_attn.capacity = args.capacity`：设置最多保留多少比例 KV。
- `layer.self_attn.budget = args.budget`：设置每步最多预取多少比例 KV。

这一步是 setup 和推理阶段真正接起来的地方。

### 4.3.6 逐条读取请求 JSONL

对应 [run_lm_eval_harness.py#L131](../accuracy/lm_eval/run_lm_eval_harness.py#L131) 到 [run_lm_eval_harness.py#L135](../accuracy/lm_eval/run_lm_eval_harness.py#L135)。

- 打开 `input_path`
- 每行 `json.loads`
- 非空行加入 `requests`

这个文件就是 `generate_task_data.py` 产出的 `results/openbookqa-5.jsonl`。

### 4.3.7 对每个 prompt 做真实前向

对应 [run_lm_eval_harness.py#L139](../accuracy/lm_eval/run_lm_eval_harness.py#L139) 到 [run_lm_eval_harness.py#L182](../accuracy/lm_eval/run_lm_eval_harness.py#L182)。

对每个请求：

- `prompt = request['prompt']`：取出 prompt 文本。
- `input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)`：tokenize。
- `logits = model(input_ids).logits.log_softmax(dim=-1)`：调模型前向，拿每个位置的 log-prob。
- `density.append(model.get_density())`：记录当前样本的平均 attention 密度。
- `values, indices = logits.squeeze(0).topk(dim=-1, k=1)`：取每个位置 top-1 token。
- `tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))`：把输入 token id 转回 token 字符串。
- `gold_indices = input_ids[:, 1:]`：真值 token 是“下一个 token”。
- `torch.gather(logits, -1, gold_indices.unsqueeze(-1))`：取每个位置上真值 token 的 log-prob。
- 组装成 OpenAI-completion 风格结果结构 `result`：供 `evaluate_task_result.py` 后续复用。

### 4.3.8 清理 `previous_hidden_states`

对应 [run_lm_eval_harness.py#L176](../accuracy/lm_eval/run_lm_eval_harness.py#L176) 到 [run_lm_eval_harness.py#L182](../accuracy/lm_eval/run_lm_eval_harness.py#L182)。

这一步很关键。因为 `modeling_opt_ours.py` 会在层与层之间传递 `previous_hidden_states`，样本与样本之间如果不清理，就会串状态。所以每处理完一个请求，都会对每层执行：

- `layer.self_attn.previous_hidden_states = None`

### 4.3.9 计算并打印 retain ratio

对应 [run_lm_eval_harness.py#L184](../accuracy/lm_eval/run_lm_eval_harness.py#L184) 到 [run_lm_eval_harness.py#L188](../accuracy/lm_eval/run_lm_eval_harness.py#L188)。

- `density = sum(density) / len(density) * 100`：计算平均保留密度百分比。
- `retain_ratio = (1 - math.sqrt(1 - density / 100)) * 100`：把密度换算成 retain ratio。
- `print("retain ratio: %.2f\n"%(retain_ratio))`：打印到终端。

### 4.3.10 写出结果 JSONL

对应 [run_lm_eval_harness.py#L190](../accuracy/lm_eval/run_lm_eval_harness.py#L190) 到 [run_lm_eval_harness.py#L192](../accuracy/lm_eval/run_lm_eval_harness.py#L192)。

把每条请求对应的模型输出写入 `output_path`。

## 4.4 `modeling_opt_ours.py` 如何实现 InfiniGen 推理

文件：[`accuracy/src/modeling_opt_ours.py`](../accuracy/src/modeling_opt_ours.py)

这是 Hello World 的真正核心实现文件。

### 4.4.1 重要类与函数

对主流程最重要的是：

- `OPTAttention.__init__` 在 [modeling_opt_ours.py#L126](../accuracy/src/modeling_opt_ours.py#L126)
- `OPTAttention.kv_cache_mask` 在 [modeling_opt_ours.py#L173](../accuracy/src/modeling_opt_ours.py#L173)
- `OPTAttention.forward` 在 [modeling_opt_ours.py#L237](../accuracy/src/modeling_opt_ours.py#L237)
- `OPTDecoder.forward` 在 [modeling_opt_ours.py#L636](../accuracy/src/modeling_opt_ours.py#L636)
- `OPTForCausalLM.get_density` 在 [modeling_opt_ours.py#L934](../accuracy/src/modeling_opt_ours.py#L934)

其他类如 `OPTModel.forward`、`OPTDecoderLayer.forward` 大多还是标准 Hugging Face 包装层，只负责把数据一层层往下传。

### 4.4.2 `OPTAttention.__init__`

关键语句在 [modeling_opt_ours.py#L149](../accuracy/src/modeling_opt_ours.py#L149) 到 [modeling_opt_ours.py#L167](../accuracy/src/modeling_opt_ours.py#L167)。

- `self.k_proj = nn.Linear(embed_dim, embed_dim+1, bias=False)`
- `self.q_proj = nn.Linear(embed_dim, embed_dim+1, bias=False)`：与 setup 版一样，使用扩维输入的权重形式。
- `self.partial_weight_q = None`：之后由 `run_lm_eval_harness.py` 从磁盘加载进来。
- `self.alpha = 4`：默认阈值。
- `self.capacity = 1.0`：默认存储比例。
- `self.budget = 0.1`：默认预取比例。
- `self.eviction_policy = "counter"`：默认淘汰策略。
- `self.previous_hidden_states = None`：用于层间传递上一层隐藏状态。
- `self.current_hidden_states = None`：保存当前层输入隐藏状态。
- `self.density = None`：保存本层当前样本的 attention 稀疏密度。

### 4.4.3 `kv_cache_mask(attn)`

定义在 [modeling_opt_ours.py#L173](../accuracy/src/modeling_opt_ours.py#L173)。

它的作用是：根据“预测得到的 attention 分数”构造一个二值保留掩码，再转成可直接加到 attention logits 上的 `0 / -10000` mask。

逐句拆开看：

- `assert self.budget < self.capacity`：约束预取比例不能超过总容量比例。
- `heads, tgt_len, src_len = attn.shape`：读出 attention 张量形状。
- `attn_mask = torch.full(attn.shape, -10000, ...)`：准备一个大负数 mask。
- `attn_mask = torch.triu(attn_mask, diagonal = 1)`：把未来位置遮掉，保留因果性。
- `fetch_mask = torch.zeros_like(attn)`：初始化“要保留哪些 KV”的掩码。
- `attn = attn + attn_mask`：把未来 token 排除出考虑范围。
- `max = torch.max(attn, dim = -1, keepdim = True)[0]`：每个 query 位置找最大 attention 分数。
- `threshold = max - self.alpha`：设定阈值。
- `mask = (attn >= threshold)`：认为高于阈值的位置都值得取。
- `fetch_num = torch.sum(mask, dim = -1)`：每个 head、每个 query 位置有多少候选。
- `fetch_num = torch.mean(..., dim = 0).to(torch.int32)`：在 head 维度上取平均，得到所有 head 公用的 fetch 数。
- `fetch_max = int(src_len * self.budget)`：每一步最多预取多少。
- `fetch_num = torch.where(fetch_num >= fetch_max, fetch_max, fetch_num)`：上限截断。
- `store_max = int(src_len * self.capacity)`：最多允许存多少 KV。

接着分三段处理：

- `fetch_mask[:, :fetch_max] = torch.tril(...)`：前几步直接全部保留，保证 warmup。
- `for i in range(fetch_max, store_max):`：容量未满时，只取 top-k。
- `for i in range(store_max, tgt_len):`：容量满了以后，还要在引入新 KV 前做淘汰。

淘汰策略默认是 `counter`：

- `counter = torch.sum(fetch_mask[:, :i + 1, :int(i / 2)], dim = 1, keepdim = True)`：统计历史上每个 KV 被命中的次数。
- `_, ind = torch.min(counter, dim = -1, keepdim = True)`：找出最少被访问的 KV。
- `attn[:, (i + 1):] = attn[:, (i + 1):].scatter(-1, ind, -10000)`：将其从未来步骤的候选中剔除。

最后：

- `density = fetch_mask.float().sum().item() / heads / (tgt_len * (tgt_len + 1) / 2)`：计算保留密度。
- `fetch_mask = torch.where(fetch_mask == 1, 0, m_inf)`：把二值掩码转成可直接加到 attention logits 上的 `0 / -10000` mask。
- `return fetch_mask, density`

### 4.4.4 `OPTAttention.forward`

定义在 [modeling_opt_ours.py#L237](../accuracy/src/modeling_opt_ours.py#L237)。

这个函数把“预测 attention 稀疏模式”和“真实 attention 计算”拼到了一起。

第一块：记录当前层输入

- `self.current_hidden_states = hidden_states.clone()`

后面 `OPTDecoder.forward` 会把它传给下一层，作为 `previous_hidden_states`。

第二块：正常计算 query/key/value

- `new_attn_in = torch.cat((hidden_states, torch.ones(...)), dim = -1)`：扩维输入。
- `query_states = torch.matmul(new_attn_in, self.q_proj.weight.data)`：计算 query。
- `if past_key_value is not None: ... else: ...`：正常生成/拼接 K/V cache。
- `past_key_value = (key_states, value_states)`：保存缓存。

这部分和标准 attention 相比，主要差异是使用扩维后的 `q_proj/k_proj` 权重格式。

第三块：用 `partial_weight_q` 预测 attention

对应 [modeling_opt_ours.py#L276](../accuracy/src/modeling_opt_ours.py#L276) 到 [modeling_opt_ours.py#L283](../accuracy/src/modeling_opt_ours.py#L283)。

只有在两个条件都满足时才执行：

- `self.previous_hidden_states is not None`
- `self.partial_weight_q is not None`

也就是：

- 前一层已经把隐藏状态传过来了
- setup 阶段生成的 `partial_weight_q` 已经加载进来了

具体步骤：

- `query = torch.matmul(self.previous_hidden_states, self.partial_weight_q)`：用上一层隐藏状态和稀疏权重近似得到一个 query。
- `attn = torch.matmul(query, key_states.transpose(1, 2))`：计算“预测 attention 分数”。
- `attn_mask, density = self.kv_cache_mask(attn)`：用预测 attention 生成保留掩码。
- `self.density = density`：记录当前层稀疏密度。

第四块：计算真实 attention logits

- `attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))`：真实 QK^T。
- 如果有 `attention_mask`，就加因果/ padding mask。

第五块：把预测出来的稀疏掩码加到真实 attention 上

对应 [modeling_opt_ours.py#L304](../accuracy/src/modeling_opt_ours.py#L304) 到 [modeling_opt_ours.py#L310](../accuracy/src/modeling_opt_ours.py#L310)。

- `attn_weights = attn_weights + attn_mask`：把“不想看的 KV”压成极小值。
- `softmax`：只在保留下来的 KV 范围内分配概率。

也就是说：InfiniGen 不是直接用近似 attention 做输出，而是先用近似 attention 预测“哪些 KV 值值得取”，再在这个掩码内做真实 attention。

第六块：标准输出投影

- `attn_output = torch.bmm(attn_probs, value_states)`
- `self.out_proj(attn_output)`

### 4.4.5 `OPTDecoder.forward`

定义在 [modeling_opt_ours.py#L636](../accuracy/src/modeling_opt_ours.py#L636)。

它与标准 OPT decoder 的最大差别在 [modeling_opt_ours.py#L793](../accuracy/src/modeling_opt_ours.py#L793) 到 [modeling_opt_ours.py#L801](../accuracy/src/modeling_opt_ours.py#L801)。

这几行的作用是：

- 每跑完第 `idx` 层
- 取出该层 `self_attn.current_hidden_states`
- 在末尾拼一个常数 1
- 传给下一层的 `self_attn.previous_hidden_states`

具体语句：

- `cur_device = self.layers[idx].self_attn.current_hidden_states.device`
- `cur_dtype = ...dtype`
- `cur_bsz, cur_tgt_len, _ = ...shape`
- `self.layers[idx + 1].self_attn.previous_hidden_states = torch.cat(...)`

这就是 InfiniGen 做“层间预测”的关键连接线。

### 4.4.6 `OPTForCausalLM.get_density()`

定义在 [modeling_opt_ours.py#L934](../accuracy/src/modeling_opt_ours.py#L934)。

逐句解释：

- `density = []`：存每层密度。
- `for l in self.model.decoder.layers:`：遍历所有层。
- `if hasattr(l.self_attn, "density") and l.self_attn.density != None:`：只统计真正算出密度的层。
- `density.append(l.self_attn.density)`：收集。
- `return sum(density)/len(density)`：返回平均密度。

这个值会被 `run_lm_eval_harness.py` 用来打印 retain ratio。

## 4.5 `evaluate_task_result.py` 如何回评结果

文件：[`accuracy/lm_eval/evaluate_task_result.py`](../accuracy/lm_eval/evaluate_task_result.py)

### 4.5.1 `json_to_key(obj)`

定义在 [evaluate_task_result.py#L8](../accuracy/lm_eval/evaluate_task_result.py#L8)。

作用：把 request 对象序列化成字符串，作为字典 key。这样可以用“原始请求内容”去精确匹配推理结果。

### 4.5.2 顶层初始化

对应 [evaluate_task_result.py#L15](../accuracy/lm_eval/evaluate_task_result.py#L15) 到 [evaluate_task_result.py#L40](../accuracy/lm_eval/evaluate_task_result.py#L40)。

这里设置：

- 输入结果文件
- 任务名
- 模型类型
- few-shot 数

然后根据 `model_type` 设置环境变量 `MODEL_NAME`，目的是让 [`tasks/eval_harness.py`](../accuracy/lm_eval/tasks/eval_harness.py) 在初始化 tokenizer 时使用合适的模型词表。

### 4.5.3 `RealRunner.__init__`

定义在 [evaluate_task_result.py#L44](../accuracy/lm_eval/evaluate_task_result.py#L44)。

作用：把 `run_lm_eval_harness.py` 生成的 result JSONL 读进内存，建立从 request 到 result 的映射。

逐句解释：

- `self.results = {}`：初始化缓存字典。
- `with open(args.result_file, 'r') as f:`：打开结果文件。
- `for line in f:`：逐行读取。
- `if line.strip() == '': continue`：跳过空行。
- `item = json.loads(line)`：解析 JSON。
- `request = item['request']`：取请求部分。
- `result = item['result']`：取推理结果部分。
- `self.results[json_to_key(request)] = result`：用请求内容做 key 存起来。

### 4.5.4 `RealRunner.eval(batch)`

定义在 [evaluate_task_result.py#L63](../accuracy/lm_eval/evaluate_task_result.py#L63)。

这个函数是“把已经算好的 logits 结果重新翻译成 lm-eval 需要的 loss/correct”的关键。

逐句看：

- `from tasks.eval_harness import tokenizer`：取前面初始化好的 tokenizer。
- `mask_loss = []`：收集每个样本的损失。
- `each_correct = []`：收集每个样本是否答对。

对 batch 中每条文本：

- 重新按 `generate_task_data.py` 里的同样格式构造 `request`
- `key = json_to_key(request)`：生成查找 key。
- `if key in self.results:`：找到这条 prompt 的推理结果。
- `token_logprobs = ...`：取每个 token 的 log-prob。
- `tokens = ...`：取 token 字符串。
- `top_logprobs = ...`：取每个位置 top-1 候选。
- `obs / target / eval_mask = batch[...]`：取 lm-eval 这边的标注信息。

然后对 `eval_mask` 为真的位置做统计：

- `correct = correct and (tokens[i+1] == next(iter(top_logprobs[i+1].keys())))`：看该位置 top-1 是否等于真值 token。
- `sum_lobprob += token_logprobs[i+1]`：累加真值 token 的 log-prob。
- `n_positive += 1`：统计参与评估的位置数。

最后：

- `avg_logprob = sum_lobprob / n_positive`：求平均 log-prob。
- `mask_loss.append(-avg_logprob)`：转成 loss。
- `each_correct.append(correct)`：记录整题是否正确。

返回：

- `{'mask_loss': ..., 'each_correct': ...}`

再由 `EvalHarnessAdaptor.loglikelihood()` 转成 lm-eval 的标准输出格式。

## 5. Hello World 运行时的完整调用链

如果你只想抓主线，可以把整个 Hello World 记成下面这条链。

### 5.1 数据准备链

`README 命令`
-> [`generate_task_data.py`](../accuracy/lm_eval/generate_task_data.py)
-> [`EvalHarnessAdaptor`](../accuracy/lm_eval/tasks/eval_harness.py)
-> 外部 `lm_eval.evaluator.evaluate`
-> 生成 `results/openbookqa-5.jsonl`

### 5.2 推理链

`README 命令`
-> [`ours.sh`](../accuracy/lm_eval/ours.sh)
-> [`run_lm_eval_harness.py`](../accuracy/lm_eval/run_lm_eval_harness.py)
-> `set_symlink("opt", "modeling_opt_ours.py")`
-> [`modeling_opt_ours.py`](../accuracy/src/modeling_opt_ours.py)
-> 读取 `../setup/weights/opt-6.7b_0.2/partial_weight_q_*.pt`
-> 输出 `results/openbookqa-5-opt-6.7b-ours.jsonl`

### 5.3 回评链

[`evaluate_task_result.py`](../accuracy/lm_eval/evaluate_task_result.py)
-> [`EvalHarnessAdaptor`](../accuracy/lm_eval/tasks/eval_harness.py)
-> 外部 `lm_eval.evaluator.evaluate`
-> 打印 `openbookqa` 的最终结果

## 6. Setup 产物和 Hello World 参数是如何一一对应的

README 的 Hello World 命令：

```sh
bash ours.sh openbookqa ../setup/opt-model/opt-6.7b facebook/opt-6.7b opt 5 0.2 4 1.0 0.2
```

与 setup 产物的对应关系如下。

- `../setup/opt-model/opt-6.7b`
  来自 [`setup.sh`](../accuracy/setup/setup.sh#L14) 到 [`setup.sh`](../accuracy/setup/setup.sh#L18) 的 `gen_opt_model.py`
- `0.2`
  对应 [`setup.sh`](../accuracy/setup/setup.sh#L29) 设定的 `PARTIAL_RATIO=0.2`
- `../setup/weights/opt-6.7b_0.2`
  来自 [`setup.sh`](../accuracy/setup/setup.sh#L31) 到 [`setup.sh`](../accuracy/setup/setup.sh#L38) 的 `gen_partial_weight.py`
- `alpha=4 capacity=1.0 budget=0.2`
  由 [`ours.sh`](../accuracy/lm_eval/ours.sh#L9) 到 [`ours.sh`](../accuracy/lm_eval/ours.sh#L12) 传入，并在 [`run_lm_eval_harness.py`](../accuracy/lm_eval/run_lm_eval_harness.py#L114) 到 [`run_lm_eval_harness.py`](../accuracy/lm_eval/run_lm_eval_harness.py#L116) 注入每层 attention

## 7. 一句话总结每个关键文件的职责

- [`accuracy/setup/setup.sh`](../accuracy/setup/setup.sh)
  总控 setup，批量生成 skewed 模型、skewing matrix、partial weight。
- [`accuracy/setup/utils.py`](../accuracy/setup/utils.py)
  通过软链接切换 `transformers` 实际使用的模型实现文件。
- [`accuracy/setup/gen_opt_model.py`](../accuracy/setup/gen_opt_model.py)
  用真实文本激活构造 skewed OPT 模型。
- [`accuracy/setup/gen_partial_weight.py`](../accuracy/setup/gen_partial_weight.py)
  生成每层 `partial_weight_q`。
- [`accuracy/src/modeling_opt_orig.py`](../accuracy/src/modeling_opt_orig.py)
  setup 生成 skewed 模型时使用的 OPT 基线实现。
- [`accuracy/src/modeling_opt_ours_setup.py`](../accuracy/src/modeling_opt_ours_setup.py)
  setup 生成 `partial_weight_q` 时使用的 OPT setup 专用实现。
- [`accuracy/lm_eval/generate_task_data.py`](../accuracy/lm_eval/generate_task_data.py)
  把 benchmark task 转成 prompt JSONL。
- [`accuracy/lm_eval/ours.sh`](../accuracy/lm_eval/ours.sh)
  串起真实推理和结果回评。
- [`accuracy/lm_eval/run_lm_eval_harness.py`](../accuracy/lm_eval/run_lm_eval_harness.py)
  读取 prompt JSONL，调用 InfiniGen 模型推理，输出 token log-prob JSONL。
- [`accuracy/src/modeling_opt_ours.py`](../accuracy/src/modeling_opt_ours.py)
  InfiniGen 正式推理实现：用 `partial_weight_q` 预测 KV 保留掩码，再做真实 attention。
- [`accuracy/lm_eval/evaluate_task_result.py`](../accuracy/lm_eval/evaluate_task_result.py)
  把推理 JSONL 回填进 lm-eval，得到最终 benchmark 分数。

## 8. 如果你要继续顺代码看，建议的阅读顺序

最推荐的顺序是：

1. 先看 [`accuracy/README.md`](../accuracy/README.md)
2. 再看 [`accuracy/setup/setup.sh`](../accuracy/setup/setup.sh)
3. 看 [`accuracy/setup/utils.py`](../accuracy/setup/utils.py)
4. 看 [`accuracy/setup/gen_opt_model.py`](../accuracy/setup/gen_opt_model.py)
5. 看 [`accuracy/src/modeling_opt_orig.py`](../accuracy/src/modeling_opt_orig.py) 里的 `OPTAttention.forward`
6. 看 [`accuracy/setup/gen_partial_weight.py`](../accuracy/setup/gen_partial_weight.py)
7. 看 [`accuracy/src/modeling_opt_ours_setup.py`](../accuracy/src/modeling_opt_ours_setup.py) 里的 `OPTAttention.forward`
8. 看 [`accuracy/lm_eval/generate_task_data.py`](../accuracy/lm_eval/generate_task_data.py)
9. 看 [`accuracy/lm_eval/tasks/eval_harness.py`](../accuracy/lm_eval/tasks/eval_harness.py)
10. 看 [`accuracy/lm_eval/ours.sh`](../accuracy/lm_eval/ours.sh)
11. 看 [`accuracy/lm_eval/run_lm_eval_harness.py`](../accuracy/lm_eval/run_lm_eval_harness.py)
12. 最后重点看 [`accuracy/src/modeling_opt_ours.py`](../accuracy/src/modeling_opt_ours.py) 的 `kv_cache_mask`、`OPTAttention.forward`、`OPTDecoder.forward`

按这个顺序看，你会最容易把“setup 生成的产物”与“推理阶段如何消费这些产物”对上。
