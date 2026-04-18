# `LLM-Research/Meta-Llama-3.1-8B` 运行文档

模型页面：

- https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B/summary

本文档基于当前机器环境，给出这只模型在本机上可行的运行方案，包括但不限于：

- `WSL2 + Conda + Transformers`
- `Windows 原生 + Conda`
- `Docker`
- `WSL2 + vLLM`
- `llama.cpp / GGUF`

## 1. 先说结论

结合这台机器的环境，最推荐的顺序是：

1. `WSL2 + Ubuntu + Conda + Transformers + 4bit`  
   这是当前机器上最现实、最稳的本地 GPU 方案。
2. `Docker + GPU 容器 + Transformers + 4bit`  
   可以做，但你当前 `Docker Engine` 没启动，要先把 Docker Desktop 拉起来。
3. `Windows 原生 + Conda + CPU/下载验证`  
   能跑，但不适合作为 8B 模型的日常本地推理方案。
4. `WSL2 + vLLM`  
   适合做 API 服务，但对你这张 `RTX 4070 12GB` 来说，这个“原始 8B 权重”方案很容易卡在显存上，除非换量化权重或更大显卡。
5. `llama.cpp / GGUF`  
   对 12GB 显存其实很友好，但前提是你愿意把原始 Hugging Face / ModelScope 权重转成 `GGUF` 并量化。

## 2. 当前环境检查结果

以下信息是我刚刚在本机上实际检查到的。

### 2.1 系统与工具

- OS: `Windows`，PowerShell `5.1.26100.7920`
- 全局 Python: `3.13.9`
- Conda: `25.9.1`
- Git: `2.47.1.windows.1`
- Git LFS: `3.6.0`
- Docker CLI: `27.2.0`
- Docker Engine: 当前未运行
- WSL: 已安装，默认发行版是 `Ubuntu`，版本是 `WSL2`

### 2.2 GPU

- 显卡: `NVIDIA GeForce RTX 4070`
- 显存: `12 GB`
- 驱动: `577.00`
- 驱动报告的 CUDA 版本: `12.9`

### 2.3 现有 Conda 环境

已有环境：

- `base`
- `infinigen`
- `llama_cpp`
- `powerinfer`
- `pytorch_gpu`

其中比较关键的是：

- `pytorch_gpu`:
  - Python `3.12.12`
  - `torch 2.5.1`
  - CUDA 可用
  - 但当前没有装 `modelscope`、`accelerate`、`bitsandbytes`
- `infinigen`:
  - Python `3.9.25`
  - `torch 2.0.1+cpu`
  - 有本仓库自己的 `transformers` 源码遮蔽风险

## 3. 最重要的避坑

### 3.1 不要在 `d:\infinigen` 仓库根目录里直接跑你的模型脚本

原因是这个仓库里自带了一个本地源码目录：

- [transformers](d:/infinigen/transformers)

如果你在 `d:\infinigen` 根目录直接运行 Python，`import transformers` 很可能会导入仓库里的本地源码，而不是你 Conda 环境里真正安装的 pip 包。

所以建议：

- 模型运行目录单独放到 `D:\llm\meta-llama-3.1-8b`
- 或者在 `C:\Users\wlh` / WSL 的 `~/llm` 下跑

### 3.2 你给的这个模型是 `base` 模型，不是 `Instruct` 模型

从模型名 `Meta-Llama-3.1-8B` 可以推断，这更像基础续写模型，而不是聊天指令模型。

这意味着：

- 它可以跑文本生成
- 但如果你想直接做对话，效果通常不如 `Instruct` 版

这里我是在根据模型命名做推断。

### 3.3 12GB 显存不适合直接跑原始 `fp16` 版 8B

经验上，`8B` 模型的：

- `fp16` 权重大约就已经接近或超过 `16GB`
- 再加上 KV cache 和运行开销，本机 `12GB` 显存不现实

所以对你这张卡，推荐优先考虑：

- `4bit`
- 或者 `GGUF` 量化
- 或者 CPU 仅做功能验证

## 4. 统一准备

下面几个方案都建议先准备一个独立目录。

### 4.1 Windows 目录

```powershell
mkdir D:\llm\meta-llama-3.1-8b -Force
mkdir D:\llm\cache\modelscope -Force
mkdir D:\llm\cache\hf -Force
cd D:\llm\meta-llama-3.1-8b
```

### 4.2 WSL 目录

```bash
mkdir -p /mnt/d/llm/meta-llama-3.1-8b
mkdir -p /mnt/d/llm/cache/modelscope
mkdir -p /mnt/d/llm/cache/hf
cd /mnt/d/llm/meta-llama-3.1-8b
```

### 4.3 建议统一设置缓存路径

PowerShell:

```powershell
$env:MODELSCOPE_CACHE="D:\llm\cache\modelscope"
$env:HF_HOME="D:\llm\cache\hf"
```

WSL:

```bash
export MODELSCOPE_CACHE=/mnt/d/llm/cache/modelscope
export HF_HOME=/mnt/d/llm/cache/hf
```

## 5. 方案 A：`WSL2 + Ubuntu + Conda + Transformers + 4bit`

这是最推荐的本地 GPU 方案。

### 5.1 适用性

适合你当前机器，因为：

- `WSL2` 已安装
- `RTX 4070 12GB` 可以尝试 `4bit`
- `vLLM` 在官方文档里明确说明 Windows 原生不支持，Windows 用户应通过 `WSL` 运行 Linux 环境

### 5.2 进入 WSL

在 PowerShell 里执行：

```powershell
wsl -d Ubuntu
```

### 5.3 安装基础依赖

如果是新装的 Ubuntu，先做：

```bash
sudo apt update
sudo apt install -y git git-lfs wget curl build-essential
git lfs install
```

如果你的 WSL 里已经有 Conda，可以直接跳到下一步。  
如果没有，就先装一个 Miniconda。

### 5.4 建新环境

```bash
conda create -n ms-llama31 python=3.12 -y
conda activate ms-llama31
```

### 5.5 安装 PyTorch 和推理依赖

先装 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

再装模型运行依赖：

```bash
pip install "transformers>=4.43" accelerate sentencepiece safetensors modelscope
pip install bitsandbytes
```

说明：

- 这里选 `Python 3.12`
- 选 `cu124` 是因为你本机驱动足够新，兼容没问题
- `bitsandbytes` 是为了做 `4bit` 加载

### 5.6 先只下载模型

在 `/mnt/d/llm/meta-llama-3.1-8b` 下新建 `download_model.py`：

```python
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download(
    model_id="LLM-Research/Meta-Llama-3.1-8B",
    cache_dir="/mnt/d/llm/cache/modelscope",
)

print(model_dir)
```

运行：

```bash
python download_model.py
```

### 5.7 4bit 本地生成脚本

新建 `run_4bit.py`：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from modelscope.hub.snapshot_download import snapshot_download

MODEL_ID = "LLM-Research/Meta-Llama-3.1-8B"

model_dir = snapshot_download(
    model_id=MODEL_ID,
    cache_dir="/mnt/d/llm/cache/modelscope",
)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.7,
        do_sample=True,
        use_cache=True,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

运行：

```bash
python run_4bit.py
```

### 5.8 如果显存还是不够

优先按这个顺序收紧：

1. `max_new_tokens` 先降到 `16`
2. prompt 尽量短
3. 先关闭采样，改成 `do_sample=False`
4. 必要时把 `device_map="auto"` 保留，并允许一部分权重落到 CPU

如果还是不行，直接转去方案 E 的 `GGUF` 路线。

## 6. 方案 B：`Windows 原生 + Conda`

这个方案可以做，但我只推荐它承担两类工作：

- 下载模型
- 做非常短的 CPU 验证

不推荐你把它当作这只 `8B` 模型的主力本地推理环境。

### 6.1 新建干净环境

不要用当前全局 `Python 3.13.9`。  
建议新建一个 `Python 3.12` 环境：

```powershell
conda create -n ms-llama31-win python=3.12 -y
conda activate ms-llama31-win
```

### 6.2 安装依赖

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.43" sentencepiece safetensors modelscope
```

如果只是下载模型，其实连 CUDA 版 PyTorch 都不是必须的。

### 6.3 下载模型

新建 `download_model.py`：

```python
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download(
    model_id="LLM-Research/Meta-Llama-3.1-8B",
    cache_dir=r"D:\llm\cache\modelscope",
)
print(model_dir)
```

运行：

```powershell
python download_model.py
```

### 6.4 CPU 验证脚本

新建 `run_cpu_smoke.py`：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download(
    model_id="LLM-Research/Meta-Llama-3.1-8B",
    cache_dir=r"D:\llm\cache\modelscope",
)

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_dir)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

运行：

```powershell
python run_cpu_smoke.py
```

说明：

- 这个方案非常慢
- 更适合做“模型文件没问题、代码能跑通”的检查

### 6.5 Git LFS 直接拉模型仓库

你本机已经装了 `git-lfs`，所以还有一个可行思路是直接克隆模型仓库。

常见的 ModelScope Git 地址格式是：

```powershell
git lfs install
git clone https://www.modelscope.cn/LLM-Research/Meta-Llama-3.1-8B.git
```

然后再把本地目录作为 `from_pretrained()` 的输入路径。

这里我是在根据 ModelScope 常见仓库地址格式做推断；如果页面要求登录、鉴权或先接受许可，先在浏览器里完成。

## 7. 方案 C：`Docker + GPU 容器`

这个方案可以做，而且很干净，但你当前机器的状态是：

- `docker --version` 有结果
- 但 `docker info` 报的是 engine 没起来

所以在开跑前，你要先：

1. 启动 Docker Desktop
2. 确认它使用 `WSL2 backend`
3. 确认 GPU 容器可用

### 7.1 先检查 Docker 是否恢复正常

```powershell
docker info
```

能正常输出服务端信息，再继续。

### 7.2 推荐的容器方式

更推荐你用较新的 `PyTorch CUDA` 基础镜像，再自己装 `modelscope`。

例如：

```powershell
docker run --gpus all -it --rm `
  -v D:\llm\meta-llama-3.1-8b:/workspace `
  -v D:\llm\cache\modelscope:/root/.cache/modelscope `
  -v D:\llm\cache\hf:/root/.cache/huggingface `
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
```

进容器后：

```bash
pip install "transformers>=4.43" accelerate sentencepiece safetensors modelscope bitsandbytes
cd /workspace
python run_4bit.py
```

这里的 `run_4bit.py` 可以直接复用方案 A 的脚本。

### 7.3 ModelScope 官方镜像

ModelScope 官方 PyPI 页面也提供了官方 Docker 镜像示例。

但需要注意：

- 页面展示的镜像偏旧
- 示例 GPU 镜像是 `cuda11.8 + torch2.0.1`

所以它是“可以用”的方案，但不是我在你这台机器上的首推方案。

## 8. 方案 D：`WSL2 + vLLM`

这个方案适合你后面要把模型变成一个本地 API 服务。

但在你这台机器上，它有两个现实限制：

1. 官方文档明确说 `vLLM` 不原生支持 Windows，要走 `WSL`
2. 你这张 `12GB` 卡对“未量化的 8B 权重”比较吃力

所以：

- 如果你坚持要 `vLLM`
- 最好准备量化模型
- 或者换更大显存卡

### 8.1 新环境

在 WSL 里：

```bash
conda create -n ms-llama31-vllm python=3.12 -y
conda activate ms-llama31-vllm
```

### 8.2 安装

官方文档给出的 CUDA 12.9 预编译 wheel 安装方式是：

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
pip install modelscope
```

### 8.3 下载模型

```bash
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
print(snapshot_download("LLM-Research/Meta-Llama-3.1-8B", cache_dir="/mnt/d/llm/cache/modelscope"))
PY
```

### 8.4 启动服务

假设模型下载到 `/mnt/d/llm/cache/modelscope/.../Meta-Llama-3.1-8B`，可尝试：

```bash
vllm serve /mnt/d/llm/cache/modelscope/LLM-Research/Meta-Llama-3.1-8B \
  --dtype half \
  --max-model-len 2048
```

但请预期：

- 很可能因为显存不足失败
- 或者必须把上下文长度压得很低

因此，这个方案在你当前机器上更适合做“后续扩展方案”，不是首选。

## 9. 方案 E：`llama.cpp / GGUF`

如果你的目标是“这张 12GB 卡上真的想舒服地本地跑 8B”，这个路线其实很值得考虑。

它的思路是：

1. 先从 ModelScope 下载原始权重
2. 转成 `GGUF`
3. 再量化成 `Q4_K_M`、`Q5_K_M` 之类
4. 用 `llama.cpp` 或兼容运行器推理

### 9.1 优点

- 对 12GB 显存更友好
- 推理资源占用更可控
- 适合桌面本地长期使用

### 9.2 缺点

- 不是“原始 ModelScope 权重拿来就跑”
- 需要额外做格式转换和量化

### 9.3 实际建议

如果你后面发现：

- `Transformers + 4bit` 还是太吃力
- 或者你更在意稳定本地推理

那就切到这条路线。

当前你已有一个叫 `llama_cpp` 的 Conda 环境，但里面还没有现成的 `llama_cpp` Python 模块，所以如果要走这条路，建议单独再整理。

## 10. 我对你这台机器的推荐执行顺序

我建议你按这个顺序实际操作：

1. 先不要在 `d:\infinigen` 根目录跑任何模型脚本
2. 先用方案 A，在 `WSL2` 里跑 `download_model.py`
3. 下载成功后再跑 `run_4bit.py`
4. 如果 4bit 还是吃紧，就别继续在原始权重上硬顶，直接改走方案 E
5. 如果你后面想要服务化部署，再评估方案 D 的 `vLLM`

## 11. 最小可执行路径

如果你只想先把它跑起来，最短路径就是下面这组动作。

### 11.1 在 PowerShell 里

```powershell
wsl -d Ubuntu
```

### 11.2 在 WSL 里

```bash
sudo apt update
sudo apt install -y git git-lfs wget curl build-essential
git lfs install

conda create -n ms-llama31 python=3.12 -y
conda activate ms-llama31

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.43" accelerate sentencepiece safetensors modelscope bitsandbytes

mkdir -p /mnt/d/llm/meta-llama-3.1-8b
cd /mnt/d/llm/meta-llama-3.1-8b
```

然后把方案 A 里的 `run_4bit.py` 保存下来，运行：

```bash
python run_4bit.py
```

## 12. 来源

我这次写文档时，除了本机实际环境检查，还参考了这些官方或一手来源：

- ModelScope PyPI 页面：安装方式、Docker 镜像、Python 版本要求  
  https://pypi.org/project/modelscope/
- ModelScope GitHub / PyPI 描述：大多数模型可通过网站、API 或 git 下载  
  https://github.com/modelscope/modelscope
- Hugging Face Transformers 官方 bitsandbytes 文档：`4bit` / `NF4` / `device_map="auto"`  
  https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
- vLLM 官方安装文档：Windows 原生不支持，建议 WSL；CUDA 12.9 wheel  
  https://docs.vllm.ai/en/latest/getting_started/installation/
  https://docs.vllm.ai/en/latest/getting_started/installation/gpu/

## 13. 你如果要我继续做什么

如果你下一步想直接开始，我最建议我继续帮你做这两件事里的一个：

1. 我直接在 `docs` 之外，再给你生成一套可复制运行的脚本文件：
   - `download_model.py`
   - `run_4bit.py`
   - `run_cpu_smoke.py`
2. 我继续替你检查 `WSL2 Ubuntu` 里的 Python / CUDA / Conda 状态，并把方案 A 细化成“逐条执行版”。
