# 基础 Transformer 与 TinyStories

所有命令从仓库根目录执行。需要可执行 VE 运行槽位及本机 NEC 工具链；先做环境检查，每次构建和硬件测试均使用温度守护。ASPEED 显示控制器不承担推理。

## 基础验证

```bash
bash scripts/check_environment.sh
python3 scripts/temperature_guard.py -- make test
bash scripts/validate_ve.sh 1 2 3
bash scripts/validate_nlc.sh 1 2 3
bash scripts/validate_fast.sh 1 2 3
bash scripts/validate_workspace.sh 1 2 3
```

验证脚本内置温度监控。基础 C11/FP32 执行器支持前置 LayerNorm、多头因果或双向注意力、残差及 ReLU 前馈；扩展接口支持偏置和部分 GPT-Neo 语义。原生 API 见 [transformer.h](../../src/transformer.h)，每个并发调用者需要独立工作区。仅推理，不支持反向传播。

## 项目内环境与框架桥接

```bash
python3 scripts/temperature_guard.py -- bash scripts/setup_env.sh
python3 scripts/temperature_guard.py -- .venv/bin/python tests/check_pytorch.py
python3 scripts/temperature_guard.py -- .venv/bin/python examples/tiny_model.py
bash scripts/validate_framework.sh
```

基础依赖固定在 [requirements.lock](../../requirements.lock)，扩展 LLM 依赖固定在 [requirements-llm.lock](../../requirements-llm.lock)。再次运行基础环境的 sync 会移除额外 LLM 包，需要时再执行 `setup_llm.sh`。

Python 桥接 [ve_transformer.py](../../python/ve_transformer.py) 使用主机 CPU float32 数组，通过有界文件调用 VE；不是 PyTorch 原生 VE 设备后端。随机两层示例的嵌入和输出头在主机，不能用于有意义的文本生成。

## TinyStories-1M

```bash
python3 scripts/temperature_guard.py -- bash scripts/setup_llm.sh
python3 scripts/temperature_guard.py -- .venv/bin/python scripts/download_tinystories.py
python3 scripts/temperature_guard.py -- make resident
bash scripts/validate_llm.sh 1 2 3
bash scripts/validate_resident.sh 1
python3 scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
  .venv/bin/python examples/trained_lm.py --backend resident --node 1 --threads 1 --new-tokens 16
```

模型为作者训练的 GPT-Neo：8 层、宽 64、16 头，交替全局及局部因果注意力。全部 Transformer 块在 VE，嵌入和词表输出头仍在主机；单卡常驻进程复用权重/工作区，没有生成缓存。三卡旧逐层路径通过生成序列对照；常驻硬件验证覆盖槽位1。

同一短提示的五轮交替测量：单 VE 常驻约316词元/秒，单线程 CPU 同工作量约232词元/秒；不能推广为大模型或全 CPU 核心吞吐优势。详细口径和限制见 [常驻测量](../research/20261002T075034Z-resident-results.md)。

基础 NLC、注意力和工作区各轮的数据及旧入口，保存在 [原 README 详细历史](../research/20261008T032306Z-readme-history.md)，不将不同阶段直接混作当前性能。
