# Qwen 文本推理与单卡优化

命令从仓库根目录执行。当前已验证路径使用 FP64 累加和 FP32 激活，读取量化权重；不能把原始低精度路径的失败结果计入验收成绩。模型、参考数组和原始日志均在本地 `build/`，仓库不包含它们。

## 构建前提

本机需要 NEC 工具链、可执行运行槽位、项目内 uv 环境。LLM 固定 llama.cpp 修订 `8d81559fa7b8bcac9f7c8b478858953486371f90`。从已有环境准备框架与精度控制对象：

```bash
python3 scripts/temperature_guard.py -- bash scripts/setup_llm.sh
bash scripts/build_qwen.sh cpu
bash scripts/build_qwen.sh ve
python3 scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
  .venv/bin/python scripts/build_qwen_precision.py
bash scripts/build_qwen_accurate.sh
```

构建脚本内置温度监控；外层包装的入口不可省去守护。这里是完整本地重建流程，编译耗时和温度须单独观察，不必为了运行已有已验收二进制重复构建。

## Qwen2.5-1.5B-Instruct

模型使用 Q4_K_M。先准备模型和基础执行器，再构建常驻及两轮优化：

```bash
python3 scripts/temperature_guard.py -- .venv/bin/python scripts/download_qwen15.py
bash scripts/build_qwen15.sh
bash scripts/build_qwen15_session.sh
bash scripts/build_qwen15_prefill.sh
bash scripts/build_qwen15_input_reuse.sh
```

单卡连续请求入口：

```bash
.venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
  .venv/bin/python python/qwen_session.py \
  --model build/models/qwen25-1.5b/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --prompt 'Write a Python function that adds two integers.' \
  --prompt '请用一句话说明二分查找的原理。' --tokens 32
```

客户端优先选用哈希匹配公开采用报告的输入复用执行器；不匹配时依次回退上一轮执行器及常驻基线。每条请求清空上下文，模型和不可变权重缓存跨请求保留；分词及模型计算在 VE，主机负责调用和收取结果。接口是本地 stdin/stdout，不提供网络服务。

当前单 VE、四线程的热态测量约4.896词元/秒，17输入/32输出请求约7.537秒；相对上一轮生成提升8.69%。这是同轮交替对照中位数，首次加载和缓存建立成本另计。FP64 权重缓存有效载荷约11.50GiB，量化文件大小不是运行内存。逐样本、精度边界及采用哈希见 [输入复用验收](../results/qwen25-1.5b-input-reuse.json)。

模型验证依赖本地既有 CPU 参考，可指定对应目录：

```bash
.venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
  .venv/bin/python tests/check_qwen15_attention.py \
  --baseline-dir build/qwen15-projection --candidate-dir build/qwen15-input-reuse --order abba
.venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
  .venv/bin/python tests/check_qwen15_boundaries.py --candidate-dir build/qwen15-input-reuse
```

## Qwen3.6-35B-A3B

这是 MoE 模型，约35B总参数、每词元约3B激活参数；当前为单 VE 文本推理，不将三卡 HBM 视为统一内存。模型准备与最初高精度验收：

```bash
python3 scripts/temperature_guard.py -- .venv/bin/python scripts/download_qwen36.py
bash scripts/validate_qwen.sh correctness --backend accurate
```

基本单卡性能与长输入边界见 [最终测量摘要](../results/qwen36-single-ve.json)。后续使用共用权重/缓存的 MTP 执行器进行普通生成与草稿验证，命令及当前验收见 [MTP 使用文档](qwen36-mtp.md)。不同版本、参考数学和输入长度的结果不要直接混作同一性能基线。

节点 trace 和 profiling 仅用于定位问题，正式速度测量须关闭。已拒绝路径、各线程扫描和历史详细表保存在 [原 README 详细历史](../research/20261008T032306Z-readme-history.md) 与四类迭代文档。
