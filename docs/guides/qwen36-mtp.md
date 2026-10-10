# Qwen3.6 原生 MTP 实验

MTP 使用模型原生预测头提出草稿词元，再由完整目标模型验证。当前覆盖贪心文本推理、同进程重置及常驻请求；没有采样、视觉、其他 VE 卡或训练最大上下文验收。实验未设为默认。

## 准备

先完成 [Qwen 框架与精度控制对象](qwen-inference.md) 及输入复用 hook 的构建。使用本地 Qwen3.6-35B-A3B UD-Q4_K_M 目标与官方匹配预测头；共享嵌入/输出张量逐字节核对，运行时共用目标的权重引用和 FP64 缓存，MTP 主体为 F32。

```bash
.venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
  .venv/bin/python scripts/prepare_qwen36_mtp_weights.py
bash scripts/convert_qwen36_mtp.sh
bash scripts/build_qwen36_mtp.sh
.venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
  .venv/bin/python tests/check_qwen36_mtp.py --boundaries
```

模型准备选取约3.47GiB BF16来源，转换中间文件约7.45GB，最终 MTP GGUF 约4.35GB。来源修订、哈希与兼容检查留档；目标全部原始张量的官方提交身份尚未确立。权重不随仓库发布。

## 常驻请求

```bash
.venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
  .venv/bin/python python/qwen_session.py --engine qwen36 \
  --executor build/qwen36-mtp/qwen-mtp-ve \
  --model build/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --mtp-model build/models/qwen36-mtp/mtp-shared-q4km.gguf \
  --auto-mtp --threads 8 --context 512 --tokens 32 \
  --prompt 'Write a Python function that adds two integers.'
```

省略 `--auto-mtp` 为固定 MTP；同时省略 `--mtp-model` 为普通常驻生成。自动策略先测普通解码，至少观察三轮草稿成本，收益不足则对本请求停用草稿；不改变目标贪心序列。客户端使用非思考聊天格式，生成文本只保留在本地。

常驻三模式各12个请求的独立 CPU 序列、跨请求重置和三类异常帧拒绝均通过。中文样本触发自动回退后仍未优于普通生成，故不默认采用。硬件最高 CPU76°C、VE57.12°C。详见 [常驻验收](../results/20261008T020107Z-qwen36-mtp-session.json)。

## 性能解释

独立 CLI 的代码提示、两草稿、正式 ABBA 生成约0.585→0.932词元/秒（+59.4%）；中文短测两草稿约−10.8%，一草稿约−13.7%。代码热态预填充约7.938→8.116秒，加载约18.009→32.075秒；解码加速不等于冷启动总请求加速。这些数值不是常驻协议的新性能基线。

逐样本和口径见 [原生 MTP 验收](../results/20261007T011801Z-qwen36-native-mtp.json)。旧验收二进制保留在本地 `build/qwen36-mtp-accepted`；新二进制以常驻验收为准。
