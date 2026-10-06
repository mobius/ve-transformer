# Qwen3.6 性能估计更新

本轮只读分析已有实测，无新增硬件运行或安装。目标限定 Qwen3.6-35B-A3B 的文本推理和约22.13GB量化权重。官方模型来源：https://huggingface.co/Qwen/Qwen3.6-35B-A3B 。单卡8线程短输入：直接量化路径约1.46 token/s、输入处理约14 token/s；CPU相同量化路径约3.14 token/s。高精度诊断路径约0.117 token/s。证据为 build/qwen-final-full-quant.log、build/qwen-block-scales-paired.log、build/qwen-final-correctness-accurate.log。所有这些VE完整分数精度门槛尚未通过，不构成已验收速度。
