# VE Transformer 推理移植

面向 NEC Vector Engine 的 Transformer、语言模型和图像模型推理项目。采用原生执行器和 NEC NLC 数学库，逐步验证正确性并按实测结果优化；不包含训练功能。

| 模型 / 功能 | 当前验证范围 |
| --- | --- |
| 基础 Transformer | 三张 VE 的前向计算与 NLC 优化已验证 |
| TinyStories-1M | 三卡生成验证；单卡常驻执行，嵌入和输出头在主机 |
| Qwen2.5-1.5B-Instruct Q4_K_M | 单 VE 完整文本推理、常驻请求和输入复用优化已验证 |
| Qwen3.6-35B-A3B | 单 VE 文本推理及原生 MTP 已验证；MTP 收益取决于提示，未设为默认 |
| SD-Turbo | 单 VE FP32 512×512 整图通过；常驻线程池优化已验证，详见文档 |

需要可运行的 VE 设备、NEC 编译器和数学库；本机验证使用 NCC 5.4.1、NLC 3.1.0。先检查环境，再执行带温度守护的基础验证：

```bash
bash scripts/check_environment.sh
bash scripts/validate_ve.sh 1 2 3
```

设备枚举不等于执行可用。全部硬件测试必须监控温度：CPU 80°C 警告、85°C 停止；VE 70°C 警告、75°C 停止；监控失效也停止。直接运行程序时用 `scripts/temperature_guard.py` 包装，具体命令见使用文档。

依赖安装在项目内 uv 环境，模型、缓存、二进制和原始日志放在被 Git 忽略的 `build/`；不安装全局软件、不提交权重或敏感信息。

- [文档导航与公开结果](docs/README.md)
- [基础 Transformer 与 TinyStories](docs/guides/transformer-tinystories.md)
- [Qwen 文本推理与单卡优化](docs/guides/qwen-inference.md)
- [Qwen3.6 MTP 实验](docs/guides/qwen36-mtp.md)
- [SD-Turbo 图像移植](docs/guides/sd-turbo.md)
- [术语表](docs/glossary.md) · [第三方来源](THIRD_PARTY.md) · [开发约定](AGENTS.md)

详细性能、复现步骤和历史迭代保存在 `docs/`；测量仅适用于各报告注明的模型、输入和运行配置。
