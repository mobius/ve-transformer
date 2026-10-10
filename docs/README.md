# 项目导航

## 使用文档

| 入口 | 内容 |
| --- | --- |
| [基础 Transformer 与 TinyStories](guides/transformer-tinystories.md) | 环境、三卡基础验证、框架桥接和小模型常驻 |
| [Qwen 文本推理](guides/qwen-inference.md) | 固定框架构建、1.5B 当前入口与单卡优化 |
| [Qwen3.6 MTP](guides/qwen36-mtp.md) | 预测头准备、常驻模式、收益与限制 |
| [SD-Turbo](guides/sd-turbo.md) | 独立图像环境、当前验证状态和参考流程 |
| [术语表](glossary.md) | 中英文术语解释 |
| [原 README 详细历史](research/20261008T032306Z-readme-history.md) | 迁移前完整记录，保留各阶段数据和失败 |

所有命令从仓库根目录执行。需要本机 NEC 工具链和可用 VE 运行槽，设备枚举不能证明可执行。全部硬件测试由温度守护包装；只观察可读风扇指标，不自行调整配置。

## 公开验收结果

- [Qwen 1.5B 输入复用优化](results/qwen25-1.5b-input-reuse.json)
- [Qwen3.6 基础单卡测量](results/qwen36-single-ve.json)
- [Qwen3.6 原生 MTP](results/20261007T011801Z-qwen36-native-mtp.json) 与 [常驻验收](results/20261008T020107Z-qwen36-mtp-session.json)
- [SD-Turbo 来源](results/20261008T021839Z-sd-turbo-model.json) 与 [首例完整 VE 图像验收](results/20261008T081632Z-sd-turbo-validation.json)：512×512 单步与12项算子通过；[六提示整图验收](results/20261008T091001Z-sd-turbo-validation.json)及[四步整图验收](results/20261008T091957Z-sd-turbo-validation.json)均通过；尚无稳态性能基准。

- [SD-Turbo 常驻线程池优化](results/20261008T102317Z-sd-turbo-pool.json)：单提示ABBA全进程延迟下降19.66%、图执行下降27.43%，四步数值复用验证通过。

- [SD-Turbo 算子计时](results/20261008T110130Z-sd-turbo-operator-profile.json)与[SiLU优化](results/20261008T114450Z-sd-turbo-silu.json)：线程池基础上单提示ABBA全进程延迟下降22.71%、图执行下降33.81%，四步与算子回归通过。

- [SD-Turbo Softmax优化](results/20261009T031452Z-sd-turbo-softmax.json)：SiLU与线程池基础上，全进程延迟下降25.52%、UNet图执行下降86.10%，四步和算子回归通过。

结果以各报告的模型、二进制、数学精度、输入和计时范围为准，不能把不同阶段直接混作同一性能基线。

## 目录与过程记录

| 目录 | 内容 |
|---|---|
| `src/` | 基础Transformer、常驻执行器、Qwen/ggml接入和VE内核 |
| `python/` | 主机框架与VE执行器桥接 |
| `framework/`、`cmake/` | 框架构建与NEC VE工具链配置 |
| `scripts/` | 项目内环境、模型下载、构建、温度监督和验证入口 |
| `tests/` | 数学正确性、模型精度和性能测量 |
| `examples/` | 小模型与训练模型推理示例 |
| `docs/research/` | 调研、测量和限制 |
| `docs/plan/` | 每轮实施与验收计划 |
| `docs/impl/` | 实现过程和验证证据 |
| `docs/architecture/` | 数学路径、生命周期和运行边界 |
| `docs/results/` | 经审计的公开测量摘要 |
| `docs/guides/` | 当前使用步骤与复现边界 |
| `docs/glossary.md` | 中英文技术术语 |

历史过程文档按UTC时间戳保留，旧状态和失败结果属于对应迭代，当前结果以对应验收为准。早期 Qwen3.6 基础单卡验收：`20261002T175953Z-qwen36-final-acceptance.md`，各过程目录均有记录；后续 MTP 与常驻请求单独验收。

`build/`集中存放模型、下载缓存、上游源码、二进制、原始日志和测量轨迹；`.venv/`为本地 LLM 依赖环境，图像参考环境另在 `build/venvs/sd-image`，均不提交。原始结果可能带有本机路径，公开结果只包含经筛选审计的字段。

上游 llama.cpp 固定修订为 `8d81559fa7b8bcac9f7c8b478858953486371f90`；图像框架独立固定来源，详见对应使用文档。模型和第三方软件遵循各自许可，不随本仓库发布。参见 [第三方来源](../THIRD_PARTY.md)。

- [累计优化结果、全部迭代及下一步](guides/optimization-history.md)：性能、失败和待验证轮次一并保留。
