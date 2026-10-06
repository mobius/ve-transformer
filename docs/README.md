# 项目导航

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
| `docs/glossary.md` | 中英文技术术语 |

历史过程文档按UTC时间戳保留，旧状态和失败结果属于对应迭代，当前结果以最终验收为准。Qwen单卡最终验收：`20261002T175953Z-qwen36-final-acceptance.md`，各过程目录均有记录。

`build/`集中存放模型、下载缓存、上游源码、二进制、原始日志和测量轨迹；`.venv/`为本地依赖环境，两者不提交。原始结果可能带有本机路径，不作为公开交付；公开结果为经过字段筛选的`results/qwen36-single-ve.json`。

复现命令见根目录README。需要本机NEC工具链和可用VE运行槽，设备枚举不能证明可执行。所有硬件测试先检查环境并由温度守护包装。本轮发布整理只做静态检查，没有重新运行硬件测试。

上游llama.cpp固定修订为`8d81559fa7b8bcac9f7c8b478858953486371f90`，在忽略提交的build目录下载。模型和第三方软件遵循各自许可，不随本仓库发布。参见根目录THIRD_PARTY.md。
