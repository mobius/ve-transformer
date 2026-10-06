# 项目开发约定

本项目由 OpenAI Codex（GPT-6，主 agent `/root`）开发。

- 优先中文沟通；新增技术术语解释并记录到 `docs/glossary.md`。
- 每次实施前判断 CPU、GPU、NEC VE 硬件及运行环境是否支持。设备枚举不等于运行可用。
- 每次迭代持续在 `docs/research`、`docs/plan`、`docs/impl`、`docs/architecture` 留档，文件名以 UTC 时间戳 `YYYYMMDDTHHMMSSZ` 开头。
- 安装软件优先项目内 uv 环境，其次 conda、podman、docker；禁止污染全局环境。
- 提交前审计代码、文档与待提交文件，不记录凭据、令牌、设备序列号、私人路径或其他敏感信息。
- 未实际在 VE 上运行的功能不得宣称已完成硬件验证；未测量不得声称性能提升。
- 当前会话未授权多 agent 委派。
- 所有硬件测试必须先检查并持续监控 CPU/VE 温度。依用户最新指示，CPU 默认 80°C 警告、85°C 停止；VE 保持 70°C 警告、75°C 停止；温度监控不可用也停止。使用 `scripts/temperature_guard.py` 或已接入该监控的验证/基准脚本，不得跳过温度监控。观察可读风扇转速和策略变化，不自行改风扇配置。阈值为项目测试策略，不是厂商极限。
