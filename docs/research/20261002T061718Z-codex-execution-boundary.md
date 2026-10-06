# 迭代 004：Codex 为何不能直接使用 VE

用户询问如何让 agent 自行执行测试。本轮仅调查执行边界，不改系统或会话配置。

- 当前会话 workspace-write 沙箱只能写项目和临时目录；主机 /dev 被私有目录覆盖，veslot 设备未暴露。
- 当前策略关闭 sandbox_approval 类别，不能通过本会话申请沙箱外执行；这不是已发起审批被拒绝，而是会话预设限制。
- 用户主机测试已证明三卡可执行。无需重装驱动或更改设备权限。
- 本机 codex --help 和 codex resume --help 确认支持 --sandbox danger-full-access、--ask-for-approval on-request 及 --cd 参数。
- 可由用户在主机启动新会话或恢复项目会话，主动选择上述执行模式；这扩大主机访问范围，不是只授权三张 VE，且托管策略可能限制此选项。
- 本 agent 不修改用户全局配置、不自行启动嵌套不受限 agent，也不尝试绕过当前隔离。

官方参考：https://developers.openai.com/codex/permissions ，https://learn.chatgpt.com/docs/sandboxing 。具体本机命令以已安装 CLI 帮助核实。
