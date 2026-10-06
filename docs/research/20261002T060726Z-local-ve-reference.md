# 迭代 002：本机已有 VE 项目调查

用户提供本机已有 NEC VE 测试项目供只读参考；未修改该项目，也未执行其安装、故障注入或压力测试脚本。为避免传播账户路径，文档不记录源目录的完整路径。

阅读内容：README、PerfTest 的 Makefile、scripts/common.sh、multi_independent.sh、perf_nlc_dgemm.sh、aveo_multi_test.c、ve_nlc_dgemm.c 及 NLC 验证记录。

发现：

- 运行槽位使用 1、2、3，执行方式为 ve_exec -N N；公共脚本优先根据服务探测槽位，失败后才回退固定编号。
- VE_LD_LIBRARY_PATH 用于卡上动态库；已有代码设置 nfort 运行库，NLC 测试另设置数学库。
- 旧文档记录三卡 NLC 双精度矩阵乘法约 1720–1768 GFLOPS。这是历史项目报告，不是本次测量，也不能用于估算当前 float32 Transformer 性能。
- 本机 NLC cblas.h、libcblas 和 BLAS 库已安装，后续矩阵乘法优化无需先安装全局软件。
- /var/opt/nec/ve/veos 下可见对应 1、2、3 的服务 socket 文件；文件存在不等于本会话可调用服务。

重新探测槽位 1、2、3：均返回 Node 'N' is Offline。读取本机 ve_exec 包装脚本确认：它先检查 /dev/veslotN 是否为链接，缺失时就直接返回这个消息，尚未启动真正的 VE 执行程序。

读取当前进程 mountinfo 确认：主机 /dev 上叠加了沙箱私有 tmpfs，只有基础设备被显式暴露，没有 veslot/VE 设备。systemctl 连接系统总线也返回 Operation not permitted。因此可以确定当前设备不可见来自会话隔离；无法据此认定宿主机卡或 VEOS 离线。

修正迭代 001 中的探测不足：sysfs ve0/ve1/ve2 不应直接当成运行槽位，/dev/veN 也不是 ve_exec 实际选择入口。当前沙箱仍不能完成任何 VE 运行测试。
