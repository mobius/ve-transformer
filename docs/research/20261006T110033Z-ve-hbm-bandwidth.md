# VE HBM 带宽规格核查

本轮只读项目历史环境记录与 NEC 官方规格，没有进行硬件压力测试或安装软件。既有环境为 96 CPU 逻辑线程、ASPEED 显示控制器、三张 8 核/48 GB 级 VE；精确子型号尚未确定，不能由 PCI 枚举名称或容量直接认定 10B。

NEC 官方 A101-1 规格中 Type 10B 的最大内存带宽为 1.22 TB/s（1220 GB/s，十进制）。官方 VE 型号页中 Type 10AE 为 1.35 TB/s、Type 10CE 为 1.00 TB/s；因此 1.22 TB/s 可作 10B 参考值，不是本机已核实值。HBM 是卡内高带宽内存，这一带宽不是主机 PCIe 传输带宽。三张卡各有本地内存，不能把三卡峰值之和当作当前单卡推理可用带宽。

当前没有独立 HBM 带宽基准或硬件流量计数结果，不能把每 token 权重负载乘生成速度当作实测 HBM 带宽或带宽利用率。最近分项计时只覆盖部分阶段，输入整理占比也不是全部推理占比。

来源：
- https://www.nec.com/en/global/solutions/hpc/sx/A101-1.html
- https://www.nec.com/en/global/solutions/hpc/sx/vector_engine.html

后续如确认精确子型号和实测持续带宽，仍须先核对环境并使用温度守护。本轮未执行此测量。
