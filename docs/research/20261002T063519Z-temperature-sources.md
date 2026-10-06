# 迭代 007：温度监控来源

用户新增要求：测试过程中监控温度，避免过热。实施前检查 CPU 与三卡温度可读取：首次完整采样 CPU 最大约 61°C，三卡温度通道最大分别为 41.25°C、39.12°C、41.38°C。这些编号为 sysfs 硬件编号，不自动对应运行槽位。

单位核实：NEC 官方驱动仓库 src/sensor_ve1.tbl 将 15..18、20..23 映射到 h 解码，24..26、28 到 i，27 到 j；src/decoder.c 的对应定义明确输出微摄氏度。监控这些已确认单位的通道，除以 1000000；14 和 19 当前读数为零，不纳入有效通道集合。CPU 仅采用 coretemp 的 temp*_input，除以 1000。

来源：https://github.com/veos-sxarr-NEC/ve_drv-kmod/blob/master/src/sensor_ve1.tbl ，https://github.com/veos-sxarr-NEC/ve_drv-kmod/blob/master/src/decoder.c 。只读研究，未复制源码到项目可提交文件。

主板 nct6793 有 SYSTIN=100°C、CPUTIN=-29.5°C 等未经映射/校准确认的读数，不能冒充 CPU 或 VE 温度；本次不将其作为这两类设备的控制输入。IPMI 设备不可见，未能另行核实主板温度。监控覆盖范围限于已确认的 CPU/VE 温度，不声称整机所有部件均已覆盖。

之前的性能运行未连续采样，不能补写其温度历史。后续测试通过温度监督入口。
