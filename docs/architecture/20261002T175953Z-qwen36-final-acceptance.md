# 已验收运行配置与边界

环境：双路Xeon、显示GPU不参与推理、三张NEC VE中使用运行槽1/物理ve0；真实模型已实际执行，不以设备枚举替代运行验证。没有全局安装或风扇配置修改。

测试限定为Qwen3.6-35B-A3B UD-Q4_K_M文本权重、4096上下文配置，FP64累加/F32激活。模型约22.13GB，稠密缓存实际有效载荷14.44GiB（元数据和工作缓冲另计）。不代表视觉、256K上下文或三卡并行已验证。

当前建议配置为已测8个工作线程、--no-mmap、--dense-cache-mib 16384；默认缓存仍为0，调用者显式选择容量。缓存仅在当前进程/模型生命周期内有效，新进程仍需要加载和建立缓存。容量预算只覆盖缓存有效载荷，不覆盖模型、索引和临时缓冲。

热态生成随输入长度从短提示0.510下降到1940词元0.427；输入处理延迟较高。当前结果支持单卡正确性和移植验证，不证明交互性能已经满足产品需求。独立CPU参考使用同数学FP64路径；4.925倍比较是VE缓存开关比较，不是相对原优化CPU量化路径的比较。

证据：build/results/20261002T162100Z-qwen-final-bench/summary.json；build/results/20261002T161007Z-qwen-dense-cache/summary.json；build/qwen-parallel-cache-build-check.log；build/qwen-final-dense-cache-bench.log；build/results/20261002T162003Z-temperature-np_0u1bs.csv及对应风扇CSV。

收尾审计完成：326份Git候选文本未发现所检查的私人路径/凭据模式；三组完整词表数据重新计算确认原门槛通过。git diff --check通过。没有进行Git提交或发布。
