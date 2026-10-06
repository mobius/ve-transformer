# 最终验收清单

环境：双路Xeon、显示GPU不参与推理、三张NEC VE中使用运行槽1/物理ve0；真实模型已实际执行，不以设备枚举替代运行验证。没有全局安装或风扇配置修改。

测试限定为Qwen3.6-35B-A3B UD-Q4_K_M文本权重、4096上下文配置，FP64累加/F32激活。模型约22.13GB，稠密缓存实际有效载荷14.44GiB（元数据和工作缓冲另计）。不代表视觉、256K上下文或三卡并行已验证。

本轮范围已完成：230项CPU/VE数学检查，包括量化格式边界、有界缓存满/半预算竞争、清理和精度原语；三提示各12步完整词表误差门槛及最高分词元一致；三次16步上下文重置；同数学未缓存32步三重复基线；1/2/4/8线程各32步三重复且与基线输出一致；16/64/128段输入各16步两重复且输出一致；全程温度和风扇监测。最终温度监督任务退出0，performance_scan_completed=true。

最终交付为当前源代码、可复现构建和验证脚本、README及四类迭代文档。最终敏感信息审计与变更空白检查在收尾阶段执行，不将当前结论扩大到其他模型或全部输入正确性。

证据：build/results/20261002T162100Z-qwen-final-bench/summary.json；build/results/20261002T161007Z-qwen-dense-cache/summary.json；build/qwen-parallel-cache-build-check.log；build/qwen-final-dense-cache-bench.log；build/results/20261002T162003Z-temperature-np_0u1bs.csv及对应风扇CSV。

收尾审计完成：326份Git候选文本未发现所检查的私人路径/凭据模式；三组完整词表数据重新计算确认原门槛通过。git diff --check通过。没有进行Git提交或发布。
