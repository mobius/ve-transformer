# 迭代038：最终基准运行预算

tests/check_qwen_model.py的run新增timeout_seconds参数；仅基线/线程/长输入正式bench入口传7200，正确性默认1800不变。当前节点采集代码不依赖此参数，不重启正在运行的硬件进程。
