# 迭代036：高精度节点路径调查

所有旧硬件检查均已终止，未自动重跑失败完整套件。trace_qwen_nodes.py新增--math accurate，启动新节点诊断，保存新CPU/VE轨迹，不改写既有CPU参考或失败分数。数据来源build/results/20261002T141651Z-qwen36；新的受温度/BMC保护诊断日志为build/qwen-accurate-node-trace.log，正在运行。
