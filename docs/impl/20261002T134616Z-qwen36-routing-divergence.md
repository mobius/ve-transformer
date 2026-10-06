# 迭代031：专家路由分叉证据

新增追踪分析--analyze-folder复用已经完成的真实硬件数据，跳过双方均为空的节点并校验数据完整性；解析相对文件夹后转换为绝对对象，再输出项目相对位置。分析证据build/results/20261002T133933Z-qwen-nodes/node-differences.json，96通道分析阶段没有BMC转速数据，不算实际风扇观测；真实采集阶段104通道中FRNT_FAN2 3900→4000RPM。启动scripts/validate_qwen.sh correctness --backend nlc，日志build/qwen-final-correctness-nlc.log，仍在运行。
