# 迭代035：高精度独立验证

验证入口新增--backend accurate，CPU/VE执行器分别独立存放，强制校验JSON的math_mode字段，防止误用旧数学路径。原quant/nlc接口与失败结果保留。已启动受温度/BMC风扇保护的correctness --backend accurate，日志build/qwen-final-correctness-accurate.log。未改变数值门限、检查步数、提示集、模型或量化权重。
