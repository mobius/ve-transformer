# 迭代033：精度控制执行器

tests/check_qwen_model.py允许指定诊断执行器binary_override，其他运行配置与结果/内存采样保留。新增tests/check_qwen_precision.py复用已存CPU轨迹、验证模型SHA，执行独立严格版本并应用原全词表门限；当前--backend nlc任务运行中。scripts/build_qwen_precision.py记录每个对象来源、额外flags和散列。
