# 完整数值验收与优化次序

本轮接续既有完整验证，工具句柄92723仍存活；实际子进程为qwen-infer-cpu-accurate，第一个提示CPU参考正在计算。温度记录持续更新，尚无完成分数。已有局部证据不足以宣布最终完成。

第一个提示case0完整12步已通过：max_abs_error=0.004188567399978638、RMSE=0.00047690944386541146、argmax一致12/12。CPU参考约0.069110 token/s，VE高精度路径0.103386 token/s，VE预填充19.742624秒。实际RSS peak_raw=24061952，仍保留原始单位，不等同卡总内存。证据build/results/20261002T151124Z-qwen36及build/qwen-final-correctness-delta-softmax.log。case1-cpu正在运行，最终套件尚未完成；未开始性能扫描。
