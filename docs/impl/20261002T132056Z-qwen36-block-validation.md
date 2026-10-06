# 迭代025：块解码独立验证

新增scripts/validate_qwen_math.sh提供可复现数学验证入口，含编译、独立CPU/VE检查和温度风扇监控。新增tests/benchmark_qwen_block_scales.py完成对照采样及实际算术输出检查；日志写入build，待测量，不宣称优化收益。CPU参考与所有VE执行器已从当前源码重新编译。
