# 迭代032：编译器精度控制实验

新增scripts/build_qwen_precision.py，在build/qwen-ve-precision复制原GGML库并精确替换六个数学对象，框架源码不变、原库和执行器不变；生成对象散列与flags清单。首轮ops.cpp触发NCC Internal Error: No realloc for STAT_temp_set_pool，编译终止无执行成绩。对该单一实验对象改为-O1 -fno-inline，保持其他五对象O3及全部精度控制flags，正在重新编译。
