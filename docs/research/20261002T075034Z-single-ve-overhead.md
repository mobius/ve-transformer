# 迭代 012：single-ve-overhead

复查CPU96线程/AVX512，ASPEED显示、三张VE1槽位和NCC/NLC可用；CPU64°C，VE38.5–40.75°C。上一轮VE1完整16词元耗时27.898s，CPU0.099s；每词元8次执行器启动。先单卡实测导出、运行器、实际块计算，不能将未测占比当作结论。
