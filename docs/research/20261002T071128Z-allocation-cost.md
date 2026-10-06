# 迭代 010：allocation-cost

CPU96线程、ASPEED显示、三VE1运行槽位1/2/3及NCC/NLC可用；温度CPU65°C，VE38.75–41.12°C。枚举后仍需执行冒烟测试。上一轮256×512分配0.807ms/总2.886ms，约28%。当前calloc清零全部临时内存，但fast路径所有数据先写后读。
