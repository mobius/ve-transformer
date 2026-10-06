# 迭代032：编译器精度控制实验

ncc --help本机5.4.1是编译默认值的直接证据：associative-math、fast-math、reciprocal-math默认开启，向量sqrt/divide默认使用倒数近似序列。实验精度flags为-fno-fast-math -fno-associative-math -fno-reciprocal-math -mvector-sqrt-instruction -mvector-floating-divide-instruction。precise-math只影响特定pow算法，未把它误解为整体IEEE精度开关。保持框架版本、张量布局、数学门限和模型不变，记录ops局部优化级别差异。
