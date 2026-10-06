# 迭代024：块缩放预解码

ve_dequant_rows将half转换从每个元素降低到每个32/256元素块，Q4/5/6整数scale预解码；保持d*(整数scale*q)-dm*min或d*(scale*q)原公式，Q8保持d*q。参数声明restrict，调用者缓冲相互独立。实现已写入，编译/硬件验证待补。
