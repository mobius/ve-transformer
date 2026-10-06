# 迭代022：VE完整权重读取

实际VE完整模型加载尝试已进入40层张量加载，随后上游mmap请求返回Invalid argument；未完成权重加载及推理。可用执行器已有--no-mmap普通读取模式，拟验证。ELF machine字段为0xfb，执行经ve_exec -N1，不是x86 CPU参考。
