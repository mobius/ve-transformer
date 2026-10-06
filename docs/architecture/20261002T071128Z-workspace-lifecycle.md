# 迭代 010：workspace-lifecycle

工作缓冲区由调用者创建和销毁，持有固定tokens/width/heads/hidden形状；causal允许每次改变。同一个工作缓冲区不能并发调用，各线程需独立实例。不保存权重和输出，不缓存结果。便捷API维持每次分配语义，复用API必须明确传入匹配实例。基准分开报告初始化时间和稳态延迟，不能把稳态收益直接等同于单次框架启动收益。

临时存储改为不初始化的malloc。norm、线性变换（BLAS beta=0）、fast注意力和FFN在读取前覆盖各自完整输出；基础注意力需要累加，所以每次调用仅清零该输出缓冲区。复用路径也执行该局部清零，防止上次调用污染本次输出。前向的allocation_ms字段在复用模式表示形状验证和必要初始化，不是创建耗时。

使用顺序：vt_workspace_create(cfg,&ws)，循环vt_forward_workspace(cfg,weights,input,output,ws,NULL)，最后vt_workspace_destroy(ws)。先检查各返回码；创建失败时输出指针设为NULL。调用者负责独立且不重叠的输入/输出和合法权重缓冲区。不得在工作区正被使用时销毁。
