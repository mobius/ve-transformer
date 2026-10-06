# 迭代 011：gpt-neo-inference

主机分词、嵌入、位置嵌入、最终LayerNorm和输出头；完整GPTNeo Transformer块在VE执行。现有float32参数通道扩展有界v2，保留v1。GPTNeo注意力不做1/sqrt(head_dim)缩放，局部层只看最近window_size个位置；GELU采用作者gelu_new的tanh公式。传统前置归一化顺序残差不变。支持单序列、无padding/外部掩码、无训练，明确拒绝不支持配置。

格式v2：8字节VTF32V2魔数，8个little-endian uint32（tokens,width,heads,hidden,causal,activation,unscaled,window），随后float32 epsilon，固定44字节头。张量依次Q/K/V/O/up/down矩阵（输入维在行）、gain1/bias1/gain2/bias2、q/k/v/out/up/down线性偏置、输入。总payload float数为4*d*d+2*d*f+9*d+f+t*d。v1读写顺序及旧API语义保留。

C新增独立vt_layer_options与vt_forward_extended；NULL options保持旧语义，NULL工作区每次分配，也可复用已验证固定形状工作区。局部窗口包含当前位置，只允许max(0,i+1-window)..i；fast矩阵路径明确清零所有禁止位置，基础路径只累加有效位置。GELU tanh与作者配置一致，不替换为ReLU或erf公式。

模型加载固定作者修订并核对下载文件SHA256清单，local_files_only=True、trust_remote_code=False、weights_only=True，不执行远端自定义代码。校验清单用于下载完整性，不视为独立发布者签名。参考框架版本锁定，导出只接受已验证的eager GPTNeoBlock。
