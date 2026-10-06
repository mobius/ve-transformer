# 迭代 006：现代框架权重接入

实施环境：CPU 和三卡执行已经验证，项目 uv 环境包含 NumPy 2.4.6、CPU 版 PyTorch 2.14.1，版本固定 requirements.lock。没有安装 CUDA 或 VE Python 框架，也未修改全局软件。

官方接口参考：https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html 。明确采用前置归一化、ReLU、无投影/FFN 偏置、epsilon=1e-5 的受限层结构，不默认支持任意 PyTorch 模型。

框架参数需要从 PyTorch 的输出×输入矩阵转为 C 的输入×输出矩阵；Q/K/V 从 packed in_proj_weight 按输出轴拆分并转置。二维输入按单序列解释，batch_first 对无批量维输入不产生额外维度。
