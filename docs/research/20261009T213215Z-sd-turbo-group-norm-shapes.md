# 第88轮：真实GroupNorm形状诊断

182条实际元数据、每请求UNet61/VAE30节点，24阶段/形状/epsilon组合、20阶段/形状组合，两epsilon约1e-5/1e-6必须区分。热GroupNorm UNet0.042688/VAE0.225581秒。10CPU/2PNG及前版字节比较通过。

性能、温度、正确性、启动错误及未完成项见[报告](../impl/20261009T213215Z-sd-turbo-group-norm-shapes.md)。
