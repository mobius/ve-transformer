# 下一阶段图像模型：SD-Turbo 单 VE 验证

建议首个完整模型使用 Stability AI 的 SD-Turbo（稠密、由 SD2.1 蒸馏），先做单步 512×512 文生图，再验证四步。官方支持少步生成且单步路径关闭提示引导，便于降低首轮完整验证成本；这不是现代视频或 DiT 的性能替代。小尺寸 256×256 仅用于算子/流程排错，不用于宣称正常图像质量。[开发者模型说明](https://huggingface.co/stabilityai/sd-turbo)

实现候选是独立固定版本的 stable-diffusion.cpp，用其已有 SD 模型加载、计算图和采样器；CPU Diffusers 为参考。先检查所选版本对 SD-Turbo 的配置/调度支持及算子列表，不能假设其 ggml 与现有 Qwen 分支二进制兼容；复用的是移植经验和已验证的 NLC/温度工具，不直接混链接两套 ABI。[原作者实现](https://github.com/leejet/stable-diffusion.cpp)、[Diffusers 官方管线](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)

准备时固定源码、依赖、官方权重完整提交与每文件 SHA256；按真实张量规模统计模型和工作区容量，不用页面简略参数量替代完整管线占用。权重与图像保存在忽略目录。模型条款按所选版本保留原文来源；本轮只是研究计划，没有下载安装或构建图像框架。
