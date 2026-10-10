# 第三方依赖说明

本仓库不打包第三方源码、模型权重、编译库或本地Python环境。

- llama.cpp / ggml：<https://github.com/ggml-org/llama.cpp>，MIT许可；固定修订见构建脚本。VE接入使用其接口与量化块布局，生成的上游适配源码留在build目录；上游许可文件随下载保留。
- Qwen3.6-35B-A3B：官方模型及第三方量化权重的来源、固定修订和校验值见`scripts/download_qwen36.py`；下载和使用按各自模型许可执行。
- stable-diffusion.cpp / 独立 ggml：<https://github.com/leejet/stable-diffusion.cpp>，MIT许可；固定主框架及 ggml 子模块修订见 `scripts/prepare_sd_source.sh`；当前兼容验收基线及其独立 ggml 修订见 `scripts/prepare_sd_baseline_source.sh`。原始源码和许可保留在本地 build，局部适配由脚本生成，不修改或分发上游工作树。
- SD-Turbo：<https://huggingface.co/stabilityai/sd-turbo>；固定官方修订、逐文件来源与哈希见 `scripts/prepare_sd_turbo.py`。下载和使用按官方模型许可执行；权重、原始配置及生成图像保存在本地 build。
- NEC NCC/NLC：由用户已有本机工具链提供，不分发其安装文件或库。
- PyTorch、Transformers及其他Python依赖：仅声明固定版本，安装于项目环境，各包遵循自身许可。

本说明不替代第三方许可，也不替项目选择新的开源许可。
