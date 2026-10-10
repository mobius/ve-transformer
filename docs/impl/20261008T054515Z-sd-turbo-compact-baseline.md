# 紧凑框架实施

由OpenAI Codex GPT-6主agent实施，无委派。独立build/vendor/sd-turbo-baseline来源与build/sd-baseline-overlay生成覆盖，不改上游。修正旧框架强制半精度Conv2d为FP32；GELU关闭半精度查表并使用erf；GroupNorm传入保存的eps，UNet残差/输出归一化1e-5、VAE和空间注意力1e-6。

以NLC BLAS和原生VE通用后端调度图，参数缓冲标记权重，实际图分配数写私有日志。保存相同噪声、调度数组及逐阶段FP32输出。验证器仅对已校验官方SD-Turbo启用显式EPS覆写；未启用时保留旧框架预测类型探测。编译开始，完整VE结果尚待执行。

首轮CMake配置报cxx_std_11未列入本地NEC工具链特征集合，尚未编译模型；CPU峰值61°C。新增基线专用工具链声明兼容的C++11/14子集，并在project加载编译器之后再次补齐特征，以避免缓存的特征列表覆盖。保留实际C++17编译方言，不修改LLM工具链。

CMake配置后，旧压缩文件辅助库的严格C标准环境缺失S_IFMT/S_IFDIR等接口宏，仅对zip目标增加_GNU_SOURCE。随后ggml数值库、CPU通用后端及NLC BLAS后端均编译完成。model.cpp报告10个日志宏错误，定位为GNU空可变参数逗号扩展；覆盖util.h四个日志宏为标准全参数包写法，数学代码不改变。温度最高70°C，未达停止阈值。

新基线11项VE算子全部通过，最大绝对误差2.08354936e-7；两项矩阵分配到NLC。CPU75°C、VE45.88°C，97个风扇通道未观察到变化。公开证据docs/results/20261008T055924Z-sd-turbo-validation.json，明确framework_variant=compact，完整图像仍待验收。测试适配旧版上采样三参数接口，标准不变。

模型元数据文件O1编译持续超过9分钟，主动停止已核对的model.cpp进程，仅该文件改O0且关闭内联；数值内核与NLC不改，构建manifest如实记录。尚未完成CLI。
