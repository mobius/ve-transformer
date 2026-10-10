# SD-Turbo 紧凑框架兼容基线

本机CPU/VE、NCC5.4.1和NLC3.1.0已检查，前轮11项真实VE基础算子通过。当前大框架的model_builders、image、diffusion-engine先后触发ccom SIGSEGV；O0和64MiB局部编译栈试验不能解决后两项。小型张量编译用例含填充、标量运算、相加与切片编译成功，仅是编译验证，不是完整模型验证。

选择独立固定上游2025-01-18提交5eb15ef4d022bef4a391de4f5f6556e81fbb5024、ggml6fcbd60bc72ac3f7ad43f78c87e535f2e6206f58，源码声明SD-Turbo和Diffusers目录加载支持。保持官方模型与独立CPU参考不变。公开源：https://github.com/leejet/stable-diffusion.cpp/tree/5eb15ef4d022bef4a391de4f5f6556e81fbb5024 。

首轮CMake配置报cxx_std_11未列入本地NEC工具链特征集合，尚未编译模型；CPU峰值61°C。新增基线专用工具链声明兼容的C++11/14子集，并在project加载编译器之后再次补齐特征，以避免缓存的特征列表覆盖。保留实际C++17编译方言，不修改LLM工具链。

CMake配置后，旧压缩文件辅助库的严格C标准环境缺失S_IFMT/S_IFDIR等接口宏，仅对zip目标增加_GNU_SOURCE。随后ggml数值库、CPU通用后端及NLC BLAS后端均编译完成。model.cpp报告10个日志宏错误，定位为GNU空可变参数逗号扩展；覆盖util.h四个日志宏为标准全参数包写法，数学代码不改变。温度最高70°C，未达停止阈值。

新基线11项VE算子全部通过，最大绝对误差2.08354936e-7；两项矩阵分配到NLC。CPU75°C、VE45.88°C，97个风扇通道未观察到变化。公开证据docs/results/20261008T055924Z-sd-turbo-validation.json，明确framework_variant=compact，完整图像仍待验收。测试适配旧版上采样三参数接口，标准不变。

模型元数据文件O1编译持续超过9分钟，主动停止已核对的model.cpp进程，仅该文件改O0且关闭内联；数值内核与NLC不改，构建manifest如实记录。尚未完成CLI。
