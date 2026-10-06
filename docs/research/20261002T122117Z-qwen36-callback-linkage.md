# 迭代021：NEC回调语言链接兼容

NEC首次完整库链接实际失败，缺少llama_set_abort_callback。源码有实现，nm证明对象导出C++修饰名，同时引用公共C名称；公共头采用ggml_abort_callback typedef，定义采用原始bool(*)(void*)类型，NCC严格区分函数语言链接。CPU GCC构建无此错误。此证据明确定位链接兼容问题，未宣称VE模型执行。
