# 迭代 011：trained-model-selection

CPU为双路Xeon Gold6252，96逻辑线程，AVX512F/BW/VL及avx512_vnni实际存在，当前PyTorch报告AVX512并使用MKL；仅确认能力，不据此宣称特定算子或INT8性能。CPU63°C，VE38.75–41°C，三运行槽位与软件存在，运行仍需冒烟。选择作者发布的TinyStories-1M，固定修订77f1b168e219585646439073245fe87e56b3023e。GPTNeo，8层、宽64、16头、FFN256、词表50257，gelu_new，线性偏置，交替全局/窗口256的局部注意力，不缩放QK点积。当前实现缺少这些选项，必须扩展语义，不能将ReLU近似替代。模型卡没有明确license字段，不推断许可、不将权重纳入仓库；本地技术验证记录来源。候选Pythia14M结构另含旋转位置和并行残差，作为后续。

来源：https://huggingface.co/roneneldan/TinyStories-1M 、https://huggingface.co/roneneldan/TinyStories-1M/blob/main/config.json 、https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py 、https://www.intel.com/content/www/us/en/developer/articles/news/second-generation-intel-xeon-processor-scalable-family-technical-overview.html
