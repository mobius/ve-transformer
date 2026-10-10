# SD-Turbo 词表编译拆分

生成独立sd_clip_vocab.c及带精确数组长度的C链接声明；model.cpp不再包含29MB vocab.hpp，CLIP load_merges保持原始内容。T5 tokenizer API明确拒绝。上一轮受监控构建CPU峰值80°C、VE最高44.25°C，97个风扇及策略观测通道无观测变化；未修改风扇策略。
