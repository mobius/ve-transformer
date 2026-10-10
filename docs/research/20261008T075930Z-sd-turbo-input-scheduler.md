# SD-Turbo 输入张量调度兼容

实际整图运行在CLIP图分配阶段触发GGML_ASSERT(buffer_id>=0)。日志已证实372份文本权重加载、词表构造及77位分词完成。旧单后端执行器传入有data但没有backend buffer及INPUT标记的ggml上下文张量；调度器无法识别其后端。首次无阶段日志的运行返回SIGSEGV，后续定位运行明确暴露分配断言，均未通过验收。
