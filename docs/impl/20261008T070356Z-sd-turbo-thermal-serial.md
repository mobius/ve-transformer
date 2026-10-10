# SD-Turbo 温度停止与串行重试

已有model.cpp实际完成编译；主模型文件曾持续生成汇编，但因温度守护停止未完成对象。没有称为编译器崩溃。将stable-diffusion.cpp与metadata单元采用O0/fno-inline；所有矩阵、卷积及激活ggml内核仍O1/NLC。CPU恢复至64°C后重新启动仅编译任务。
