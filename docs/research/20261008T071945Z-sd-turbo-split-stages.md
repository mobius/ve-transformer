# SD-Turbo 三组件拆分验证入口

旧兼容框架的整体stable-diffusion.cpp在O0/fno-inline、C++11下仍触发NCC ccom SIGSEGV，CPU峰值74°C、VE最高43.75°C，属于实际编译器失败而非温度停止。元数据已编译通过，数值库及算子也已通过。进一步缩小编译单元。
