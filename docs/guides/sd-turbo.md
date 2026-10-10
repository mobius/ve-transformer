# SD-Turbo 图像移植

当前官方 FP32 SD-Turbo 已通过单 VE、512×512 的六提示单步及一提示四步整图验证，12项基础/输入回归算子也通过独立对照。尚无稳态秒/图、物理HBM峰值或图像质量基准结论。

## 独立环境与来源

图像环境位于 `build/venvs/sd-image`，不改变 LLM 的 `.venv`。框架是独立固定修订的 stable-diffusion.cpp/ggml，模型为官方 SD-Turbo 固定提交；三组件共约1.290B参数、FP32权重有效载荷4.805GiB，工作区另计。参见 [来源与配置摘要](../results/20261008T021839Z-sd-turbo-model.json)。

从已存在的项目基础环境准备：

```bash
bash scripts/prepare_sd_baseline_source.sh
bash scripts/install_sd_reference.sh
.venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
  .venv/bin/python scripts/prepare_sd_turbo.py
bash scripts/build_sd_baseline.sh
bash scripts/check_sd_baseline_ops.sh
```

构建和算子检查内置温度监控。当前验收使用2025年1月的固定兼容基线，来源见 `scripts/prepare_sd_baseline_source.sh`，读取官方 FP32 权重。较新框架及旧基线的整体多模型入口在本机 NCC 编译时发生编译器崩溃；当前使用独立 CLIP、UNet、VAE 编译单元的固定验证入口，已通过实际整图验收。较新框架仍保留为未完成实验。

默认单任务50%编译占空比，可用 `SD_BUILD_DUTY_PERCENT` 调整，温度阈值不改变。固定旧图像库采用其要求的 C++11，元数据加载及模型流程文件采用 O0 并关闭内联，ggml 数值内核保持 O1，矩阵计算使用 NLC。主机先校验官方权重并导出张量索引；VE直接读取原始FP32字节，三个神经网络的计算仍在VE执行。完整构建后，来源、索引及二进制哈希写入 `build/sd-baseline-ve/manifest.json`。尚无性能收益结论。

## 正确性验证流程

CPU 参考导出当前采用10%限速（早期独立单例使用25%），参考耗时不充当正常 CPU 性能基线。示例命令生成带时间戳的本地参考目录：

```bash
.venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- \
  env VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1 SD_REFERENCE_DUTY_PERCENT=10 \
  .venv/bin/python scripts/cpu_duty.py --percent 10 -- taskset -c 0-3 \
  build/venvs/sd-image/bin/python tests/export_sd_reference.py --cases 6 --steps 1
```

完整 VE 构建完成后，将导出的相对目录传给 `tests/check_sd_turbo.py --reference`，同样必须通过温度守护并设置监督标记。测试逐级比较相同初始噪声、文本嵌入、每步输入/预测、最终潜在数组与 VAE 输出，未通过不记录为整图验收。

例如复用本地已导出的参考目录（其他机器需替换为自己导出的目录）：

```bash
.venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- \
  env VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tests/check_sd_turbo.py --framework compact \
  --reference build/results/20261008T081800Z-sd-reference --threads 8
```

数值验收使用模型加载、诊断数组输出和独立参考对照，因此该过程耗时不作为稳态秒/图指标。

当前结果：

| 验证 | 结果 |
| --- | --- |
| [六提示单步](../results/20261008T091001Z-sd-turbo-validation.json) | 30项阶段比较通过；最大归一化L2误差1.41×10⁻⁵，解码最大6.60×10⁻⁶ |
| [一提示四步](../results/20261008T091957Z-sd-turbo-validation.json) | 11项阶段比较及四个时间步通过；解码误差1.09×10⁻⁶ |
| 温度 / 风扇 | VE验证CPU最高77°C、VE最高53.50°C；可读风扇及策略通道未观察到变化，完整策略固定性未建立 |
| 诊断进程耗时 | 单步均值266.828秒，四步526.571秒；含权重加载与诊断输出，不是稳态秒/图 |

用同样流程再导出 `--cases 1 --steps 4` 的CPU参考，再将其目录传给VE检查器，可复现四步对照；不要与编译或VE任务并行。检查器要求源代码/二进制/官方权重哈希一致，并验证三网络确实分配矩阵节点到NLC。


## 首轮线程池优化

[实测结果](../results/20261008T102317Z-sd-turbo-pool.json)：同一二进制、单提示单步ABBA，全进程均值264.240→212.289秒，延迟下降19.66%；三网络图执行191.826→139.215秒，下降27.43%。七个诊断数组和PNG四次逐字节一致。权重加载均值两组均约59.68秒。额外一提示四步11项比较通过，耗时382.847秒；四步未做本轮ABBA。

组件内复用原生VE线程池，NLC执行时空闲线程阻塞等待；权重仍逐组件加载释放。推荐在已验证流程显式增加 `--threadpool persistent --profile`；`disposable`保留原生命周期并仍为脚本默认值。例如：

```bash
.venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- \
  env VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tests/check_sd_turbo.py --framework compact \
  --reference build/results/20261008T023813Z-sd-reference --threads 8 \
  --threadpool persistent --profile
```

这项优化的性能对照限于首提示、两次/模式，使用包含诊断输出的流程，不是常驻权重后的稳态指标。VAE与UNet通用算子仍是主要耗时，下一轮按[计划](../plan/20261008T102317Z-sd-turbo-pool-verified.md)进一步拆分计时，并先测内存再设计常驻权重。

本轮限定官方 SD-Turbo、512×512、Euler、引导系数1和固定噪声。已经查看实际图像；机器人示例缺少清晰气球、帆船示例没有天空，尚未建立质量基准。详见[最终记录](../research/20261008T091957Z-sd-turbo-final-validation.md)。已完成阶段/算子计时、线程池、SiLU与Softmax优化验证；im2col已完成优化验收，权重常驻仍待实施。

编译与重型CPU参考必须串行。一次分CPU插槽的并发参考触及停止线，监控已终止任务；不能假设换核心即可降低温度。

当前 `build/sd-baseline-ve/bin/sd` 为固定验证入口：必须由对照脚本提供完整噪声、调度数组与诊断目录；不提供完整上游SDK API或通用生产文生图接口。

## 第二轮 SiLU 优化

[实测结果](../results/20261008T114450Z-sd-turbo-silu.json)：在8线程常驻池基础上，同二进制单提示ABBA，全进程216.835→167.596秒（延迟下降22.71%），图执行145.630→96.387秒（下降33.81%），VAE图执行下降50.92%。权重加载约57.8秒不变。候选两次各五项CPU对照及额外四步11项均通过；四步进程326.387秒，未做四步ABBA。舍入结果与原实现有细微差异。

显式在上述常驻池命令增加 `--silu ve`；`--silu generic` 保留原实现并为默认值。独立候选启用NCC向量数学范围检查和直接除法，修正首版极端输入错误。单独运行边界测试：`bash scripts/check_sd_silu.sh`。`--op-profile` 可增加算子级诊断，额外开销未量化，本轮性能ABBA没有启用它。

本轮CPU/VE最高80°C/53.25°C，CPU曾触及警告线但未到停止线，可读风扇未观察到变化。范围仍限单提示、两次每组及包含诊断的加载流程。下一轮先优化UNet Softmax，再优化VAE im2col；详见[实施计划](../plan/20261008T114450Z-sd-turbo-silu-verified.md)。

## 第三轮 Softmax 优化

[实测结果](../results/20261009T031452Z-sd-turbo-softmax.json)：固定SiLU ve及8线程常驻池，仅切换Softmax，同二进制单提示ABBA，全进程169.065→125.915秒（延迟下降25.52%），图执行96.638→51.655秒（下降46.55%），UNet图执行50.312→6.995秒（下降86.10%）。权重加载58.993→60.805秒，无收益；缓存未控制，性能不是常驻稳态指标。额外四步11项通过，耗时148.966秒，四步没有ABBA。

在前述参考对照命令中使用 `--threadpool persistent --profile --silu ve --softmax ve`。`--softmax generic`为默认原路径；独立边界测试用 `bash scripts/check_sd_softmax.sh`。候选仅向量化FP32指数，保持原顺序double累加以及最大值、掩码和缩放逻辑；24组直接测试及12算子回归通过。四步解码归一化L2误差1.45e-6，舍入与原路径有微小差异。

本轮CPU/VE最高79°C/53.875°C。状态/策略通道未观察到变化，但sysfs转速读数均为零且BMC不可用，无法建立物理转速和完整策略结论。下一轮先优化VAE im2col，再测HBM与设计权重常驻，见[下一轮计划](../plan/20261009T031452Z-sd-turbo-softmax-verified.md)。单提示的收益不推广到其他模型或任意尺寸。

## 第四轮 im2col 优化

[实测结果](../results/20261009T053046Z-sd-turbo-im2col.json)：固定SiLU/Softmax ve与8线程常驻池，同二进制ABBA全进程131.625→92.942秒（下降29.39%），图执行59.277→20.512秒（下降65.40%），VAE图执行下降73.08%。七个诊断数组及PNG在四次运行中逐字节一致。额外四步11项、六提示30项、16组边界及12算子回归均通过。

验证命令显式使用 `--threadpool persistent --profile --silu ve --softmax ve --im2col ve`；默认仍为generic。数据整理候选支持连续二维FP32张量，非连续或一维布局回退；独立测试分别运行 `bash scripts/check_sd_im2col.sh` 和 `bash scripts/check_sd_im2col_graph.sh`，均内置温度保护。CPU最高82°C（有警告，未停止），VE最高51.125°C；风扇物理策略未建立结论。

权重加载仍约59秒，下一轮先测模型内存占用，再实现权重常驻，见[计划](../plan/20261009T053046Z-sd-turbo-im2col-verified.md)。本轮耗时含加载与诊断，不能解释为生产稳态秒/图。

## 内存预算测量

[连续采样结果](../results/20261009T053237Z-sd-turbo-memory.json)：单提示单步全节点采样最高5.3223 GiB，退出回到128 MiB基线；528个样本、最大间隔0.20106秒。不是模型独占内存或瞬时峰值，尚未验证权重常驻。下一轮将复用三组件权重，并按请求释放工作区。

## 第五轮：权重常驻与共享线程池

[本轮报告](../impl/20261009T071320Z-sd-turbo-resident-verified.md)：两请求全进程延迟下降32.83%，后续请求下降65.43%；六提示、四步及算子验收完成，节点采样最高9.8223 GiB，退出恢复128 MiB。显式候选不默认启用，限固定512、FP32、顺序请求。

使用已准备的六提示参考集，验证入口自动固定共享8线程池及已验收算子候选；必须通过温度保护运行：

```bash
VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1 .venv/bin/python scripts/temperature_guard.py -- .venv/bin/python tests/check_sd_resident.py --mode resident --cases 0,1,2,3,4,5 --reference build/results/20261008T081800Z-sd-reference
```

使用 `--mode reload` 对照重载路径。原生常驻模式要求共享线程池，当前入口不是并发服务。

## 第六轮：常驻 CLIP 分词器

[最终报告](../impl/20261009T080313Z-sd-turbo-tokenizer-verified.md)：固定权重常驻，只切换分词器复用，两请求总延迟下降8.08%，后续请求下降31.14%（32.539→22.407秒），首次无明显收益。107项CPU对照通过，六提示和四步已验收，候选默认关闭。只覆盖单VE、FP32、512、顺序1/4步；计时含诊断、缓存未控制。

```bash
VE_TRANSFORMER_TEMPERATURE_SUPERVISED=1 .venv/bin/python scripts/temperature_guard.py --interval .2 --bmc-fans --post-seconds 5 -- .venv/bin/python scripts/sample_ve_memory.py --timeout 900 -- .venv/bin/python tests/check_sd_resident.py --reference build/results/20261008T081800Z-sd-reference --mode resident --tokenizer resident --cases 0,1,2,3,4,5
```

`--tokenizer request`选择每请求构造路径。四步使用已冻结四步参考集`build/results/20261008T083810Z-sd-reference`和`--cases 0,0`。串行性能对照在同样温度守护下运行`scripts/benchmark_sd_resident.py --comparison tokenizer --reference build/results/20261008T081800Z-sd-reference --cases 0,0`；实际基准汇总异常与独立恢复证据完整保留在本轮报告。下一轮先测真实矩阵，再评估NLC并行，不宣称未测收益。

## 矩阵计算分项计时

[定位报告](../impl/20261009T081518Z-sd-turbo-nlc-profile-verified.md)：热请求VAE矩阵库调用约8.278秒，六组大矩阵占84.15%，下一步单独比较顺序和多线程NLC。此轮10项CPU对照通过，尚无多线程收益结论。

在第六轮温度守护命令的环境前缀增加`SD_NLC_PROFILE=1`，并使用`--cases 0,0`，可记录每次实际矩阵调用的m/n/k、行跨度、转置和耗时。默认关闭；诊断输出开销未独立测量，不能当作生产性能基准。

## 第七轮：通用算子与NLC共用固定OpenMP线程

[验证报告](../impl/20261009T094430Z-sd-turbo-nlc-abba.md)：同源码两构建ABBA，一步热请求22.384→18.194秒（延迟下降18.72%），原生双请求进程下降7.72%。共102项独立CPU阶段检查通过，含六提示和四步；四步没有ABBA。前一逐BLAS图切换线程方案因严重准备开销被拒绝，保留失败报告；当前统一固定8线程避免该停滞。默认仍原顺序方案。

显式构建候选：`SD_NLC_MODE=openmp SD_GGML_OPENMP=1 bash scripts/build_sd_baseline.sh`，构建脚本内置温度/风扇保护及低占空比。在第六轮温度守护验证命令添加`--nlc-threads unified`；顺序构建用`SD_NLC_MODE=sequential SD_GGML_OPENMP=0`，验证使用默认`single`。统一构建必须与unified检查模式匹配；不要沿用逐图staged模式。

固定8线程同时用于通用算子和NLC，没有独立的pthread工作线程池；共享池日志表示调度元数据生命周期。两构建ABBA可用`scripts/benchmark_sd_runtime.py`，要求两个完整冻结构建目录和冻结参考集，并必须置于温度守护中运行，详见报告。CPU最高80°C有警告但未停止，VE53.25°C；候选节点采样峰值9.8848GiB，结束128MiB，风扇物理策略未验证。下一轮继续定位VAE通用算子与矩阵的完整图取舍。

## 第八轮：连续空间平面广播加乘

[正式报告](../impl/20261009T104156Z-sd-turbo-binary-abba.md)：同二进制ABBA只切换广播加乘，热请求18.157→13.943秒，延迟下降23.21%；102项独立CPU阶段检查、504组内核和16组真实图检查，ABBA输出数组/PNG逐字节相同。四步候选热请求24.602秒，但没有四步ABBA。两个请求的原生进程均值下降9.69%，包含加载波动，不能作为内核加载收益或固定生产收益。

沿用第七轮统一OpenMP8构建与温度守护验证命令，添加`--binary-scalar ve`；默认`generic`保留原路径。候选仅接收连续FP32主输入/输出、空间前两维广播1、合法原地或独立缓冲，其他布局/别名回退，逐阶段实际分派计数严格核验。直接边界测试用`bash scripts/check_sd_binary.sh`，真实图分派与回退用`bash scripts/check_sd_binary_graph.sh`，均内置温度守护。

同二进制对照在温度守护下运行`scripts/benchmark_sd_resident.py --comparison binary --reference build/results/20261008T081800Z-sd-reference --cases 0,0`；固定权重/分词器常驻和统一8线程，保持算子/形状计时关闭。正式测试CPU最高76°C、VE53.12°C，风扇可读状态无变化但物理策略未确认，节点峰值9.8848GiB、结束128MiB。下一轮优先继续减少VAE矩阵阶段或严格erf GELU耗时。

### VAE4线程显式选择

单VE上的SD-Turbo FP32 512×512一步验证中，在统一OpenMP模式与既有广播加乘优化下，`--vae-threads 4`将热请求均值13.976→13.059秒（降低6.56%）。这同时将VAE通用算子和NLC矩阵团队设为4，下一请求CLIP恢复8；须配合`--nlc-threads unified --binary-scalar ve`使用。默认仍8。六提示/四步正确性已验证，但正式性能对照仅覆盖一个一步提示；不可外推到所有模型或生产服务。参见[本轮报告](../impl/20261009T111710Z-sd-turbo-vae-abba.md)。

### VAE仅矩阵4线程

统一OpenMP模式下，显式`--vae-threads 8 --vae-blas-threads 4 --nlc-threads unified --binary-scalar ve`使通用算子保持8，VAE矩阵图进入4、退出恢复8。同二进制对照中，一步热请求13.081→12.292秒（比全VAE4降低6.03%）；六提示/四步正确性通过。仍为显式候选配置，不默认启用，不外推到其他模型。[本轮报告](../impl/20261009T113726Z-sd-turbo-vae-selective-abba.md)。

### GELU向量核显式选择

在常驻权重/分词器、通用8/VAE矩阵4及既有广播加乘下，额外--gelu ve使用同FP32 erf公式的受检查向量核。一步同二进制对照12.442→11.046秒（延迟降低11.22%），模型102项CPU阶段检查与实际对象102个用例通过。默认generic；输出有小幅舍入差异，所测图像67个像素变化、最大1灰度级。[本轮报告](../impl/20261009T121339Z-sd-turbo-gelu-abba.md)、[累计结果](optimization-history.md)。原始结果审计使用已有build/venvs/sd-image/bin/python执行scripts/record_sd_gelu_result.py。
