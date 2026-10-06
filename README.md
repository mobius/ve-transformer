# VE Transformer 推理移植

目录导航与复现边界见 [项目导航](docs/README.md)，可公开的单卡结果见 [最终测量摘要](docs/results/qwen36-single-ve.json)。历史迭代文档按原时间戳保留。

已在本机三张 NEC VE（运行槽位 1、2、3）上验证 Transformer 前向推理，并完成线性层及注意力的 NLC 优化。计算采用 C11、float32，支持多头自注意力、因果或双向模式、两次前置 LayerNorm、残差连接和 ReLU 前馈网络。PyTorch 参数可以导出到原生执行器；两层小模型也已在三卡验证。已接入作者训练的 TinyStories-1M，并在三卡验证英文逐步生成；尚未进行广泛质量评估。

```bash
git status                         # .git 指向现有 .git-local，无需专用入口
bash scripts/check_environment.sh # 设备与软件能力检查
python3 scripts/temperature_guard.py -- make test # 标准库独立小规模参考
bash scripts/validate_ve.sh 1 2 3   # 基础 CPU/三卡验证
bash scripts/benchmark_sweep.sh 1 2 3 # 基础三档规模对照
bash scripts/setup_env.sh          # uv 项目内环境，仅用于扩展参考与框架接入
bash scripts/validate_nlc.sh 1 2 3  # 五种形状逐元素验证及基础/NLC 性能对照
make build/infer-cpu build/infer-ve build/infer-ve-nlc
python3 scripts/temperature_guard.py -- .venv/bin/python tests/check_pytorch.py
python3 scripts/temperature_guard.py -- .venv/bin/python examples/tiny_model.py
```

依赖安装使用项目 `.venv`，版本固定在 `requirements.lock`，缓存存放于 `build/uv-cache`。CPU 基础参考无需安装依赖；扩展验证使用主机侧 NumPy 2.4.6 和 CPU 版 PyTorch 2.14.1。VE 原生程序使用已经安装的 NCC 5.4.1 和 NLC 3.1.0，脚本只为子进程设置运行库路径。

重启并开放主机执行权限后，agent 已可直接访问 `/dev/veslot1–3`。原来的 Offline 来自会话设备隔离；sysfs 硬件编号不能直接代替运行槽位编号。所有卡上结果均来自真实执行。脚本逐卡运行，原始结果保存到忽略提交的 `build/results/`。

验证和规模对照脚本已自动接入温度监督：以 0.5 秒间隔采样 CPU coretemp 与三卡温度；CPU 80°C 警告、85°C 中止，VE 70°C 警告、75°C 中止。监控缺失、读数无效或读取失败也中止；触发保护时只终止本次任务的进程组。CSV 温度记录保存在 `build/results/`。这些是保守的项目测试阈值，不是厂商温度上限。直接运行原生程序或框架测试时须用 `temperature_guard.py -- 命令` 包装；CPU-only 测量加 `--cpu-only`。该机制不能检测采样间隔内的短暂温度尖峰，也不能保证驱动立即完成卡上任务清理。

首次 NLC 优化结果（平均单层延迟 ms；三卡范围）：

| 位置数×特征维 | 基础 VE | NLC VE | 相对基础 VE 加速 |
| --- | ---: | ---: | ---: |
| 32×64 | 1.566–1.600 | 0.434–0.441 | 3.60–3.63 倍 |
| 128×256 | 48.417–49.424 | 14.212–14.472 | 3.41–3.42 倍 |
| 256×512 | 202.278–206.531 | 62.712–63.863 | 3.23 倍 |

该对照是六次线性变换的矩阵乘法替换效果，注意力计算保持基础实现。NLC 配置最多 8 个线程，三档预热 5 次、分别重复 50、10、5 次。结果不是与优化 CPU 库的性能比较，也不包含框架导出、进程启动和文件传输。完整数值及限制见 `docs/research/20261002T062015Z-nlc-results.md`。

原生 API 见 `src/transformer.h`：调用者传入权重及独立输入/输出缓冲区，保证长度、形状和不重叠。单序列、不支持训练；旧接口无投影偏置，扩展接口支持偏置及部分 GPT-Neo 语义。便捷接口每次调用分配临时内存并计入层内时间；另提供工作缓冲区复用接口，详见下文。基准参数入口为 `./build/benchmark-cpu tokens width heads hidden repeats`，上限分别为 1024、2048、2048、8192、1000，头数必须整除特征维数。P50/P95 是最近秩分位数，少量重复只能粗略观察。

PyTorch 接入见 `python/ve_transformer.py`：`infer_layer(layer, x, node=1, backend='fast', causal=True)` 接受处于 eval 模式的 `torch.nn.TransformerEncoderLayer`，要求前置归一化、ReLU、无投影/FFN 偏置、LayerNorm epsilon=1e-5，输入为 CPU float32 的 `[位置数, 特征维]`。不支持额外掩码、批量维或反向传播；不兼容的配置明确拒绝。内部导出有界 `.vtf` 文件、执行本地原生程序并读取结果，临时文件自动删除。它不是 PyTorch 的原生 VE 设备后端，每次调用有启动、文件读写和权重传输开销。

两层示例在主机执行嵌入和输出头，在指定 VE 执行 Transformer 层。三卡输出与 PyTorch float64 参考的最大误差均为 3.26e-07，最大分数对应的输出编号一致。示例使用随机未训练权重，不能用于有意义的文本生成。

所有迭代记录在 `docs/{research,plan,impl,architecture}`；术语在 `docs/glossary.md`。已阅读参考仓库 https://github.com/mobius/uni-framework 的 README 与 VE 执行模块，参考 NCC 编译、运行槽位选择和 Python 调度方式，未复制其代码或修改系统/参考项目。

注意力优化版使用 `make fast` 编译，`bash scripts/validate_fast.sh 1 2 3` 执行三卡正确性和性能验证，`bash scripts/validate_framework.sh` 执行 PyTorch 对照及两层示例。桥接默认后端仍为 nlc，选择 fast 需显式传参。

| 位置数×特征维 | 原 NLC 延迟 ms | fast 延迟 ms | 相对原 NLC 加速 |
| --- | ---: | ---: | ---: |
| 32×64 | 0.430–0.437 | 0.128–0.130 | 3.33–3.36 倍 |
| 128×256 | 14.456–14.716 | 0.919–0.930 | 15.72–15.83 倍 |
| 256×512 | 63.657–64.766 | 2.877–2.905 | 22.12–22.29 倍 |

以上为三卡逐卡单层实测，预热 5 次、分别重复 100/50/20 次，8 线程，不包含框架传输。fast 用矩阵乘法执行注意力，分头复用一个位置数平方的分数缓冲区，1024 个位置时约 4 MiB。这组结果来自逐次分配的原生层执行；后续工作缓冲区和已训练模型结果见下文。

风扇观察只读记录 sysfs 控制值及 NEC 服务状态；可加 `--bmc-fans` 使用已授权的 `sudo -n ipmitool` 读取实际转速，每约 5 秒一次，超时 3 秒，读取耗时会延长采样周期。凭据不写入脚本、环境变量或文件；读取不可用明确记录，温度读取失效仍中止。使用示例：`python3 scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- bash scripts/validate_framework.sh --temperature-supervised`。只观察转速不能证明管理控制器内部风扇曲线未变化。

内存优化（迭代010）已在三卡实测：相对上一版fast，32×64延迟降低约10–11%，128×256约22%，256×512约28%。主要收益来自去掉不必要的整块清零；复用内存提供较小额外收益。完整三路径对照见 `docs/research/20261002T071128Z-workspace-results.md`。

`vt_workspace_create` / `vt_forward_workspace` / `vt_workspace_destroy` 提供相同形状连续推理的临时内存复用；每个并发调用者需独立实例，因果模式可切换。原有API维持每次分配。基准末尾增加 `--workspace`，创建时间单列，首次页面触达包含在预热中。`bash scripts/validate_workspace.sh 1 2 3` 自动温度监督、只读风扇观察，并验证重复调用。若本地保留旧基准二进制，会额外执行旧版本对照，否则只运行当前两路径。旧单层Python桥接仍每次启动执行器；已训练模型另提供下文的常驻接口。

已训练语言模型入口：

```bash
bash scripts/setup_llm.sh
.venv/bin/python scripts/download_tinystories.py
bash scripts/validate_llm.sh 1 2 3
python3 scripts/temperature_guard.py --bmc-fans -- .venv/bin/python examples/trained_lm.py --backend fast --node 1 --new-tokens 16
```

新增依赖锁为 `requirements-llm.lock`，以基础锁约束版本，安装在现有 `.venv`。基础 `setup_env.sh` 执行 sync 后会移除额外LLM依赖；需要时再运行 `setup_llm.sh`。模型、分词文件及下载缓存全部存于忽略的 `build/`，下载固定作者修订并生成文件SHA256清单；加载只读本地权重，不执行远端模型代码。作者卡没有明确许可字段，仓库不附带权重。

该模型采用 GPT-Neo：8层、宽64、16头，GELU tanh、线性偏置、交替全局及窗口256局部因果注意力，不缩放QK点积。完整Transformer块在VE，其余嵌入和输出头在CPU。v2有界格式保留v1兼容；不支持批量、额外掩码、训练或生成缓存。示例限制总长256，原生导出最多1024；验证包含259位置跨窗口情况。

三卡16词元生成与CPU词元编号完全一致，完整输出分数对照通过。当前生成约0.55–0.60词元/s，1线程PyTorch默认接口约162词元/s（一次短序列测量、都无缓存，CPU当时为全部位置计算词表头）；原VE桥接每词元启动8次进程。该表述对应首次移植结果；单VE常驻优化见下文。实测及温度见 `docs/research/20261002T072334Z-trained-lm-results.md`。

单VE常驻执行已在槽位1验证，默认示例改为常驻：

```bash
make resident
python3 scripts/temperature_guard.py --bmc-fans -- .venv/bin/python examples/trained_lm.py --backend resident --node 1 --threads 1 --new-tokens 16
bash scripts/validate_resident.sh 1
```

`python/resident_ve.py` 的 `GPTNeoResident` 通过 `with` 创建/关闭一次工作进程，保持全部8个Transformer块权重和最大容量工作缓冲区。主机嵌入和输出头保持CPU；每步只往返一次特征，卡上连续执行全部块，无生成缓存。支持变长请求及最后位置输出；单会话不支持并发调用。退出/超时会清理会话，进程继承温度监督进程组。1线程是该小模型的默认，可选1/2/4/8。

同9词元提示生成16词元：旧逐层桥接约0.576词元/s，常驻约309–318词元/s（各线程3次短序列平均），约537–552倍；首次导出和启动约0.1–0.2秒，单列且未计入稳态。用户入口单次约290词元/s，实际短样本存在波动。补测CPU同样只算最后位置词表头，1线程CPU约232词元/s，单VE常驻约316词元/s（5轮交替测量，约1.36倍），词元完全一致。不能推广到大模型、长序列或CPU所有核心最优吞吐。

详细分项、线程和限制见 `docs/research/20261002T075034Z-resident-results.md`。目前常驻实现的硬件验证覆盖单卡槽位1；其他卡的旧推理验证仍有效。原 `--backend fast` 保留逐层桥接对照，`--backend torch` 现在只算最后位置输出头，便于相同工作量比较。

Qwen3.6-35B-A3B 文本推理已在单VE完成完整模型加载和四词元烟测（391及结束标记）；完整数值对齐、稳定性能和优化仍在验证。模型工厂的VE构建仅支持`qwen35moe`；CPU构建保留上游完整工厂。以下命令使用固定llama.cpp版本、项目内构建目录及固定第三方量化权重，约需22.1GB下载空间：

```bash
.venv/bin/python scripts/download_qwen36.py
bash scripts/build_qwen.sh cpu
bash scripts/build_qwen.sh ve
bash scripts/validate_qwen_math.sh
bash scripts/validate_qwen.sh smoke --backend quant
bash scripts/validate_qwen.sh full --backend quant
bash scripts/validate_qwen.sh correctness --backend nlc
```

默认`quant`路径使用原内部量化激活并验证优化点积；备选`nlc`路径保留float32激活并分块反量化权重，采用相同数学CPU参考。两条路径不能混用参考来宣称数值相同。测试保留完整词表对齐、三次同进程缓存重置、线程扫描及长提示词；生成吞吐与加载、预填充时间分别记录。所有构建/执行命令由温度保护包裹；实际BMC风扇读取需要当前终端已有sudo授权，读取不可用会明确留档。权重、下载缓存、参考轨迹和原始日志都保存在被Git忽略的`build/`。

实际单VE槽位1、8线程、27输入词元、4生成词元的三次短请求，排除首次后直接量化路径约1.464词元/秒，块缩放NLC路径约0.207词元/秒；这是短请求测量，完整词表与长输入验收仍在进行。结果见`docs/research/20261002T132803Z-qwen36-route-measured.md`。独立数学检查覆盖CPU/VE各76项，包含最大128行×16384列的解码块。

后续12步全词表检查中，优化与原始VE路径均未通过CPU对齐门限，正在定位中间节点差异；1.464词元/秒尚不是数值验收通过的成绩。失败详情见`docs/research/20261002T133558Z-qwen36-numerical-investigation.md`。

推理执行器可加`--profile`输出预填充/生成的节点耗时聚合到stderr，用于定位瓶颈。该选项会改变图调度，只用于诊断，正式吞吐必须关闭；已在真实CPU模型上验证开关前后四步全词表logit完全一致，VE尚待验证。

VE整模型执行使用`--no-mmap`普通文件读取；当前VE运行环境对上游文件映射请求返回Invalid argument。CPU参考仍可使用文件映射。加载时间独立记录，不能计入稳定生成吞吐。

执行器默认复用工作线程；`--fresh-threads`用于对照原线程创建行为。`--test-abort`在正式请求前检查非空中止回调，清除回调后按正常路径重新推理；该检查不计入请求时间。模型验证另保存5秒间隔的`ve-ps RSS`原始采样，采样峰值不代表严格的瞬时峰值或整张卡的内存占用。

`--node-trace PREFIX`记录计算节点的连续F32/I32中间值及元数据；纯视图、纯权重和大于16MiB的节点不保存。仅用于数值定位，正式速度测量必须关闭。

已通过本轮验收的`accurate`路径使用FP64累加、对应高精度归一化/激活函数和NLC DGEMM/DGEMV，激活存储仍为float32，量化权重文件不变。它使用独立CPU高精度参考，不能冒充原float32/INT8路径已经通过；原路径的失败结果保留。该路径的CPU/VE独立数学检查已通过206项。统一状态更新和专家Softmax后，第一个真实提示的12步完整词表检查通过（最大误差0.004189、RMSE 0.000477、最高分词元一致12/12），对应VE约0.103 token/s；第二提示也已通过12步（最大误差0.001738、RMSE 0.000221、最高分词元一致12/12）；第三提示也已通过12步（最大误差0.001386、RMSE 0.000210、最高分词元一致12/12），三个提示共36步全部通过；三次16步缓存重置检查也已通过，完整正确性套件退出0；本轮最终性能优化与扫描已完成，正式结果见下文。已验收CPU/VE执行器保留在`build/qwen-accurate-baseline`。证据为`build/qwen-final-correctness-delta-softmax.log`。执行方式：

```bash
# 精度控制对象与原构建分开存放；仅ops.cpp实验对象使用-O1以避免NCC内部错误。
python3 scripts/temperature_guard.py --bmc-fans -- .venv/bin/python scripts/build_qwen_precision.py
bash scripts/build_qwen_accurate.sh
bash scripts/validate_qwen.sh correctness --backend accurate
```

有界稠密权重缓存通过230项CPU/VE检查。执行器增加`--dense-cache-mib`（默认0），缓存命中跳过重复解码和FP64转换。完整缓存验证已通过三提示36步和三次16步重复请求。并行填充版稳态约0.508 token/s，约为关闭缓存同数学基线的4.9倍，缓存实际保留14.44GiB；首次预填充约45–47秒，稳态预填充约10秒。最终1/2/4/8线程扫描已完成，各线程三次32词元输出均与未缓存基线相同；三档长输入扫描也已完成，每档两次16词元输出一致，温度监督任务退出0。正式短输入结果如下（热态取后两次的中位数）：

| 工作线程 | 热态生成（词元/秒） | 热态预填充（秒，21词元输入） | 相对同精度未缓存基线 |
|---:|---:|---:|---:|
| 1 | 0.361515 | 28.590036 | 3.49倍 |
| 2 | 0.485214 | 16.322276 | 4.69倍 |
| 4 | 0.470930 | 10.753569 | 4.55倍 |
| 8 | 0.509636 | 9.896059 | 4.93倍 |

已完成的260词元输入测得热态预填充99.093178秒、生成0.494542词元/秒；980词元输入两次输出一致，热态预填充424.361415秒、生成0.470626词元/秒。1940词元输入两次输出一致，热态预填充1031.422392秒、生成0.426516词元/秒；8线程仅为本次已测短输入配置中最快。证据为`build/results/20261002T162100Z-qwen-final-bench/summary.json`。可通过温度监督运行专项验证：

```bash
.venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- .venv/bin/python tests/check_qwen_dense_cache.py
```

最终长输入测量及运行边界：

| 输入词元 | 首次输入处理（秒） | 热态输入处理（秒） | 热态生成（词元/秒） |
|---:|---:|---:|---:|
| 21 | 45.445821 | 9.896059 | 0.509636 |
| 260 | 129.870469 | 99.093178 | 0.494542 |
| 980 | 455.720340 | 424.361415 | 0.470626 |
| 1940 | 1061.780509 | 1031.422392 | 0.426516 |

测试全程CPU峰值81°C、运行卡峰值62.875°C，CPU曾达到80°C警告线，但所有采样低于停止阈值。BMC前置风扇1–4观察到转速变化，没有修改配置，也不能据此断言控制策略改变。缓存在同一模型进程内复用；新进程仍有约18秒加载和首次建立缓存成本。

复现最终性能扫描（使用已完成验收的产物目录）：

```bash
.venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- .venv/bin/python tests/benchmark_qwen_dense_cache.py --accepted build/results/20261002T161007Z-qwen-dense-cache
```

该测试固定运行槽1、4096上下文、16384MiB缓存，自动扫描线程并测长输入；需要当前终端具备读取BMC风扇的sudo授权。结果只覆盖本文列出的文本模型和测试输入。

## Qwen2.5-1.5B量化单卡测试

官方Qwen2.5-1.5B-Instruct Q4_K_M，文件约1.12GB。高精度路径通过三提示共24步完整词表对照及三次32步输出检查；原量化路径分数误差超标，不作为通过精度验收的性能。1/2/4/8线程热态生成分别约2.16/2.99/3.78/3.04词元/秒，四线程本次最快。采用FP64累加/FP32激活、有界稠密缓存；缓存有效载荷约11.50GiB，不能用量化文件大小代表运行内存。完整数据见[公开摘要](docs/results/qwen25-1.5b-single-ve.json)。

```bash
.venv/bin/python scripts/download_qwen15.py
bash scripts/build_qwen15.sh
.venv/bin/python scripts/temperature_guard.py --bmc-fans --post-seconds 5 -- .venv/bin/python tests/check_qwen15.py
```

构建入口复用此前Qwen框架构建及高精度对象，需先按上文完成对应构建；创建适配器限定固定上游ABI和qwen2架构。所有产物独立放在build/qwen15-ve，不替换35B执行器。此次测试仅运行槽1、上下文配置2048、固定生成步数；不覆盖其他卡或全部模型输入。测试CPU峰值82°C、运行卡56.25°C，BMC读取不可用，原生可读风扇通道留档。
