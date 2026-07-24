# STLQ 从零配置与常用 CFG 手册

Last updated: 2026-06-01

这份文档面向第一次接触 STLQ 的使用者，目标是：

- 配好 CMake / CUDA / BLAS 环境，以及可选 HDF5 兼容环境
- 编译 `stlq_main`
- 配好数据集路径
- 跑通 SIFT1M 风格的训练、baseset 编码、linkage 构建和 eval
- 理解 `configs/basic_sift1m_linux.cfg` 里显式写出的常用配置

README 是完整百科；本文是上手路线图。

## 1. 先理解两个模式

STLQ 有两条主流程。

### Large / streaming 模式

```ini
large.enabled = 1
```

这是当前主线。它面向大数据和磁盘流水线，训练、base 编码、base list-order store、linkage_list、disk eval 都围绕可复用磁盘 artifact 展开。

核心状态机大致是：

```text
config bootstrap
  -> runtime / CUDA / BLAS 初始化
  -> train 或 load train result
  -> large workspace / artifact contract
  -> base_basic 编码
  -> base_list list-order store
  -> linkage_list 构建
  -> disk eval
```

推荐新实验优先用 large 模式。

### Non-large / in-memory 模式

```ini
large.enabled = 0
```

这是较早的从Julia代码复刻过来，并优化的内存版流程。它能用于小数据集和快速功能验证，但功能不如 large 模式完整，尤其是新的 disk artifact、coeff codec、disk-native eval、cleanup/checkpoint 等能力主要围绕 large 模式维护。

适用场景：

- 小数据集
- 只想快速试算法主干
- 不需要完整 large artifact / disk eval 链路

不建议用 non-large 做最终性能或大数据实验。

## 2. 环境依赖

最低依赖：

- CMake 3.16+
- C++20 编译器
  - Linux: GCC 9+ 或兼容版本
  - Windows: MSVC 2022 Build Tools
    - 如果 CUDA Toolkit 对 VS 2022 版本有要求，可从微软 release history 选择/更新到合适版本：
      <https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#updating-your-installation-to-a-specific-release>
    - 安装器内建议勾选 MSVC v143 x64/x86 build tools、Windows 10/11 SDK、C++ CMake tools for Windows
- BLAS/LAPACK
  - Intel oneAPI MKL，推荐，性能应该会好一点，毕竟是针对性优化。
  - 或 OpenBLAS
- HDF5 C library，可选
  - `STLQ_ENABLE_HDF5=ON` 时用于兼容 `.h5` 训练结果和 non-large 的旧式结果文件
  - large 主流程可以使用 native `.stlqbin` 训练检查点，不强制依赖 HDF5，但在hdf5阅读器下可比较方便的观察结果。可以去官网下载该依赖，但是需要注册一下。
- NVIDIA CUDA Toolkit，可选，但 large/linkage 性能实验通常需要，在NVIDIA官网可按引导下载安装。

CUDA 说明：

- `STLQ_ENABLE_CUDA=ON` 只是编译 CUDA 代码。
- cfg 里的 `runtime.use_cuda=1` 才是在运行时启用 CUDA。
- 二者都需要打开，GPU 路径才会真的跑。

## 3. CMake 配置

Linux上默认的cmake配置包括CUDA，MKL，HDF5，release模式。注意MKL环境变量的配置，以及CUDA的编译器路径。
如果你使用JetBrain Clion IDE，那么如果配置好了以上依赖，只需要打开release cmake配置，在cmake options中类似传入-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc即可（如果你不希望额外使用HDF5依赖，那么传递-DSTLQ_ENABLE_HDF5=OFF。MKL也并非必须，可用OpenBLAS替换）。

以下是命令行环境的示例配置：
### Linux + CUDA + MKL

```bash
cmake -S . -B cmake-build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTLQ_ENABLE_CUDA=ON \
  -DSTLQ_ENABLE_HDF5=OFF \
  -DSTLQ_USE_MKL=ON \
  -DMKLROOT=/opt/intel/oneapi/mkl/latest \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSTLQ_CUDA_ARCHITECTURES=89
```

已测试过的环境 ：

| GPU                         | `STLQ_CUDA_ARCHITECTURES`   | System      | CPU                   |
|-----------------------------|-----------------------------|-------------|-----------------------|
| RTX 3090 / Ampere           | `86`                        | Win11       | Intel Ultra7 265K     |
| RTX 4090 / Ada              | `89`                        | Ubuntu20.04 | Intel Xeon Gold 6226R |
| RTX 5060 Laptop / Blackwell | `120`                       | Win11       | AMD Ryzen 9 9955HX    |

如果不确定GPU架构，可以先用本机 CUDA 工具查询 GPU 型号，再看对应的 sm_xxx。

### Linux + CPU / OpenBLAS

```bash
cmake -S . -B cmake-build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTLQ_ENABLE_CUDA=OFF \
  -DSTLQ_ENABLE_HDF5=OFF \
  -DSTLQ_USE_MKL=OFF
```

如果 CMake 找不到 OpenBLAS，可以显式传：

```bash
-DSTLQ_OPENBLAS_ROOT=/path/to/OpenBLAS
```

### Windows + MSVC + CUDA + MKL

下面这套命令与 README 保持一致，适用于 Windows CMD + Ninja + oneAPI MKL + 可选 HDF5。
示例按 RTX 3090 写 `STLQ_CUDA_ARCHITECTURES=86`。

命令从 STLQ 仓库目录内执行；`path\to\STLQ` 表示替换成你本机 STLQ 仓库所在目录。

```bat
cd /d path\to\STLQ

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64

"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" ^
  -S . -B cmake-build-win -G Ninja ^
  -DSTLQ_ENABLE_CUDA=ON ^
  -DSTLQ_USE_MKL=ON ^
  -DMKLROOT="C:/Program Files (x86)/Intel/oneAPI/mkl/latest" ^
  -DHDF5_ROOT="C:/Program Files/HDF_Group/HDF5/2.0.0" ^
  -DSTLQ_CUDA_ARCHITECTURES=86 ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe" ^
  -DCMAKE_BUILD_TYPE=Release
```

注意：

- STLQ 使用 `STLQ_*` CMake 变量。
- Windows 上 `STLQ_USE_MKL` 默认是 `OFF`，如果要用 MKL 必须显式设为 `ON`。
- Ninja + CUDA 需要先执行 `VsDevCmd.bat`，让 `cl.exe` 在环境里可见。
- MKL 构建建议先执行 oneAPI `setvars.bat`，同时传入 `MKLROOT`。
- HDF5 是可选依赖。上面的命令启用默认 `STLQ_ENABLE_HDF5=ON` 并提供 `HDF5_ROOT`；如果不用 HDF5，添加 `-DSTLQ_ENABLE_HDF5=OFF`，并删除 `-DHDF5_ROOT=...`。

### Windows + OpenBLAS

如果不用 MKL，需要提供 OpenBLAS 根目录：

```bat
cmake -S STLQ -B STLQ\cmake-build-win -G Ninja ^
  -DSTLQ_ENABLE_CUDA=OFF ^
  -DSTLQ_ENABLE_HDF5=OFF ^
  -DSTLQ_USE_MKL=OFF ^
  -DSTLQ_OPENBLAS_ROOT="C:/path/to/OpenBLAS" ^
  -DCMAKE_BUILD_TYPE=Release
```

### 可选 HDF5 兼容

默认建议新实验使用 native train result：

```ini
io.result_format = "bin"
io.train_file = "./results/SIFT1M/test/test_train.stlqbin"
```

如果要兼容 `.h5` 文件：

```bash
cmake -S . -B cmake-build-release \
  -DSTLQ_ENABLE_HDF5=ON \
  -DHDF5_ROOT=/path/to/hdf5
```

`io.result_format` 可取：

- `auto`: 默认文件名保持 `.h5` 兼容；构建时没有 HDF5 时，保存会自动写 native STLQ bin 内容，读取会先识别 native magic
- `bin`: native STLQ result，默认文件名使用 `.stlqbin`，不依赖 HDF5
- `hdf5`: HDF5 兼容 train result，需要 `STLQ_ENABLE_HDF5=ON`

建议新配置显式写 `io.result_format = "bin"`；如果只是拿旧 cfg 做对比，先保持 `auto`，避免因为输出路径后缀变化影响 run identity 或 artifact 复用判断。

## 4. 编译

通用方式：

```bash
cmake --build cmake-build-release -j
```

只编译主程序：

```bash
cmake --build cmake-build-release --target stlq_main -j
```

默认不构建 `tests/` 下的测试目标。需要测试时，configure 时显式打开：

```bash
cmake -S . -B cmake-build-release -DBUILD_TESTING=ON ...
cmake --build cmake-build-release --target test_train_result_bin_io -j
ctest --test-dir cmake-build-release --output-on-failure
```

Windows Ninja：

```bat
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" ^
  --build cmake-build-win -j 18
```

视情况加 `--clean-first` 做干净重编。

如果使用 Makefile generator，也可以：

```bash
make -C cmake-build-release -j
```

## 5. 运行主程序

Linux：

```bash
./cmake-build-release/stlq_main --config configs/basic_sift1m_linux.cfg
```

Windows：

```bat
cd cmake-build-win
set "PATH=C:\Program Files\HDF_Group\HDF5\2.0.0\bin;%PATH%"
stlq_main.exe --config ..\configs\basic_desktop_win.cfg
```

如果构建时关闭了 HDF5，可省略 `PATH` 中的 HDF5 `bin`。

命令行临时覆盖 cfg：

```bash
./cmake-build-release/stlq_main \
  --config configs/basic_sift1m_linux.cfg \
  --set dataset.data_root=/data/pq-dataset \
  --set runtime.omp_threads=32
```

建议第一次跑时把输出放到 build 目录下，避免污染源码目录：

```bash
./cmake-build-release/stlq_main \
  --config configs/basic_sift1m_linux.cfg \
  --set large.output_dir=cmake-build-release/outputs_smoke \
  --set large.tmp_dir=cmake-build-release/tmp_smoke
```

## 6. 配好数据集路径

`basic_sift1m_linux.cfg` 里最重要的是：

```ini
dataset.data_root = "/media/dell01/hdd0/download/pq-dataset"
dataset.ntrain = 100000
dataset.nbase = 1000000
dataset.groundtruth_add1 = 0
```

当 `dataset.name = "SIFT1M"` 或未显式设置时，默认文件相对 `dataset.data_root`：

```text
sift/sift_learn.fvecs
sift/sift_base.fvecs
sift/sift_query.fvecs
sift/sift_groundtruth.ivecs
```

因此你的目录应该类似：

```text
/data/pq-dataset/
  sift/
    sift_learn.fvecs
    sift_base.fvecs
    sift_query.fvecs
    sift_groundtruth.ivecs
```

如果文件不在默认位置，可以显式指定：

```ini
dataset.train_path = "my_sift/learn.fvecs"
dataset.base_path = "my_sift/base.fvecs"
dataset.query_path = "my_sift/query.fvecs"
dataset.groundtruth_path = "my_sift/gt.ivecs"
```

相对路径会从 `dataset.data_root` 解析；绝对路径也可以。

常见格式：

- `fvecs`: float vectors
- `bvecs` / `siftbin`: uint8 vectors
- `fbin`: float binary matrix
- `ivecs` / `ibin`: groundtruth ids

如果自动识别不符合你的文件，可用 large 格式覆盖：

```ini
large.train_format = auto
large.base_format = auto
large.query_format = auto
large.gt_format = auto
```

常见可选值：`auto`, `fvecs`, `bvecs`, `fbin`, `ivecs`, `ibin`。

## 7. `basic_sift1m_linux.cfg` 常用配置导读

这一章按 `basic_sift1m_linux.cfg` 的显式配置解释“参数进入哪段控制流、影响什么产物”。完整参数表见
`README.md`，字段定义见 `include/stlq/common/config.h`。
注意，很多bool参数使用0，1简化，也可显式写false,true.

本节按状态机阶段组织 `basic_sift1m_linux.cfg` 的显式配置：

| cfg 分组 | 主要控制流 | 主要产物/影响 |
|---|---|---|
| `dataset.*`, `model.*` | 数据 reader、模型形状归一化 | 数据规模、root/list 数、码本层数、store hash |
| `large.*` | large workspace、store 构建、cleanup、codec | run root、tmp、base/list/linkage store、磁盘峰值 |
| `advanced/train/base/eval *.enabled` | 顶层 stage 开关 | 是否 train、base encode、linkage build、eval-only |
| `runtime.*` | CUDA/CPU 调度、异步 IO、hybrid base encode | 速度、显存/RAM峰值、微小浮点差异 |
| `io.*` | train/base/linkage 结果保存和加载 | HDF5/native bin、resume/eval-only 输入 |
| `train.*`, `train.kmeans.*` | init、kmeans、OPQ/R 迭代、训练 linkage | `C_root`、`C_one`、训练质量和耗时 |
| `base.encode.*`, `base.linkage.*` | baseset basic 编码、baseset linkage | `base_basic`、`base_list`、`linkage_list` |
| `virtual.*`, `hnsw.*` | linkage candidate/virtual structure | linkage 深度、候选质量、构建时间 |
| `eval.*` | base/linkage disk eval | recall/QPS、norm2 cache、eval archive |

部分参数是模型/算法超参，会改变结果和 artifact hash；部分参数属于运行调度、IO 或日志配置，理论上不改变语义，但可能因
CPU/GPU 后端、并行顺序和浮点非结合性带来很小差异。

### 7.1 数据集与模型规模

```ini
dataset.ntrain = 100000
dataset.nbase = 1000000
model.m = 5
model.h0_one = 256
model.h0 = 256
```

- `dataset.ntrain`: 训练阶段使用多少 train/learn 向量。
- `dataset.nbase`: baseset 使用多少 base 向量。
- `model.m`: 量化层数，也影响编码长度和 coeff 布局。
- `model.h0`: root/IVF 层的 codebook size。常规实验只需要改这个值。
- `model.h0_one`: `C_one[0]` 的 codebook size，large 流程通常保持 `<=256`，因为 `code0_one` 存成 `u8`。


### 7.2 Large 磁盘流水线

```ini
large.enabled = 1
large.output_dir = ./outputs_opt
large.tmp_dir = ./tmp_opt
large.train_block = 200000
large.base_block = 200000
large.write_vector_bucket = 1
large.profile_timing = 0
large.archive_log = 1
```

- `large.enabled=1`: 使用 streaming/out-of-core 主流程。
- `large.output_dir`: 最终 run artifact 根目录。
- `large.tmp_dir`: 临时文件目录。
- `large.train_block`: 训练流式读块大小。
- `large.base_block`: base 编码流式读块大小。
- `large.write_vector_bucket`: 写 cluster-friendly raw vector bucket，用于后续 linkage build 的按簇顺序读取。
- `large.profile_timing`: 打开后打印更细阶段计时，排查性能时非常重要。
- `large.archive_log`: 保存运行日志到 run root。

`train_block/base_block` 属于性能和内存调度参数，理论上不改变算法语义，但可能因并行顺序、浮点归约、CPU/GPU 路径差异带来微小数值差异。

### 7.3 Coeff codec 与清理策略

```ini
large.linkage_coeff_codec.enabled = 1
large.linkage_coeff_codec.bits_per_layer = [6,6,6,6,6]
large.linkage_coeff_codec.p_first_candidates = [99.93]
large.linkage_coeff_codec.p_rest_candidates = [99.93]
large.linkage_coeff_codec.granularity = cluster
large.linkage_coeff_codec.use_weighted_quantile = 1
large.linkage_coeff_codec.q_refine_sweeps = 3
large.linkage_coeff_codec.q_refine_max_layer = 0
large.base_linkage_store_coeffs_f32 = 1
large.cleanup.enabled = 1
large.cleanup.preset = eval_int8_min
large.protect_existing_outputs = 0
```

- `large.linkage_coeff_codec.enabled`: 为 linkage coeff 构建 int8/bit-packed codec。
- `bits_per_layer`: 每层 coeff 量化 bit 数（默认为6）；bit 越多通常误差越小、codec payload 越大。
- `p_first_candidates`, `p_rest_candidates`: 选择每层量化范围的分位候选（可调，一般99.93到99.98）。第 0 层和其余层可分开控制，避免极端 coeff 拉大范围。
- `granularity`: coeff codec 的 Huffman stream 粒度。`cluster` 表示每个 `(cluster, root/linkage group)` 建一个 Huffman model/bitstream，并把该 group 的所有层串到同一个 stream；`layer` 表示每个 `(cluster, root/linkage group, layer)` 单独建 Huffman model/bitstream。
- `use_weighted_quantile`: 构建量化范围时考虑权重，面向 linkage coeff 分布更偏斜的场景。
- `q_refine_sweeps`: coeff 量化后的 refinement 轮数。
- `q_refine_max_layer`: 从第 0 层开始最多 refinement 多少层；`0` 表示所有层。
- `large.base_linkage_store_coeffs_f32`: 是否在 `linkage_list` 中保留 float coeff payload（最终形态是不需要的，只是为了开发使用，如果你想后续反复调整coeff压缩参数，这个还是可以保留的，只是存储占用比较大）。
- `large.cleanup.*`: 按阶段删除可重建中间产物，节省磁盘。
- `large.protect_existing_outputs`: 防止覆盖已有 run root产物。调试时可设 `0`，正式复用/防误删时可设 `1`。

控制流位置：

1. `base.linkage.enabled=1` 先构建 `linkage_list` 的 float 逻辑产物。
2. `large.linkage_coeff_codec.enabled=1` 再从 `linkage_list` 的 coeff 构建 codec payload、scale、offset/size 和 `coeff_hash.u64`。
3. `eval.linkage.coeff_mode="int8"` 时，disk eval 使用 codec 路径；`float` 使用 f32 coeff（开发使用，测试第一阶段编码效果，没什么特别的意义）；`both` 两者都跑。
4. `large.cleanup.preset=eval_int8_min` 会在安全点删除可重建中间 store 和 float payload，只保留 int8 eval 所需内容。

`coeff codec` 会影响 int8 eval 的结果和存储大小，是功能/质量相关配置。`cleanup` 不改变正确性，但会影响后续是否还能重建 float/int8 codec；例如删除 float coeff 后，若 codec 也缺失，就无法再从 float payload 重建。

#### 大数据 cleanup 与容量预估

SIFT1B / BigANN 这类数据集默认会把原始 base 向量按 IVF bucket / list-order 重新写成临时副本。这样 linkage build 可以按 cluster 顺序顺读，否则直接在原始文件上做大量随机读取会非常慢。

容量粗估，以 SIFT1B `1e9 x 128 uint8` 为例：

- 原始 base 文件本身约 `128 GB`，`bvecs` 每条带 4 字节维度头时约 `132 GB`。
- `large.write_vector_bucket=1` 时，会在构建过程中写一份 bucket/list-order raw 副本，SIFT1B uint8 约再占 `128-132 GB`。这是为了顺序读 cluster，属于峰值占用；cleanup 会在安全阶段删除它。
- `base_basic` 的 code、coeff、cluster id、IVF CSR 等通常是几十 GB；`m=5` 时 code 约 `5 GB`，float coeff 约 `20 GB`，再加 shard/索引/检查点开销。但float系数是中间结果，在最小配置下会被删掉。
- `linkage_list` 的 parent/LOUDS/coeff/codec 取决于 virtual、depth、coeff codec 和是否保留 f32 payload。`eval_int8_min` 会删除 float coeff 和可重建中间 store，最终主要保留 int8 codec、LOUDS/meta/hash/log/eval 结果，通常是 `20 GB 量级`。
- 按默认生产口径 `cleanup.enabled=1 + preset=eval_int8_min` 估算，不计原始数据集本身，SIFT1B 的构建峰值通常按 `250-350 GB` 理解更合理；考虑文件系统碎片、临时峰值、日志/eval 归档和失败重跑余量，`output_dir/tmp_dir` 所在盘预留 `400-500 GB` 一般是实用建议。
- 如果保留 `eval_both`，会额外保留 float linkage coeff/parent 等，最终 footprint 会增加几十 GB；如果使用 `dev_all`、关闭 cleanup、保留多次 run、或把 raw 以 f32 形式落盘，才需要按 `1 TB` 甚至更高规划。
- 在 `eval_int8_min` 和数据库 norm2 LUT 模式下，最终常驻结果通常约 `20 GB` 量级。`base_basic` 属于构建中间产物，主要用于生成最终链式结构和开发检查。

内存粗估：

- large 训练/build 不会把 1B base 全量读入内存。SIFT1B 通常会把 `dataset.ntrain` 手动调到 `1e7 / 3e7 / 5e7`，并把 root cluster 数 `model.h0` 调到 `65536` 或 `100000` 级别；资源估算应按目标数据集规模计算，而不是按 SIFT1M 示例的 `ntrain=100000, nlist=256` 外推。
- 训练 reader 会尝试缓存 `ntrain` 范围到 Host RAM。SIFT/BIGANN u8 路径约为 `ntrain * 128` 字节：
  - `ntrain=1e7`: `1.19 GiB`
  - `ntrain=3e7`: `3.58 GiB`
  - `ntrain=5e7`: `5.96 GiB`
  如果训练输入以 f32 形式进入，则上面数字乘 4。缓存 OOM 时会降级为 pass-through，只是训练 I/O 会变慢。
- `train.init_samples=5000000` 只控制 CPU fallback/reservoir init 的采样上限；CUDA streaming init 仍按 `dataset.ntrain` 的 reader 视图流式训练。
- 单个 base block 在 `base_block=200000, d=128` 时：uint8 raw 约 `24.4 MiB`，f32 工作块约 `97.7 MiB`。`runtime.basic_async_io_depth=6` 的 u8 prefetch 队列约 `146 MiB`；`runtime.basic_async_io_mb=1024` 是上限，不代表默认一定常驻 1 GiB。
- hybrid reorder/result 队列按每 block 约 `30 MiB` 级别估算，`reorder_depth=64` 的理论上限是数 GB；稳定运行通常低于这个上限。
- baseset linkage build 按 cluster 工作。对 SIFT1B `nbase=1e9, d=128`：
  - `nlist=65536`: 平均每簇约 `15259` 向量，raw u8 工作集约 `1.86 MiB`
  - `nlist=100000`: 平均每簇约 `10000` 向量，raw u8 工作集约 `1.22 MiB`
  实际峰值看 cluster skew、virtual 比例和候选临时 buffer，但不再是 `nlist=256` 时的几百 MiB 单簇量级。
- 默认 int8 linkage eval 且 `eval.linkage.preload_clusters_io_threads=0` 时，会预加载 cluster 数据以换 QPS。`eval_int8_min` 下 resident cache 主要是 `real_ids`、parent、codes、int8 coeff、norm2 LUT 和 virtual 相关数组。按 real vector 粗估：
  - real-only 下界约 `17-20 bytes/vector`
  - 加上 virtual、容器/哈希表、cluster metadata、alignment 后，SIFT1B 常见应按 `24-32 GB` 级别规划
  - `nlist=100000` 会比 `65536` 多一些 per-cluster 容器开销，但主要项仍是按 `nbase` 线性增长
- 因此 SIFT1B 常用配置更具体的口径是：只跑 train/build，`24 GB` 空闲RAM较稳；完整默认 train/build + int8 preload eval，`32 GB` 空闲RAM比较宽松。
- 3090 这类 `24 GB VRAM` 足够运行默认 CUDA 配置，不需要预先调低参数。只有在自定义提高 pool/cache/chunk/batch、或同机有其他进程占显存并实际 OOM 时，才考虑降低 CUDA budget。

这些数字是工程估算，真实峰值受 `model.m`、`model.h0`、virtual 比例、coeff codec、cleanup preset、是否保留 f32 payload、raw 文件格式影响。大数据第一次跑建议先开 `large.profile_timing=1`，观察日志里的 store build、cleanup 和 eval 阶段。

### 7.4 训练流程开关

```ini
advanced.eval_only = 0
train.init_enabled = 1
train.enabled = 1
base.encode.enabled = 1
base.linkage.enabled = 1
train.ckpt.enabled = 0
train.ckpt.every_R = 5
train.ckpt_after_init_basic = 0
train.exit_after_ckpt_init_basic = 0
large.base_basic_checkpoint = 0
large.linkage_checkpoint = 0
```

含义：

- `train.enabled=1`: 运行训练；为 `0` 时加载已有 train result。
- `train.init_enabled=1`: 运行 init 阶段；为 `0` 时依赖已有的init训练结果/检查点。
- `base.encode.enabled=1`: 构建 `base_basic` / `base_list` 所需的基础编码。
- `base.linkage.enabled=1`: 构建 baseset `linkage_list`。
- `train.ckpt.enabled`: 训练 OPQ/R 迭代 checkpoint 开关；用于长训练中断后恢复。
- `train.ckpt.every_R`: 每多少轮 R/OPQ 迭代保存一次训练 checkpoint。
- `train.ckpt_after_init_basic`: init basic 完成后额外保存一个检查点，方便跳过legacy_root_init模式下昂贵的 init/basic。
- `train.exit_after_ckpt_init_basic`: 写完 init-basic checkpoint 后直接退出，用于分阶段跑任务。
- `large.base_basic_checkpoint`: `base_basic` 编码阶段的 block-level checkpoint/resume。
- `large.linkage_checkpoint`: baseset `linkage_list` 构建阶段的 per-cluster checkpoint/resume。
- `advanced.eval_only=1`: 便捷 recall-only 模式，会强制：
  - `train.enabled=false`
  - `base.encode.enabled=false`
  - `base.linkage.enabled=false`
  - eval 开关保持不变

enable参数可自由组合，用于不同目的。

典型完整训练：

```ini
advanced.eval_only = 0
train.enabled = 1
base.encode.enabled = 1
base.linkage.enabled = 1
eval.linkage.enabled = 1
```

典型只 eval 链式编码：

```ini
advanced.eval_only = 1
eval.base.enabled = 0
eval.linkage.enabled = 1
```

只 eval 前必须已有可复用 artifact，例如 train result、`linkage_list/`、store hash、必要的 norm2 / codec payload。

checkpoint 只改变恢复能力和阶段切分，不应改变算法目标。它会写额外 sidecar / checkpoint 文件，并且恢复时会检查 store hash，防止把不兼容配置的产物混在一起。

### 7.5 Runtime / CUDA 性能配置

```ini
runtime.omp_threads = 48
runtime.use_cuda = 1
runtime.cuda_allow_tf32 = 1
runtime.cuda_mode = "fast"
runtime.cuda_pool_size = 32
runtime.cuda_pool_size_init_linkage = 24
```

- `runtime.omp_threads`: CPU OpenMP 线程数。
- `runtime.use_cuda`: 运行时启用 CUDA。
- `runtime.cuda_mode="strict"`: 更偏复现；TF32 默认关闭，cuBLAS atomics 禁用。
- `runtime.cuda_mode="fast"`: 更偏速度（区别不算很大）；TF32 和非严格 cuBLAS 行为启用。
- `runtime.cuda_allow_tf32`: strict 模式下是否允许 TF32；fast 模式会启用更偏吞吐的 CUDA 行为（区别不算很大）。
- `runtime.cuda_pool_size`: linkage build 的 CUDA ctx pool 大小。
- `runtime.cuda_pool_size_init_linkage`: init-linkage 单独 pool 大小。

这些配置主要影响性能，同时也会影响浮点执行路径。结果应在误差范围内，但不保证 bitwise 完全一致。

### 7.6 Linkage CUDA 调度参数

相关参数包括：

```ini
runtime.cuda_linkage_single_gpu_min_candidates = 16
runtime.cuda_linkage_many_nodes_max_pairs = 8192
runtime.cuda_linkage_many_nodes_preflush_target_pairs = 64
runtime.cuda_linkage_mem_budget_mb = 512
runtime.cuda_linkage_large_root_xc0_chunk_mb = 256
runtime.cuda_linkage_same_layer_window_min_pairs = 32
runtime.cuda_linkage_same_layer_block_nodes = 32
runtime.cuda_linkage_same_layer_window_max_pending_nodes = 64
runtime.cuda_linkage_same_layer_preflush_single_shot = 1
runtime.cuda_linkage_same_layer_window_max_pairs = 8192
runtime.cuda_linkage_same_layer_single_gpu_min_candidates = 6
runtime.cuda_linkage_same_layer_preflush_target_pairs = 0
runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable = 1
runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_pairs = 8

runtime.cuda_linkage_many_nodes_max_pairs_init_linkage = 8192
runtime.cuda_linkage_chunk_max_pairs_init_linkage = 0
runtime.cuda_linkage_mem_budget_mb_init_linkage = 512
runtime.cuda_linkage_eval_async_pinned_mb_init_linkage = 256
runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage = 256
runtime.cuda_linkage_init_same_layer_tiny_forced_cpu_enable = 1
```

它们控制 linkage build 中 GPU candidate evaluation 的 batching、chunking、VRAM 上限，以及小规模 tail 是否走 CPU。

原则：

- 不改变候选集合和目标函数。
- 主要改变 GPU 调度形状、kernel 调用次数、显存峰值和 CPU/GPU fallback。
- 可能带来 CPU/GPU 浮点细微差异。
- GPU 适合大批量 candidate evaluation、GEMM、many-nodes batch 和 large-root 矩阵计算。
- CPU 适合小规模 tail / forced flush / sparse window：这类任务的 kernel launch、H2D/D2H、cuBLAS 同步和 ctx pool 竞争成本可能大于计算本身。
- CPU 处理小任务不会改变候选集合或目标函数；同一批候选由 CPU backend 执行，以避免 GPU 被碎片化 tiny batch 拖慢。
- 该协同方式使 GPU 保持处理大 batch，CPU 同时推进小 cluster 或小 window，整体吞吐通常更稳定。
- 默认配置以 `24 GB VRAM` 级别显卡为推荐目标，通常不需要调整。只有在自定义放大 CUDA 预算或实际 OOM 时，才优先降低：
  - `runtime.cuda_pool_size`
  - `runtime.cuda_linkage_many_nodes_max_pairs`
  - `runtime.cuda_linkage_mem_budget_mb`
  - `runtime.cuda_linkage_large_root_xc0_chunk_mb`

常见 key 对应关系：

| Key | 控制位置 | 作用 |
|---|---|---|
| `runtime.cuda_linkage_single_gpu_min_candidates` | 单节点候选 eval | 候选数低于阈值时不值得上 GPU，可能走 CPU |
| `runtime.cuda_linkage_many_nodes_max_pairs` | many-nodes batch | 限制一次 GPU 调用的 `(node,parent)` pair 数，控制吞吐/显存 |
| `runtime.cuda_linkage_many_nodes_preflush_target_pairs` | many-nodes 预 flush | 候选积累到目标 pair 数附近时提前发 GPU，减少 tiny call |
| `runtime.cuda_linkage_mem_budget_mb` | CUDA evaluator | 显存预算提示，用于内部 chunk/workspace 控制 |
| `runtime.cuda_linkage_large_root_xc0_chunk_mb` | large-root root scan | 控制 `xC0`/root 相关 GEMM chunk，越大通常调用更少但显存更高 |
| `runtime.cuda_linkage_eval_async_pinned_mb` | async candidate eval | pinned host staging 预算，用于 CPU/GPU overlap |
| `runtime.cuda_linkage_same_layer_window_min_pairs` | same-layer window | good-cluster 同层动态候选窗口的预 flush pair 阈值 |
| `runtime.cuda_linkage_same_layer_block_nodes` | same-layer window | 每个同层窗口覆盖的 node 数 |
| `runtime.cuda_linkage_same_layer_window_max_pending_nodes` | same-layer window | 同层动态 pending node 上限 |
| `runtime.cuda_linkage_same_layer_preflush_single_shot` | same-layer window | 允许窗口内动态候选用 single-shot batch 预 flush |
| `runtime.cuda_linkage_same_layer_window_max_pairs` | same-layer window | 同层窗口内一次 flush 的 pair 上限 |
| `runtime.cuda_linkage_same_layer_single_gpu_min_candidates` | same-layer per-node | 同层动态候选单节点 GPU 阈值；`0` 时复用通用阈值 |
| `runtime.cuda_linkage_same_layer_preflush_target_pairs` | same-layer preflush | 专门的同层预 flush 目标；`0` 表示关闭该额外触发 |
| `runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable` | forced flush tail | 小规模 forced flush 直接由 CPU 处理，避免 GPU launch 成本 |
| `runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_pairs` | forced flush tail | tiny CPU 路径允许的最大 pair 数 |

带 `_init_linkage` 后缀的是 init-linkage 专用覆盖。为 `0` 时通常表示禁用该覆盖或复用普通值；显式设置时只影响 init 阶段，不影响后续 iteration/base linkage。

### 7.7 Async IO / hybrid base encode

```ini
runtime.linkage_async_io = true
runtime.linkage_async_io_depth = 6
runtime.basic_async_io = 1
runtime.basic_async_io_depth = 6
runtime.basic_async_io_mb = 1024
runtime.basic_hybrid_enable = 1
runtime.basic_hybrid_cpu_stride = 15
runtime.basic_hybrid_cpu_threads = 48
runtime.basic_hybrid_reorder_depth = 64
runtime.basic_hybrid_inflight_mb = 0
runtime.basic_async_write_depth = 8
```

- `linkage_async_io`: linkage build 按 cluster 预取 IO。
- `basic_async_io`: base 编码按 block 预取 IO。
- `basic_hybrid_enable`: base encode 同时用 GPU 和 CPU lane。
- `basic_hybrid_cpu_stride`: 大致每隔多少 block 给 CPU lane（这个为了榨取性能，需要单独测试baseset的基础编码阶段，看gpu和cpu编码一个block各自需要多久，然后设置这个分配比例）。
- `basic_hybrid_cpu_threads`: CPU lane 使用的线程数。
- `basic_hybrid_reorder_depth`: hybrid 结果乱序完成后的重排缓冲深度。
- `basic_hybrid_inflight_mb`: hybrid in-flight block 内存预算；`0` 表示不按 MB 额外限制。
- `basic_async_write_depth`: async writer 的队列深度，影响写盘 overlap 和内存峰值。

这些是性能/吞吐调度参数。理论上不改变语义，但 hybrid/并行顺序可能带来极小浮点差异。

### 7.8 训练、编码、linkage 超参数总览

下面几组都属于结果相关超参数，不应只把 `train.*` 看成训练超参。完整质量面包括 init、训练迭代、base basic 编码、baseset linkage、virtual/HNSW。

#### 7.8.1 Train init / OPQ / kmeans

```ini
train.init_samples = 5000000
train.seed = 1985326
train.use_opq_rotation = 1
train.max_R_iters = 25
train.ils_iters = 8
train.icm_iters = 4
train.log_metrics = 1
train.log_linkage_pre_c1 = 0
train.kmeans_iters = 40
train.kmeans_init = "kmeansll"
train.kmeansll_oversample = 16
train.kmeansll_rounds = 8
train.kmeansll_candidate_cap = 0
train.kmeans_streaming = 1
train.kmeans_annealing_factor = 0.9
train.kmeans_cache_xnorm_device = 0
train.kmeans_pin_host_x = 1
train.kmeans_large_k_threshold = 4096
train.kmeans_large_k_top_coarse = 1
train.kmeans_force_disable_device_weights = 0
train.kmeans_anneal_mode = onepass
train.kmeans_device_cache_mb = 20480
```

- `train.init_samples`: init / seeding 使用样本数。
- `train.seed`: 训练随机种子。
- `train.use_opq_rotation`: 是否更新 OPQ/global rotation。
- `train.max_R_iters`: init 后全局 R/OPQ 迭代轮数。
- `train.ils_iters`, `train.icm_iters`: train basic 编码 refinement 强度。
- `train.log_metrics`: 训练中输出 MSE/linkage 等指标，方便观察质量走势，关闭可以省去相关计算以加速。
- `train.log_linkage_pre_c1`: 输出 C_one 更新前的 linkage 指标，用于诊断，不建议普通跑法频繁打开。
- `train.kmeans_iters`: kmeans Lloyd 迭代轮数。
- `train.kmeans_init`: 初始化方法。`kmeansll` 是 kmeans||，比随机更稳，但随机方法可能也会更好。
- `train.kmeansll_oversample`: kmeans|| 每轮过采样倍率。
- `train.kmeansll_rounds`: kmeans|| seeding 轮数。
- `train.kmeansll_candidate_cap`: kmeans|| 候选 cap；`0` 表示不额外 cap。
- `train.kmeans_streaming`: 使用 streaming kmeans，不把训练集当成单个大矩阵处理。
- `train.kmeans_annealing_factor`: streaming/anneal 更新中的退火系数。
- `train.kmeans_cache_xnorm_device`: 尝试缓存 norm 到 GPU，减少重复计算/传输。
- `train.kmeans_pin_host_x`: 用 pinned host memory 改善 H2D 传输。
- `train.kmeans_large_k_threshold`: K 大于阈值时启用 large-K 分层/近似策略。
- `train.kmeans_large_k_top_coarse`: large-K 分层候选展开数量。
- `train.kmeans_force_disable_device_weights`: 强制禁用 device weights 路径，用于兼容/排查。
- `train.kmeans_anneal_mode`: anneal 执行模式，`onepass` 是当前 cfg 的流式口径，虽然输出的是最后一轮的近似，但足够用了。
- `train.kmeans_device_cache_mb`: GPU 上缓存训练块/归一化数据的预算，显存足够时减少 I/O 和重复转换，非常有用。

这些多数是模型训练超参数，会影响最终结果。

#### 7.8.2 Init-linkage 与训练迭代 linkage

```ini
train.linkage.num_layers = 10
train.linkage.depth_k = 6
train.linkage.knn_k = 15
train.linkage.use_ils = 1
train.linkage.ils_rounds = 4
train.linkage.icm_round = 1
train.init_linkage_ils_rounds = 5
train.init_linkage_icm_round = 1
train.init_linkage_mode = legacy_root_only
train.init_linkage_hybrid_varroot_rounds = 2
```

- `train.linkage.*`: 训练迭代中的 linkage 配置。
- `train.init_linkage_*`: init-linkage 的专用覆盖。
- `train.init_linkage_mode`:
  - `legacy_root_only`: 基本模式，仅用单一C_root码本初始化C_one码本，作为基线。
  - `hybrid`: 只使用 C_root，但前几轮算法中保持默认首层可扰动，后续固定下来，用于平衡质量和速度。
  - `fast`: 更激进的快速模式，使用迭代流程中一致的双码本初始化（复制了一份C_root作为C_one）。

这些是质量/速度超参数，可能明显改变 linkage 结构和训练结果。

#### 7.8.3 Base basic 编码与 baseset linkage

```ini
base.encode.seed = 38251450
base.encode.use_abs = true
base.encode.ils_iters = 32
base.encode.icm_iters = 4
base.encode.H_beam = 2
base.linkage.use_ils = 1
base.linkage.num_layers = 12
base.linkage.depth_k = 11
base.linkage.knn_k = 30
base.linkage.ils_rounds = 4
base.linkage.icm_round = 1
```

- `base.encode.*`: 对 1M/1B base 向量做基础编码。
- `base.encode.seed`: base basic 编码的随机种子，用于 ILS/beam 中需要随机扰动或 tie-break 的路径，保证可复现。
- `base.encode.use_abs`: basic 编码目标使用 ABS 形式的 residual/coeff 处理；应和训练出的模型及 eval 路径保持一致。
- `base.encode.ils_iters`, `base.encode.icm_iters`: basic 编码内的 ILS/ICM refinement 强度。
- `base.encode.H_beam`: beam 宽度，常用 `2`；更大可能更准但更慢。
- `base.linkage.*`: baseset `linkage_list` 构建参数。
- `base.linkage.use_ils`: baseset linkage 编码中启用 ILS refinement；关闭后速度可能变快，但结构质量通常下降。
- `depth_k`, `knn_k`, `num_layers`: 控制 linkage 候选搜索和深度结构。

这些是结果相关超参数。尤其 `base.linkage.*` 会影响最终 `linkage_list` 和 recall。

#### 7.8.4 Virtual 与 HNSW

```ini
virtual.enabled = 1
hnsw.M = 48
hnsw.candidate_multiplier_good = 1
hnsw.candidate_multiplier_bad = 1
```

- `virtual.enabled`: 开启 virtual node / virtual-root augmentation，保持开启。
- `hnsw.M`: HNSW 图连接度。
- `candidate_multiplier_good/bad`: good/bad cluster 候选扩张倍率，保持1为不扩张。

这些是结构/搜索超参数，保持默认即可，或者有需要可以调一调hnsw.M。他们会影响 linkage 质量、构建时间和内存。

### 7.9 IO 保存/加载 contract

```ini
io.save_train = 1
io.save_base = 0
io.save_linkage = 0
io.pre_fix = "sift1m_trans"
io.hdf5_layout = "julia"
io.index_dtype = "int"
io.load_date = "20260528"
io.load_seq = ""
```

- `io.save_train`: 保存训练结果，后续 `train.enabled=0` 或 `advanced.eval_only=1` 时可加载复用。
- `io.save_base`: 保存 non-large / legacy base 编码结果。large 主流程的 `base_basic/base_list` 主要由 disk store 管理。
- `io.save_linkage`: 保存 non-large / legacy linkage 结果。large 主流程的 `linkage_list` 主要由 disk store 管理。
- `io.pre_fix`: 实验名前缀，会进入 run 目录命名。常见目录形如 `<pre_fix>_m<model.m>__<hash>`。
- `io.hdf5_layout`: HDF5 矩阵布局兼容项。`julia` 用于兼容 Julia 侧保存的列主序布局；`cxx` 用于 C++ 风格布局。
- `io.index_dtype`: HDF5 索引 dtype 兼容项。`int` 是默认路径；`uint` 用于兼容旧 Julia 文件。
- `io.load_date`: 加载已有结果时选择日期目录；`""` 表示当天，`YYYYMMDD` 表示指定日期，`raw` 表示直接使用显式文件路径。
- `io.load_seq`: 同一天多次运行时选择具体序号；为空时使用默认序号解析。

控制流位置：

1. 完整训练时，`io.pre_fix` 参与 run root 命名，`io.save_train=1` 会把训练结果保存为后续复用输入。
2. `train.enabled=0` 或 `advanced.eval_only=1` 时，程序根据 `io.pre_fix/io.load_date/io.load_seq` 定位已有 train/base/linkage artifact。
3. large 模式优先使用 disk store 和 store hash 管理 `base_basic`、`base_list`、`linkage_list`；`io.save_base/io.save_linkage` 更多是 non-large/legacy 保存开关。
4. `io.hdf5_layout/io.index_dtype` 只影响 HDF5 读写兼容，不应该改变算法语义；如果 layout/dtype 配错，会表现为加载失败或结果明显不对。

## 8. Eval 路径

### 8.1 Base disk IVF eval

```ini
eval.base.enabled = 1
eval.base.use_ivf = true
eval.base.nprobe = 16
```

用于评估 `base_list` / IVF base recall，只是开发时使用。一般只关心 linkage，可以设：

```ini
eval.base.enabled = 0
```

### 8.2 Linkage disk eval

```ini
eval.linkage.enabled = 1
eval.linkage.use_ivf_disk = true
eval.linkage.nprobe = [64]
eval.linkage.query_block = 1000
eval.linkage.coeff_mode = "int8"
```

- `eval.linkage.nprobe`: 可以是单值，也可以是列表。
- `query_block`: query 分块大小，影响内存和吞吐。
- `coeff_mode`:
  - `float`: 使用 float coeff payload
  - `int8`: 使用 coeff codec
  - `both`: 两者都评估
  - `auto`: 自动选择可用路径

如果 `coeff_mode=int8`，需要 `linkage_coeff_codec` 已存在或 float coeff payload 仍可用于重建 codec。

### 8.3 Probe backend: exact / hier2 / hnsw

```ini
eval.linkage.ivf_probe_mode = exact
eval.linkage.ivf_hier2_top_coarse = 8
eval.linkage.ivf_hnsw_M = 32
eval.linkage.ivf_hnsw_ef_construction = 200
eval.linkage.ivf_hnsw_ef_search = 32
```

- `exact`: 精确 coarse probe，适合基准。
- `hier2`: 两级 coarse probe，用 `ivf_hier2_top_coarse` 控制展开数量。
- `hnsw`: 快速搜索需要探查的簇，降低probe生成开销，可把 `nprobe` 和 `ef_search` 都设成列表，输出批量结果和 Pareto summary。

HNSW batch 示例：

```ini
eval.linkage.ivf_probe_mode = hnsw
eval.linkage.nprobe = [16,32,64,128]
eval.linkage.ivf_hnsw_ef_search = [32,64,128]
```

### 8.4 Disk-native / LOUDS-native eval

```ini
eval.linkage.parent_louds_native_eval = 0
```

- `0`: 常规 disk linkage eval 路径。
- `1`: LOUDS-native CPU eval 路径，直接查询 LOUDS，不物化完整 parent array。

LOUDS-native 当前是 CPU 路径，适合验证 LOUDS 结构和低内存 eval，不是 GPU scan 加速路径。

### 8.5 Eval 性能与缓存

```ini
eval.disk_norm2_mode = lut
eval.disk_norm2_lut_kmeans_niter = 25
eval.linkage.gpu_norm_enable = 1
eval.linkage.gpu_scan_enable = 0
eval.linkage.gpu_scan_cache_mb = 20480
eval.linkage.norm2_store_float = 1
eval.linkage.norm2_store_int8 = 1
eval.linkage.preload_clusters_io_threads = 0
eval.linkage.louds_huffman_bench_times = 1
eval.linkage.archive_eval_result = 1
```

- `disk_norm2_mode=lut`: 使用 LUT 压缩 norm2，更省 IO/内存，精度损失不大（但也分数据集，比如GIST1M 960维受到的影响比其他数据集更大，但这个更多由训练超参数决定效果，以及coeff压缩效果，和这个lut模式关系不大，保持lut即可）。
- `gpu_norm_enable`: GPU 准备 norm2，优化加速预准备项。
- `gpu_scan_enable`: 实验性 GPU local scan/top-k 路径，没有深度优化，只是测试性观察效果，不是主要功能。
- `gpu_scan_cache_mb`: GPU scan 路径的 cache/workspace 预算；只有 `gpu_scan_enable=1` 时才有实际意义。
- `norm2_store_*`: eval 后保存 norm2 cache，后续 eval 可复用。
- `preload_clusters_io_threads`:
  - `<0`: lazy on-demand （cpu流水线不满载，稍慢）
  - `0`: 自动线程数 preload
  - `>0`: 指定 preload 线程数，通常提高 eval 吞吐，但会增加内存占用。
- `louds_huffman_bench_times`: 首个 query block 上测 LOUDS/Huffman 解码成本。
- `archive_eval_result`: 保存 recall/timing 到 `<run_root>/eval_result/`。

### 8.6 Benchmark repeat

```ini
eval.base.warmup = 0
eval.base.repeat = 1
eval.linkage.warmup = 0
eval.linkage.repeat = 1
eval.bench.quiet = 0
```

- `warmup`: 预热，不计入最终统计。
- `repeat`: 重复次数，会输出 median/avg QPS。
- `bench.quiet=1`: 减少详细日志，偏 QPS 输出。

性能测试建议设置：

```ini
eval.linkage.warmup = 1
eval.linkage.repeat = 3
eval.bench.quiet = 1
```

## 9. 常见运行配方

### 9.1 完整 SIFT1M large 流程

```bash
./cmake-build-release/stlq_main  --config configs/basic_sift1m_linux.cfg 
```
可额外--set需要修改的参数，也可直接修改cfg

会执行：

```text
train init
global train iterations
base_basic
base_list
linkage_list
linkage disk eval
```

### 9.2 只做 init/base smoke

```bash
./cmake-build-release/stlq_main \
  --config configs/basic_sift1m_linux.cfg \
  --set large.output_dir=cmake-build-release/outputs_smoke \
  --set large.tmp_dir=cmake-build-release/tmp_smoke \
  --set train.max_R_iters=0 \
  --set base.linkage.enabled=0 \
  --set eval.base.enabled=0 \
  --set eval.linkage.enabled=0
```

用途：

- 验证环境、CUDA、BLAS、可选 HDF5、数据路径
- 不构建 expensive `linkage_list`

### 9.3 只 eval 已有 linkage_list

```bash
./cmake-build-release/stlq_main \
  --config configs/basic_sift1m_linux.cfg \
  --set advanced.eval_only=1 \
  --set eval.base.enabled=0 \
  --set eval.linkage.enabled=1 \
  --set eval.linkage.coeff_mode=int8
```

要求：

- cfg 指向相同 run root / artifact
- train result 可加载
- `linkage_list/` 存在
- 若 eval int8，coeff codec 存在或可从 float coeff 重建

### 9.4 对比 coeff float / int8

```bash
./cmake-build-release/stlq_main \
  --config configs/basic_sift1m_linux.cfg \
  --set advanced.eval_only=1 \
  --set eval.linkage.coeff_mode=both
```

会分别归档 `float` 和 `int8` eval 结果。

### 9.5 HNSW probe 网格 eval

```bash
./cmake-build-release/stlq_main \
  --config configs/basic_sift1m_linux.cfg \
  --set advanced.eval_only=1 \
  --set eval.linkage.ivf_probe_mode=hnsw \
  --set 'eval.linkage.nprobe=[16,32,64]' \
  --set 'eval.linkage.ivf_hnsw_ef_search=[32,64,128]' \
  --set eval.linkage.repeat=3 \
  --set eval.bench.quiet=1
```

输出会包含 batch summary 和 Pareto summary。

## 10. 结果与 artifact 位置

large 模式下常见目录：

```text
<large.output_dir>/<run-tag>/
  config_snapshot.txt
  original_config/
  store_hashes.txt
  base_basic/
  base_list/
  linkage_list/
  eval_result/
  log/
```

常见 artifact：

- `base_basic/`: base 基础编码，按 shard/bucket 存。
- `base_list/`: list-order store，供 linkage build 和 disk base eval 使用。
- `linkage_list/`: linkage graph、coeff、LOUDS、codec 等。
- `eval_result/`: recall / metrics 归档。
- `store_hashes.txt`: store identity，用于判断能否复用。

### 10.1 run 目录 hash 后缀从哪里来

large 模式的目录通常长这样：

```text
outputs_opt/SIFT1M/sift1m_trans_m5__0x<hash>/
```

这里的 `<hash>` 是 large run identity，目前等于最终 `linkage_list_hash_identity`。它不是简单的 cfg 文本 hash，而是由产物语义逐级推出来：

```text
base_basic_hash
  -> base_list_hash_identity
  -> linkage_list_hash_identity
  -> run 目录后缀
```

其中：

- `base_basic_hash`: 受训练产物 `R/C_root`、base encode 参数、base 数据规模/类型、store layout 影响。
- `base_list_hash_identity`: 受 `base_basic_hash`、`nlist/ntotal`、list-order store layout 影响。
- `linkage_list_hash_identity`: 受训练产物 `R/C_root/C_one`、`base_list_hash_identity`、`base.linkage.*`、`virtual.*`、`hnsw.*` 影响。
- `coeff_codec_hash`: 只用于校验 linkage coeff codec，依赖 `linkage_list_hash_identity` 和 `large.linkage_coeff_codec.*`，不是 run 目录后缀。
- `eval_result/recall_*_0x<hash>.txt`: 这是 eval archive hash，只表示 eval/report 设置，不表示 base/linkage 产物身份。

如果目录后缀变了，先看：

```text
<run_root>/store_hashes.txt
```

从 `base_basic_hash`、`base_list_hash_identity`、`linkage_list_hash_identity` 往下比。更完整的 hash 分类见 README 的 “Store Hashing / Reuse” 和 “Hash Taxonomy”。

如果 `large.cleanup.enabled=1`，部分中间文件会在安全阶段被删除。删除策略由 `large.cleanup.preset` 和 override 控制。

## 11. 结果复现与微小差异

这些配置通常属于**超参数**，会明显改变结果：

- `model.m`, `model.h0`, `model.h0_one`
- `train.max_R_iters`, `train.kmeans_*`
- `train.linkage.*`, `base.linkage.*`
- `base.encode.*`
- `virtual.*`
- `hnsw.*`
- `large.linkage_coeff_codec.*`
- `eval.linkage.coeff_mode`, `eval.linkage.ivf_probe_mode`

这些配置主要属于**性能/调度参数**，理论上不改变语义，但可能引入微小浮点差异：

- `runtime.omp_threads`
- `runtime.cuda_mode`, `runtime.cuda_allow_tf32`
- `runtime.cuda_pool_size*`
- `runtime.cuda_linkage_*` batching/chunking/window/tiny-CPU 参数
- `runtime.linkage_async_io*`
- `runtime.basic_async_io*`
- `runtime.basic_hybrid_*`
- `large.train_block`, `large.base_block`
- `eval.*.warmup`, `eval.*.repeat`, `eval.bench.quiet`
- `eval.linkage.gpu_norm_enable`, `eval.linkage.gpu_scan_enable`

如果要做严格基准对比：

- 固定 cfg 文件和 `--set`
- 固定 CUDA mode
- 固定线程数
- 固定 build type 和 CUDA architecture
- 保留 `config_snapshot.txt`
- 比较 `store_hashes.txt`、manifest、eval archive hash

## 12. 排错清单

### CMake 找不到 BLAS

如果你想用 MKL，确认：

```bash
-DSTLQ_USE_MKL=ON
-DMKLROOT=/path/to/mkl/latest
```

如果你想用 OpenBLAS，确认：

```bash
-DSTLQ_USE_MKL=OFF
-DSTLQ_OPENBLAS_ROOT=/path/to/OpenBLAS
```

### CMake 找不到 CUDA / cl.exe

Windows 上从 VS Native Tools Prompt 启动；Linux 上显式传：

```bash
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### 运行时说没有 CUDA

确认两层开关：

```text
CMake: -DSTLQ_ENABLE_CUDA=ON
cfg:   runtime.use_cuda = 1
```

### 找不到数据文件

先检查：

```ini
dataset.data_root = ...
dataset.name = ...
dataset.train_path/base_path/query_path/groundtruth_path
```

SIFT1M 默认要求 `dataset.data_root/sift/*.fvecs|*.ivecs`。

### Eval-only 找不到 artifact

`advanced.eval_only=1` 不会重新训练或重建 base/linkage。它要求已有 artifact 可用，并且 cfg 中的 run identity 与 artifact 匹配。

优先检查：

- `large.output_dir`
- `io.pre_fix`
- `io.load_date`, `io.load_seq`
- `io.train_file`
- `io.linkage_file`
- `<run_root>/linkage_list/meta.bin`
- `<run_root>/store_hashes.txt`

### 显存不足

优先降低：

```ini
runtime.cuda_pool_size
runtime.cuda_pool_size_init_linkage
runtime.cuda_linkage_many_nodes_max_pairs
runtime.cuda_linkage_many_nodes_max_pairs_init_linkage
runtime.cuda_linkage_mem_budget_mb
runtime.cuda_linkage_large_root_xc0_chunk_mb
large.train_block
large.base_block
```

也可以先关掉某些并发：

```ini
runtime.linkage_async_io = false
```
