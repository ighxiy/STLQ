# Master Query-owner Top-k Finalize QPS（2026-09-14）

## 结论

Master 的 CPU 生产 int8 coefficient-codec 扫描已把 Top-k finalize 固定为 query-owner 执行；不增加配置开关。`parent_louds_native_eval=true` 的原生 LOUDS 扫描也采用同一固定语义。

修改不改变候选、距离、tie 或输出排序规则，只把每个 query 的私有 heap 从“并行扫描后交给单线程统一 finalize”改为“仍由该 query 的 OpenMP owner finalize”。持久率和索引内存均不变。

Release+CUDA+LTO 的同机 A/B/A 结果表明：

- M5 materialized-parent，nprobe=8：55,515.1 / 67,261.9 / 54,543.8 QPS，owner 相对两条 serial 均值为 1.2223x。
- M5 LOUDS-native，nprobe=8：28,273.4 / 31,605.5 / 28,167.4 QPS，owner 为 1.1200x。
- M10 materialized-parent，nprobe=8：40,284.8 / 46,564.5 / 39,576.9 QPS，owner 为 1.1661x。
- 所有 A/B/A 腿的完整已输出 Recall 点完全相同；M5 的 materialized 与 LOUDS-native 在 nprobe=8、24 的 Recall 也完全相同。

在干净 `cc75297917360ea6c288be405bc3fc6a48b050ed` 的 LSQ++ M8、`dev_dense_scan_variant=baseline`、Release+LTO 对照中，Master M5 从 nprobe=4 到 64 的所有测量点均同时具有更高 QPS、R@1 和 R@10。nprobe=1、2 的 QPS 仍略低，不能写成全低-probe 支配。

## 修改边界

- `src/eval/recall_linkage_disk.cpp`：`RunScanTopK_Int8CoeffCpuAllTasks` 中，每个 OpenMP query iteration 在扫描完 probe 后立即 finalize 自己的 heap；删除并行区后的串行 finalize loop。
- `src/eval/recall_linkage_disk_parent_louds_native.cpp`：原生 LOUDS 的 query iteration 同样持有 heap 直到 finalize；删除串行尾部。
- 不增加 runtime/config 开关；这是 Master 的固定生产行为。
- GPU scan 的已有并行 merge/finalize 路径未改。
- materialized-parent 的 float-coefficient 诊断扫描不属于本次从 DEV 迁移的 production int8 特性，未扩展其实现边界；LOUDS-native 的共享循环天然覆盖其 int8/float 两个分支。

## 构建与控制变量

目标仓库：

- Master 工作树基线提交：`6ec98190d769bc26a5d3e3d7eef93d28c3300a13`。
- 论文副本基线提交：`f7592253d160833a95aeccc0c772823716579842`。
- 两个目标仓库修改后的两份扫描源码逐字节相同。

构建均为：

```bash
cmake -S . -B <build> \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTLQ_ENABLE_CUDA=ON \
  -DSTLQ_ENABLE_LTO=ON \
  -DSTLQ_USE_MKL=ON
cmake --build <build> --target stlq_main -j 16
```

二进制 SHA-256：

| binary | SHA-256 |
| --- | --- |
| Master owner | `477343e5738a50c411121ea71360eaae5fa719a621b4575f7a767913a877db3c` |
| Master serial control | `7f627e0a00c66f81e650f44802622a1a0dd659f465768105ce82d04e56f33d9b` |
| Paper owner | `f5aba064edf7d065c0f127d888dd362c00f8a84de8fba99d461e2dc679ee8a03` |
| Clean LSQ++ baseline | `6896c851921065d67088a9b7204201fdfae5c212a65ec720cd1cca21b7f5064a` |

评估固定条件：

- SIFT1M 官方 10,000 queries，Top-100；
- 32 个物理 core，`OMP_NUM_THREADS=32`、`OMP_PLACES=cores`、`OMP_PROC_BIND=spread`；
- `numactl --physcpubind=0-31 --interleave=all`；
- MKL/OpenBLAS 单线程；
- exact IVF256，`query_block=1000`，CPU scan，production int8 coeff + norm2 LUT；
- 每点 1 次 warmup、5 次计时，表中使用 median core QPS；
- 复用冻结 Master M5/M10 索引，没有重新训练或编码；
- 未运行 CTest。

冻结产物和全部日志仅位于：

```text
cmake-build-release/query_owner_finalize_20260914/
```

## A/B/A 原始结果

顺序均为 serial A / owner B / serial A2。

| 模型与扫描路径 | nprobe | serial A QPS | owner QPS | serial A2 QPS | owner / serial均值 | Recall R@1/R@10/R@50/R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M5 materialized | 8 | 55,515.1 | 67,261.9 | 54,543.8 | 1.2223x | 38.88/81.34/92.60/94.00 |
| M5 materialized | 24 | 24,005.2 | 25,288.0 | 24,689.1 | 1.0386x | 39.45/83.86/96.85/98.62 |
| M5 LOUDS-native | 8 | 28,273.4 | 31,605.5 | 28,167.4 | 1.1200x | 38.88/81.34/92.60/94.00 |
| M5 LOUDS-native | 24 | 11,021.0 | 11,556.4 | 11,078.8 | 1.0458x | 39.45/83.86/96.85/98.62 |
| M10 materialized | 8 | 40,284.8 | 46,564.5 | 39,576.9 | 1.1661x | 57.63/91.79/93.98/94.03 |

LOUDS-native 自身仍显著慢于 materialized-parent；本特性只移除其串行 finalize，不把 LOUDS decode/parent-access 的其他成本伪装成本次收益。

## Master M5 与 LSQ++ baseline 完整曲线

两边均为同机 Release+LTO、32 threads、相同 query block、同样的 warmup/repeat。LSQ++ 使用 M8 IVF256、generic `baseline` scanner，不启用 `dev_window16`。

| nprobe | Master R@1 | Master R@10 | Master R@50 | Master R@100 | Master QPS | LSQ++ R@1 | LSQ++ R@10 | LSQ++ R@50 | LSQ++ R@100 | LSQ++ QPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.37 | 46.46 | 48.93 | 49.07 | 267,586.0 | 25.02 | 48.94 | 54.76 | 55.45 | 286,214.3 |
| 2 | 33.07 | 63.08 | 68.30 | 68.66 | 181,347.0 | 29.77 | 62.16 | 71.85 | 72.97 | 182,215.0 |
| 4 | 37.04 | 74.75 | 83.24 | 84.07 | 112,701.0 | 32.58 | 71.34 | 84.89 | 86.76 | 107,746.9 |
| 8 | 38.88 | 81.34 | 92.60 | 94.00 | 67,261.9 | 33.51 | 75.55 | 91.70 | 94.18 | 61,297.7 |
| 9 | 39.05 | 82.03 | 93.51 | 94.96 | 58,471.0 | 33.61 | 75.91 | 92.52 | 95.10 | 56,211.3 |
| 10 | 39.08 | 82.30 | 94.05 | 95.59 | 53,149.5 | 33.61 | 76.21 | 93.04 | 95.70 | 49,878.4 |
| 11 | 39.20 | 82.68 | 94.73 | 96.32 | 50,504.8 | 33.73 | 76.50 | 93.56 | 96.26 | 46,835.8 |
| 12 | 39.34 | 82.97 | 95.21 | 96.85 | 47,759.6 | 33.78 | 76.62 | 93.89 | 96.58 | 43,896.8 |
| 16 | 39.41 | 83.52 | 96.23 | 97.93 | 37,908.3 | 33.76 | 76.96 | 94.44 | 97.30 | 34,027.1 |
| 24 | 39.45 | 83.86 | 96.85 | 98.62 | 25,288.0 | 33.80 | 77.13 | 94.81 | 97.79 | 23,080.6 |
| 32 | 39.45 | 83.99 | 97.00 | 98.85 | 19,947.8 | 33.80 | 77.19 | 94.98 | 97.99 | 17,764.9 |
| 48 | 39.45 | 84.02 | 97.13 | 98.98 | 14,127.7 | 33.81 | 77.20 | 95.07 | 98.10 | 12,255.0 |
| 64 | 39.44 | 84.02 | 97.15 | 99.01 | 10,925.4 | 33.81 | 77.20 | 95.07 | 98.10 | 9,242.5 |

## 论文副本验证

论文副本的同源 Release+CUDA+LTO 可执行文件复用同一 M5 冻结索引：

| path | nprobe | repeat | median QPS | Recall R@1/R@10/R@50/R@100 |
| --- | ---: | ---: | ---: | --- |
| materialized | 8 | 3 | 65,491.5 | 38.88/81.34/92.60/94.00 |
| LOUDS-native | 8 | 3 | 31,510.5 | 38.88/81.34/92.60/94.00 |

这证明论文副本不只完成编译，也实际进入了两条修改后的扫描路径。
