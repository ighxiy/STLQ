# Master Query-owner Top-k Finalize QPS（2026-09-14）

## 结论

Master 的 CPU 生产 int8 coefficient-codec 扫描已把 Top-k finalize 固定为 query-owner 执行；不增加配置开关。`parent_louds_native_eval=true` 的原生 LOUDS 扫描也采用同一固定语义。

修改不改变候选、距离、tie 或输出排序规则，只把每个 query 的私有 heap 从“并行扫描后交给单线程统一 finalize”改为“仍由该 query 的 OpenMP owner finalize”。持久率和索引内存均不变。

Release+CUDA+LTO 的同机 A/B/A 结果表明：

- M5 materialized-parent，nprobe=8：55,515.1 / 67,261.9 / 54,543.8 QPS，owner 相对两条 serial 均值为 1.2223x。
- M5 LOUDS-native，nprobe=8：28,273.4 / 31,605.5 / 28,167.4 QPS，owner 为 1.1200x。
- M10 materialized-parent，nprobe=8：40,284.8 / 46,564.5 / 39,576.9 QPS，owner 为 1.1661x。
- 所有 A/B/A 腿的完整已输出 Recall 点完全相同；M5 的 materialized 与 LOUDS-native 在 nprobe=8、24 的 Recall 也完全相同。

LSQ++ 对照已经纠正并同时覆盖历史使用的 rate-aligned M9：

- 原报告误把 `m8_scan_kernel=0` 的通用 M8 scanner 写成历史 M8 baseline。历史 M8 baseline 实际为 `m8_scan_kernel=1`、`dev_dense_scan_variant=baseline`；此前“p1、p2 均略慢”的结论作废。
- 正确 M8 baseline 下，Master M5 在 p1 快 2.41%，p2 慢 1.08%；其余 11 个测量点中 8 个更快，p10、p24、p32 分别慢 1.91%、1.44%、0.57%。两者已是同一性能量级，不应写成全曲线严格支配。
- M9 通用 baseline 下，Master 仅 p1 慢 4.53%；从 p2 到 p64 的所有共同测量点均更快。Master 的 R@1 全部更高，R@10 从 p8 起更高。
- M9 `dev_window16` 下 LSQ++ 全点更快；这是显式打开 LSQ++ 的 DEV fixed-M 扫描优化，不是原始 baseline。该端点用于解释历史 DEV 对照，不能冒充 owner-only 公平对照。

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

## Master M5 与 LSQ++ M8 历史 baseline 完整曲线

两边均为同机 Release+LTO、32 threads、相同 query block、同样的 warmup/repeat。LSQ++ 使用 M8 IVF256、历史固定 M8 scanner：`m8_scan_kernel=1`、`dev_dense_scan_variant=baseline`。`dev_window16` 未启用。

| nprobe | Master R@1 | Master R@10 | Master R@50 | Master R@100 | Master QPS | LSQ++ R@1 | LSQ++ R@10 | LSQ++ R@50 | LSQ++ R@100 | LSQ++ QPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.37 | 46.46 | 48.93 | 49.07 | 267,586.0 | 25.02 | 48.94 | 54.76 | 55.45 | 261,278.9 |
| 2 | 33.07 | 63.08 | 68.30 | 68.66 | 181,347.0 | 29.77 | 62.16 | 71.85 | 72.97 | 183,321.4 |
| 4 | 37.04 | 74.75 | 83.24 | 84.07 | 112,701.0 | 32.58 | 71.34 | 84.89 | 86.76 | 105,360.5 |
| 8 | 38.88 | 81.34 | 92.60 | 94.00 | 67,261.9 | 33.51 | 75.55 | 91.70 | 94.18 | 62,756.3 |
| 9 | 39.05 | 82.03 | 93.51 | 94.96 | 58,471.0 | 33.61 | 75.91 | 92.52 | 95.10 | 58,114.6 |
| 10 | 39.08 | 82.30 | 94.05 | 95.59 | 53,149.5 | 33.61 | 76.21 | 93.04 | 95.70 | 54,184.5 |
| 11 | 39.20 | 82.68 | 94.73 | 96.32 | 50,504.8 | 33.73 | 76.50 | 93.56 | 96.26 | 48,495.1 |
| 12 | 39.34 | 82.97 | 95.21 | 96.85 | 47,759.6 | 33.78 | 76.62 | 93.89 | 96.58 | 45,936.7 |
| 16 | 39.41 | 83.52 | 96.23 | 97.93 | 37,908.3 | 33.76 | 76.96 | 94.44 | 97.30 | 36,466.1 |
| 24 | 39.45 | 83.86 | 96.85 | 98.62 | 25,288.0 | 33.80 | 77.13 | 94.81 | 97.79 | 25,657.1 |
| 32 | 39.45 | 83.99 | 97.00 | 98.85 | 19,947.8 | 33.80 | 77.19 | 94.98 | 97.99 | 20,061.5 |
| 48 | 39.45 | 84.02 | 97.13 | 98.98 | 14,127.7 | 33.81 | 77.20 | 95.07 | 98.10 | 14,115.5 |
| 64 | 39.44 | 84.02 | 97.15 | 99.01 | 10,925.4 | 33.81 | 77.20 | 95.07 | 98.10 | 10,329.6 |

错误的 `m8_scan_kernel=0` 控制仍保留为 `logs/lsqpp_m8_generic_m8off_control_curve.log`，只作通用 scanner 消融，不再称为 LSQ++ M8 baseline。正确日志为 `logs/lsqpp_m8_baseline_curve.log`。

## Master M5 与 rate-aligned LSQ++ M9

M9 使用同一 IVF256 冻结产物。两条 LSQ++ 腿只改变 `dev_dense_scan_variant`；两者所有 probe 的 Recall 和 result hash 完全相同。`baseline` 是 M9 通用 row scanner，`dev_window16` 是后来迁入 LSQ++、由开关控制的 fixed-M DEV scanner。

| nprobe | Master M5 Recall R@1/R@10/R@50/R@100 | Master owner QPS | LSQ++ M9 Recall R@1/R@10/R@50/R@100 | M9 baseline QPS | M9 `dev_window16` QPS |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | 26.37/46.46/48.93/49.07 | 267,586.0 | 26.30/50.82/55.13/55.65 | 280,285.2 | 424,279.4 |
| 2 | 33.07/63.08/68.30/68.66 | 181,347.0 | 31.66/64.71/72.37/73.28 | 167,991.5 | 302,467.4 |
| 4 | 37.04/74.75/83.24/84.07 | 112,701.0 | 35.09/74.96/85.86/87.27 | 105,208.7 | 216,713.9 |
| 8 | 38.88/81.34/92.60/94.00 | 67,261.9 | 36.42/79.63/92.92/94.90 | 59,418.4 | 133,070.6 |
| 9 | 39.05/82.03/93.51/94.96 | 58,471.0 | 36.59/80.13/93.83/95.86 | 52,010.0 | 121,397.9 |
| 10 | 39.08/82.30/94.05/95.59 | 53,149.5 | 36.63/80.40/94.37/96.46 | 48,925.8 | 117,587.9 |
| 11 | 39.20/82.68/94.73/96.32 | 50,504.8 | 36.68/80.72/94.85/97.04 | 45,299.9 | 109,392.0 |
| 12 | 39.34/82.97/95.21/96.85 | 47,759.6 | 36.75/80.90/95.18/97.39 | 41,023.8 | 100,416.8 |
| 16 | 39.41/83.52/96.23/97.93 | 37,908.3 | 36.75/81.27/95.82/98.10 | 30,628.4 | 80,926.4 |
| 24 | 39.45/83.86/96.85/98.62 | 25,288.0 | 36.79/81.54/96.25/98.58 | 21,644.1 | 58,883.2 |
| 32 | 39.45/83.99/97.00/98.85 | 19,947.8 | 36.83/81.64/96.40/98.80 | 16,907.2 | 45,241.9 |
| 48 | 39.45/84.02/97.13/98.98 | 14,127.7 | 36.83/81.71/96.49/98.89 | 11,697.2 | 30,610.8 |
| 64 | 39.44/84.02/97.15/99.01 | 10,925.4 | 36.82/81.71/96.49/98.89 | 8,713.7 | 24,050.9 |

当前 owner-only Master 对 M9 baseline 的准确表述是：p1 QPS 低 4.53%，p2 及以后全点更快；p8 为 1.1320x，p24 为 1.1684x。它不支配 M9 `dev_window16`，因为后者已打开另一项大幅扫描优化。

## 与历史 DEV 记录的关系

- `QUERY_OWNER_TOPK_FINALIZE_20260828.md` 的通用 master-codec owner-only 实验只测了 p8、p24，没有给出 p1、p2，也没有证明 owner 单项在全低 probe 支配 M9。
- `M5_CAUSAL_DENSE_SORTED_ROUTE_SCAN_20260827.md` 中 p1 从 196,469 到 375,999 QPS 的 owner A/B，底层已经是 `causal_dense_sorted` scanner；owner 是该 A/B 的唯一变化，但绝对端点不是原生 Master scanner 加 owner。
- `M5_M10_FIXED_CODEC_AOSOA16_DENSE_SCAN_20260828.md` 的最终 rate-aligned 对手确实是 LSQ++ M9 `dev_window16`，但正式表使用 IVF1024，并且 DEV5 也使用完整 fixed-M root/linked AoSoA16 scanner。它验证的是双方优化端点，不是当前 IVF256 Master owner-only 对 M9 baseline。
- 最近 LSQ++ 提交 `cc75297917360ea6c288be405bc3fc6a48b050ed` 的主要 fixed-M 改动受 `dev_dense_scan_variant=dev_window16` 控制；本报告的 M8/M9 baseline 均关闭它。该提交另有一个无条件、语义不变的 IVF 选择暂存复用（`static thread_local`），两条当前 M9 腿共同包含，不能解释二者约 2 倍的 scanner 差距。

## 论文副本验证

论文副本的同源 Release+CUDA+LTO 可执行文件复用同一 M5 冻结索引：

| path | nprobe | repeat | median QPS | Recall R@1/R@10/R@50/R@100 |
| --- | ---: | ---: | ---: | --- |
| materialized | 8 | 3 | 65,491.5 | 38.88/81.34/92.60/94.00 |
| LOUDS-native | 8 | 3 | 31,510.5 | 38.88/81.34/92.60/94.00 |

这证明论文副本不只完成编译，也实际进入了两条修改后的扫描路径。
