# STLQ

CPU-first C++ implementation of STLQ with pluggable CUDA kernels, designed for billion-scale vector quantization.

## Features

- **Hybrid CPU/GPU Architecture**: Critical compute paths (k-means, encoding) accelerated by CUDA (cuBLAS/TF32).
- **Large-Scale Streaming**: Out-of-core pipeline supports large datasets with CUDA tuning.
- **Host RAM Caching**: Large training subsets can be cached in host RAM to reduce repeated disk I/O.
- **Hierarchical Initialization**: RVQ-like initialization with spherical k-means.
- **Advanced Quantization**: Beam search, ICM optimization, and linkage quantization.

#### `runtime.cuda_mode` — Reproducibility vs Speed

| Value | Effect |
|-------|--------|
| `"strict"` | TF32 **off** by default; cuBLAS atomics disabled (`CUBLAS_ATOMICS_NOT_ALLOWED`). Maximizes numerical reproducibility. GPU GEMM path is still used (no CPU fallback). |
| `"fast"` | TF32 **on** (Ampere+: ~1.5–2× GEMM throughput); cuBLAS atomics **allowed** (`CUBLAS_ATOMICS_ALLOWED`). Run-to-run results may vary by small floating-point deltas. |

**Default**: `"fast"`.

#### `runtime.cuda_allow_tf32` — Fine-grained TF32 override

| Value | Effect |
|-------|--------|
| `false` (default) | TF32 off when `cuda_mode="strict"`. |
| `true` | TF32 on when `cuda_mode="strict"` (useful for reproducibility experiments where you still want TF32 speed). |

**Interaction with `cuda_mode`**:
- `cuda_mode="fast"` **unconditionally enables TF32** and cuBLAS atomics, regardless of `cuda_allow_tf32`.
- `cuda_mode="strict"` uses `cuda_allow_tf32` to decide whether TF32 is enabled.
  Atomics are always disabled in strict mode.

In summary: `fast` ≠ just TF32. It also unlocks non-deterministic GEMM algorithms (`CUBLAS_ATOMICS_ALLOWED`), which `strict + cuda_allow_tf32=true` does **not** enable.

- `train.kmeans_device_cache_mb`: If GPU VRAM is large enough, this can cache the normalized training block directly on GPU (for example, `2048` MB).

## License

STLQ is distributed under the BSD 3-Clause License. See [`LICENSE`](LICENSE).
Third-party components keep their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## AI Assistance Disclosure

STLQ includes human-authored code developed with AI assistance. AI tools are
acknowledged for transparency and are not listed as copyright holders in the
license text.

Estimated share of AI-assisted contribution:

- OpenAI GPT: 90%
- DeepSeek: 7.5%
- GitHub Copilot: 2.5%

## Library Facade

Applications can use STLQ through the stable facade header:

### CLI-compatible entry

Use this when the embedding application wants exactly the same behavior as
`stlq_main`: `--config`, `--set`, config snapshots, stage switches, eval-only,
and logging all follow the command-line path.

```cpp
#include "stlq/stlq.h"

int main(int argc, char** argv) {
    return stlq::RunStlq(argc, argv);
}
```

The overload can also return the effective normalized config:

```cpp
stlq::StlqRunResult result;
const int rc = stlq::RunStlq(argc, argv, &result);
```

### Config-owned entry

Use this when the caller has already parsed or constructed `stlq::Config`.
The config is copied intentionally, so `result.effective_config` records defaults
and eval-only overrides without mutating the caller's object.

```cpp
stlq::Config cfg = stlq::DefaultConfig(true);

stlq::StlqRunOptions options;
options.dump_config_path = "outputs/config_snapshot.txt";

stlq::StlqRunResult result;
const int rc = stlq::RunStlq(cfg, options, &result);
```

### Stage-oriented request

Use `StlqRunRequest` when the caller wants to select a top-level pipeline mode
without manually editing all train/base/linkage/eval switches.

```cpp
stlq::StlqRunRequest req;
req.config = stlq::DefaultConfig(true);
req.mode = stlq::StlqRunMode::kEvalOnly;

stlq::StlqRunResult result;
const int rc = stlq::RunStlq(req, &result);
```

`StlqRunMode` values:

| Mode | Meaning |
|------|---------|
| `kUseConfig` | Preserve the stage switches already present in `Config`. |
| `kFullPipeline` | Enable train, base encode, and linkage build; eval switches remain config-owned. |
| `kTrainOnly` | Train/save/load only, then stop before base/linkage/eval stages. |
| `kBuildBaseOnly` | Load an existing train result and build base artifacts only. |
| `kBuildLinkageOnly` | Load existing train/base artifacts and build linkage artifacts only. |
| `kEvalOnly` | Force the same semantics as `advanced.eval_only=true`. |

`StlqRunOptions` fields:

| Field | Purpose |
|-------|---------|
| `dump_config_path` | Optional path for writing the effective config snapshot. |
| `frozen_original_config_text` | Preserves the original config text for reproducible snapshots/hashes. |
| `frozen_original_config_name` | Records the original config source name. |
| `linkage_nprobe_batch` | Optional batch override for linkage eval `nprobe` sweeps. |
| `linkage_ef_search_batch` | Optional batch override for linkage eval `efSearch` sweeps. |

`StlqRunResult` contains:

| Field | Purpose |
|-------|---------|
| `exit_code` | The returned process-style status code. |
| `error` | Error text when the facade catches a failure path. |
| `effective_config` | The config after defaults, request mode, and eval-only normalization. |

The facade is a thin orchestration layer over the same staged train/base/linkage/eval
pipeline used by `stlq_main`; it does not reimplement algorithm logic.
Internal HNSW implementation headers and optional HDF5 C-library headers are kept
out of the public include surface; downstream users should include STLQ headers,
not `third_party/hnswlib` or `<hdf5.h>` directly.
General pipeline/orchestration headers also avoid pulling CUDA provider headers;
CUDA runtime/cuBLAS types are limited to dedicated CUDA module headers.
That boundary is intentional: the stable library facade is CUDA-agnostic, while
low-level CUDA scheduler/encoder modules may still expose CUDA handles internally
where doing so preserves the existing high-performance execution path. A future
SDK-grade CUDA API should be designed as a separate opaque-handle layer instead
of mechanically rewriting the current scheduler hot path.

## Build

For a first-time setup and practical config walkthrough, start with
[`GETTING_STARTED_GUIDE.md`](GETTING_STARTED_GUIDE.md).
The guide focuses on environment setup, dataset paths, `basic_sift1m_linux.cfg`,
large/non-large workflow, and common train/eval recipes.

### Requirements
- CMake 3.16+
- C++20 compiler:
  - GCC 9+
  - MSVC 19.29+
- CUDA Toolkit 11.0+ (optional, for GPU support)
- Intel MKL or OpenBLAS

## Compilation Guide

This project supports CPU-only builds and CUDA builds from the same `CMakeLists.txt`.

Developer-only checks and probes live under `tests/` and `tests/checks/`.
They are not built by default; enable them with `-DBUILD_TESTING=ON`.
User-facing command-line tools live under `cmd/` and remain part of the normal
build.

### Common CMake Options

| Option | Default | Meaning |
|---|---:|---|
| `STLQ_ENABLE_CUDA` | `ON` | Build CUDA kernels and GPU paths |
| `STLQ_ENABLE_HDF5` | `ON` | Enable legacy HDF5 result IO compatibility |
| `STLQ_USE_MKL` | `ON` on Linux, `OFF` on Windows | Use MKL instead of OpenBLAS |
| `STLQ_ENABLE_LTO` | `ON` | Enable link-time optimization in `Release`, `RelWithDebInfo`, `MinSizeRel` |
| `STLQ_CUDA_ARCHITECTURES` | `89` | CUDA SM target(s), e.g. `89` for RTX 4090, `86` for RTX 3090. Synced to `CMAKE_CUDA_ARCHITECTURES`. |

### What `STLQ_ENABLE_LTO` Does

`STLQ_ENABLE_LTO=ON` enables release-build whole-program optimization:

- GCC host code: `-flto`
- MSVC host code: `/GL` + `/LTCG`

It is enabled by default for:

- `Release`
- `RelWithDebInfo`
- `MinSizeRel`

It is intentionally not applied to Debug builds.

Practical effect:

- usually reduces executable size somewhat
- sometimes improves runtime by allowing cross-translation-unit inlining / dead-code elimination
- often increases link time noticeably

Current note:

- `STLQ_ENABLE_LTO` controls host C/C++ LTO only.
- CUDA device LTO is intentionally not provided because it slightly hurt both binary size and runtime in project testing.

### Standard Linux Build

CPU or GPU build with Release config:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTLQ_USE_MKL=ON
cmake --build build -j
```

### CUDA Build on Linux

Example for RTX 4090 / SM89:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTLQ_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSTLQ_CUDA_ARCHITECTURES=89
cmake --build build -j
```

If CMake cannot find `nvcc`, set `CUDACXX` or pass `-DCMAKE_CUDA_COMPILER=...`.

### Windows Build Notes

- CUDA on Windows normally requires the MSVC host toolchain expected by your CUDA Toolkit version.
- `STLQ_USE_MKL` defaults to `OFF` on Windows to avoid toolchain/link surprises; enable it explicitly when oneAPI MKL is installed and initialized.
- HDF5 is optional. If it is not installed, configure with `-DSTLQ_ENABLE_HDF5=OFF`.
- The command examples below assume the current directory is the parent directory that contains `STLQ\`.
  If you are already inside the repository, use `-S . -B cmake-build-win` and build `cmake-build-win` directly.

Recommended dependencies:

- Visual Studio 2022 Build Tools compatible with your CUDA Toolkit. If you need a specific/current VS 2022 release,
  use Microsoft's release-history installer page:
  <https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#updating-your-installation-to-a-specific-release>.
  In the installer, select:
  - MSVC v143 x64/x86 build tools
  - Windows 10/11 SDK
  - C++ CMake tools for Windows
- NVIDIA CUDA Toolkit.
- Intel oneAPI MKL or OpenBLAS.
- HDF5, only if `STLQ_ENABLE_HDF5=ON`.

Example configure command for Windows CMD, Ninja, oneAPI MKL, optional HDF5, and an RTX 3090 (`sm_86`):

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

Build:

```bat
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" ^
  --build cmake-build-win -j 18
```

Add `--clean-first` to the build command when you need a clean rebuild.

Run:

```bat
cd cmake-build-win
set "PATH=C:\Program Files\HDF_Group\HDF5\2.0.0\bin;%PATH%"
stlq_main.exe --config ..\configs\basic_desktop_win.cfg
```

If HDF5 is disabled, omit the HDF5 `PATH` line and configure with `-DSTLQ_ENABLE_HDF5=OFF`.

### Known Windows CUDA Limitation

Windows builds are supported, including CUDA builds. However, large linkage build can issue many small GPU tasks
when cluster-local candidate batches are fragmented. Based on the observed behavior on a `consumer` 3090 GPU setup
and the Windows display-driver execution model, these small-task launches may have much higher latency than the same
path on Linux. We have not validated this with a `professional` NVIDIA card or a compute-driver/TCC-style setup, so the
limitation should be treated as an inferred environment issue rather than a resolved STLQ kernel issue. For serious
large-linkage benchmarking, Linux remains the recommended environment.

### Windows MinGW + OpenBLAS (MKL doesn't support MinGW; CPU-only example)

```bash
cmake -S . -B build \
  -DSTLQ_USE_MKL=OFF \
  -DSTLQ_ENABLE_CUDA=OFF
cmake --build build -j
```

### Size Reduction: `strip`

After a successful Release build on Linux, you can additionally run:

```bash
strip build/stlq_main
```

What `strip` does:

- removes symbol tables and other non-runtime metadata from the executable
- usually reduces on-disk file size
- often helps packaging / deployment size

What `strip` does **not** usually do:

- it does not materially improve numeric performance
- it does not reduce embedded CUDA fatbin device code itself
- it does not reduce runtime memory in a meaningful way for this project

Tradeoff:

- backtraces become less informative
- debugging and profiler symbolization become worse
- `gdb`, `addr2line`, and similar tools lose detail

Typical workflow:

- keep the unstripped binary for development
- strip only release artifacts intended for deployment / benchmarking distribution

## Usage

Run the main application using a config file:
```bash
./build/stlq_main --config configs/basic_sift1m_linux.cfg
```

Override config values via command line:
```bash
./build/stlq_main --config configs/basic_sift1m_linux.cfg --set dataset.ntrain=1000000 --set train.kmeans_iters=50
```

### Eval Metrics

STLQ supports three evaluation metric modes through `eval.metric_mode`:

- `0`: legacy top1 recall only, reported as `r@k`
- `1`: top1 recall + `top<g>_r@k'`
- `2`: top1 recall + `top<g>_r@k'` + `ndcg<g>@k'`

Use `eval.metric_gt_topks` to choose the GT set sizes `g`, for example:

```cfg
eval.metric_gt_topks = [10, 50]
```

Definitions are fixed and shared with `lsq_gpu_qps`:

- `r@k`
  - legacy top1 recall: whether the exact top1 GT id appears in the returned top-`k`
- `top<g>_r@k'`
  - let `G_g(q)` be the exact GT top-`g` set
  - let `A_k'(q)` be the returned top-`k'` set
  - per query: `|G_g(q) ∩ A_k'(q)| / g`
  - final score: average over queries
  - reports are only shown for `k' >= g`
- `ndcg<g>@k'`
  - standard binary-relevance NDCG evaluated against the exact GT top-`g` set
  - relevance is `1` iff a returned id belongs to `G_g(q)`, else `0`
  - `IDCG@k'` uses `min(g, k')` relevant items
  - when `IDCG@k = 0`, the query contribution is explicitly `0`
  - the common binary top-k set recall `top<g>_r@g` is the `k'=g` point
    of the `top<g>_r@k'` curve

Example:

```cfg
eval.metric_mode = 2
eval.metric_gt_topks = [10, 50]
dataset.k = 100
```

Console output keeps the old `r@k` lines and, for modes `1/2`, adds:

- `top10_r@10`, `top10_r@20`, ..., `top50_r@50`, ...
- `ndcg10@10`, `ndcg10@20`, ..., `ndcg50@50`, ...

Archive behavior:

- `eval.metric_mode = 0`
  - keeps the legacy archive file:
    - `eval_result/recall_<label>_0x....txt`
- `eval.metric_mode = 1/2`
  - writes a separate metrics archive:
    - `eval_result/metrics_<label>_0x....txt`

The new metrics archive does not replace the old recall archive naming, so older recall-only workflows remain compatible.

### Linkage Depth Report Tool

`report_linkage_depth` reads an existing `linkage_list` artifact and prints a depth-distribution report.
It does not rebuild linkages or touch eval hot paths.

Typical Windows usage from the project root:

```powershell
.\cmake-build-win\report_linkage_depth.exe `
  --out_dir .\cmake-build-win\outputs_opt\SIFT1M\test15_m5__0x3279c41f7c895b00 `
  --mean_bucket_width 0.1 `
  --topk 20 `
  --report .\cmake-build-win\outputs_opt\SIFT1M\test15_m5__0x3279c41f7c895b00\linkage_depth_report.txt
```

You can also point it directly at a `linkage_list` directory:

```powershell
.\cmake-build-win\report_linkage_depth.exe `
  --linkage_list_dir .\cmake-build-win\outputs_opt\SIFT1M\test15_m5__0x3279c41f7c895b00\linkage_list
```

The report includes:

- global node depth histogram (`node_depth_hist`)
- global mean/max depth
- histogram of per-cluster mean depth
- top clusters by `max_depth`
- top clusters by `mean_depth`

### Int8 Reconstruction Error Tool

`report_linkage_int8_recon_mse` reads an existing `linkage_list` + coeff codec store and computes
the full-database reconstruction error for the **int8 coeff codec** path.
It reconstructs each real base vector from the on-disk Huffman-decoded int8 coefficients and
compares it against the rotated raw base vector used by stlq.

Important:

- The primary output is **squared error per vector**:
  - `SE(x) = sum((x_hat - x)^2)` over all dimensions
- This tool no longer reports per-dimension averaged `MSE = SE / d`
- The summary file reports `mean_se`, `min_se`, `max_se`

Typical Linux usage from the project root:

```bash
./cmake-build-release/report_linkage_int8_recon_mse \
  --run_root ./cmake-build-release/outputs_opt/SIFT1M/sift1m_2_m5__0x9f68368d9ff63813 \
  --threads 32 \
  --block_cols 8192 \
  --use_cuda 0
```

If you only need the aggregate MSE/summary and do not want to keep the large per-gid file:

```bash
./cmake-build-release/report_linkage_int8_recon_mse \
  --run_root <run_root> \
  --threads 32 \
  --write_se 0
```

Outputs are written under `<run_root>/int8_recon_mse/`:

- `stlq_int8_recon_se_by_gid.f32`
  - length = `dataset.nbase`
  - indexed by global base id (`gid`)
  - value = full-vector squared error for that base vector
- `stlq_int8_recon_mse_summary.txt`
  - aggregate summary including `mean_se`, throughput, coverage, and timing breakdown

Notes on flags:

- `--block_cols` controls the base read/rotate block size (larger = fewer IO calls, more memory).
- `--fast_dp 1` is not recommended. It enables a faster path that first DP-reconstructs **f16** reconstructed vectors per `gid`
  into tmp recon buckets, then does one `dot(x, x_hat)` per base vector in Phase2.
  This typically speeds up Phase2 a lot, but writes large tmp files under `--tmp_dir`.

Notes:

- `--use_cuda 1` only accelerates the batch rotation of raw base vectors; coeff decode and linkage
  reconstruction remain on CPU.
- The tool resolves the train HDF5 in the same way as other analysis tools:
  it first prefers `run_state.txt: effective_train_h5`, then falls back to
  `config_snapshot.txt` `io.train_file + io.load_date/load_seq`.

### Cluster Angle Export Tool

`export_stlq_cluster_angles` exports per-vector angle pairs for one selected
cluster from an existing large STLQ run. It is intended for plotting whether
linkage reconstruction preserves each vector's angular position relative to the
first-layer IVF/root center.

Each CSV row corresponds to the same base vector:

- `raw_angle_deg`: angle between the raw base vector and the selected
  first-layer root center.
- `recon_angle_deg`: angle between that vector's STLQ linkage reconstruction
  and the same root center.

CSV schema:

```csv
raw_angle_deg,recon_angle_deg
```

Typical Linux usage from the build directory:

```bash
./export_stlq_cluster_angles \
  --run_root outputs_opt/SIFT1M/sift1m_trans_m5__0x4b881fa77f11a471 \
  --cluster_id 0 \
  --out cluster0_angles_stlq.csv
```

Notes:

- `--cluster_id` defaults to `0`.
- `--out` defaults to `<run_root>/analysis/cluster_<cid>_angles.csv`.
- Raw vectors are read from the configured base dataset and rotated by
  `train.R` before angle computation, because STLQ codebooks/linkage artifacts
  live in the rotated OPQ space.
- Reconstruction uses retained `linkage_list` int8 coefficient codec artifacts
  through the same coeff-codec/provider path used by runtime eval. This keeps
  the tool usable after cleanup presets remove float coefficient arrays.

### Query ADC Distance Export Tool

`export_stlq_query_adc` exports per-vector distance pairs for one selected
query and one selected first-layer cluster. It is intended for plotting how the
STLQ chain ADC estimate compares with the exact raw-vector distance to the same
query.

Each CSV row corresponds to the same base vector:

- `adc_distance`: STLQ chain ADC squared-distance estimate,
  `||q||^2 + norm_hat - 2 * dot_hat`.
- `true_distance`: exact squared L2 distance between the raw base vector and
  the query, after applying the same OPQ rotation used by STLQ artifacts.

CSV schema:

```csv
adc_distance,true_distance
```

Typical Linux usage from the build directory:

```bash
./export_stlq_query_adc \
  --run_root outputs_opt/SIFT1M/sift1m_trans_m5__0x4b881fa77f11a471 \
  --cluster_id 0 \
  --query_id 0 \
  --out query0_cluster0_adc_stlq.csv
```

Notes:

- `--cluster_id` and `--query_id` default to `0`.
- `--out` defaults to
  `<run_root>/analysis/query_<qid>_cluster_<cid>_adc.csv`.
- The ADC estimate uses the retained `linkage_list` coeff codec and the same
  runtime norm source used by eval (`r_norm2` or its LUT-backed form).
- Distances are exported as squared L2 values so they match the ranking formula
  used by ADC without applying a square root.

## Performance & Optimization

## Codebook Update (C_root vs C_one)

This project has **two distinct codebook-update mechanisms** in the large/streaming pipeline.
They solve different linear systems, and their **memory scaling** is very different.

### `C_root` update (large-root / streaming)

`C_root` is updated from the streaming **basic encode** store, and must handle a potentially huge
root layer (`h0_root` can be 65536). The implementation uses a **streaming Schur-elimination**
style update that avoids materializing a dense `(h0_root + H_small) × (h0_root + H_small)` system.

Implementation notes:
- Accumulates only the small-layer normal equations in a dense form.
- Uses **packed-upper** storage for symmetric matrices (saves ~2× memory vs full `H×H`).
- Stores `T` in a **transposed** layout during accumulation to reduce strided writes.

This is why `C_root` update can stay fast even when `h0_root` is very large.

### `C_one` update (exact joint LS)

`C_one` is updated from the streaming **linkage_list** store using an **exact joint least-squares**
solve over the flattened one-codebook size:

- `H_one = sum_l h_l(one)` (with `h_0(one)=model.h0_one`)
- Dense normal equation: `G ∈ R^{H_one×H_one}`, `T ∈ R^{H_one×d}`

This update is typically bottlenecked by the accumulation pattern:
“sparse per-sample features → scatter-add into a dense `G` and `T`”.

#### Memory model

With the same packed/transposed tricks as `C_root`:
- `G` stored as packed-upper doubles: `H_one*(H_one+1)/2`
- `T` stored as transposed doubles: `H_one*d`

So the **per-accumulator** memory is approximately:

`bytes_per_accum ≈ 8 * ( H_one*(H_one+1)/2  +  H_one*d )`

#### Sharded accumulation (new)

To avoid allocating one full accumulator per OpenMP thread (which would scale as
`O(omp_threads * H_one^2)` and can explode for `m=10`), `C_one` update uses a **sharded**
accumulator:

- Only `shards` accumulators are allocated.
- Clusters are parallelized across these `shards` workers.
- At the end, shard 1..(shards-1) are reduced into shard 0.

Config:
- `runtime.c_one_update_shards`
  - `0`: auto (conservative; currently `min(omp_max, 16)`).
  - `>0`: explicit number of shard workers (bounds RAM by `O(shards * H_one^2)`).
- `runtime.c_one_update_shards_init_linkage`
  - `0`: use `runtime.c_one_update_shards`
  - `>0`: override for init-stage `C_one` update only.

Tuning guidance:
- For `m=5` (moderate `H_one`): `shards ≈ omp_max` is often OK, but `8..16` is usually sufficient.
- For `m=10`: prefer `shards=8` or `16` to keep RAM bounded.

## Terminology: `large.enabled` vs “large-root”

This repo uses two similar-sounding concepts:

- **`large.enabled` (pipeline switch)**: enables the out-of-core / streaming pipeline in
  `src/quantizer/encoder_train_streaming.cpp` (`TrainQuantizerStreamingLarge`), used for
  very large `dataset.ntrain` / `dataset.nbase` (e.g. SIFT1B).
- **“large-root” (model shape)**: means the **root codebook is very large**, typically
  `model.h0 >= 4096` (equivalently, normalized `model.h_vec[0] >= 4096`; `h0_root` is large).
  This is orthogonal to `large.enabled`:
  you can have `large.enabled=true` with small root, or `large.enabled=false` with large root
  (though the non-large pipeline usually assumes the full training set fits in RAM).

When logs mention “large-root”, it is referring to the **`h0_root` size**, not `large.enabled`.

## Linkage List Stores: `code0_width_bytes`

`linkage_list` stores split codes on disk:
- small-layer codes (layers `1..m-1`) are stored as **u8**, with `small_code_width_bytes=1`
- layer0 codes for depth>0 nodes (“code0_one”) use `code0_width_bytes`

`LinkageListStore` supports `code0_width_bytes ∈ {1,2,4}`:
- `1` for `h0_root <= 256`
- `2` for `h0_root <= 65536`
- `4` for `h0_root > 65536`

End-to-end support notes:
- `code0_one` can be stored/loaded as `u32` when `code0_width_bytes=4`.
- Small-layer codes remain `u8` (`small_code_width_bytes=1`), and the legacy `FullCode` path stays <=16-bit for
  backward compatibility of on-disk layouts that pack per-layer codes.

### Feature Note: `h0_root > 65536` (U32 root routing)

When `model.h0` (root / IVF nlist; normalized `model.h_vec[0]`) exceeds `65536`,
**root-layer codes no longer fit in `u16`**.
This repo supports such models by **keeping small-layer codes unchanged** and storing only `code0_one` as `u32`
where needed via `code0_width_bytes=4`.

Support matrix (current behavior):

- **Large / streaming pipeline (`large.enabled=true`)**:
  - **GPU training + encoding**: supported (init-linkage / iteration-linkage / baseset linkage) via `code0_width_bytes=4`.
  - **Disk recall / evaluation**: supported (CPU lookup & GPU scan paths understand `code0_width_bytes=4`).
- **CPU-only training**:
  - Not a primary target for `h0_root > 65536`. The CPU module supports reading/evaluating `code0_width_bytes=4`,
    but CPU-only training/encoding paths may still assume `h0_root <= 65535`.
- **Non-large (“full-precomp” / legacy full-code) pipeline**:
  - **Not supported for `h0_root > 65535`**: legacy in-memory `FullCode` and some packed on-disk layouts assume
    the root-layer fits in `<=16-bit`. For `h0_root > 65536`, use the large/streaming + split-store routes above.

## Checkpoint/Resume (BaseSet `linkage_list` build)

The BaseSet linkage build (`linkage_list` store generation) supports **checkpoint + resume** to make it robust against
interruptions (Ctrl+C, power loss) and to avoid redoing already-finished IVF clusters when running in OpenMP
cluster-parallel mode.

### Enable

Config key:

```cfg
large.linkage_checkpoint = 1
```

Notes:
- This checkpoint is for the BaseSet `linkage_list` stage (where clusters can finish out-of-order).
  It is intentionally **not** used for training/iteration linkage stores, because their write plan/layout can change
  between iterations.
- Resume is triggered by the presence of checkpoint files in the `linkage_list` output directory.

### Checkpoint Files

These files live inside the `linkage_list` output directory:

- `ckpt_linkage_done.u8`
  - A **byte-per-cluster done bitmap** of length `nlist`.
  - `done[cid] = 1` means cluster `cid` has been successfully written and will be **skipped** on resume.

- `ckpt_linkage_stats_v1.bin`
  - A fixed-size **per-cluster stats table** (record `cid` is stored at a deterministic offset).
  - Used to restore progress/aggregate stats deterministically when resuming, without recomputing already-done clusters.

### Crash Safety / Commit Semantics

- A cluster is marked done **only after** its outputs are successfully written.
  - If the process is interrupted mid-cluster, that cluster remains `done=0` and will be recomputed on the next run.
- For the coefficient codec store, a crash can leave unused tail bytes in the payload file, but correctness is preserved:
  retrying the cluster overwrites the per-cluster offsets/sizes, so readers will not reference the stale tail.

### Resume Safety Guard

The driver validates that the existing output directory matches the current run signature (store hash). If the hash
does not match, resume is refused to avoid mixing incompatible layouts/configs. To force a clean rebuild, delete the
`linkage_list` output directory.

## Checkpoint/Resume (BaseSet `base_basic` encoding)

The BaseSet basic encoding stage (`base_basic` store generation) supports **checkpoint + resume** for robustness.
This is separate from the `linkage_list` checkpoint: `base_basic` writes sequential global IDs, so the checkpoint is
block-based rather than per-cluster.

### Enable

Config key:

```cfg
large.base_basic_checkpoint = 1
```

### Commit Semantics

- Checkpoints are committed **only at `large.base_block` boundaries**.
- Each commit happens only after the writer flushes data for the completed block.

### Resume Semantics

- Resume is triggered by the presence of the checkpoint file in the `base_basic` output directory.
- On resume, outputs are **truncated back** to the last committed `next_id` (including append-only `bucket_*.bin` files),
  then encoding continues from that `next_id`.
- The driver validates the `base_basic` store hash before resuming; if it mismatches, resume is refused.

### Checkpoint Files

These files live under the `base_basic` output directory:

- `checkpoint/base_basic/ckpt.bin`
  - Binary checkpoint record containing:
    - `next_id`: next global base ID to encode
    - per-bucket committed byte sizes (used to truncate `bucket_*.bin` safely)

> Tip: When `large.protect_existing_outputs=true`, a pre-existing `base_basic/` directory is only accepted for
> resume if the checkpoint file exists; otherwise the run errors to avoid accidental overwrites.

## Disk Artifact Cleanup (`large.cleanup.*`)

The large/streaming pipeline generates several **fully-regenerable** intermediate on-disk stores during
training and BaseSet encoding. For billion-scale datasets (e.g. SIFT1B) these can consume hundreds of GB.

A staged cleanup framework allows automatic deletion of regenerable intermediates at safe pipeline points.
Cleanup settings are purely operational — they do **not** participate in store hashes, do **not** change
training/encoding semantics, and are excluded from the train-resume signature.

### Quick Start

```cfg
large.cleanup.enabled = 1
large.cleanup.preset  = eval_int8_min # or eval_both / eval_float_min / dev_all
```

### Config Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `large.cleanup.enabled` | bool | `false` | Master switch. When `false`, no cleanup is performed. |
| `large.cleanup.preset` | string | `"eval_both"` | Preset that sets default keep/delete flags (see below). Shipped example configs use `eval_int8_min` for minimum int8-only eval footprint. |
| `large.cleanup.keep_base_basic` | int | `-1` | Override: `-1` = use preset, `0` = delete, `1` = keep |
| `large.cleanup.keep_base_basic_cluster_id` | int | `-1` | Override for `base_basic/cluster_id.u32` |
| `large.cleanup.keep_base_basic_ivf_lists` | int | `-1` | Override for `base_basic/ivf_offsets.u64` + `base_basic/ivf_ids.u32` (IVF CSR lists) |
| `large.cleanup.keep_base_list` | int | `-1` | Override for `base_list/` directory |
| `large.cleanup.keep_base_list_raw` | int | `-1` | Override for `base_list/raw_u8.bin` / `raw_f32.bin` |
| `large.cleanup.keep_linkage_list_f32` | int | `-1` | Override for `linkage_list/*.f32` coeff files |
| `large.cleanup.keep_linkage_parent_u32` | int | `-1` | Override for `linkage_list/parent.u32` |

Explicit `keep_*` flags (`0` or `1`) always override the preset default for that artifact.

### Presets

Presets control which artifacts are **kept** vs **deleted** after successful pipeline stages.
Note: "eval_both" refers to supporting **both float and int8 linkage eval** (not basic eval + linkage eval).

Presets are intentionally **progressive** (monotonic in deletion):
`none` < `dev_all` < `eval_both` < `eval_float_min` < `eval_int8_min`.
Each later preset deletes everything the previous one would, plus more.

| Preset | base_basic | base_basic ivf_* | base_list | base_list raw | linkage_list.f32 | parent.u32 | Use case |
|--------|-----------|------------------|-----------|---------------|-----------------|------------|----------|
| `none` | keep | keep | keep | keep | keep | keep | No deletion (same as `enabled=false`) |
| `dev_all` | **delete** | keep | keep | keep | keep | keep | Development: delete base_basic payload (and `cluster_id.u32`) while keeping reusable derived stores |
| `eval_int8_min` | **delete** | **delete** | **delete** | **delete** | **delete** | **delete** | Minimum footprint for int8-only linkage eval (needs `linkage_coeff_codec`) |
| `eval_float_min` | **delete** | **delete** | **delete** | **delete** | keep | **delete** | Minimum footprint for float-only linkage eval |
| `eval_both` | **delete** | **delete** | **delete** | **delete** | keep | keep | Both float and int8 linkage eval |
| `custom` | keep | keep | keep | keep | keep | keep | All kept by default; use explicit `keep_*=0` to delete individually |

### What Gets Deleted (and When)

Cleanup is staged: each stage only runs after its dependent artifacts are fully built.

**Stage 0 — After training completes** (guarded by `large.cleanup.enabled`):
- `train_basic/`, `train_ivf/`, `train_list/`, `train_linkage_init_list/`, `train_linkage_list/`
- These are workspace intermediates; the canonical artifact is the saved train result (`results/*.h5`).

**Stage 1 — After `base_list` is ready**:
- If `keep_base_basic = false`: delete `base_basic/` payload files (`bucket_*.bin`, `codes_shard_*.bin`, `coeffs_shard_*.bin`).
  Metadata is preserved (`meta.bin`, `hash.u64`, `cluster_id.u32`, `ivf_offsets.u64`, `ivf_ids.u32`).
  - `cluster_id.u32` is the per-vector root list id (u32, global-id order).
    It is the raw input used to build IVF CSR lists, and is useful for debugging/rebuilding.
  - `ivf_offsets.u64` + `ivf_ids.u32` are the derived IVF CSR lists (required by `base_list/` alignment and **base** disk IVF recall).
    They are also used during `linkage_list` build in current code. Disk linkage recall uses `linkage_list` only and does **not** require IVF CSR lists.
  - You may drop `cluster_id.u32` via `large.cleanup.keep_base_basic_cluster_id=0` once IVF CSR lists exist, but **do not**
    auto-delete `ivf_offsets.u64` / `ivf_ids.u32` if you plan to run **base** disk IVF recall or rebuild linkages.
- If `keep_base_list_raw = false`: delete `base_list/raw_u8.bin` and `base_list/raw_f32.bin` **only when** `base.linkage.enabled=false`.
  (When `base.linkage.enabled=true`, raw vectors are required to build `linkage_list`, so raw deletion is deferred to Stage 2.)

**Stage 2 — After `linkage_list` (+ optional coeff codec) is ready**:
- If `keep_base_basic_ivf_lists = false`: delete `base_basic/ivf_offsets.u64` and `base_basic/ivf_ids.u32`.
- If `keep_base_basic = false`, `keep_base_basic_cluster_id = false`, and `keep_base_basic_ivf_lists = false`:
  delete the **entire** `base_basic/` directory (including leftover `meta.bin`, `meta.json`, `hash.u64`, and `checkpoint/`).
- If `keep_base_list = false`: delete entire `base_list/` directory.
- Else if `keep_base_list_raw = false`: delete `base_list/raw_u8.bin` and `base_list/raw_f32.bin`.
- If `keep_linkage_list_f32 = false`: delete `linkage_list/*.f32` coefficient files.
- If `keep_linkage_parent_u32 = false` and LOUDS representation exists: delete `linkage_list/parent.u32`.

Under `eval_int8_min`, deleting `linkage_list/*.f32` does **not** prevent int8 coeff-codec recall.
It only disables float-coeff recall paths. Cleanup also rewrites `linkage_list/meta.bin`
so `store_coeffs_f32=0` matches the on-disk files.

For `eval_only` runs that do **linkage-only** disk recall, `base_basic/` is no longer required
once cleanup has removed it; the run can proceed from `linkage_list/` + train result alone.

### Related: `large.base_linkage_store_coeffs_f32`

| Key | Type | Default |
|-----|------|---------|
| `large.base_linkage_store_coeffs_f32` | bool | `true` |

Controls whether the baseset `linkage_list` build writes float coefficient files (`.f32`).
When `false`, linkage_list meta sets `store_coeffs_f32=0` and `.f32` files are never created,
saving significant disk space.

**Constraint**: if set to `false`, `large.linkage_coeff_codec.enabled` must be `true` (otherwise
no linkage eval is possible).

> Training-stage linkage_list always stores float coeffs regardless of this flag (needed for codebook updates).

### Notes
- `large.protect_existing_outputs` controls whether existing outputs can be overwritten/rebuilt; it is
  independent of cleanup behavior.
- If you need to debug or re-run store conversion without rebuilding, set `large.cleanup.enabled=false`.
- Cleanup only targets the specific directories/files listed above. Any extra/undocumented files you place under the
  run root are preserved by presets unless they match one of the explicit delete rules.

### Parent Storage: `large.linkage_store_parent_u32`

The large BaseSet `linkage_list` store always generates a succinct parent representation (`parent_louds.bin`) for disk recall.
The legacy materialized parent array (`parent.u32`) can be **very large** (e.g., ~4GB for SIFT1B), so by default we avoid
storing it in BaseSet outputs.

- `large.linkage_store_parent_u32` (bool, default `false`):
  - `false`: do **not** write `linkage_list/parent.u32`; parent pointers are decoded from `parent_louds.bin` on-demand when needed.
  - `true`: also write `linkage_list/parent.u32` (useful for legacy paths / A/B benchmarks / debugging).

Notes:
- Training-stage `train_linkage_list` may still keep `parent.u32` for update stages even if this is `false` (training outputs are treated as temporary).
- This setting does not change linkage semantics/results; it only changes which on-disk payload files are stored.

> [!NOTE]
> **Streaming / Large Scale Pipeline Behavior**:
> The streaming training pipeline (`TrainQuantizerStreamingLarge`) computes `is_bad_cluster` after the first iteration (iter=0) by streaming through the linkage_list and computing per-cluster mean MSE. This enables virtual node weighting in subsequent iterations, similar to the non-large pipeline.
> Note: Unlike the non-large pipeline which computes `is_bad_cluster` during initialization, the streaming pipeline defers this computation to after the first full iteration due to the streaming data flow constraints.


### Memory Optimization (Host RAM Cache)
For large-scale RVQ initialization (e.g., `ntrain=1e8`, ~13GB+), disk I/O significantly impacts performance during multi-iteration training (e.g. 70 iters).
- **Mechanism**: The first iteration reads from disk and caches the training subset in RAM. Subsequent iterations serve data directly from RAM using zero-copy (or fast `memcpy`) transfers to GPU.
- **Recommendation**: For billion-scale tasks using large training subsets (100M+ vectors), we recommend a machine with **32GB+ RAM**.
- **Behavior**: If system RAM is insufficient, the cache automatically disables (fallback to disk streaming), preventing OOM crashes.
- **Performance**: Reduces I/O latency to near-zero. A typical large-scale 70-iteration initialization can see >20x total speedup.

### Memory Notes (Base `bucket_*.bin` buffering + Linux page cache)

In large-scale mode, Base/basic encoding can optionally write per-vector records into bucket files
(`base_basic/bucket_*.bin`) to accelerate later per-cluster processing. This is controlled by:

- `large.write_basic_to_bucket` (bool): append per-vector **quantization codes + coefficients**
  (`codes_small[m-1 bytes]` + `a[m floats]`) to each `bucket_XXXX.bin`.  **Required** for list-order
  store building and disk-based recall evaluation (downstream tools read bucket files per cluster).
- `large.write_vector_bucket` (bool): additionally append the **raw original vector** (`uint8`, `d` bytes)
  to each `bucket_XXXX.bin`.  Makes bucket files self-contained for in-bucket vector reranking, but
  increases IO volume by `d` bytes per vector (~8× overhead vs codes-only for SIFT-128).
  **Independent** of `write_basic_to_bucket`; either or both flags may be set.
- `large.cluster_bucket_size` (cluster ids per bucket; default `256`)
- `large.bucket_flush_mb` (per-bucket in-memory flush threshold; default `256`)

Bucket record layout (all fields present for vectors routed to that bucket):
```
uint32  global_id
uint32  cluster_id
[write_vector_bucket=true]  uint8   x[d]               (raw vector, u8)
[write_basic_to_bucket=true] uint8  codes_small[m-1]   (small-layer codes, layers 1..m-1)
[write_basic_to_bucket=true] float32 a[m]              (coefficients, all m layers)
```
When neither flag is set, only the global-order shard files are written (`codes_shard_*.bin`,
`coeffs_shard_*.bin`), which require sorted seeks for per-cluster access; bucket files are empty.

Important semantics (this is often the reason “RAM usage looks high” on SIFT1B even when the pipeline is streaming):

- **`bucket_flush_mb` is a size threshold, not a “bucket finished” signal.**
  The writer flushes a bucket only when its buffer reaches `bucket_flush_mb` *or* when the writer is closed at the end
  of the stage. If a bucket’s buffer never reaches the threshold, its data can remain in memory until `Close()`/destructor.
- **Bucket buffers are per-bucket and their capacity is retained after flush.**
  Implementation uses `std::vector<uint8_t>` per bucket; `FlushBucket()` does `buf.clear()` (does not shrink capacity).
  Once a bucket grows, its `capacity()` typically stays high for the rest of the stage.
- **How many buckets can exist at once?**
  `nbucket = ceil(h0_root / large.cluster_bucket_size)`. For example, `h0_root=65536` with `cluster_bucket_size=256`
  gives `nbucket=256`. Buckets are opened lazily, but for large BaseSets most buckets will be touched and will allocate.
- **Rule-of-thumb RAM footprint (writer buffers only).**
  If many buckets grow near the threshold, a rough lower bound is:
  `RAM_buf ≈ nbucket × large.bucket_flush_mb`.
  In practice, allocator growth (reserve to ~1.5×) can push this higher (e.g., tens of GB → ~80–100GB).
- **Linux page cache can dominate “used memory” graphs.**
  Large sequential reads/writes will populate the OS page cache and show as high “used” RAM even if your process RSS is
  much smaller. This is normal and typically beneficial; memory is reclaimed under pressure.

Small-memory tuning guidance:
- To bound peak RSS: reduce `large.bucket_flush_mb` and/or increase `large.cluster_bucket_size` (fewer buckets).
- To save disk space and write time at the cost of slower per-cluster random access:
  - Disabling `large.write_vector_bucket` saves `d` bytes per vector (safe unless raw vectors are needed per-bucket).
  - Disabling `large.write_basic_to_bucket` saves `(m-1)+4m` bytes per vector but makes list-order store
    building and recall evaluation significantly slower (random seeks into shard files instead of sequential
    bucket reads).  **Not recommended** unless storage is critically constrained.

### CUDA Tuning
- `runtime.cuda_mode="strict"`: Maximize reproducibility. TF32 is off by default; large-root beam-search fast paths (`H_beam=2/4`) avoid computing `xC_small` via GPU GEMM.
- `runtime.cuda_beam_deterministic=true`: Force the above “beam reproducibility” behavior even when `runtime.cuda_mode="fast"` (slower but more stable run-to-run).
- `runtime.cuda_mode="fast"`: Enables TF32 instructions on Ampere+ GPUs (significantly faster GEMM).
- `train.kmeans_device_cache_mb`: If your GPU VRAM is large enough, you can set this to cache the normalized dataset directly on GPU (e.g. `2048` MB).

### GPU ICM Limitation (Non-root `hj`)
Current CUDA ICM kernels are intentionally optimized for the common production shape:
- **Root layer (layer 0)** may be large (`h0`), and is handled by the dedicated **large-root** path.
- **Non-root layers (layer >= 1)** are currently supported only when `model.h_vec[layer] == 256`.

If any non-root `hj != 256`, the **GPU ICM path** will fail fast with an error. Some higher-level pipelines may
catch this and fall back to the CPU ICM implementation; others may treat it as fatal (depending on where ICM is used).

### CUDA Linkage Build (Large Pipeline)
When `runtime.use_cuda=true`, the large/streaming linkage-build pipeline can offload **candidate-parent evaluation**
to CUDA (this does not change the linkage semantics, but may introduce tiny floating-point differences).

#### CPU↔GPU Interaction Flow (Basic + Linkage)

This project is intentionally **hybrid CPU/GPU** in the large/streaming pipelines:

- CPU owns **control flow**, legality rules, and **commit-order** semantics (especially same-layer).
- GPU owns **hot numeric kernels** (GEMM + ILS/ICM + LS-cost) on batched workloads.

This section summarizes what lives on CPU vs GPU and what transfers happen.

##### Basic Encoding (streaming)
Entry points are in `src/quantizer/encode_base_streaming.cpp` and are used to build on-disk
basic-code stores for train/base (bucketed/list-order stores depending on pipeline stage).

High-level flow (per block):
1) **CPU** reads a block of raw vectors (`uint8` or `float`) from disk.
2) **CPU→GPU** (optional): upload the block (or tiles of the block) when CUDA kernels are enabled.
3) **GPU** computes the heavy math:
   - `xC` / `xC_small` (depending on precomp strategy and large-root handling)
   - beam search init
   - per-sample (or batched) ICM/ILS refinement
   - optional caching/pinning (when enabled) reduces repeated H2D and improves overlap.
4) **GPU→CPU**: copy out compact results (codes + coeffs) and write to stores.

Notes:
- Logs may mention `Streaming path: full-precomp` when the codebook precomp strategy materializes full `xC`
  for the block on GPU (bounded by block/tile sizes).
- For large-root shapes, the implementation may avoid materializing full `xC` and instead use specialized
  small-`H_beam` kernels to reduce memory traffic.

##### Linkage Build (streaming, `large.enabled=true`)
The linkage build is a **per-cluster** process with strict semantics:
- same-layer commit-order dependence must be preserved
- a node can only use already-committed same-layer parents

We use a split design:
- **CPU** builds/filters candidate lists (HNSW query + legality + depth constraints) and performs sequential commits.
- **GPU** evaluates candidate parents by running residual encoding (ILS/ICM/LS-cost) and returning the best.

Key device-side optimization: **device-resident `R_full`**
- For each cluster, we maintain a device mirror `d_R_full` (reconstruction matrix) and gather parent columns
  by candidate ids on GPU. This avoids CPU packing of `Rp(d×Kp)` for every node.

Per cluster flow:
1) **CPU** constructs `Xrot` for the cluster (read raw → rotate/convert).
2) **CPU** initializes codes/coeffs from the basic store and reconstructs host `R_full` (cluster-local).
3) **CPU→GPU** uploads `R_full` once via `CudaLinkageUploadClusterRfull(...)`.
4) For each node (commit-order preserved by CPU):
   - **CPU** builds candidate parent ids:
     - inner / cross-layer candidates have no same-layer dependence and can be batched
     - same-layer candidates depend on committed frontier and must be evaluated at the correct time
   - **CPU→GPU** uploads a small `X_block` (d×B) plus compact candidate structures:
     - `cand_parent_local_flat`, `pair_node`, `cand_offsets` (many-nodes batching)
   - **GPU**:
     - gathers parent recon from `d_R_full`
     - builds residuals + norm2
     - runs `LinkageEncodeBatchCuda*` (ILS/ICM/LS-cost) on all (node,candidate) pairs
     - reduces to per-node argmin (stable tie-breaking)
   - **GPU→CPU** returns only “best packs”:
     - best parent id, best cost, best `(B,a)` for the residual encoding
   - **CPU** commits the chosen parent/codes/coeffs and updates host `R_full[:, node]`
   - **CPU→GPU** updates the corresponding `d_R_full` column(s) via `CudaLinkageUpdateClusterRfullColumn(...)`
     (or the batched `CudaLinkageUpdateClusterRfullColumnsBatch(...)`)

Thread-safety note (why we don’t “just use one CUDA provider everywhere”):
- The linkage stages are **OpenMP cluster-parallel**. A single `CudaStreamKernels` (one cuBLAS handle + shared scratch)
  is not thread-safe under that execution model.
- The linkage GPU path therefore uses a **context pool** (`CudaStreamKernelsPool`) so each worker thread can acquire
  an independent CUDA context/stream/handle for candidate evaluation.
- Some per-cluster prep steps (e.g., rotate/convert) may still run on CPU to avoid per-thread GPU provider contention
  and unnecessary H2D/D2H churn; the dominant speedups come from batching candidate-eval and keeping `R_full` on device.

Init-linkage vs iterative linkage:
- **Init-linkage (C_root-only, forced root)**: uses a forced-root CUDA encoder path (`*ForcedRoot*`) so layer0 stays fixed.
  Three modes available: `legacy_root_only` (full VarRoot ILS), `hybrid` (first round VarRoot, rest ConstRoot), `fast` (two-codebook).
  See [Init-Linkage Mode](#init-linkage-mode-traininit_linkage_mode) below.
- **Iterative linkage (C_root + C_one)**: uses the two-codebook linkage pipeline with mature batching/window scheduling.

Relevant keys:
- `runtime.cuda_pool_size` (int): number of per-thread CUDA contexts for linkage build (OpenMP cluster-parallel). Higher can be faster but increases VRAM usage.
- `runtime.cuda_pool_size_init_linkage` (int): init-linkage override for the CUDA ctx pool size. `0` uses `runtime.cuda_pool_size`.
- `runtime.cuda_linkage_single_gpu_min_candidates` (int): per-node candidate count threshold. If a node has `Kp < min_kp`, use CPU eval (GPU launch/H2D overhead usually dominates).
- `runtime.cuda_linkage_single_gpu_max_candidates` (int): per-node candidate cap. If `Kp > max_kp`, skip the per-node GPU eval (fallback path depends on stage and batch settings).
- `runtime.cuda_linkage_use_device_rfull` (bool, Stage-2): keep per-cluster `R_full` on GPU and gather parent residuals by candidate ids. Avoids CPU packing `Rp(d×Kp)` every node.
- `runtime.cuda_linkage_inner_many_nodes_enable` (bool, Stage-3): batch **inner-candidates** across many nodes in the same layer into one GPU call. Same-layer candidates stay serial (commit-order dependency).
- `runtime.cuda_linkage_inner_many_nodes_target_nodes` (int, Stage-3): max nodes per micro-batch.
- `runtime.cuda_linkage_many_nodes_max_pairs` (int, Stage-3): max total `(node,candidate)` pairs (`Npairs`) per micro-batch; lower = lower VRAM peak, more flushes.
- `runtime.cuda_linkage_many_nodes_max_pairs_init_linkage` (int): init-linkage override for `runtime.cuda_linkage_many_nodes_max_pairs`. `0` uses the non-init value.
- `runtime.cuda_linkage_many_nodes_preflush_target_pairs` (int, Stage-3e+): opportunistic preflush target (`Npairs`) used by some schedulers to avoid emitting too many tiny many-nodes calls (does not change semantics).
- `runtime.cuda_linkage_eval_async_pinned_mb` (int): per-CUDA-ctx host pinned staging budget. When `>0`, some linkage stages may enqueue GPU candidate-eval and overlap CPU work before syncing (uses extra pinned host memory; does not change semantics).
- `runtime.cuda_linkage_eval_async_pinned_mb_init_linkage` (int): init-linkage override for `runtime.cuda_linkage_eval_async_pinned_mb`. `0` uses the non-init value.
- `runtime.cuda_linkage_same_layer_window_min_pairs` (int, Stage-3c):
  - `<=0`: disable same-layer dynamic window/pending many-nodes batching (good clusters).
  - `>0`: enable window/pending many-nodes batching; acts as an *opportunistic preflush* minimum `Npairs` (avoid too many tiny GPU calls).
  - Note: commit-time **forced flush** is allowed to run on GPU even when `Npairs < min_pairs` (to avoid CPU fallback), and dyn-before batching is not gated by `min_pairs`.
- `runtime.cuda_linkage_same_layer_block_nodes` (int, Stage-3c): window size (nodes per block) for same-layer dynamic window/pending batching (good clusters).
- `runtime.cuda_linkage_same_layer_single_gpu_min_candidates` (int, Stage-3c): per-node GPU threshold for window-local same-layer dynamic candidates. `0` uses `runtime.cuda_linkage_single_gpu_min_candidates`.
- `runtime.cuda_linkage_same_layer_preflush_single_shot` (bool, Stage-3c): enable incremental window-local same-layer dynamic many-nodes batching (good clusters).
- `runtime.cuda_linkage_same_layer_window_max_pairs` (int, Stage-3c): max `Npairs` per window-local dynamic flush. `0` uses `runtime.cuda_linkage_many_nodes_max_pairs`.
- `runtime.cuda_linkage_same_layer_window_max_pending_nodes` (int, Stage-3c): max nodes per window-local dynamic flush. `0` uses the remaining window size.
- `runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable` (bool, Stage-3f): **Tiny-CPU mixed path** for commit-time
  **forced flush** in the good-cluster same-layer dynamic window.  When the selected flush slice is
  tiny (`Npairs <= tiny_cpu_max_pairs`), skip the GPU many-nodes call and evaluate the remaining tail
  on CPU instead.  Semantics are preserved: the same objective function (residual LS + ICM/ILS) is
  evaluated on the same candidates; only the arithmetic backend changes (CPU fp32 vs CUDA).
  - **Iterative rounds** (`BuildLinkageTwoCodebookVirtualStreamingByCluster`): requires full precomp
    `BuildPrecomp(C_one)` — always built automatically when
    `runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable=true` in the large pipeline
    (because `C_one` is small, the memory cost is negligible).
  - **Init-linkage** (`BuildLinkageOneCodebookVirtualInitStreamingByCluster`): behaviour depends on `h0_root`:
    - `h0_root <= 256`: full precomp of `C_root` is built automatically (cheap); CPU tail uses it directly.
    - `h0_root > 256`: requires `runtime.cuda_linkage_init_same_layer_tiny_forced_cpu_enable=true`
      (see below); a dedicated
      fixed-root reduced precomp is used instead (full precomp would be prohibitive at large `h0_root`).
  - **Expected result difference**: enabling tiny-CPU produces **slightly different** linkage assignments
    than pure-GPU mode.  This is **by design, not a bug**: nodes whose tiny forced-flush was deferred
    use CPU fp32 arithmetic (vs CUDA fp32) and a per-node deterministic RNG for ILS perturbation (vs the
    GPU ILS sequence).  The objective values are the same; only non-associativity of fp32 and ILS seed
    ordering differ.  Recall changes are small (typically < 0.1% recall@1 on SIFT1B) and the speedup
    from avoiding many tiny GPU kernel launches is substantial.
- `runtime.cuda_linkage_init_same_layer_tiny_forced_cpu_enable` (bool): **Init-linkage only** (`h0_root > 256`,
  `large.enabled=true`).  Allows the tiny-CPU forced-flush deferral even when full-precomp is
  unavailable (which it always is for large `h0_root`).  A dedicated evaluator
  (`BuildPrecompInitLinkageFixedRootTinyCpu`) builds a reduced Gram matrix with `h_vec[0]=1`
  (only the forced root code `forced_root_code`), so CPU evaluation is cheap and memory-bounded.
  Layer-0 ILS perturbation is suppressed (`ils_layer_lo=1`) since there is only one root-code option.
  Off by default; enable only when `runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable=true`
  is also set.
- `runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_pairs` (int, Stage-3f): gate for tiny-CPU deferral:
  a flush slice is eligible only when `total_Npairs <= max_pairs`.  Non-positive value disables the
  entire tiny-CPU path even if `runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable=true`.
- `runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_nodes` (int, Stage-3f): optional safety cap on the
  number of nodes in the flush slice.  `0` disables the cap (any slice size is eligible).
- `runtime.cuda_linkage_bad_frozen_window_enable` (bool, Stage-3b): enable bad-cluster window batching of frozen candidates (virtual + window-start predecessors).
- `runtime.cuda_linkage_bad_frozen_window_nodes` (int, Stage-3b): max nodes per bad-cluster window.
- `runtime.cuda_linkage_bad_frozen_window_max_pairs` (int, Stage-3b): max frozen `Npairs` per window-batch.

###### CUDA VRAM and budgeting notes (large/streaming linkage)
There are a few different “budget knobs” that affect different allocations. They are **not** a single global VRAM limit.
In practice, total VRAM usage is roughly:

`VRAM_total ≈ (VRAM_per_ctx_peak × runtime.cuda_pool_size) + shared_caches + other_cuda_modules`

Key points:

- **Per-ctx vs shared allocations**
  - The linkage GPU candidate-eval path maintains a per-ctx workspace (see `eval_candidates_cuda.cu`), including a
    device-resident `d_R_full (d×cluster_size)` mirror and various batch buffers. This scales roughly with the
    largest cluster the ctx has seen.
  - Some CUDA encode precompute caches are process-wide shared (see comments in `linkage_encode_cuda.cu`); these do not
    scale with `cuda_pool_size` but still contribute to VRAM usage.

- **`runtime.cuda_pool_size` multiplies per-ctx peak**
  - Increasing `cuda_pool_size` allows more per-ctx workspaces to exist concurrently. This often dominates VRAM in
    large-scale runs because `d_R_full` scales with the largest cluster size.

- **Candidate-eval chunking knobs (`eval_candidates_cuda.cu`)**
  - `runtime.cuda_linkage_chunk_max_pairs`:
    - If `>0`, it is a hard cap for **internal pair chunking** inside the CUDA many-nodes evaluator.
    - This reduces per-ctx peak VRAM at the cost of more kernel launches/sync per logical eval.
  - `runtime.cuda_linkage_mem_budget_mb`:
    - Used **only when** `cuda_linkage_chunk_max_pairs<=0`.
    - It is a *hint* used to derive a safe internal chunk cap (heuristic) to reduce peak VRAM per ctx; it is not a strict
      global memory limit and does not account for all other CUDA modules.
  - Init-linkage overrides:
    - `runtime.cuda_linkage_chunk_max_pairs_init_linkage`, `runtime.cuda_linkage_mem_budget_mb_init_linkage` apply to init-linkage only.

- **Many-nodes batch shape knobs (affect peak sizes by changing `B`/`Npairs`)**
  - `runtime.cuda_linkage_inner_many_nodes_target_nodes`, `runtime.cuda_linkage_many_nodes_max_pairs`,
    `runtime.cuda_linkage_same_layer_window_max_pairs`, `runtime.cuda_linkage_bad_frozen_window_max_pairs`:
    - These indirectly control peak workspace sizes by limiting `B`/`Npairs` of individual CUDA calls.

- **Large-root (split-root) root-selection cap (`linkage_encode_cuda.cu`)**
  - `runtime.cuda_linkage_large_root_xc0_chunk_mb` controls the temporary `xC0_chunk` matrix used by the **large-root**
    encoder root selection (tiled exact scan over `h0`). This is separate from the linkage candidate-eval workspace and
    matters mainly when `h0_root` is large (e.g. 4096+).
  - `runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage` is the init-linkage override. `0` uses the non-init value.

- **Host pinned memory (not VRAM)**
  - `runtime.cuda_linkage_eval_async_pinned_mb` is a per-ctx **host pinned** staging budget used by async overlap APIs.
    This does not directly consume VRAM, but it does consume locked host memory.

- **Batched `d_R_full` updates**
  - The streaming linkage build may batch per-node `d_R_full` column updates and apply them via a scatter kernel.
    This adds small per-ctx staging buffers on GPU (typically `O(d×B)` floats + `O(B)` ints) and small pinned host buffers.
    Compared to `d_R_full (d×cluster_size)`, this overhead is negligible.

###### Future multi-GPU linkage build

Multi-GPU linkage build is not implemented yet, but the current cluster-parallel linkage architecture is a good fit for
it. The intended direction is to keep algorithm bodies consuming `CudaCtx*` and replace the single-device
`CudaStreamKernelsPool` boundary with a multi-device scheduling pool:

```text
MultiCudaLinkagePool
  device 0 -> CudaStreamKernelsPool(size=N0)
  device 1 -> CudaStreamKernelsPool(size=N1)
```

The existing shared CUDA precompute caches must remain device-owned: cache keys include the CUDA device id, and each GPU
gets its own copy of shared device data. Raw device pointers are never shared across GPUs. See
`docs/MULTI_GPU_LINKAGE_PLAN.md` for the feasibility plan and constraints.

### Large-root Optimization (`h0_root >= 4096`)
For very large `h0_root` (root codebook size), the naive “full-precomp” path becomes expensive
because it tends to allocate/compute `xC` with height `H = sum(h_vec)` and thus scales with `h0_root`.

We implement a **semantic-preserving split precompute**:

- Files:
  - `include/stlq/quantizer/precomp_large_root.h`
  - `src/quantizer/precomp_large_root.cpp`
  - `src/quantizer/encode_base_streaming.cpp`
- Idea:
  - Keep layer0 (`C0`) separate and flatten layers 1.. into `C_small`.
  - Precompute:
    - `G_small = C_small' * C_small`
    - `G0S = C0' * C_small` (cross-term for root vs small layers)
    - norms for layer0 and small layers
  - During streaming encode/training, compute only `xC_small = C_small' * Xblk` (bounded by tile),
    and use specialized “large-root” beam/ICM routines to avoid materializing the full `xC`.
- CUDA fast path:
  - For common `H_beam` values (`2` / `4`) we have specialized CUDA kernels that work directly from
    `xC_small` (and optionally device-resident `xC_small`) to keep the root stage efficient.

## Configuration Guide

The config file uses a simple line-based `key = value` format. See `configs/basic_sift1m_linux.cfg` for a SIFT1M starting point.
- **Comments**: everything after `#` or `//` on a line is ignored.
- **Arrays**: `[v1, v2, ...]`

### Dataset and Basics
- `dataset.name`: e.g. "SIFTSMALL", "SIFT1M", "SIFT1B", "DEEP1M", "DEEP10M", "MSONG", "GLOVE100".
- `dataset.data_root`: Path to data directory.
- `dataset.ntrain`: Number of training vectors to use.
- `model.m` / `model.h0`: Number of codebooks (layers) and root/IVF size.
  Non-root layers default to 256. Use `model.h_vec` only for advanced per-layer
  overrides.

### K-means Initialization (`train.kmeans`)
- `train.kmeans_iters`: Number of Lloyd's iterations.
- `train.kmeans_streaming`: Critical for large datasets (must be `true`).
- **Dataset vs Seed Size**:
  - `dataset.ntrain`: The number of vectors used for ALL assignment/update steps.
  - `train.init_samples`: Only used for **initial centroid seeding** (e.g. kmeans++).
- **Performance Knobs**:
  - `train.kmeans_cache_xnorm_device`: Best-effort full `X_norm` device cache.
  - `train.kmeans_pin_host_x`: Best-effort pinned host memory for uploads.
  - **Streaming Ping-Pong**: When streaming + pinned memory are enabled, we look for `large.profile_timing=false` to enable double-buffered overlap of H2D copies and compute.
- **Large-Scale Hierarchical Assignment** (`train.kmeans_large_k_*`):
  - Used when `K >= threshold`.
  - `hier2_train`: Use 2-level search during training assignment (less compute, slightly different centers).
  - `hier2_encode`: Use 2-level search during RVQ encode-to-store.
  - Recommendation: Keep `*_train` and `*_encode` consistent.

### Large-scale Streaming (`large.*`)
When `large.enabled=true`, the pipeline runs in out-of-core mode designed for 1e8+ train / 1e9+ base.
- `large.output_dir`: Root directory for experiment workspace.
- `large.train_block` / `large.base_block`: Block sizes for readers.
- `large.write_basic_to_bucket` (bool, **recommended `true`**): Append per-vector quantization output
  — `codes_small[m-1]` (u8, small-layer codes for layers 1..m-1) + `a[m]` (f32, coefficients for all
  m layers) — into each `bucket_XXXX.bin` file.  Required for list-order store building and disk-based
  recall evaluation.  Without this, downstream tools must do sorted random seeks into the global-order
  shard files to read codes/coeffs per cluster.
- `large.write_vector_bucket` (bool, optional): Additionally append the raw original vector (`uint8`,
  `d` bytes) into each `bucket_XXXX.bin`.  Enables in-bucket exhaustive vector reranking without
  a separate dataset read.  Increases per-vector bucket record size by `d` bytes (e.g. +128 bytes on
  SIFT-128 vs ~5-9 bytes for codes+coeffs with m=4; ~14×–16× extra IO).  Usually not needed unless a
  specific downstream tool requires raw vectors per cluster.
  Both flags are **independent**: either, neither, or both may be set; they control different fields in
  the same bucket record (see "Bucket record layout" in the On-disk Stores section).
- `large.profile_timing`: Print per-block stage timings.
- `large.protect_existing_outputs`: If `true`, forbid auto delete/rebuild of existing large outputs (manual delete only).

**Workspace Lifecycle**:
- Large outputs live under a **hash-suffixed run root**:
  - `large.output_dir/<dataset.name>/<io.pre_fix>__0x<run_hash>/`
  - `large.tmp_dir/<dataset.name>/<io.pre_fix>__0x<run_hash>/`
- `run_hash` is derived from **semantic store hashes** (not from `eval.*`, logging, or runtime/perf toggles like TF32).
- When a build stage is enabled, we generally **rebuild in-place** under the current run root.
  - Set `large.protect_existing_outputs=true` to prevent any automatic delete/rebuild.

## Large Pipeline: On-disk Stores & Temp Files

When `large.enabled=true`, we do not materialize full matrices like `X`/`B`/`a` for train/base. Instead we build
several **reusable on-disk stores** under a stable “workspace root”.

### Workspace Roots

The app normalizes `io.pre_fix` to include `_m{model.m}` (see `NormalizeIoPrefixWithM`), so the actual root is:

- **Outputs root**: `large.output_dir/<dataset.name>/<io.pre_fix>__0x<run_hash>/`
- **Temp root**: `large.tmp_dir/<dataset.name>/<io.pre_fix>__0x<run_hash>/`

Example (m=5): `outputs_opt/SIFT1B/test_m5__0x0123abcd/` and `tmp/SIFT1B/test_m5__0x0123abcd/`.

The program logs this at runtime:
- `Large run tag: <io.pre_fix>__0x<run_hash>`

Config archival note:
- `config_snapshot.txt` under the run root is generated from the in-memory effective config.
- The copied original `*.cfg` backup is frozen from `io.config_file` at process startup, then written into the
  hash-suffixed run root once `run_hash` becomes known. Later edits to the source cfg file do not affect that backup.

### Store Hashing / Reuse (`hash.u64`)

Some stores are keyed by a deterministic hash written as `hash.u64` in the store directory.
This enables:
- **eval-only reuse**: do not rebuild large stores if they match the expected hash
- **stale output detection**: error out (or warn in eval-only) when outputs don’t match the current config/model

Hashing code lives in `src/pipeline/large_store_hash.cpp`.

Notes:
- Hash is intended to represent the **semantic identity** of a store.
- Performance-only knobs (example: `base.encode.base_block`, pinned/ping-pong toggles, TF32/strict/fast) should not change `run_hash`.
- `hnsw.ef_construction_cap` is special-cased: when it is `<=0`, it is treated as disabled and does not
  affect store hashes/run tags; when it is `>0`, it is folded into the linkage-list identity.

#### Hash Taxonomy

The suffix in a large run directory is the final large-store identity:

```text
<io.pre_fix>__0x<run_hash>
```

Today `run_hash` is the same value as `linkage_list_hash_identity`, because the large run root is organized around
the final `linkage_list` product.  The dependency chain is:

```text
trained model for base
  = TrainResult.R + C_root

base_basic_hash
  = hash(trained model for base,
         dataset/store shape,
         base dtype and nbase cap,
         base.encode semantic knobs,
         base_basic store layout knobs)

base_list_hash_identity
  = hash(base_basic_hash,
         nlist,
         ntotal,
         model.m,
         list-order/raw-store layout knobs)

trained model for linkage
  = TrainResult.R + C_root + C_one

linkage_list_hash_identity
  = hash(trained model for linkage,
         base_list_hash_identity,
         model/linkage/virtual/hnsw semantic knobs)

run_hash_identity
  = linkage_list_hash_identity

coeff_codec_hash
  = hash(linkage_list_hash_identity,
         large.linkage_coeff_codec.* semantic knobs)
```

That means a small change in training output (`R`, `C_root`, or `C_one`) cascades into `base_basic_hash`,
`base_list_hash_identity`, `linkage_list_hash_identity`, and finally the run-directory suffix.

Common hash files and what they mean:

| Hash | Where | Meaning |
|---|---|---|
| `run_hash_identity` | `<run_root>/store_hashes.txt`, run directory suffix | Final large-run identity; currently equals `linkage_list_hash_identity`. |
| `base_basic_hash` | `<run_root>/store_hashes.txt`, `base_basic/hash.u64` | Identity of the sharded/bucketed base-basic store. Depends on `TrainResult.R + C_root` and base encode/store semantics. |
| `base_list_hash_identity` | `<run_root>/store_hashes.txt`, `base_list/hash.u64` | Identity of the list-order base store. Depends on `base_basic_hash`, `nlist`, `ntotal`, and list-order layout choices. |
| `linkage_list_hash_identity` | `<run_root>/store_hashes.txt`, `linkage_list/hash.u64` | Identity of the linkage graph/store. Depends on `TrainResult.R + C_root + C_one`, `base_list_hash_identity`, and linkage/virtual/HNSW semantics. |
| `coeff_codec_hash` | `<run_root>/store_hashes.txt`, `linkage_list/coeff_hash.u64` | Identity of the int8/bit-packed coeff codec built from a specific linkage store and codec config. |
| norm2 cache hash | sidecar `*.hash.u64` next to norm/LUT caches | Validates eval norm/LUT caches against the current linkage or codec source hash. |
| eval archive hash | `eval_result/recall_<label>_0x<hash>.txt` or `metrics_<label>_0x<hash>.txt` | Identity of eval/report settings, not the large store itself. |
| train resume signature | train-result metadata / sidecar | Compatibility check for resuming training. Excludes operational paths, eval knobs, runtime knobs, and some checkpoint controls. |
| config snapshot hash | internal diagnostics | Hash of the effective config snapshot; broader than store identity and not the run-directory suffix. |

`store_hashes.txt` is the quickest way to answer “why did my run directory suffix change?”  Compare it with a
previous run from top to bottom.  If `base_basic_hash` changed, inspect training output and base encode/store
inputs first.  If only `base_list_hash_identity` changed, inspect list-order shape/layout.  If only
`linkage_list_hash_identity` changed, inspect linkage/virtual/HNSW semantic knobs or `C_one`.

### Train-side Stores (large training pipeline)

These are created by `TrainQuantizerStreamingLarge` in `src/quantizer/encoder_train_streaming.cpp`.
They are internal “pipeline artifacts” and are currently overwritten/truncated on rebuild (no `hash.u64` yet).

#### `train_basic/` (BaseBasicStore: bucketed + sharded base encodings)

Purpose: streaming basic encode output for the training set, written once and reused for:
- IVF list build (`train_ivf`)
- list-order store build (`train_list`)
- per-cluster linkage build (`train_linkage_list` / `train_linkage_init_list`)

Files (see `src/io/base_store.cpp`):
- `meta.bin`
- `cluster_id.u32` (u32, length = ntrain): root cluster id (IVF list id) per vector, in **global id order**
- `codes_shard_XXXX.bin` (u8, records for layers 1..m-1; by global id shard)
- `coeffs_shard_XXXX.bin` (f32, records for layers 0..m-1; by global id shard)
- `bucket_XXXX.bin` (binary bucket files, grouped by `cluster_id` ranges)

Bucket record layout (fixed-size per vector; fields are **conditional** on the write flags):
```
uint32  global_id                                  (always present)
uint32  cluster_id                                 (always present)
uint8   x[d]                                       (only if write_vector_bucket=true)
uint8   codes_small[m-1]                           (only if write_basic_to_bucket=true; layers 1..m-1)
float32 a[m]                                       (only if write_basic_to_bucket=true; all m layers)
```
- Record size = `8 + d*(write_vector_bucket) + (m-1+4*m)*(write_basic_to_bucket)` bytes.
- Both flags default to `false`.  `write_basic_to_bucket=true` is required for list-order stores
  (`train_list/`, `base_list/`) and recall evaluation.
- `meta.bin` encodes both flags as bits 0–1 of the `flags` field so readers know the record layout.

#### `train_ivf/` (IvfListsStore: CSR lists for the training set)

Purpose: list membership for train vectors, derived from `train_basic/cluster_id.u32`.

Files (see `src/io/ivf_lists.cpp`):
- `ivf_offsets.u64` (length = nlist+1)
- `ivf_ids.u32` (length = offsets.back(), global ids in cluster order)

#### `train_list/` (BaseListStore: list-order contiguous basic encodings)

Purpose: store `(codes_small, a, optional raw_u8)` in **list order**, aligned with `train_ivf`.
This avoids random reads across shards during per-cluster processing.

Files (see `src/io/base_list_store.cpp`):
- `meta.bin`
- `codes.bin` (u8, column-major `(m-1)×ntotal`)
- `coeffs.bin` (f32, column-major `m×ntotal`)
- optional `raw_u8.bin` (u8, column-major `d×ntotal`) when `large.write_vector_bucket=true`

#### `train_linkage_init_list/` (LinkageListStore: init-linkage, C_root-only)

Purpose: “init linkage” used by the baseline R=I stage to:
- compute `is_bad_cluster` from a C_root-only linkage
- run the first exact-LS update for `C_one`

Notes:
- This init-linkage is **single-codebook** and uses fixed-root semantics (root code is the list id `cid`).
- The store uses a fixed-length `depth_offsets` array per cluster (padded), so you must not infer depth from
  `depth_offsets.size()` (see the init-linkage summary logic in `encoder_train_streaming.cpp`).

Files (see `src/io/linkage_list_store.cpp`):
- `meta.bin`
- `real_offsets.u64`, `virt_offsets.u64`, `depth_offsets_offsets.u64`
- `depth_offsets.u32`, `real_ids.u32`
- optional `parent.u32` (dense parent pointer array; legacy/debug)
- optional `parent_louds.bin` + `parent_louds_offsets.u64` (succinct parent encoding; preferred for disk recall)
- `codes.bin` (small-layer codes, bytes = `m_codes * n_real * small_code_width_bytes` where `small_code_width_bytes=1`)
- `code0_one.bin` (root layer of “one” code for depth>0 nodes; init-linkage uses zeros)
- if `store_coeffs_f32=true`: `coeffs.f32`, `a0.f32`, `virt_coeffs.f32`, `virt_a0.f32`
- `virt_codes.bin` (virtual nodes small layer codes; init-linkage uses empty virtual by default)

#### `train_linkage_list/` (LinkageListStore: two-codebook linkage for training)

Purpose: the main per-cluster linkage store used by later training rounds, including the
streaming update of `C_one` from `train_linkage_list` (depth>0 only).

Same file set as `train_linkage_init_list/`, but here `code0_one.bin` is meaningful (u8) and this folder may include
virtual nodes depending on the virtual config.

### Base-side Stores (large base encoding + linkage build)

These are created by `stlq_main` through the staged pipeline in `src/pipeline/` and are designed to be reusable.

#### `base_basic/` (BaseBasicStore)

Purpose: out-of-core base encoding output (codes/coeffs + optional raw vectors).

Files: same layout as `train_basic/`, plus:
- `hash.u64` (store identity used for reuse/stale detection)
- `ivf_offsets.u64` + `ivf_ids.u32` (base IVF CSR lists derived from `cluster_id.u32` and stored alongside `base_basic/` in current code)

Notes:
- `cluster_id.u32` is the raw per-vector assignment. `ivf_offsets.u64` + `ivf_ids.u32` are the derived CSR form
  used by downstream stages that need list membership in cluster order.
- When cleanup deletes `base_basic` “payload”, it removes the bulky bucket/shard files but keeps IVF CSR files,
  and keeps or deletes `cluster_id.u32` depending on `large.cleanup.keep_base_basic_cluster_id`.
- Once cleanup also deletes `cluster_id.u32` and `ivf_*`, the remaining `base_basic/` metadata/checkpoint files are removed too,
  so the whole `base_basic/` directory disappears.

#### `base_list/` (BaseListStore)

Purpose: list-order contiguous store aligned with base IVF lists (for linkage build and **base** disk IVF recall).

Files: same as `train_list/`, plus `hash.u64`.

#### `linkage_list/` (LinkageListStore: virtual-mode two-codebook linkage)

Purpose: the final disk linkage store used for linkage recall, and optionally for coefficient compression.

Files: same as `train_linkage_list/`, plus:
- `hash.u64` (store identity)

Notes:
- For BaseSet disk recall in large mode, `parent_louds.bin` + `parent_louds_offsets.u64` are expected to exist.
  `parent.u32` is optional (see `large.linkage_store_parent_u32`).

Optional coefficient codec store (enabled by `large.linkage_coeff_codec.enabled=true`):
- `coeff_meta.bin`
- `coeff_scales.f32`
- `coeff_lens.u8`
- `coeff_payload.bin`
- `coeff_payload_offs.u64`
- `coeff_payload_sizes.u32`
- `coeff_hash.u64` (validates codec against the expected `linkage_list` hash)

### Temp Directories (large.tmp_dir)

These are best-effort scratch spaces; most are safe to delete between runs.

- `tmp/<dataset>/<io.pre_fix>/ivf_tmp/`:
  - created when building IVF lists
  - contains `pairs_bucket_<b>.bin` during the build (removed when `keep_tmp=false`)
- `tmp/<dataset>/<io.pre_fix>/train_ivf_tmp/`: same as above but for the training IVF build
- `tmp/<dataset>/<io.pre_fix>/base_list_tmp/` and `tmp/<dataset>/<io.pre_fix>/train_list_tmp/`:
  - currently directories only (kept for API compatibility); the list builder no longer writes temp files
- `train.kmeans_tmp_dir` (or fallback `large.tmp_dir`):
  - used by streaming k-means for spill/debug artifacts (depending on mode)
  - may create: `kmeans_costs_f32.bin`, `kmeans_assign_u16.bin`/`kmeans_assign_u32.bin`,
    `kmeans_weights_in_f32.bin`, `kmeans_weights_out_f32.bin`

### Mixed Mode Architecture

> [!IMPORTANT]
> **Non-Large Pipeline Using Large Features**:
> The standard (non-large) training pipeline in `encoder_train.inc` can optionally use some `large.*` features. This creates a "mixed mode" that blurs the boundary between pipelines.

**Mixed Mode Usage in Non-Large Pipeline:**

| Feature | Config Flags | Effect |
|---------|--------------|--------|
| **Block Size** | `large.train_block` | Controls streaming block size for K-means (used even when `kmeans_streaming=true` in non-large) |
| **Temp Directory** | `large.tmp_dir` | Fallback for K-means temp files if `train.kmeans_tmp_dir` is empty |
| **Train Format** | `large.train_format` | Specifies `bvecs`/`fvecs` format when streaming reader is enabled |
| **Profile Timing** | `large.profile_timing` | Enables per-block timing logs for streaming K-means initialization |

**Key Differences Between Pipelines:**

| Aspect | Non-Large (`encoder_train.inc`) | Large (`encoder_train_streaming.cpp`) |
|--------|--------------------------------|---------------------------------------|
| **RVQ Init** | Full beam+ICM+ILS+UpdateCodebooks+BuildLinkage | Streaming K-means on full dataset (out-of-core) |
| **C_one Init** | Via `BuildLinkageOneInit` + `UpdateCOneWeighted` | Via simple SphericalKmeans on C_root[0] |
| **Data Flow** | In-memory (requires full training data in RAM) | Streaming (disk-based, multi-pass) |
| **`is_bad_cluster`** | Computed via `BuildBadClusterMaskQuantile` during init | Computed via `ComputeBadClusterMaskFromLinkageListStreaming` after iter=0 |

### IO Policies & HDF5 (`io.*`)
- **Filename Policy**: Appends `_YYYYMMDD` (and `_01` if exists).
- **Loading Policy**: `io.load_date` ("YYYYMMDD" or empty for today) and `io.load_seq`.
- If a stage is disabled (e.g. `train.enabled=false`), it attempts to load results from `io.*_file`.
- `io.save_train`: Writes codebooks/rotation.
- `io.save_base` / `io.save_linkage`: Only used for **in-memory** pipelines. Large-scale mode writes to disk stores instead.
- Default HDF5 results paths are organized as `results/<dataset.name>/<io.pre_fix>/...` to avoid cluttering all H5 files
  in a single dataset directory.

### Init-Linkage Mode (`train.init_linkage_mode`)

The init-linkage stage builds a single-codebook linkage using only `C_root` to bootstrap `C_one`.
Three modes are available:

| Mode | `train.init_linkage_mode` | Description |
|---|---|---|
| **Legacy** | `legacy_root_only` (default) | Single-codebook linkage with C_root. All ILS rounds use VarRoot (layer0 perturbable + full root ICM scan over h0). Highest init quality; most expensive for large h0. |
| **Hybrid** | `hybrid` | Same legacy C_root-only linkage path, but with hybrid ILS policy: the **first** GPU ILS round runs full VarRoot (allows root perturbation + root ICM GEMM over h0), **subsequent** ILS rounds downgrade to ConstRoot (layer0 frozen, root ICM GEMM skipped). Significant GPU time savings for large h0 while preserving high-quality initial root assignments from the first round. |
| **Fast** | `fast_init` / `fast` | Seeds C_one from C_root[0] via spherical k-means mapping (h0_root → h0_one), then runs the two-codebook linkage builder (like iteration-linkage). Fastest; trades some init quality for speed. |

**Hybrid ILS internals** (GPU path):

In `legacy_root_only` mode, each ILS round of the large-root CUDA encoder does:
1. **Perturb** all layers (including root/layer0) → `LinkageCopyAndPerturbLargeRootCodesWithSampleIdsFixed`
2. **Rebuild** root-dependent terms (`xC0`, `norm0`, `g0s`) via `ComputeXc0AndNorm0FromRootCodes` + `GatherG0sFromTable`
3. **ICM** over all layers, including `jlayer==0` root ICM (exact scan over h0 → tiled GEMM `C0^T * re0`, expensive)
4. **Accept** improved candidate (VarRoot accept copies root + small codes + cost)
5. **Rebuild** current `g0s` (root may have changed)

In `hybrid` mode, the first round (outer=0) runs the full VarRoot sequence above.
Subsequent rounds (outer≥1) run a ConstRoot variant:
1. **Perturb** only layers 1..m-1 (root frozen) → `LinkageCopyAndPerturbLargeRootCodesSkipRootWithSampleIdsFixed`
2. **Copy** root terms from current best via D2D memcpy (no GEMM)
3. **ICM** skips `jlayer==0` entirely (no root scan GEMM)
4. **Accept** improved candidate (VarRoot accept copies root unchanged + small codes + cost)
5. **Skip** `g0s` rebuild (root unchanged)

This eliminates the dominant cost (root ICM tiled GEMM over h0) from all but the first ILS round.

**Related config keys:**
- `train.init_linkage_icm_round` (int, default -1): override `base.linkage.icm_round` for init-linkage only. `-1` = use base value.
- `train.init_linkage_ils_rounds` (int, default -1): override `base.linkage.ils_rounds` for init-linkage only. `-1` = use base value.

### Training Resume & Checkpoints (`train.init_enabled`, `train.ckpt.*`)

The large/streaming pipeline supports resuming OPQ/global-R iterations from a previously saved train HDF5.

- `train.init_enabled=true` (default): run the full init stage (RVQ init + basic init + init-linkage), then run OPQ/global-R iterations.
- `train.init_enabled=false`: skip init stage and **resume** from `io.train_file` selected by `io.load_date` + `io.load_seq`,
  then run only OPQ/global-R iterations.
  - In resume mode, `train.max_R_iters` is interpreted as **additional** iterations to run (not an absolute total).
- Periodic checkpoints:
  - `train.ckpt.enabled=true`
  - `train.ckpt.every_R = N` (N>0) saves a checkpoint every N completed OPQ/global-R iterations.
  - Additionally, when periodic checkpoints are enabled and `train.init_enabled=true`, the large pipeline also writes
    an **init checkpoint** at `R_iters=0` right after the init stage completes. This makes init results resumable even
    if the run stops before the first periodic R checkpoint (e.g. before `R_iters=N`).

Checkpoint metadata:
- The train HDF5 stores `meta/train/R_iters` (completed iterations) and `meta/train/resume_sig_hex` (a filtered config signature).
- When resuming, the current `ComputeTrainResumeSigU64(config)` is compared against the checkpoint signature to catch semantic mismatches.
  This signature intentionally ignores knobs like `train.max_R_iters`, `train.init_enabled`, `train.ckpt.*`, and all `runtime.*` / `io.*` / `large.*`.
- For convenience (no HDF5 viewer needed), each saved checkpoint also writes sidecars:
  - `<train.h5>.checkpoint.txt`
  - `<train.h5>.config_snapshot.txt`

### Evaluation (`eval.*`)
- **Base Recall**: `eval.base.enabled=true`, `eval.base.use_ivf=true` (scans IVF lists from disk).
- **Linkage Recall**: `eval.linkage.enabled=true`, `eval.linkage.use_ivf_disk=true`.
  - Disk recall uses per-cluster reconstruction and only returns real IDs.
  - `eval.bench.quiet=true`: suppress detailed timing breakdowns and keep only QPS-oriented eval logs.
  - `eval.linkage.archive_eval_result=true`: archive recall + timing summaries into `eval_result/recall_<label>_0x<hash>.txt`.
  - `large.archive_log=true`: archive the full runtime console log (training start through eval end) into `<run_root>/log/run_0001.log`, `<run_root>/log/run_0002.log`, ... without overwriting older logs.
  - Experimental CUDA recall knobs (eval-only; do not affect large store hashes):
    - `eval.linkage.gpu_scan_enable`: GPU local top-k per (query,cluster), CPU merge.
    - `eval.linkage.gpu_scan_max_nc`: safety bound for the GPU path.
    - `eval.linkage.gpu_norm_enable`: GPU norm2-prep for per-cluster `r_norm2` (coeff codec / int8 path; CPU fallback otherwise).

### Coefficient Compression (`large.linkage_coeff_codec.*`)
Optional compression for linkage coefficients:
- `bits_per_layer`: signed bits per layer (or size 1 to broadcast).
- `granularity`: `cluster` (one stream per group) or `layer`.
- `q_refine_*`: Options for ICM-like quantization refinement.
- `eval.linkage.coeff_mode`: `float`, `int8`, or `both` during evaluation.

### Advanced: Fixed Bounds
- **Max Layers**: `m <= 16` (compile-time bound).
- **Beam Width**: Specialized fast CUDA kernels for `H=2` and `H=4`.
- **BLAS Threading**: Default single-threaded to avoid OpenMP conflicts; temporarily increased for large isolated GEMMs.

## Complete Configuration Reference

This section is the **authoritative config encyclopedia** for [include/stlq/common/config.h](include/stlq/common/config.h).
Older topic-specific sections above are still useful for tuning guidance, but the tables below are the complete index.

Conventions:

- “Config key” means a normal key that can appear in a config file or be passed by `--set key=value`.
- “Internal/metadata” means the field exists in `config.h` but is not a normal user-facing knob.
- Defaults below are the C++ defaults in `config.h` / `DefaultConfig()` before dataset/config-file overrides.
- If two keys interact strongly, the main linkage is called out in **Notes**.

### `dataset.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `dataset.name` | string / `SIFT1M` | Dataset preset name. | Drives default file paths and some dataset-specific defaults. Built-in presets include `SIFTSMALL`, `SIFT1M`, `GIST1M`, `SIFT1B`, `DEEP1M`, `DEEP10M`, `MSONG`, `GLOVE100`. |
| `dataset.data_root` | string / `../data` | Root directory for dataset files. | Relative paths below are resolved from here. |
| `dataset.train_path` | string / empty | Explicit train/learn vector file. | Alias: `dataset.learn_path`. Empty means “use dataset preset default”. |
| `dataset.base_path` | string / empty | Explicit base vector file. | Empty means “use dataset preset default”. |
| `dataset.query_path` | string / empty | Explicit query vector file. | Empty means “use dataset preset default”. |
| `dataset.groundtruth_path` | string / empty | Explicit groundtruth file. | Alias: `dataset.gt_path`. |
| `dataset.groundtruth_add1` | bool-int / `0` | Whether GT ids are 1-based and need `-1` normalization. | Useful for some legacy binary GT files. |
| `dataset.ntrain` | int / `100000` | Train-set truncation count. | Sets `dataset.ntrain_set=true` internally. |
| `dataset.nbase` | int / `1000000` | Base-set truncation count. | Sets `dataset.nbase_set=true` internally. |
| `dataset.nquery` | int / `10000` | Query count. | Sets `dataset.nquery_set=true` internally. |
| `dataset.k` | int / `100` | Recall top-k. | Sets `dataset.k_set=true` internally. |

### `model.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `model.m` | int / `5` | Number of quantization layers. | Affects `C_root`, `C_one`, linkage depth, coeff layout, etc. |
| `model.h0` | int / `256` | Root/IVF layer codebook size. | Used when `model.h_vec` is not explicitly set; layers `1..m-1` default to `256`. |
| `model.h_vec` | int array / empty | Advanced per-layer codebook size vector. | If set, this overrides `model.h0`; short arrays are resized to `m` by repeating the last value, and long arrays are truncated. The normalized full vector is what enters store hashes. |
| `model.h0_one` | int / `256` | Layer-0 size of `C_one`. | Usually kept `<=256` so `code0_one` can remain `u8`. |

### `runtime.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `runtime.omp_threads` | int / `0` | OpenMP thread count. | `0` means use environment / runtime default. |
| `runtime.use_cuda` | bool / `false` | Enable CUDA kernels. | Requires building with `STLQ_ENABLE_CUDA=ON`. |
| `runtime.cuda_device` | int / `0` | CUDA device ordinal. | Used by GPU paths and ctx pools. |
| `runtime.cuda_mode` | string / `strict` | CUDA numeric policy. | `strict` prioritizes reproducibility; `fast` enables faster/non-strict GEMM behavior. |
| `runtime.cuda_allow_tf32` | bool / `false` | Allow TF32 GEMM in strict mode. | Ignored by `fast`, which already enables TF32. |
| `runtime.cuda_cublas_workspace_mb` | int / `0` | cuBLAS workspace hint in MiB. | Can affect GEMM algorithm choice and determinism. |
| `runtime.cuda_pool_size` | int / `4` | CUDA ctx pool size for linkage stages. | Larger can improve throughput but uses more VRAM. |
| `runtime.cuda_pool_size_init_linkage` | int / `0` | Init-linkage-only ctx pool size override. | `0` means reuse `runtime.cuda_pool_size`. |
| `runtime.cuda_linkage_single_gpu_min_candidates` | int / `16` | Minimum candidate count before GPU candidate eval is worthwhile. | Used by linkage build dispatch heuristics. |
| `runtime.cuda_linkage_single_gpu_max_candidates` | int / `0` | Maximum candidate count for selected GPU path. | `0` means uncapped. |
| `runtime.cuda_linkage_use_device_rfull` | bool / `true` | Keep `R_full` on device for candidate gathering. | Reduces CPU packing overhead in CUDA linkage build. |
| `runtime.cuda_linkage_inner_many_nodes_enable` | bool / `true` | Enable many-nodes batching for inner-candidate evaluation. | Preserves semantics; only changes batching. |
| `runtime.cuda_linkage_inner_many_nodes_target_nodes` | int / `64` | Target node count per many-nodes batch. | Works with `runtime.cuda_linkage_many_nodes_max_pairs`. |
| `runtime.cuda_linkage_many_nodes_max_pairs` | int / `65536` | Max `(node,parent)` pairs per many-nodes batch. | Main throughput/memory cap. |
| `runtime.cuda_linkage_many_nodes_max_pairs_init_linkage` | int / `0` | Init-linkage-only batch pair override. | `0` means use `runtime.cuda_linkage_many_nodes_max_pairs`. |
| `runtime.cuda_linkage_many_nodes_preflush_target_pairs` | int / `2048` | Opportunistic preflush target in pair count. | Does not skip work; only controls when to flush. |
| `runtime.linkage_async_io` | bool / `false` | Enable async cluster IO prefetch for linkage build. | Needs list-order raw payload availability. |
| `runtime.linkage_async_io_depth` | int / `6` | Prefetch queue depth for linkage build. | Higher depth uses more RAM. |
| `runtime.basic_async_io` | bool / `false` | Enable async block prefetch for streaming basic encoding. | Independent of `runtime.linkage_async_io`. |
| `runtime.basic_async_io_depth` | int / `2` | Basic-encode prefetch queue depth. | Higher depth uses more RAM. |
| `runtime.basic_async_io_mb` | int / `0` | RAM budget for basic async prefetch. | `0` disables budget enforcement. |
| `runtime.basic_async_write` | bool / `true` | Enable async writer thread for streaming basic encoding. | Mainly useful for GPU non-hybrid path. |
| `runtime.basic_async_write_depth` | int / `4` | Max in-flight tiles for async writer. | Bounds pending write memory. |
| `runtime.basic_async_write_mb` | int / `0` | RAM budget for async writer queue. | `0` disables budget enforcement. |
| `runtime.precomp_large_root_g0s_transpose` | bool / `true` | Build `G0S^T` CPU cache for large-root precomp. | Speeds CPU ICM/ILS, costs RAM. |
| `runtime.precomp_large_root_g0s_transpose_max_mb` | int / `0` | RAM cap for the transpose cache. | `0` means uncapped. |
| `runtime.basic_hybrid_enable` | bool / `false` | Run streaming basic encoding with parallel GPU + CPU lanes. | Requires ordered writeback via reorder buffer. |
| `runtime.basic_hybrid_cpu_stride` | int / `0` | Send roughly 1/stride blocks to CPU lane. | `0` means effectively GPU-only even if hybrid is on. |
| `runtime.basic_hybrid_cpu_threads` | int / `0` | CPU-lane thread count in hybrid mode. | `0` means reuse global/runtime default. |
| `runtime.basic_hybrid_reorder_depth` | int / `6` | Reorder-buffer depth for hybrid encoding. | Must be large enough to absorb lane skew. |
| `runtime.basic_hybrid_inflight_mb` | int / `0` | RAM budget for hybrid in-flight blocks. | `0` disables budget enforcement. |
| Runtime key compatibility | alias-compatible | This branch documents the renamed linkage runtime keys. | Old aliases are still accepted by the parser for backward compatibility. |
| `runtime.cuda_linkage_same_layer_window_min_pairs` | int / `1` | Pair threshold for same-layer dynamic window batching. | Used in good-cluster same-layer dynamic scheduling. |
| `runtime.cuda_linkage_same_layer_tiny_forced_cpu_enable` | bool / `false` | Route very small forced flushes to CPU. | Changes execution backend only, not semantics. |
| `runtime.cuda_linkage_init_same_layer_tiny_forced_cpu_enable` | bool / `false` | Allow tiny-CPU same-layer fallback in large-root init-linkage. | Init-stage specific escape hatch. |
| `runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_pairs` | int / `16` | Max pairs for the tiny-CPU path. | Non-positive disables the tiny-CPU route. |
| `runtime.cuda_linkage_same_layer_tiny_forced_cpu_max_nodes` | int / `0` | Optional max nodes for tiny-CPU slices. | `0` means no explicit node cap. |
| `runtime.cuda_linkage_same_layer_block_nodes` | int / `64` | Same-layer dynamic window size. | Larger reduces launches, may reduce responsiveness. |
| `runtime.cuda_linkage_same_layer_single_gpu_min_candidates` | int / `0` | GPU threshold override for residual dynamic candidates. | `0` means reuse `runtime.cuda_linkage_single_gpu_min_candidates`. |
| `runtime.cuda_linkage_same_layer_preflush_single_shot` | bool / `false` | Batch dynamic candidate views inside the current window. | Preserves commit order. |
| `runtime.cuda_linkage_same_layer_window_max_pairs` | int / `0` | Pair cap for window-local dynamic flush. | `0` means use the regular batch cap. |
| `runtime.cuda_linkage_same_layer_window_max_pending_nodes` | int / `0` | Cap pending nodes in same-layer dynamic window. | `0` means uncapped / use full remaining window. |
| `runtime.cuda_linkage_same_layer_preflush_target_pairs` | int / `0` | Dedicated pair threshold for same-layer dynamic preflush. | `0` disables opportunistic preflush; negative means legacy fallback behavior. |
| `runtime.cuda_linkage_same_layer_preflush_target_nodes` | int / `0` | Dedicated node-count trigger for same-layer dynamic preflush. | Helps sparse dynamic candidate patterns. |
| `runtime.cuda_linkage_bad_frozen_window_enable` | bool / `true` | Enable batching for bad-cluster / virtual-front frozen candidates. | Window-local dynamic part remains serial for correctness. |
| `runtime.cuda_linkage_bad_frozen_window_nodes` | int / `64` | Window size for bad-cluster batching. | Used with `runtime.cuda_linkage_bad_frozen_window_max_pairs`. |
| `runtime.cuda_linkage_bad_frozen_window_max_pairs` | int / `20000` | Pair cap for bad-window batches. | Controls temporary memory and launch granularity. |
| `runtime.cuda_linkage_chunk_max_pairs` | int / `0` | Hard chunk cap inside many-nodes evaluator. | `0` means no manual chunking. |
| `runtime.cuda_linkage_chunk_max_pairs_init_linkage` | int / `0` | Init-linkage-only evaluator chunk cap. | `0` means reuse the normal cap. |
| `runtime.cuda_linkage_mem_budget_mb` | int / `0` | VRAM budget hint for linkage build. | Used to derive safe chunking heuristics. |
| `runtime.cuda_linkage_mem_budget_mb_init_linkage` | int / `0` | Init-linkage-only VRAM budget hint. | `0` means reuse the normal budget. |
| `runtime.cuda_linkage_eval_async_pinned_mb` | int / `0` | Pinned host memory budget for async linkage eval overlap. | `0` disables this async staging path. |
| `runtime.cuda_linkage_eval_async_pinned_mb_init_linkage` | int / `0` | Init-linkage-only pinned budget. | `0` means reuse the normal value. |
| `runtime.cuda_linkage_large_root_xc0_chunk_mb` | int / `256` | Large-root root-selection `xC0` chunk size in MiB. | Bigger chunks reduce GEMM count but raise VRAM pressure. |
| `runtime.cuda_linkage_large_root_xc0_chunk_mb_init_linkage` | int / `0` | Init-linkage-only `xC0` chunk override. | `0` means reuse the normal value. |
| `runtime.cuda_linkage_wait_for_ctx` | bool / `true` | Wait for a free CUDA ctx instead of falling back to CPU. | Does not affect explicit tiny-CPU routing. |
| `runtime.c_one_update_shards` | int / `0` | Worker/shard count for exact `C_one` LS accumulation. | `0` means auto; trades RAM for speed. |
| `runtime.c_one_update_shards_init_linkage` | int / `0` | Init-linkage-only `C_one` shard override. | `0` means reuse the normal shard count. |

### `train.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `train.enabled` | bool / `true` | Run quantizer training. | If false, load train artifacts from `io.train_file`. |
| `train.init_enabled` | bool / `true` | Run the expensive init stage before global OPQ iterations. | False means “resume from an existing train checkpoint”. |
| `train.ckpt.enabled` | bool / `false` | Enable periodic train checkpoints. | Nested under `train.ckpt.*`. |
| `train.ckpt.every_R` | int / `0` | Checkpoint period in completed `R` iterations. | `0` disables periodic checkpoints. |
| `train.use_opq_rotation` | bool / `true` | Enable OPQ/global rotation update. | Turning this off keeps `R=I` after init. |
| `train.log_metrics` | bool / `true` | Print training MSE/linkage summary metrics. | Disabling trims extra analysis passes. |
| `train.log_linkage_pre_c1` | bool / `false` | Print linkage summary before updating `C_one`. | Diagnostic-only extra scan. |
| `train.exit_after_rvq_init` | bool / `false` | Stop immediately after RVQ init. | Profiling / debugging aid. |
| `train.ckpt_after_init_basic` | bool / `false` | Write an extra checkpoint after init basic / `C_root` update. | Intended for profiling init-linkage separately. |
| `train.exit_after_ckpt_init_basic` | bool / `false` | Exit after writing the above checkpoint. | Leaves a pre-init-linkage checkpoint on disk. |
| `train.ils_iters` | int / `8` | Training-stage basic encode ILS iterations. | Used by large streaming train-basic pass. |
| `train.icm_iters` | int / `4` | Training-stage basic encode ICM iterations. | Pairs with `train.ils_iters`. |
| `train.perturb_k` | int / `3` | Training-stage perturbation count. | Used by ILS-style refinement. |
| `train.max_R_iters` | int / `20` | Number of global OPQ / `R` iterations. | Init stage is separate; this counts only global rounds. |
| `train.kmeans_iters` | int / `40` | Main k-means iterations for codebook training. | Applies to RVQ init / related training stages. |
| `train.kmeans_tol` | double / `1e-6` | k-means convergence tolerance. | Used by solver-side stopping. |
| `train.kmeans_init` | string / `random` | k-means init strategy. | Typical values: `random`, `kmeans++`, `kmeans||`. |
| `train.init_samples` | int / `100000` | Sample count used for initialization/seeding. | Used by some init modes and samplers. |
| `train.kmeans_initial_weight` | double / `1.0` | Initial adaptive sample weight. | Part of annealed k-means weighting. |
| `train.kmeans_min_weight` | double / `0.1` | Minimum adaptive sample weight. | Lower bound during annealing. |
| `train.kmeans_outlier_quantile` | double / `0.85` | Outlier quantile for adaptive weighting. | Used when annealing is enabled. |
| `train.kmeans_cost_threshold` | double / `0.115` | Cost threshold for weight shaping. | Interacts with quantile/annealing logic. |
| `train.kmeans_annealing_factor` | double / `0.9` | Weight annealing decay factor. | Larger means slower decay. |
| `train.kmeans_warmup_iters` | int / `2` | Warmup iterations before annealing. | Used by adaptive-weight modes. |
| `train.kmeans_quantile_bins` | int / `65536` | Histogram bins for quantile approximation. | Streaming anneal precision/speed trade-off. |
| `train.kmeans_anneal_no_spill` | bool / `true` | Use no-spill two-pass anneal scan instead of spill-to-disk path. | Streaming-only optimization. |
| `train.kmeans_anneal_mode` | string / `onepass` | Adaptive anneal mode. | `onepass` is faster; `twopass` is legacy two-scan mode. |
| `train.kmeans_anneal_weights_device` | bool / `true` | Prefer device-resident global weights. | Best-effort; may fall back if memory is tight. |
| `train.kmeans_force_disable_device_weights` | bool / `false` | Force-disable device-side weights. | Debug/testing switch. |
| `train.kmeans_streaming` | bool / `false` | Use out-of-core streaming k-means. | Required for very large training sets. |
| `train.kmeans_cache_xnorm_device` | bool / `true` | Cache full normalized `X_norm` on GPU if it fits. | Speeds moderate-sized streaming runs. |
| `train.kmeans_device_cache_mb` | int / `0` | Partial GPU cache size for normalized training data. | `0` disables prefix caching. |
| `train.kmeans_pin_host_x` | bool / `true` | Keep a pinned host mirror of `X`. | Reduces repeated H2D cost when full device cache is not used. |
| `train.kmeans_bvecs_use_u8` | bool / `true` | For `.bvecs`, read `u8` blocks and normalize on GPU. | Saves CPU-side u8→f32 expansion. |
| `train.kmeans_tmp_dir` | string / empty | Temp directory for streaming k-means artifacts. | Empty falls back to `large.tmp_dir`. |
| `train.kmeansll_gpu_enable` | bool / `true` | Enable GPU-first kmeans|| seeding. | Used only when `train.kmeans_init=kmeans||`. |
| `train.kmeansll_rounds` | int / `4` | Number of kmeans|| rounds. | More rounds improve seeding quality. |
| `train.kmeansll_oversample` | double / `16.0` | kmeans|| oversampling factor. | Larger gives more seeding candidates. |
| `train.kmeansll_candidate_cap` | int / `0` | Cap on kmeans|| candidate center count. | `0` means auto. |
| `train.kmeansll_seed` | uint64 / `0` | Seed for kmeans|| seeding. | `0` means derive from the main RNG. |
| `train.kmeansll_hier_enable` | bool / `false` | Enable hierarchical seeding for very large `K`. | Coarse→fine init path. |
| `train.kmeansll_hier_threshold` | int / `16384` | Threshold to activate hierarchical seeding. | Relevant only if `train.kmeansll_hier_enable=true`. |
| `train.kmeansll_hier_fixed_k1` | int / `256` | Coarse `K1` for hierarchical seeding. | `<=0` reserved for auto in the future. |
| `train.kmeans_large_k_threshold` | int / `8192` | Threshold to enable large-`K` hierarchical assignment. | Used by coarse→fine assign path. |
| `train.kmeans_large_k_hier2_enable` | bool / `true` | Enable 2-level coarse→fine assignment for large `K`. | Assignment optimization only. |
| `train.kmeans_large_k_fixed_k1` | int / `0` | Fixed coarse `K1` for large-`K` assignment. | `0` means auto. |
| `train.kmeans_large_k_k1_min` | int / `128` | Lower bound for auto `K1`. | Used when `fixed_k1=0`. |
| `train.kmeans_large_k_k1_max` | int / `1024` | Upper bound for auto `K1`. | Used when `fixed_k1=0`. |
| `train.kmeans_large_k_k1_pow2` | bool / `true` | Snap auto `K1` to a power of two. | Convenience for implementation/layout. |
| `train.kmeans_large_k_top_coarse` | int / `1` | Number of top coarse groups to expand. | Recall/speed trade-off in hier2 assignment. |
| `train.kmeans_large_k_hier2_train` | bool / `true` | Use hier2 during training assignment. | Train-time speed/quality trade-off. |
| `train.kmeans_large_k_hier2_encode` | bool / `true` | Use hier2 during encoding. | Encode-time speed/quality trade-off. |
| `train.save_init_codes` | bool / `false` | Save RVQ init codes for debugging. | Optional diagnostic output. |
| `train.init_codes_path` | string / empty | Output path for saved init codes. | Used only if `train.save_init_codes=true`. |
| `train.encode_only_after_layer` | bool / `false` | Debug flag for partial encode flow after a layer boundary. | Specialized developer/testing knob. |
| `train.seed` | int / `1985326` | Main training RNG seed. | Affects sampling, k-means init, etc. |
| `train.init_linkage_mode` | string / `legacy_root_only` | Init-linkage strategy. | Main values: `legacy_root_only`, `fast_init`, `hybrid`. |
| `train.init_linkage_icm_round` | int / `-1` | Init-linkage-specific ICM rounds. | `-1` means reuse `base.linkage.icm_round`. |
| `train.init_linkage_ils_rounds` | int / `-1` | Init-linkage-specific ILS rounds. | `-1` means reuse `base.linkage.ils_rounds`. |
| `train.init_linkage_hybrid_varroot_rounds` | int / `1` | Number of VarRoot rounds in hybrid init-linkage mode. | Only relevant when `train.init_linkage_mode=hybrid`. |

### `train.linkage.*` and `base.linkage.*` (shared `LinkageBuildConfig`)

The two prefixes share the same fields but apply to different phases:

- `train.linkage.*`: training-stage linkage build / iteration-linkage.
- `base.linkage.*`: baseset linkage build.

| Key suffix | Type / default | Meaning | Notes |
|---|---|---|---|
| `.enabled` | bool / `true` | Enable that linkage-build stage. | `base.linkage.enabled=false` skips baseset `linkage_list` build. |
| `.root_percentile` | double / `0.01` | Root-candidate percentile/selection policy. | Root search heuristic. |
| `.num_layers` | int / `16` | Max linkage layers. | Independent of `model.m`; this is the linkage depth budget. |
| `.max_depth` | int / `40` | Maximum parent depth. | Hard stop for linkage expansion. |
| `.knn_k` | int / `15` | KNN candidate count. | Candidate frontier size. |
| `.depth_k` | int / `15` | Candidate count per depth expansion. | Controls breadth of depth exploration. |
| `.icm_round` | int / `1` | ICM rounds inside linkage refinement. | Quality/speed trade-off. |
| `.use_ils` | bool / `true` | Enable ILS linkage refinement. | If false, rely on simpler linkage construction. |
| `.ils_rounds` | int / `4` | ILS rounds. | Applies only if `.use_ils=true`. |
| `.ils_perturb_layers` | int / `3` | Perturbed layers during ILS. | Search aggressiveness knob. |
| `.seed` | int / `38251450` | RNG seed for that linkage stage. | Separate from `train.seed`. |

### `base.encode.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `base.encode.enabled` | bool / `true` | Run baseset basic encoding. | If false, base artifacts must be loaded from disk. |
| `base.encode.use_abs` | bool / `true` | Use absolute-value-aware encode path. | Training basic pass sets its own override (`false`). |
| `base.encode.H_beam` | int / `2` | Beam width in prefix beam search. | Specialized fast paths exist for small widths. |
| `base.encode.ils_iters` | int / `32` | ILS iterations for base encoding. | Quality/speed trade-off. |
| `base.encode.icm_iters` | int / `4` | ICM iterations for base encoding. | Used after beam init. |
| `base.encode.perturb_k` | int / `3` | Perturbation count in ILS. | Search aggressiveness knob. |
| `base.encode.hnorms` | int / `256` | Norm/cache sizing knob for encode backend. | Backend implementation detail. |
| `base.encode.seed` | int / `38251450` | RNG seed for baseset basic encoding. | Affects randomized refinement order. |

### `virtual.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `virtual.enabled` | bool / `false` | Enable virtual-root augmentation. | Adds virtual nodes in linkage construction. |
| `virtual.virtual_ratio` | double / `0.10` | Ratio of virtual nodes to real nodes. | Combined with min/max bounds. |
| `virtual.good_fraction` | double / `0.6` | Fraction of clusters considered “good”. | Used by bad/good cluster heuristics. |
| `virtual.min_virtual` | int / `1` | Minimum virtual nodes per cluster. | Lower clamp. |
| `virtual.max_virtual` | int / `2000` | Maximum virtual nodes per cluster. | Upper clamp. |
| `virtual.alpha_bad` | float / `0.3` | Weight/scaling for bad clusters. | Affects virtual-node weighting and some LS updates. |
| `virtual.use_fixed_virtual_per_cluster` | bool / `false` | Ignore ratio and use a fixed count per cluster. | If true, use `virtual.fixed_virtual_per_cluster`. |
| `virtual.fixed_virtual_per_cluster` | int / `256` | Fixed virtual node count per cluster. | Active only if `use_fixed_virtual_per_cluster=true`. |
| `virtual.umap_knn_k` | int / `50` | Neighbor count in UMAP-like virtual augmentation. | Construction heuristic. |
| `virtual.local_connectivity` | int / `1` | UMAP-like local connectivity parameter. | Affects virtual graph density. |
| `virtual.overlap_thr` | double / `0.9` | Overlap threshold for virtual augmentation. | Controls merge/selection behavior. |
| `virtual.prefer_peaks` | bool / `true` | Prefer peak-like anchors when building virtual roots. | Heuristic behavior. |
| `virtual.anchor_neighbor_k` | int / `16` | Anchor neighbor count. | Used in anchor/virtual root generation. |

### `hnsw.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `hnsw.M` | int / `48` | HNSW out-degree / graph connectivity. | Used in graph-based candidate generation. |
| `hnsw.candidate_multiplier_good` | int / `1` | Candidate multiplier for good clusters. | Heuristic search expansion knob. |
| `hnsw.candidate_multiplier_bad` | int / `1` | Candidate multiplier for bad clusters. | Heuristic search expansion knob. |
| `hnsw.ef_construction_cap` | int / `0` | Optional cap for internal linkage-build HNSW construction `ef`. | `<=0` disables the cap and preserves legacy run-tag/hash behavior; `>0` clamps internal linkage-build graph construction and becomes store-identity relevant. |

### `io.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `io.pre_fix` | string / `test` | Run/output prefix. | Feeds default file names and large run tags. |
| `io.config_file` | string / empty | Metadata copy of the `--config` path. | Usually set by CLI parsing; not commonly written manually. |
| `io.train_file` | string / empty/unset | Train-result HDF5 path. | If not explicitly set, derived from dataset + prefix. |
| `io.base_file` | string / empty/unset | Base-result HDF5 path. | If not explicitly set, derived from dataset + prefix. |
| `io.linkage_file` | string / empty/unset | Linkage-result HDF5 path. | If not explicitly set, derived from dataset + prefix. |
| `io.save_train` | bool / `false` | Save train result HDF5. | Also used by large-streaming train path. |
| `io.save_base` | bool / `false` | Save base result HDF5. | Non-large/in-memory flows mainly. |
| `io.save_linkage` | bool / `false` | Save linkage result HDF5. | Non-large/in-memory flows mainly. |
| `io.hdf5_layout` | string / `julia` | Matrix layout convention in HDF5. | `julia` and `cxx` are supported. |
| `io.load_date` | string / empty | Date suffix used when loading dated artifacts. | `raw` means “load exact file path without date/seq suffix”. |
| `io.load_seq` | string / empty | Sequence suffix used when loading dated artifacts. | Digits only; empty means “no sequence suffix”. |
| `io.index_dtype` | string / `int` | Index-array dtype for HDF5 IO. | Typical values: `int`, `uint`. |

### `advanced.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `advanced.eval_only` | bool / `false` | Convenience override for recall-only runs. | Forces `train.enabled=false`, `base.encode.enabled=false`, and `base.linkage.enabled=false` at startup; eval toggles are left unchanged. |

### `large.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|--|
| `large.enabled` | bool / `false` | Enable large/out-of-core pipeline. | Main switch for streaming train/base/linkage paths. |
| `large.train_format` | string / `auto` | Train-file format override. | Resolved by the shared dataset reader factory. Typical values: `auto`, `fvecs`, `bvecs`, `fbin`. |
| `large.base_format` | string / `auto` | Base-file format override. | Resolved by the same factory for large base encoding and analysis tools. |
| `large.query_format` | string / `auto` | Query-file format override. | Resolved by the same factory for large disk-eval query loading. |
| `large.gt_format` | string / `auto` | Groundtruth format override. | Resolved by the same factory. Typical values: `auto`, `ivecs`, `ibin`. |
| `large.output_dir` | string / `outputs` | Root output directory for large pipeline artifacts. | Run roots live under this tree. |
| `large.tmp_dir` | string / `tmp` | Temp/work directory for large pipeline stages. | Used for bucket/list/IVF build scratch data. |
| `large.train_block` | int / `200000` | Streaming block size for training vectors. | Affects memory and IO granularity. |
| `large.base_block` | int / `200000` | Streaming block size for base vectors. | Affects memory and IO granularity. |
| `large.base_shard_size` | int / `2000000` | Base-basic output shard size. | Controls `codes_shard_*`, `coeffs_shard_*` layout. |
| `large.cluster_bucket_size` | int / `256` | Number of clusters per bucket group. | Used to avoid too many small files. |
| `large.bucket_flush_mb` | int / `256` | Flush threshold for bucket buffers. | IO buffering/memory trade-off. |
| `large.write_vector_bucket` | bool / `true` | Store raw vectors in bucket-friendly form. | Helps later per-cluster processing. |
| `large.write_basic_to_bucket` | bool / `true` | Store basic codes/coeffs in bucket-friendly form. | Helps list/linkage construction. |
| `large.base_linkage_store_coeffs_f32` | bool / `true` | Keep float coeff payloads in baseset `linkage_list`. | If false, int8 coeff codec must exist for linkage eval. |
| `large.profile_timing` | bool / `false` | Print stage-level timing for streaming pipeline internals. | Debug/profiling only. |
| `large.archive_log` | bool / `false` | Archive full runtime console logs under `<run_root>/log/`. | File suffix will auto increase |
| `large.protect_existing_outputs` | bool / `false` | Refuse auto-delete/rebuild of existing outputs. | Safety switch for expensive runs. |
| `large.linkage_store_parent_u32` | bool / `false` | Also persist `linkage_list/parent.u32`. | Default prefers LOUDS-only parent storage to save disk. |
| `large.linkage_checkpoint` | bool / `false` | Enable checkpoint/resume for baseset `linkage_list` build. | Large streaming linkage build only. |
| `large.base_basic_checkpoint` | bool / `false` | Enable checkpoint/resume for baseset `base_basic` encoding. | Sequential block-based resume. |

#### `large.cleanup.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `large.cleanup.enabled` | bool / `false` | Enable staged cleanup of regenerable disk artifacts. | Independent of store hashes and training semantics. |
| `large.cleanup.preset` | string / `eval_both` | Cleanup preset. | Main values: `none`, `dev_all`, `eval_both`, `eval_float_min`, `eval_int8_min`, `custom`. |
| `large.cleanup.keep_base_basic` | tri-state int / `-1` | Override whether to keep `base_basic/`. | `-1` means use preset, `0` delete, `1` keep. |
| `large.cleanup.keep_base_basic_cluster_id` | tri-state int / `-1` | Override whether to keep `base_basic/cluster_id.u32`. | Useful once IVF CSR lists already exist. |
| `large.cleanup.keep_base_basic_ivf_lists` | tri-state int / `-1` | Override whether to keep `base_basic/ivf_offsets.u64` + `ivf_ids.u32`. | Needed for base IVF eval and some rebuilds. |
| `large.cleanup.keep_base_list` | tri-state int / `-1` | Override whether to keep `base_list/`. | `-1` means use preset. |
| `large.cleanup.keep_base_list_raw` | tri-state int / `-1` | Override whether to keep raw vector payloads under `base_list/`. | Only deletable once dependent stages are complete. |
| `large.cleanup.keep_linkage_list_f32` | tri-state int / `-1` | Override whether to keep `linkage_list/*.f32`. | Needed for float-coeff recall or codec rebuild. |
| `large.cleanup.keep_linkage_parent_u32` | tri-state int / `-1` | Override whether to keep `linkage_list/parent.u32`. | Redundant if LOUDS is available. |

#### `large.linkage_coeff_codec.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `large.linkage_coeff_codec.enabled` | bool / `false` | Enable int8+scale+Huffman coefficient codec. | Needed when dropping float coeff payloads. |
| `large.linkage_coeff_codec.granularity` | string / `cluster` | Codec stream granularity. | Common values: `cluster`, `layer`. |
| `large.linkage_coeff_codec.bits_per_layer` | int array / `[7]` | Signed quantization bits per layer. | Size-1 input is broadcast to all layers. |
| `large.linkage_coeff_codec.p_first_candidates` | double array / `[99.5,99.8,100.0]` | Scale-quantile candidates for the first coeff group/layer. | Search space for codec calibration. |
| `large.linkage_coeff_codec.p_rest_candidates` | double array / `[99.5,99.8,100.0]` | Scale-quantile candidates for remaining groups/layers. | Search space for codec calibration. |
| `large.linkage_coeff_codec.use_weighted_quantile` | bool / `false` | Use weighted quantiles when fitting scale. | Can improve skewed distributions. |
| `large.linkage_coeff_codec.allow_clip` | bool / `true` | Allow clipping during coefficient quantization. | Accuracy/bitrate trade-off. |
| `large.linkage_coeff_codec.fit_scale` | bool / `true` | Fit quantization scale instead of using a trivial rule. | Usually should stay enabled. |
| `large.linkage_coeff_codec.q_refine_sweeps` | int / `3` | Refinement sweeps for quantized coefficients. | `0` disables refinement. q is the integer coefficient. |
| `large.linkage_coeff_codec.q_refine_max_layer` | int / `0` | Max layer count from beginning to refine. | `0` means all layers. |
| `large.linkage_coeff_codec.q_refine_step_limit` | int / `1` | Max $\Delta q$ per refinement update. | `0` means unlimited. |

### `eval.*`

| Key | Type / default | Meaning | Notes |
|---|---|---|---|
| `eval.base.enabled` | bool / `true` | Run base recall evaluation. | Large pipeline base eval can run from disk-IVF store. |
| `eval.linkage.enabled` | bool / `true` | Run linkage recall evaluation. | Uses `linkage_list` disk recall paths. |
| `eval.base.use_ivf` | bool / `false` | Use IVF-style base recall instead of full scan. | Large pipeline usually uses IVF-from-disk. |
| `eval.base.nprobe` | int / `8` | `nprobe` for base IVF recall. | Higher improves recall, lowers QPS. |
| `eval.base.warmup` | int / `0` | Warmup runs for base recall benchmark. | Warmup runs are not reported. |
| `eval.base.repeat` | int / `1` | Timed repeat count for base recall benchmark. | Repeats are summarized by median/average. |
| `eval.linkage.use_ivf_disk` | bool / `false` | Enable disk-IVF linkage recall. | Main large-pipeline linkage eval mode. |
| `eval.linkage.nprobe` | int or int-list / `8` | `nprobe` for linkage IVF recall. | May be a scalar like `32` or a batch list like `[16, 32, 64]`. Batch mode only affects linkage disk recall; preload mode reuses one provider load across probes, lazy mode keeps probes isolated. |
| `eval.linkage.query_block` | int / `256` | Query batch/block size for linkage disk recall. | Affects memory and scheduling. |
| `eval.linkage.warmup` | int / `0` | Warmup runs for linkage recall benchmark. | Warmup runs are not reported. |
| `eval.linkage.repeat` | int / `1` | Timed repeat count for linkage recall benchmark. | Repeats are summarized by median/average. |
| `eval.bench.quiet` | bool / `false` | Suppress detailed bench timings and print QPS-focused logs. | Still prints summary/QPS, not full silence. |
| `eval.linkage.coeff_mode` | string / `auto` | Which coeff representation(s) to evaluate. | Common values: `auto`, `float`, `int8`, `both`. |
| `eval.linkage.parent_louds_enable` | bool / `true` | Prefer LOUDS parent decoding during eval. | Alias family also exists in parser. |
| `eval.linkage.parent_louds_native_eval` | bool / `false` | Use the separate LOUDS-native CPU eval path. | This path queries LOUDS directly instead of materializing `parent[]`. CPU-only in current implementation. |
| `eval.linkage.parent_louds_select_stride` | int / `128` | LOUDS select index sampling stride. | Smaller is faster but larger on disk. |
| `eval.linkage.parent_louds_rank_words_per_super_log2` | int / `4` | LOUDS rank superblock size (`log2(words)`). | Rank/select index tuning knob. |
| `eval.linkage.parent_louds_build_indices` | bool / `false` | Build LOUDS rank/select indices when deserializing. | Only needed for random parent queries (rank/select); the LOUDS-native scan hot path uses a sequential decoder and does not require indices. |
| `eval.linkage.preload_clusters_io_threads` | int / `0` | Parallel I/O thread count for preloading all clusters into RAM before eval. | `<0` = disabled (lazy on-demand loading); `0` = auto ; `>0` = exact N threads. Preload mode also fully loads reusable disk caches such as float norm2 / norm2 LUT; lazy mode reads only touched cluster slices. |
| `eval.linkage.louds_huffman_bench_times` | int / `0` | LOUDS+Huffman CPU decode benchmark on first query block. | `0` = disabled: no wall QPS output; `>0` = run N OMP-parallel decode iterations on the first block's active clusters (size x). Constant time formula: `(avg_wall / x) * linkage_nprobe * nq` added to core wall for QPS. Raw Huffman payloads are freed after the benchmark (or immediately each block when 0). |
| `eval.linkage.gpu_scan_enable` | bool / `false` | Enable experimental GPU local scan/top-k in linkage recall. | Eval-only optimization. |
| `eval.linkage.gpu_scan_tile256` | bool / `false` | Use tile-256 variant for GPU scan kernel. | A/B tuning knob. |
| `eval.linkage.gpu_scan_max_nc` | int / `16384` | Max cluster size allowed on GPU scan path. | Safety/memory guard. |
| `eval.linkage.gpu_scan_cache_mb` | int / `0` | Persistent per-cluster GPU cache size for scan path. | `0` disables caching. |
| `eval.linkage.gpu_norm_enable` | bool / `false` | Enable GPU norm2 preparation in linkage recall. | Eval-only optimization. |
| `eval.disk_norm2_mode` | string / `float` | Disk-recall norm2 representation. | `float` = full float norm2; `lut` = 1D-kmeans centers + uint8 assignment codes. |
| `eval.disk_norm2_lut_kmeans_niter` | int / `25` | Iteration count for the 1D-kmeans used by `eval.disk_norm2_mode=lut`. | Eval-only prep knob. |
| `eval.linkage.ivf_probe_mode` | string / `exact` | Linkage-eval probe selection backend. | Main values: `exact`, `hier2`, `hnsw` (disk-cached centroid HNSW; avoids coarse GEMM). |
| `eval.linkage.ivf_hier2_top_coarse` | int / `1` | Number of coarse groups to expand in `hier2`. | Recall/speed trade-off. |
| `eval.linkage.ivf_hier2_kmeans_niter` | int / `25` | k-means iterations for building hier2 coarse groups. | Relevant only when `ivf_probe_mode=hier2`. |
| `eval.linkage.ivf_hnsw_M` | int / `32` | HNSW out-degree / graph connectivity for centroid probe. | Relevant only when `ivf_probe_mode=hnsw`. Cache dir: `<run_root>/ivf_hnsw/` (alongside `linkage_list/`). |
| `eval.linkage.ivf_hnsw_ef_construction` | int / `200` | HNSW construction `ef` for centroid probe. | Relevant only when `ivf_probe_mode=hnsw`. |
| `eval.linkage.ivf_hnsw_ef_search` | int or int-list / `64` | HNSW runtime `ef_search` for centroid probe. | May be a scalar like `64` or a batch list like `[32, 64, 128]`. When `eval.linkage.ivf_probe_mode=hnsw`, linkage disk recall evaluates the full `nprobe x ef_search` grid and prints a Pareto summary by `r@1`. |
| `eval.linkage.norm2_store_float` | bool / `false` | Persist float-coeff linkage-disk norm2 cache artifacts after eval. | In `float` mode this is `norm2_float.f32`; in `lut` mode this is `norm2_float_lut_*`. |
| `eval.linkage.norm2_store_int8` | bool / `false` | Persist int8-coeff linkage-disk norm2 cache artifacts after eval. | In `float` mode this is `norm2_int8.f32`; in `lut` mode this is `norm2_int8_lut_*`. |
| `eval.linkage.archive_eval_result` | bool / `false` | Save recall/timing summaries under `<run_root>/eval_result/`. | Uses an eval-config hash in file names. |
| `eval.linkage.archive_log` | bool / `false` in struct | Legacy eval-log archival field. | **Not a normal active key** in the current parser; runtime log archival is handled by `large.archive_log`. |

### Internal / metadata fields in `config.h`

These fields exist in `config.h` but are **not ordinary tuning knobs**. They are recorded here so every field in the header is still explained in README.

| Field | Kind | Meaning | Notes |
|---|---|---|---|
| `dataset.ntrain_set` | internal bool | Whether `dataset.ntrain` was explicitly set by user/config. | Used to distinguish defaults from explicit truncation. |
| `dataset.nbase_set` | internal bool | Whether `dataset.nbase` was explicitly set. | Same purpose as above. |
| `dataset.nquery_set` | internal bool | Whether `dataset.nquery` was explicitly set. | Same purpose as above. |
| `dataset.k_set` | internal bool | Whether `dataset.k` was explicitly set. | Same purpose as above. |
| `io.train_file_set` | internal bool | Whether `io.train_file` was explicitly set. | Prevents auto-regeneration of the derived path later. |
| `io.base_file_set` | internal bool | Whether `io.base_file` was explicitly set. | Prevents auto-regeneration of the derived path later. |
| `io.linkage_file_set` | internal bool | Whether `io.linkage_file` was explicitly set. | Prevents auto-regeneration of the derived path later. |

### Alias / compatibility keys worth knowing

| Alias | Canonical meaning | Notes |
|---|---|---|
| `dataset.learn_path` | `dataset.train_path` | Legacy naming compatibility. |
| `dataset.gt_path` | `dataset.groundtruth_path` | Short alias. |

For datasets that ship only one base vector file and no separate learn file
(for example `MSONG`), the preset may default train and base to the same
physical file. For a quick overlap test, point train and base to the same file
and cap training with `dataset.ntrain`.

For a strict train/base split, pre-split the vector files and generate a
groundtruth file against the exact base file used by `dataset.base_path`:

```cfg
dataset.name = "MSONG"
dataset.data_root = "/path/to/pq-dataset"
dataset.train_path = "msong/splits/train_100k.fvecs"
dataset.base_path = "msong/splits/base_noself.fvecs"
dataset.query_path = "msong/splits/query.fvecs"
dataset.groundtruth_path = "msong/splits/groundtruth_base_noself.ivecs"
dataset.ntrain = 100000
dataset.nbase = 894185
```

When the base file changes, the groundtruth must be regenerated for that exact
base id space. A groundtruth file computed against a different full base file is
not valid after removing vectors or reordering ids. Use a distinct
`dataset.name` or `io.pre_fix` for different split families so large-store
reuse remains explicit.

`GLOVE100` is based on ANN-Benchmarks `glove-100-angular.hdf5`. The preset
expects the HDF5 file to be converted first into row-L2-normalized fvecs/ivecs:

```bash
cd /media/dell01/hdd0/download/pq-dataset/glove100
python3 convert_glove100_hdf5.py --overwrite
```

The converted files live under `glove100/converted/`. STLQ evaluates them with
the normal L2 fvecs path; because the vectors are normalized, L2 ranking is
equivalent to cosine/angular ranking from the source dataset.
| `eval.linkage_parent_louds_enable` | `eval.linkage.parent_louds_enable` | Backward-compatible flat form. |
| `eval.parent_louds_select_stride` | `eval.linkage.parent_louds_select_stride` | Backward-compatible flat form. |
| `eval.parent_louds_rank_words_per_super_log2` | `eval.linkage.parent_louds_rank_words_per_super_log2` | Backward-compatible flat form. |
| `eval.parent_louds_build_indices` | `eval.linkage.parent_louds_build_indices` | Backward-compatible flat form. |

### LOUDS-native eval path (`eval.linkage.parent_louds_native_eval`)

The project has two separate CPU eval behaviors when `linkage_list` contains `parent_louds.bin`:

- **decode-to-parent path** (default): deserialize LOUDS, decode parent pointers, then scan using a compact cached `parent[]` (`u16`/`u32`).
- **LOUDS-native path**: scan directly from the succinct structure without persisting a full `parent[]` cache.

Important behavior details:

- The LOUDS blob on disk stores only the serialized LOUDS payload (`Header + words`).
- Rank/select helper indices are **not** stored on disk; they are rebuilt in memory only when `eval.linkage.parent_louds_build_indices=true`.
- `eval.linkage.parent_louds_select_stride` and `eval.linkage.parent_louds_rank_words_per_super_log2` only affect eval when indices are built (random parent queries / debugging paths).
- The LOUDS-native scan implementation primarily uses a sequential parent decoder and does not require rank/select indices.
- `eval.disk_norm2_mode="float"` keeps the exact per-sample `norm2` path and, when
  `eval.linkage.norm2_store_float` / `eval.linkage.norm2_store_int8` is enabled, stores/reuses
  `norm2_float.f32` / `norm2_int8.f32`.
- `eval.disk_norm2_mode="lut"` switches disk recall to the large-memory-style 1D-kmeans norm2 LUT:
  each cluster stores a small center table plus assignment codes instead of a full float norm2 array.
  The linkage disk cache files are:
  `norm2_float_lut_codes.u8`,
  `norm2_float_lut_centers.f32`, `norm2_float_lut_center_offsets.u64`
  (and the corresponding `norm2_int8_lut_*` names for the int8 coeff path).
- The LUT scan path is intentionally fixed to `uint8` codes; the requested center count is capped at 256.
- LUT caches are controlled by the same `eval.linkage.norm2_store_float` / `eval.linkage.norm2_store_int8`
  switches. In LUT mode the float `norm2_*.f32` files are not written or reused.
- Both float caches and LUT caches are reused only when their sidecars `*.hash.u64` and `*.count.u64`
  match the current `linkage_list/hash.u64` and full non-empty-cluster coverage. Partial caches are
  ignored; they are not valid for later larger-`nprobe` evals. When the corresponding
  `eval.linkage.norm2_store_*` switch is enabled, session preparation proactively builds a full-cache
  artifact if one is not already present.
- Cache loading follows the same preload/lazy policy as the cluster payload path:
  preload mode reads the whole float/LUT cache into RAM once; lazy mode reads only the selected
  clusters' slices on demand and then reuses the in-process per-cluster cache on later hits.
- Changing these eval knobs does **not** rewrite `parent_louds.bin`; it only changes the transient in-memory index shape used by the eval process.

Current performance notes:

- The LOUDS-native path is kept as a separate implementation for A/B and profiling.
- It is CPU-only in the current codebase.
- Recent optimizations include packed rank-directory storage and a sequential parent decoder for scan hot loops, but the classic decoded-parent path may still be faster on some workloads.
