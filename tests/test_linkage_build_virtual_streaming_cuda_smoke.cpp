#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#include "stlq/linkage/linkage_builder_virtual_streaming.h"
#include "stlq/pipeline/app_utils.h"
#include "stlq/common/config.h"
#include "stlq/core/blas.h"
#include "stlq/core/kernel_provider_cpu.h"
#include "stlq/io/base_list_store.h"
#include "stlq/io/base_store.h"
#include "stlq/io/bvecs_reader.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/dataset_io.h"
#include "stlq/io/ivf_lists.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/encode_base_streaming.h"
#include "stlq/quantizer/encoder.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#pragma comment(lib, "dbghelp.lib")
#endif

namespace {

std::string TmpDir() {
    const auto root = std::filesystem::path("cpp_to_further_process") / ".test_tmp";
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    const std::uint64_t tag =
        static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const auto dir = root / ("run_" + std::to_string(tag));
    std::filesystem::create_directories(dir, ec);
    return dir.string();
}

void WriteBvecs(const std::string& path,
                int d,
                const stlq::ColMajorMatrix<std::uint8_t>& X) {
    assert(X.rows == d);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    assert(out.is_open());
    for (int i = 0; i < X.cols; ++i) {
        const std::int32_t dim = d;
        out.write(reinterpret_cast<const char*>(&dim), sizeof(std::int32_t));
        out.write(reinterpret_cast<const char*>(X.Col(i)), d);
    }
    assert(out.good());
}

stlq::ColMajorMatrix<float> Identity(int d) {
    stlq::ColMajorMatrix<float> I(d, d);
    for (int j = 0; j < d; ++j) {
        for (int i = 0; i < d; ++i) {
            I(i, j) = (i == j) ? 1.0f : 0.0f;
        }
    }
    return I;
}

stlq::CodebookPack RandomCodebookPack(std::mt19937& rng,
                                        int d,
                                        int m,
                                        const std::vector<int>& h_vec) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    stlq::CodebookPack C;
    C.d = d;
    C.h_vec = h_vec;
    C.books.resize(static_cast<std::size_t>(m));
    for (int l = 0; l < m; ++l) {
        const int h = (l < static_cast<int>(h_vec.size())) ? h_vec[static_cast<std::size_t>(l)] : 0;
        C.books[static_cast<std::size_t>(l)] = stlq::ColMajorMatrix<float>(d, h);
        for (int c = 0; c < h; ++c) {
            float* col = C.books[static_cast<std::size_t>(l)].Col(c);
            for (int r = 0; r < d; ++r) {
                col[r] = dist(rng);
            }
        }
    }
    return C;
}

#if defined(_WIN32)
static void PrintWinStackTrace() {
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    if (!SymInitialize(proc, nullptr, TRUE)) {
        std::fprintf(stderr, "[STLQ_TEST] SymInitialize failed: gle=%lu\n", GetLastError());
        return;
    }
    void* frames[128];
    const USHORT n = CaptureStackBackTrace(0, 128, frames, nullptr);
    std::fprintf(stderr, "[STLQ_TEST] Stack trace (%u frames):\n", static_cast<unsigned>(n));

    alignas(SYMBOL_INFO) unsigned char sym_buf[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO* sym = reinterpret_cast<SYMBOL_INFO*>(sym_buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;

    IMAGEHLP_LINE64 line;
    std::memset(&line, 0, sizeof(line));
    line.SizeOfStruct = sizeof(line);

    for (USHORT i = 0; i < n; ++i) {
        const DWORD64 addr = reinterpret_cast<DWORD64>(frames[i]);
        DWORD64 disp = 0;
        if (SymFromAddr(proc, addr, &disp, sym)) {
            DWORD disp_line = 0;
            if (SymGetLineFromAddr64(proc, addr, &disp_line, &line)) {
                std::fprintf(stderr, "  #%02u %s +0x%llx (%s:%lu)\n",
                             static_cast<unsigned>(i),
                             sym->Name,
                             static_cast<unsigned long long>(disp),
                             line.FileName ? line.FileName : "?",
                             static_cast<unsigned long>(line.LineNumber));
            } else {
                std::fprintf(stderr, "  #%02u %s +0x%llx\n",
                             static_cast<unsigned>(i),
                             sym->Name,
                             static_cast<unsigned long long>(disp));
            }
        } else {
            std::fprintf(stderr, "  #%02u 0x%llx\n",
                         static_cast<unsigned>(i),
                         static_cast<unsigned long long>(addr));
        }
    }
    std::fflush(stderr);
}

static LONG WINAPI UnhandledExceptionFilterFn(EXCEPTION_POINTERS* ep) {
    if (ep && ep->ExceptionRecord) {
        std::fprintf(stderr,
                     "[STLQ_TEST] Unhandled exception: code=0x%08lx addr=%p\n",
                     static_cast<unsigned long>(ep->ExceptionRecord->ExceptionCode),
                     ep->ExceptionRecord->ExceptionAddress);
    } else {
        std::fprintf(stderr, "[STLQ_TEST] Unhandled exception\n");
    }
    PrintWinStackTrace();
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

}  // namespace

int main() {
    using namespace stlq;

#if defined(_WIN32)
    SetUnhandledExceptionFilter(UnhandledExceptionFilterFn);
#endif

    omp_set_dynamic(0);
    omp_set_num_threads(1);
#if defined(_OPENMP) && (_OPENMP >= 200805)
    omp_set_max_active_levels(1);
#else
    omp_set_nested(0);
#endif
    BlasSetThreads(1);

    Config cfg = DefaultConfig(/*virtual_mode=*/true);
    NormalizeHVec(&cfg);

    cfg.runtime.omp_threads = 1;
    cfg.runtime.use_cuda = true;
    cfg.runtime.cuda_device = 0;
    cfg.runtime.cuda_pool_size = 1;
    cfg.runtime.cuda_linkage_use_device_rfull = true;
    cfg.runtime.cuda_linkage_batch_inner_enabled = true;

    cfg.train.enabled = false;
    cfg.base.encode.enabled = true;

    cfg.large.enabled = true;
    cfg.large.base_block = 2000;          // single block
    cfg.large.base_shard_size = 1000000;  // no sharding
    cfg.large.cluster_bucket_size = 8;
    cfg.large.bucket_flush_mb = 8;
    cfg.large.write_vector_bucket = false;
    cfg.large.write_basic_to_bucket = true;

    cfg.model.m = 5;
    cfg.model.h_vec = {16, 16, 16, 16, 16};

    const int d = 16;
    const int n = 2000;

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist_u8(0, 255);

    ColMajorMatrix<std::uint8_t> X_u8(d, n);
    for (int i = 0; i < n; ++i) {
        std::uint8_t* col = X_u8.Col(i);
        for (int r = 0; r < d; ++r) {
            col[r] = static_cast<std::uint8_t>(dist_u8(rng));
        }
    }

    const std::string tmp = TmpDir();
    const std::string bvecs_path = (std::filesystem::path(tmp) / "base.bvecs").string();
    WriteBvecs(bvecs_path, d, X_u8);

    io::BvecsReader base_reader;
    std::string err;
    if (!base_reader.Open(bvecs_path, &err)) {
        LogError(err);
        return 1;
    }
    io::DatasetVectorReader fallback_reader;
    fallback_reader.format = io::VectorFileFormat::kBvecs;
    fallback_reader.is_u8 = true;
    if (!fallback_reader.bvecs.Open(bvecs_path, &err)) {
        LogError(err);
        return 1;
    }
    fallback_reader.d = fallback_reader.bvecs.d();
    fallback_reader.n = fallback_reader.bvecs.n();
    fallback_reader.path = fallback_reader.bvecs.path();

    // Random codebooks.
    CodebookPack C_root = RandomCodebookPack(rng, d, cfg.model.m, cfg.model.h_vec);
    CodebookPack C_one = RandomCodebookPack(rng, d, cfg.model.m, cfg.model.h_vec);

    Precomp pre_root;
    if (!BuildPrecomp(C_root, &pre_root)) {
        LogError("BuildPrecomp(C_root) failed");
        return 2;
    }

    io::BaseBasicStoreConfig store_cfg;
    store_cfg.dir = (std::filesystem::path(tmp) / "base_basic").string();
    store_cfg.d = d;
    store_cfg.m = cfg.model.m;
    store_cfg.h_vec = cfg.model.h_vec;
    store_cfg.shard_size = cfg.large.base_shard_size;
    store_cfg.bucket_size = cfg.large.cluster_bucket_size;
    store_cfg.bucket_flush_mb = cfg.large.bucket_flush_mb;
    store_cfg.write_vector_bucket = cfg.large.write_vector_bucket;
    store_cfg.write_basic_to_bucket = cfg.large.write_basic_to_bucket;

    io::BaseBasicWriter writer;
    if (!writer.Open(store_cfg, &err)) {
        LogError(err);
        return 3;
    }

    CpuStreamKernels stream_kernels;
    const ColMajorMatrix<float> R = Identity(d);
    float stream_beam = 0.0f;
    float stream_final = 0.0f;
    if (!EncodeBaseStreaming(cfg, base_reader, R, C_root, pre_root,
                             /*pre_large=*/nullptr,
                             &stream_kernels,
                             &writer,
                             &stream_beam, &stream_final, &err)) {
        LogError(err);
        return 4;
    }
    if (!writer.Close(&err)) {
        LogError(err);
        return 5;
    }

    io::IvfListsPaths ivf_paths;
    ivf_paths.offsets_u64 = (std::filesystem::path(store_cfg.dir) / "ivf_offsets.u64").string();
    ivf_paths.ids_u32 = (std::filesystem::path(store_cfg.dir) / "ivf_ids.u32").string();
    const std::string ivf_tmp = (std::filesystem::path(tmp) / "ivf_tmp").string();
    if (!io::BuildIvfListsFromClusterIdFile((std::filesystem::path(store_cfg.dir) / "cluster_id.u32").string(),
                                           cfg.model.h_vec[0],
                                           cfg.large.cluster_bucket_size,
                                           ivf_paths,
                                           ivf_tmp,
                                           /*keep_tmp=*/false,
                                           &err)) {
        LogError(err);
        return 6;
    }
    io::IvfListsReader ivf;
    if (!ivf.Open(ivf_paths, &err)) {
        LogError(err);
        return 7;
    }

    const std::string base_list_dir = (std::filesystem::path(tmp) / "base_list").string();
    const std::string base_list_tmp = (std::filesystem::path(tmp) / "base_list_tmp").string();
    if (!io::BuildBaseListStoreFromBasicBuckets(store_cfg.dir, ivf, base_list_dir, base_list_tmp,
                                               /*keep_tmp=*/false,
                                               /*profile_timing=*/false,
                                               &err)) {
        LogError(err);
        return 8;
    }
    io::BaseListReader base_list;
    if (!base_list.Open(base_list_dir, &err)) {
        LogError(err);
        return 9;
    }

    TrainResult train;
    train.C_root = C_root;
    train.C_one = C_one;
    train.R = R;
    train.is_bad_cluster.assign(static_cast<std::size_t>(ivf.nlist()), false);

    io::LinkageListStoreConfig linkage_store_cfg;
    linkage_store_cfg.dir = (std::filesystem::path(tmp) / "linkage_list").string();
    linkage_store_cfg.nlist = ivf.nlist();
    linkage_store_cfg.m_codes = std::max(0, cfg.model.m - 1);
    linkage_store_cfg.code0_width_bytes = 1;
    linkage_store_cfg.store_coeffs_f32 = true;
    linkage_store_cfg.store_parent_u32 = true;
    linkage_store_cfg.store_parent_louds = false;

    VirtualUmapReencodeEncodeConfig umap_cfg;
    LinkageDepthStats depth_stats;
    MseStats mse_stats;
    if (!BuildLinkageTwoCodebookVirtualStreamingByCluster(cfg,
                                                        train,
                                                        umap_cfg,
                                                        base_list,
                                                        ivf,
                                                        &fallback_reader,
                                                        /*allow_random_fallback=*/false,
                                                        &stream_kernels,
                                                        linkage_store_cfg,
                                                        &depth_stats,
                                                        &mse_stats,
                                                        &err)) {
        LogError(err);
        return 10;
    }

    LogInfo("OK: depth.max=" + std::to_string(depth_stats.max_depth) +
            " depth.mean=" + std::to_string(depth_stats.mean_depth) +
            " mse.mean=" + std::to_string(mse_stats.mean));
    return 0;
}
