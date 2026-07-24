#include <cassert>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#include "stlq/common/config.h"
#include "stlq/core/blas.h"
#include "stlq/core/kernel_provider_cpu.h"
#include "stlq/core/kernels_cpu.h"
#include "stlq/io/base_list_store.h"
#include "stlq/io/base_store.h"
#include "stlq/io/bvecs_reader.h"
#include "stlq/io/dataset_io.h"
#include "stlq/io/ivf_lists.h"
#include "stlq/quantizer/encode_base_streaming.h"
#include "stlq/quantizer/encoder.h"

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

}  // namespace

int main() {
    using namespace stlq;

    omp_set_dynamic(0);
    omp_set_num_threads(1);
#if defined(_OPENMP) && (_OPENMP >= 200805)
    omp_set_max_active_levels(1);
#else
    omp_set_nested(0);
#endif
    BlasSetThreads(1);

    Config cfg = DefaultConfig(/*virtual_mode=*/true);
    cfg.runtime.omp_threads = 1;
    cfg.train.enabled = false;
    cfg.base.encode.enabled = true;
    cfg.base.encode.use_abs = true;
    cfg.base.encode.H_beam = 2;
    cfg.base.encode.icm_iters = 2;
    cfg.base.encode.ils_iters = 3;
    cfg.base.encode.perturb_k = 2;
    cfg.base.encode.seed = 1234;

    cfg.large.enabled = true;
    // Keep a single block so streaming path matches in-mem call structure.
    cfg.large.base_block = 2000;
    cfg.large.base_shard_size = 1000000;
    cfg.large.cluster_bucket_size = 8;
    cfg.large.bucket_flush_mb = 8;
    cfg.large.write_vector_bucket = false;
    cfg.large.write_basic_to_bucket = true;

    cfg.model.m = 3;
    cfg.model.h_vec = {16, 16, 16};

    const int d = 16;
    const int n = 2000;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_u8(0, 255);
    std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

    ColMajorMatrix<std::uint8_t> X_u8(d, n);
    for (int i = 0; i < n; ++i) {
        std::uint8_t* col = X_u8.Col(i);
        for (int r = 0; r < d; ++r) {
            col[r] = static_cast<std::uint8_t>(dist_u8(rng));
        }
    }

    Dataset base_dataset;
    io::ConvertBvecsToF32(X_u8, &base_dataset.Xb);

    // Random codebooks (fixed seed).
    CodebookPack C_root;
    C_root.d = d;
    C_root.h_vec = cfg.model.h_vec;
    C_root.books.resize(static_cast<std::size_t>(cfg.model.m));
    for (int l = 0; l < cfg.model.m; ++l) {
        const int h = cfg.model.h_vec[static_cast<std::size_t>(l)];
        C_root.books[static_cast<std::size_t>(l)] = ColMajorMatrix<float>(d, h);
        for (int c = 0; c < h; ++c) {
            float* col = C_root.books[static_cast<std::size_t>(l)].Col(c);
            for (int r = 0; r < d; ++r) {
                col[r] = dist_f(rng);
            }
        }
    }

    Precomp pre;
    std::string err;
    if (!BuildPrecomp(C_root, &pre)) {
        return 2;
    }

    CpuKernels kernels;
    BaseEncoding base_inmem;
    float inmem_beam = 0.0f;
    float inmem_final = 0.0f;
    if (!EncodeBase(cfg, base_dataset, C_root, pre, &kernels, &base_inmem,
                    &inmem_beam, &inmem_final, &err)) {
        return 3;
    }

    // Write temp bvecs file and run streaming encode into BaseBasicStore.
    const std::string tmp = TmpDir();
    const std::string bvecs_path = (std::filesystem::path(tmp) / "base.bvecs").string();
    WriteBvecs(bvecs_path, d, X_u8);

    io::BvecsReader base_reader;
    if (!base_reader.Open(bvecs_path, &err)) {
        return 4;
    }
    if (base_reader.d() != d || static_cast<int>(base_reader.n()) != n) {
        return 5;
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
        return 6;
    }

    CpuStreamKernels stream_kernels;
    float stream_beam = 0.0f;
    float stream_final = 0.0f;
    const ColMajorMatrix<float> R = Identity(d);
    if (!EncodeBaseStreaming(cfg, base_reader, R, C_root, pre,
                             /*pre_large=*/nullptr,
                             &stream_kernels,
                             &writer,
                             &stream_beam, &stream_final, &err)) {
        return 7;
    }
    if (!writer.Close(&err)) {
        return 70;
    }

    // Build IVF CSR lists from cluster_id.u32.
    io::IvfListsPaths ivf_paths;
    ivf_paths.offsets_u64 = (std::filesystem::path(store_cfg.dir) / "ivf_offsets.u64").string();
    ivf_paths.ids_u32 = (std::filesystem::path(store_cfg.dir) / "ivf_ids.u32").string();
    const std::string ivf_tmp = (std::filesystem::path(tmp) / "ivf_tmp").string();
    if (!io::BuildIvfListsFromClusterIdFile((std::filesystem::path(store_cfg.dir) / "cluster_id.u32").string(),
                                           cfg.model.h_vec[0],
                                           cfg.large.cluster_bucket_size,
                                           ivf_paths, ivf_tmp,
                                           /*keep_tmp=*/false, &err)) {
        return 8;
    }
    io::IvfListsReader ivf;
    if (!ivf.Open(ivf_paths, &err)) {
        return 9;
    }
    if (static_cast<int>(ivf.ntotal()) != n) {
        return 10;
    }

    // Build list-order store from buckets and verify content equals in-mem encoding.
    const std::string base_list_dir = (std::filesystem::path(tmp) / "base_list").string();
    const std::string base_list_tmp = (std::filesystem::path(tmp) / "base_list_tmp").string();
    if (!io::BuildBaseListStoreFromBasicBuckets(store_cfg.dir, ivf, base_list_dir, base_list_tmp,
                                               /*keep_tmp=*/false,
                                               /*profile_timing=*/false,
                                               &err)) {
        return 11;
    }

    io::BaseListReader base_list;
    if (!base_list.Open(base_list_dir, &err)) {
        return 12;
    }
    if (static_cast<int>(base_list.meta().ntotal) != n) {
        return 13;
    }

    std::vector<Code> codes_all;
    std::vector<float> coeffs_all;
    if (!base_list.ReadCodesSpan(0, static_cast<std::uint32_t>(n), &codes_all, &err)) {
        return 14;
    }
    if (!base_list.ReadCoeffsSpan(0, static_cast<std::uint32_t>(n), &coeffs_all, &err)) {
        return 15;
    }

    int list_pos = 0;
    for (int cid = 0; cid < ivf.nlist(); ++cid) {
        std::vector<std::uint32_t> ids;
        if (!ivf.ReadList(cid, &ids, &err)) {
            return 16;
        }
        for (std::uint32_t gid : ids) {
            if (gid >= static_cast<std::uint32_t>(n)) {
                return 17;
            }
            const int gi = static_cast<int>(gid);
            const Code* code_disk = codes_all.data() + static_cast<std::size_t>(list_pos) * (cfg.model.m - 1);
            const float* a_disk = coeffs_all.data() + static_cast<std::size_t>(list_pos) * cfg.model.m;

            // Compare small codes (layers 1..m-1).
            for (int l = 1; l < cfg.model.m; ++l) {
                const Code want = base_inmem.B(l, gi);
                const Code got = code_disk[l - 1];
                if (want != got) {
                    return 18;
                }
            }
            // Compare coeffs.
            for (int l = 0; l < cfg.model.m; ++l) {
                const float want = base_inmem.a(l, gi);
                const float got = a_disk[l];
                const float diff = std::fabs(want - got);
                if (diff > 1e-4f) {
                    return 19;
                }
            }
            ++list_pos;
        }
    }
    if (list_pos != n) {
        return 20;
    }

    // Hybrid basic encoding sanity: enable ordered-writer reorder buffer and run with 2 blocks.
    // This exercises the hybrid scheduler while keeping outputs identical to in-mem encoding.
    Config cfg_hybrid = cfg;
    cfg_hybrid.large.base_block = 1000;  // force 2 blocks
    cfg_hybrid.runtime.basic_hybrid_enable = true;
    cfg_hybrid.runtime.basic_hybrid_cpu_stride = 2;
    cfg_hybrid.runtime.basic_hybrid_reorder_depth = 4;
    cfg_hybrid.runtime.basic_hybrid_cpu_threads = 1;

    io::BaseBasicStoreConfig store_cfg_hybrid = store_cfg;
    store_cfg_hybrid.dir = (std::filesystem::path(tmp) / "base_basic_hybrid").string();

    io::BaseBasicWriter writer_hybrid;
    if (!writer_hybrid.Open(store_cfg_hybrid, &err)) {
        return 21;
    }
    float stream_beam_hybrid = 0.0f;
    float stream_final_hybrid = 0.0f;
    if (!EncodeBaseStreaming(cfg_hybrid, base_reader, R, C_root, pre,
                             /*pre_large=*/nullptr,
                             &stream_kernels,
                             &writer_hybrid,
                             &stream_beam_hybrid, &stream_final_hybrid, &err)) {
        return 22;
    }
    if (!writer_hybrid.Close(&err)) {
        return 23;
    }

    io::IvfListsPaths ivf_paths_h;
    ivf_paths_h.offsets_u64 = (std::filesystem::path(store_cfg_hybrid.dir) / "ivf_offsets.u64").string();
    ivf_paths_h.ids_u32 = (std::filesystem::path(store_cfg_hybrid.dir) / "ivf_ids.u32").string();
    const std::string ivf_tmp_h = (std::filesystem::path(tmp) / "ivf_tmp_hybrid").string();
    if (!io::BuildIvfListsFromClusterIdFile((std::filesystem::path(store_cfg_hybrid.dir) / "cluster_id.u32").string(),
                                           cfg_hybrid.model.h_vec[0],
                                           cfg_hybrid.large.cluster_bucket_size,
                                           ivf_paths_h, ivf_tmp_h,
                                           /*keep_tmp=*/false, &err)) {
        return 24;
    }
    io::IvfListsReader ivf_h;
    if (!ivf_h.Open(ivf_paths_h, &err)) {
        return 25;
    }
    if (static_cast<int>(ivf_h.ntotal()) != n) {
        return 26;
    }

    const std::string base_list_dir_h = (std::filesystem::path(tmp) / "base_list_hybrid").string();
    const std::string base_list_tmp_h = (std::filesystem::path(tmp) / "base_list_tmp_hybrid").string();
    if (!io::BuildBaseListStoreFromBasicBuckets(store_cfg_hybrid.dir, ivf_h, base_list_dir_h, base_list_tmp_h,
                                               /*keep_tmp=*/false,
                                               /*profile_timing=*/false,
                                               &err)) {
        return 27;
    }

    io::BaseListReader base_list_h;
    if (!base_list_h.Open(base_list_dir_h, &err)) {
        return 28;
    }
    std::vector<Code> codes_all_h;
    std::vector<float> coeffs_all_h;
    if (!base_list_h.ReadCodesSpan(0, static_cast<std::uint32_t>(n), &codes_all_h, &err)) {
        return 29;
    }
    if (!base_list_h.ReadCoeffsSpan(0, static_cast<std::uint32_t>(n), &coeffs_all_h, &err)) {
        return 30;
    }
    int list_pos_h = 0;
    for (int cid = 0; cid < ivf_h.nlist(); ++cid) {
        std::vector<std::uint32_t> ids;
        if (!ivf_h.ReadList(cid, &ids, &err)) {
            return 31;
        }
        for (std::uint32_t gid : ids) {
            if (gid >= static_cast<std::uint32_t>(n)) {
                return 32;
            }
            const int gi = static_cast<int>(gid);
            const Code* code_disk = codes_all_h.data() + static_cast<std::size_t>(list_pos_h) * (cfg_hybrid.model.m - 1);
            const float* a_disk = coeffs_all_h.data() + static_cast<std::size_t>(list_pos_h) * cfg_hybrid.model.m;
            for (int l = 1; l < cfg_hybrid.model.m; ++l) {
                const Code want = base_inmem.B(l, gi);
                const Code got = code_disk[l - 1];
                if (want != got) {
                    return 33;
                }
            }
            for (int l = 0; l < cfg_hybrid.model.m; ++l) {
                const float want = base_inmem.a(l, gi);
                const float got = a_disk[l];
                const float diff = std::fabs(want - got);
                if (diff > 1e-4f) {
                    return 34;
                }
            }
            ++list_pos_h;
        }
    }
    if (list_pos_h != n) {
        return 35;
    }

    return 0;
}
