#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "stlq/io/linkage_list_store.h"
#include "stlq/succinct/parent_louds.h"

namespace {

std::string TempDirPath() {
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    const std::filesystem::path dir = base / "stlq_test_linkage_list_parent_louds";
    return dir.string();
}

void RemoveAllQuiet(const std::string& dir) {
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(dir), ec);
}

void WriteTinyStore(const std::string& dir) {
    using stlq::io::LinkageListStoreConfig;
    using stlq::io::LinkageListWriter;

    RemoveAllQuiet(dir);

    LinkageListStoreConfig cfg;
    cfg.dir = dir;
    cfg.nlist = 3;
    cfg.m_codes = 2;
    cfg.code0_width_bytes = 1;
    cfg.store_coeffs_f32 = false;
    cfg.store_parent_louds = true;
    cfg.parent_louds_select_stride = 128;

    std::string err;
    LinkageListWriter w;
    assert(w.Open(cfg, &err));

    // Cluster 0: non-empty + virtual roots appended (requires internal BFS mapping).
    {
        const int n_real = 6;
        const int n_virt = 2;
        const std::vector<std::uint32_t> real_ids = {10, 11, 12, 13, 14, 15};
        // Depth order (real-only): roots (0,1), depth1 (2,3), depth2 (4,5).
        // NOTE: parent is stored in the *full local-id space* (virtual-front):
        //   local = [virt(0..n_virt-1), real(0..n_real-1)+n_virt]
        // so real parents are shifted by +n_virt.
        const std::vector<std::uint32_t> parent = {0, 0, 3, 4, 5, 6};
        const std::vector<std::uint32_t> depth_offsets = {0, 2, 4, 6};
        std::vector<std::uint8_t> codes_small(static_cast<std::size_t>(cfg.m_codes) *
                                              static_cast<std::size_t>(n_real) *
                                              static_cast<std::size_t>(LinkageListStoreConfig::kSmallCodeWidthBytes),
                                              std::uint8_t(0));
        std::vector<std::uint8_t> code0_one(static_cast<std::size_t>(n_real), std::uint8_t(0));
        std::vector<std::uint8_t> virt_codes(static_cast<std::size_t>(cfg.m_codes) *
                                             static_cast<std::size_t>(n_virt) *
                                             static_cast<std::size_t>(LinkageListStoreConfig::kSmallCodeWidthBytes),
                                             std::uint8_t(0));
        assert(w.AppendCluster(/*cid=*/0,
                               real_ids,
                               parent,
                               depth_offsets,
                               codes_small,
                               /*coeffs_small_depth_order=*/{},
                               code0_one,
                               /*a0_depth_order=*/{},
                               virt_codes,
                               /*virt_coeffs_small=*/{},
                               /*virt_a0=*/{},
                               n_real,
                               n_virt,
                               &err));
    }

    // Cluster 1: empty cluster.
    {
        assert(w.AppendCluster(/*cid=*/1,
                               /*real_ids=*/{},
                               /*parent=*/{},
                               /*depth_offsets=*/{0},
                               /*codes_small=*/{},
                               /*coeffs_small=*/{},
                               /*code0_one=*/{},
                               /*a0=*/{},
                               /*virt_codes=*/{},
                               /*virt_coeffs=*/{},
                               /*virt_a0=*/{},
                               /*n_real=*/0,
                               /*n_virt=*/0,
                               &err));
    }

    // Cluster 2: non-empty real-only.
    {
        const int n_real = 4;
        const std::vector<std::uint32_t> real_ids = {20, 21, 22, 23};
        // roots: 0,1; 2->0; 3->2
        const std::vector<std::uint32_t> parent = {0, 0, 1, 3};
        const std::vector<std::uint32_t> depth_offsets = {0, 2, 3, 4};
        std::vector<std::uint8_t> codes_small(static_cast<std::size_t>(cfg.m_codes) *
                                              static_cast<std::size_t>(n_real) *
                                              static_cast<std::size_t>(LinkageListStoreConfig::kSmallCodeWidthBytes),
                                              std::uint8_t(0));
        std::vector<std::uint8_t> code0_one(static_cast<std::size_t>(n_real), std::uint8_t(0));
        assert(w.AppendCluster(/*cid=*/2,
                               real_ids,
                               parent,
                               depth_offsets,
                               codes_small,
                               /*coeffs_small_depth_order=*/{},
                               code0_one,
                               /*a0_depth_order=*/{},
                               /*virt_codes_small_bytes=*/{},
                               /*virt_coeffs_small=*/{},
                               /*virt_a0=*/{},
                               n_real,
                               /*n_virt=*/0,
                               &err));
    }

    assert(w.Finish(&err));
}

void VerifyStoreParentLOUDS(const std::string& dir) {
    using stlq::io::LinkageListReader;
    using stlq::succinct::ParentLOUDS;
    (void)dir;

    LinkageListReader r;
    std::string err;
    assert(r.Open(dir, &err));
    assert(r.meta().store_parent_louds != 0);

    for (int cid = 0; cid < r.nlist(); ++cid) {
        std::vector<std::uint32_t> real_ids;
        std::vector<std::uint32_t> parent_u32;
        std::vector<std::uint32_t> depth_offsets;
        std::vector<std::uint8_t> codes_small;
        std::vector<std::uint8_t> code0_one;
        std::vector<std::uint8_t> virt_codes;
        assert(r.ReadCluster(cid,
                             &real_ids,
                             &parent_u32,
                             &depth_offsets,
                             &codes_small,
                             /*coeffs_small_depth_order=*/nullptr,
                             &code0_one,
                             /*a0_depth_order=*/nullptr,
                             &virt_codes,
                             /*virt_coeffs_small=*/nullptr,
                             /*virt_a0=*/nullptr,
                             &err));
        const std::size_t n_real = real_ids.size();
        const std::size_t n_virt =
            (r.meta().m_codes > 0 && r.meta().small_code_width_bytes > 0)
                ? (virt_codes.size() /
                   (static_cast<std::size_t>(r.meta().m_codes) *
                    static_cast<std::size_t>(r.meta().small_code_width_bytes)))
                : 0;
        const std::size_t n_total = n_real + n_virt;

        std::vector<std::uint8_t> blob;
        assert(r.ReadClusterParentLOUDSBlob(cid, &blob, &err));
        if (n_real == 0) {
            // Empty cluster: parent LOUDS may be stored as an empty blob or as a valid v1 header+words
            // for n_total==0 (for positioned writes / fixed planning). Either is acceptable.
            if (!blob.empty()) {
                ParentLOUDS louds;
                louds.Deserialize(blob);
                assert(louds.n_nodes() == 0);
                std::vector<std::uint32_t> parent_all;
                louds.DecodeParent1Based(&parent_all, /*n_out=*/0);
                assert(parent_all.empty());
            }
            continue;
        }
        assert(!blob.empty());

        ParentLOUDS louds;
        louds.Deserialize(blob);

        std::vector<std::uint32_t> parent_all;
        louds.DecodeParent1Based(&parent_all, n_total);
        assert(parent_all.size() == n_total);
        std::vector<std::uint32_t> parent_real(n_real);
        std::memcpy(parent_real.data(), parent_all.data() + n_virt, n_real * sizeof(std::uint32_t));
        assert(parent_real == parent_u32);
    }
}

void RewriteMetaAsLegacyChainStore(const std::string& dir) {
    const auto meta_path = std::filesystem::path(dir) / "meta.bin";
    std::fstream meta(meta_path, std::ios::binary | std::ios::in | std::ios::out);
    assert(meta.is_open());
    meta.write("CHAINLST", 8);
    assert(meta.good());
}

}  // namespace

int main() {
    const std::string dir = TempDirPath();
    WriteTinyStore(dir);
    VerifyStoreParentLOUDS(dir);
    RewriteMetaAsLegacyChainStore(dir);
    VerifyStoreParentLOUDS(dir);
    RemoveAllQuiet(dir);
    return 0;
}
