#include <cassert>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "stlq/io/linkage_list_store.h"

namespace {

std::string TempDirPath() {
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    const std::filesystem::path dir = base / "stlq_test_linkage_list_parent_louds_no_u32";
    return dir.string();
}

void RemoveAllQuiet(const std::string& dir) {
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(dir), ec);
}

void WriteTinyStoreNoParentU32(const std::string& dir) {
    using stlq::io::LinkageListStoreConfig;
    using stlq::io::LinkageListWriter;

    RemoveAllQuiet(dir);

    LinkageListStoreConfig cfg;
    cfg.dir = dir;
    cfg.nlist = 2;
    cfg.m_codes = 2;
    cfg.code0_width_bytes = 1;
    cfg.store_coeffs_f32 = false;
    cfg.store_parent_u32 = false;
    cfg.store_parent_louds = true;
    cfg.parent_louds_select_stride = 128;

    std::string err;
    LinkageListWriter w;
    assert(w.Open(cfg, &err));

    // Cluster 0: non-empty + virtual roots appended.
    {
        const int n_real = 6;
        const int n_virt = 2;
        const std::vector<std::uint32_t> real_ids = {10, 11, 12, 13, 14, 15};
        // Same convention as other tests: parent is stored in the full local-id space (virtual-front).
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

    // Cluster 1: non-empty real-only.
    {
        const int n_real = 4;
        const std::vector<std::uint32_t> real_ids = {20, 21, 22, 23};
        const std::vector<std::uint32_t> parent = {0, 0, 1, 3};
        const std::vector<std::uint32_t> depth_offsets = {0, 2, 3, 4};
        std::vector<std::uint8_t> codes_small(static_cast<std::size_t>(cfg.m_codes) *
                                              static_cast<std::size_t>(n_real) *
                                              static_cast<std::size_t>(LinkageListStoreConfig::kSmallCodeWidthBytes),
                                              std::uint8_t(0));
        std::vector<std::uint8_t> code0_one(static_cast<std::size_t>(n_real), std::uint8_t(0));
        assert(w.AppendCluster(/*cid=*/1,
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

    const std::filesystem::path p = std::filesystem::path(dir) / "parent.u32";
    assert(!std::filesystem::exists(p));
}

void VerifyReadClusterDecodesParentFromLOUDS(const std::string& dir) {
    using stlq::io::LinkageListReader;
    (void)dir;

    LinkageListReader r;
    std::string err;
    assert(r.Open(dir, &err));
    assert(r.meta().store_parent_louds != 0);
    assert(r.meta().store_parent_u32 == 0);

    // Expected parents for the two clusters.
    const std::vector<std::vector<std::uint32_t>> expect_parent = {
        {0, 0, 3, 4, 5, 6},
        {0, 0, 1, 3},
    };

    for (int cid = 0; cid < r.nlist(); ++cid) {
        std::vector<std::uint32_t> real_ids;
        std::vector<std::uint32_t> parent;
        std::vector<std::uint32_t> depth_offsets;
        std::vector<std::uint8_t> codes_small;
        std::vector<std::uint8_t> code0_one;
        std::vector<std::uint8_t> virt_codes;
        assert(r.ReadCluster(cid,
                             &real_ids,
                             &parent,
                             &depth_offsets,
                             &codes_small,
                             /*coeffs_small_depth_order=*/nullptr,
                             &code0_one,
                             /*a0_depth_order=*/nullptr,
                             &virt_codes,
                             /*virt_coeffs_small=*/nullptr,
                             /*virt_a0=*/nullptr,
                             &err));
        assert(parent == expect_parent[static_cast<std::size_t>(cid)]);
    }
}

}  // namespace

int main() {
    const std::string dir = TempDirPath();
    WriteTinyStoreNoParentU32(dir);
    VerifyReadClusterDecodesParentFromLOUDS(dir);
    RemoveAllQuiet(dir);
    return 0;
}

