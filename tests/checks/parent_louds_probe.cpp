#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "stlq/eval/linkage_cluster_provider.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/succinct/parent_louds.h"

namespace {

struct ZeroNormProvider final : stlq::eval::IClusterNormProvider {
    bool ComputeNorm2(const stlq::eval::ClusterView& cv,
                      std::vector<float>* out_r_norm2,
                      std::string* /*err*/) override {
        if (!out_r_norm2) {
            return false;
        }
        out_r_norm2->assign(static_cast<std::size_t>(std::max(0, cv.n_real)), 0.0f);
        return true;
    }
};

int ParseIntArg(const char* s, int fallback) {
    if (!s || !(*s)) return fallback;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return fallback;
    return static_cast<int>(v);
}

bool StartsWith(const std::string& s, const char* pref) {
    return s.rfind(pref, 0) == 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: stlq_parent_louds_probe <linkage_list_dir> [--num=N] [--rank_log2=K] [--prefer=0|1]\n";
        return 2;
    }

    std::string dir = argv[1];
    int num = 8;
    int rank_log2 = 4;
    bool prefer = true;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i] ? argv[i] : "";
        if (StartsWith(a, "--num=")) {
            num = ParseIntArg(a.c_str() + 6, num);
        } else if (StartsWith(a, "--rank_log2=")) {
            rank_log2 = ParseIntArg(a.c_str() + 12, rank_log2);
        } else if (StartsWith(a, "--prefer=")) {
            prefer = (ParseIntArg(a.c_str() + 9, prefer ? 1 : 0) != 0);
        } else {
            std::cerr << "Unknown arg: " << a << "\n";
            return 2;
        }
    }
    if (num < 1) num = 1;
    if (rank_log2 < 1) rank_log2 = 1;
    if (rank_log2 > 10) rank_log2 = 10;

    stlq::io::LinkageListReader reader;
    std::string err;
    if (!reader.Open(dir, &err)) {
        std::cerr << "LinkageListReader::Open failed: " << err << "\n";
        return 1;
    }
    const auto meta = reader.meta();
    std::cout << "linkage_list: dir=" << dir << "\n";
    std::cout << "meta: nlist=" << meta.nlist
              << " m_codes=" << meta.m_codes
              << " small_code_width_bytes=" << meta.small_code_width_bytes
              << " code0_width_bytes=" << meta.code0_width_bytes
              << " store_parent_louds=" << meta.store_parent_louds
              << " select_stride=" << meta.parent_louds_select_stride
              << " rank_log2=" << meta.parent_louds_rank_words_per_super_log2
              << "\n";

    ZeroNormProvider norm;
    stlq::eval::ClusterProvider provider;
    if (!provider.Open(reader,
                       /*coeff_codec=*/nullptr,
                       &norm,
                       /*use_coeff_codec=*/false,
                       /*prefer_parent_louds=*/prefer,
                       static_cast<std::uint32_t>(std::max(1, meta.parent_louds_select_stride)),
                       static_cast<std::uint32_t>(rank_log2),
                       /*parent_louds_build_indices=*/false,
                       /*adaptive_parent_u16_storage=*/true,
                       /*use_norm2_lut=*/false,
                       /*use_norm2_lut_global=*/false,
                       /*norm2_lut_h=*/256,
                       /*norm2_lut_kmeans_niter=*/25,
                       /*profile_prep_stats=*/true,
                       &err)) {
        std::cerr << "ClusterProvider::Open failed: " << err << "\n";
        return 1;
    }

    double decode_sec = 0.0;

    const int nlist = reader.nlist();
    const int max_cid = std::min(nlist, num);
    for (int cid = 0; cid < max_cid; ++cid) {
        stlq::eval::ClusterView cv;
        stlq::eval::ClusterProvider::PrepStats st;
        if (!provider.GetCluster(cid, &cv, &st, &err)) {
            std::cerr << "GetCluster cid=" << cid << " failed: " << err << "\n";
            return 1;
        }
        decode_sec += st.parent_louds_decode_sec;
        std::cout << "cid=" << cid << " n_real=" << cv.n_real
                  << " parent_louds_decode_sec=" << st.parent_louds_decode_sec << "\n";
    }

    std::cout << "TOTAL: parent_louds_decode_sec=" << decode_sec
              << "\n";
    return 0;
}
