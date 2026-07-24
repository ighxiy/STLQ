#include "stlq/pipeline/large_store_hash.h"

#include <filesystem>
#include <fstream>

namespace stlq::app {
namespace {

struct Hash64 {
    std::uint64_t h = 1469598103934665603ull;

    void AddBytes(const void* data, std::size_t len) {
        const auto* p = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < len; ++i) {
            h ^= static_cast<std::uint64_t>(p[i]);
            h *= 1099511628211ull;
        }
    }

    void AddU64(std::uint64_t v) { AddBytes(&v, sizeof(v)); }
    void AddU32(std::uint32_t v) { AddBytes(&v, sizeof(v)); }
    void AddI32(std::int32_t v) { AddBytes(&v, sizeof(v)); }
    void AddBool(bool v) {
        const std::uint8_t b = v ? 1 : 0;
        AddBytes(&b, sizeof(b));
    }
    void AddF32(float v) { AddBytes(&v, sizeof(v)); }
    void AddF64(double v) { AddBytes(&v, sizeof(v)); }

    void AddStr(const std::string& s) {
        AddU64(static_cast<std::uint64_t>(s.size()));
        if (!s.empty()) {
            AddBytes(s.data(), s.size());
        }
    }

    void AddVecI32(const std::vector<int>& v) {
        AddU64(static_cast<std::uint64_t>(v.size()));
        for (int x : v) {
            AddI32(x);
        }
    }
};

std::uint64_t HashMatF32(const ColMajorMatrix<float>& M) {
    Hash64 hh;
    hh.AddI32(M.rows);
    hh.AddI32(M.cols);
    if (!M.data.empty()) {
        hh.AddBytes(M.data.data(), M.data.size() * sizeof(float));
    }
    return hh.h;
}

std::uint64_t HashTrainModelForBase(const TrainResult& tr) {
    Hash64 hh;
    hh.AddStr("train_model_for_base_v1");
    hh.AddU64(HashMatF32(tr.R));
    for (const auto& book : tr.C_root.books) {
        hh.AddU64(HashMatF32(book));
    }
    return hh.h;
}

std::uint64_t HashTrainModelForLinkage(const TrainResult& tr) {
    Hash64 hh;
    hh.AddStr("train_model_for_linkage_v1");
    hh.AddU64(HashMatF32(tr.R));
    for (const auto& book : tr.C_root.books) {
        hh.AddU64(HashMatF32(book));
    }
    for (const auto& book : tr.C_one.books) {
        hh.AddU64(HashMatF32(book));
    }
    return hh.h;
}

void HashLinkageCfg(Hash64* hh, const LinkageBuildConfig& c) {
    hh->AddF64(c.root_percentile);
    hh->AddI32(c.num_layers);
    hh->AddI32(c.max_depth);
    hh->AddI32(c.knn_k);
    hh->AddI32(c.depth_k);
    hh->AddI32(c.icm_round);
    hh->AddBool(c.use_ils);
    hh->AddI32(c.ils_rounds);
    hh->AddI32(c.ils_perturb_layers);
    hh->AddI32(c.seed);
}

}  // namespace

std::string StoreHashPath(const std::string& dir) {
    return (std::filesystem::path(dir) / "hash.u64").string();
}

std::string CodecHashPath(const std::string& linkage_list_dir) {
    return (std::filesystem::path(linkage_list_dir) / "coeff_hash.u64").string();
}

bool ReadU64File(const std::string& path, std::uint64_t* out) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    std::string s;
    in >> s;
    if (s.empty()) {
        return false;
    }
    try {
        std::size_t pos = 0;
        int base = 10;
        if (s.size() > 2 && (s[0] == '0') && (s[1] == 'x' || s[1] == 'X')) {
            base = 16;
        }
        const auto v = static_cast<std::uint64_t>(std::stoull(s, &pos, base));
        if (pos == 0) {
            return false;
        }
        *out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool WriteU64FileHex(const std::string& path, std::uint64_t v, std::string* error) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (error) {
            *error = "Failed to write: " + path;
        }
        return false;
    }
    out << "0x" << std::hex << v << std::dec << "\n";
    if (!out) {
        if (error) {
            *error = "Failed to write: " + path;
        }
        return false;
    }
    return true;
}

bool ReadNorm2SourceHash(const std::string& linkage_list_dir,
                         bool use_coeff_codec,
                         std::uint64_t* out) {
    if (!out) {
        return false;
    }
    std::uint64_t store_hash = 0;
    if (!ReadU64File(StoreHashPath(linkage_list_dir), &store_hash)) {
        return false;
    }
    if (!use_coeff_codec) {
        *out = store_hash;
        return true;
    }
    std::uint64_t codec_hash = 0;
    if (!ReadU64File(CodecHashPath(linkage_list_dir), &codec_hash)) {
        return false;
    }
    Hash64 hh;
    hh.AddStr("linkage_norm2_source_v1");
    hh.AddU64(store_hash);
    hh.AddU64(codec_hash);
    *out = hh.h;
    return true;
}

std::uint64_t ComputeBaseBasicStoreHash(const Config& cfg,
                                        const TrainResult& tr,
                                        const io::BaseBasicStoreConfig& store_cfg,
                                        bool base_is_u8) {
    Hash64 hh;
    hh.AddStr("base_basic_store_v1");
    hh.AddU64(HashTrainModelForBase(tr));
    hh.AddStr(cfg.dataset.name);
    // NOTE: dataset.data_root is intentionally excluded from the store identity so
    // relocating the same dataset (e.g. SSD <-> HDD or different mount points) does not force a
    // full rebuild. The dataset should be disambiguated by `dataset.name` (and, if needed, by
    // using distinct `io.pre_fix` or a distinct dataset name).
    hh.AddBool(base_is_u8);
    // IMPORTANT: base encoding may cap ntotal by `dataset.nbase` (see EncodeBaseStreaming),
    // so this must be part of the store identity.
    hh.AddBool(cfg.dataset.nbase_set);
    hh.AddI32(cfg.dataset.nbase);
    hh.AddI32(store_cfg.d);
    hh.AddI32(store_cfg.m);
    hh.AddVecI32(store_cfg.h_vec);
    hh.AddI32(store_cfg.shard_size);
    hh.AddI32(store_cfg.bucket_size);
    hh.AddI32(store_cfg.bucket_flush_mb);
    hh.AddBool(store_cfg.write_vector_bucket);
    hh.AddBool(store_cfg.write_basic_to_bucket);

    // Base encoding knobs (exclude enable flag).
    hh.AddBool(cfg.base.encode.use_abs);
    hh.AddI32(cfg.base.encode.H_beam);
    hh.AddI32(cfg.base.encode.ils_iters);
    hh.AddI32(cfg.base.encode.icm_iters);
    hh.AddI32(cfg.base.encode.perturb_k);
    hh.AddI32(cfg.base.encode.hnorms);
    hh.AddI32(cfg.base.encode.seed);

    return hh.h;
}

std::uint64_t ComputeBaseListStoreHash(const Config& cfg,
                                       std::uint64_t base_basic_hash,
                                       int nlist,
                                       std::uint64_t ntotal) {
    Hash64 hh;
    hh.AddStr("base_list_store_v1");
    hh.AddU64(base_basic_hash);
    hh.AddI32(nlist);
    hh.AddU64(ntotal);
    hh.AddI32(cfg.model.m);
    hh.AddI32(std::max(0, cfg.model.m - 1));
    hh.AddBool(cfg.large.write_basic_to_bucket);
    hh.AddBool(cfg.large.write_vector_bucket);
    return hh.h;
}

std::uint64_t ComputeLinkageListStoreHash(const Config& cfg,
                                        const TrainResult& tr,
                                        std::uint64_t base_list_hash,
                                        int nlist) {
    Hash64 hh;
    hh.AddStr("linkage_list_store_v1");
    hh.AddU64(HashTrainModelForLinkage(tr));
    hh.AddU64(base_list_hash);
    hh.AddI32(nlist);
    hh.AddI32(cfg.model.m);
    hh.AddVecI32(cfg.model.h_vec);
    hh.AddI32(cfg.model.h0_one);

    HashLinkageCfg(&hh, cfg.base.linkage);

    hh.AddBool(cfg.virtual_cfg.enabled);
    hh.AddF64(cfg.virtual_cfg.virtual_ratio);
    hh.AddF64(cfg.virtual_cfg.good_fraction);
    hh.AddI32(cfg.virtual_cfg.min_virtual);
    hh.AddI32(cfg.virtual_cfg.max_virtual);
    hh.AddF32(cfg.virtual_cfg.alpha_bad);
    hh.AddBool(cfg.virtual_cfg.use_fixed_virtual_per_cluster);
    hh.AddI32(cfg.virtual_cfg.fixed_virtual_per_cluster);
    hh.AddI32(cfg.virtual_cfg.umap_knn_k);
    hh.AddI32(cfg.virtual_cfg.local_connectivity);
    hh.AddF64(cfg.virtual_cfg.overlap_thr);
    hh.AddBool(cfg.virtual_cfg.prefer_peaks);
    hh.AddI32(cfg.virtual_cfg.anchor_neighbor_k);

    hh.AddI32(cfg.hnsw.M);
    hh.AddI32(cfg.hnsw.candidate_multiplier_good);
    hh.AddI32(cfg.hnsw.candidate_multiplier_bad);
    if (cfg.hnsw.ef_construction_cap > 0) {
        hh.AddI32(cfg.hnsw.ef_construction_cap);
    }

    return hh.h;
}

std::uint64_t ComputeLinkageCoeffCodecHash(const Config& cfg,
                                         std::uint64_t linkage_list_hash) {
    Hash64 hh;
    hh.AddStr("linkage_coeff_codec_v1");
    hh.AddU64(linkage_list_hash);
    hh.AddStr(cfg.large.linkage_coeff_codec.granularity);
    hh.AddVecI32(cfg.large.linkage_coeff_codec.bits_per_layer);
    hh.AddU64(static_cast<std::uint64_t>(cfg.large.linkage_coeff_codec.p_first_candidates.size()));
    for (double p : cfg.large.linkage_coeff_codec.p_first_candidates) hh.AddF64(p);
    hh.AddU64(static_cast<std::uint64_t>(cfg.large.linkage_coeff_codec.p_rest_candidates.size()));
    for (double p : cfg.large.linkage_coeff_codec.p_rest_candidates) hh.AddF64(p);
    hh.AddBool(cfg.large.linkage_coeff_codec.use_weighted_quantile);
    hh.AddBool(cfg.large.linkage_coeff_codec.allow_clip);
    hh.AddBool(cfg.large.linkage_coeff_codec.fit_scale);
    hh.AddI32(cfg.large.linkage_coeff_codec.q_refine_sweeps);
    hh.AddI32(cfg.large.linkage_coeff_codec.q_refine_max_layer);
    hh.AddI32(cfg.large.linkage_coeff_codec.q_refine_step_limit);
    return hh.h;
}

}  // namespace stlq::app
