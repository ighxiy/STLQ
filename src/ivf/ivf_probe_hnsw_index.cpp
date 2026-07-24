#include "stlq/ivf/ivf_probe_hnsw_index.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <omp.h>

#include "hnswlib/hnswlib.h"
#include "hnswlib/space_ip.h"

namespace stlq::ivf {

namespace {

static std::string Trim(std::string s) {
    std::size_t lo = 0;
    while (lo < s.size() && std::isspace(static_cast<unsigned char>(s[lo]))) ++lo;
    std::size_t hi = s.size();
    while (hi > lo && std::isspace(static_cast<unsigned char>(s[hi - 1]))) --hi;
    return s.substr(lo, hi - lo);
}

static bool StartsWith(const std::string& s, const char* p) {
    const std::size_t n = std::strlen(p);
    return s.size() >= n && s.compare(0, n, p) == 0;
}

static bool ParseKeyValueLine(const std::string& line_in, std::string* out_k, std::string* out_v) {
    if (!out_k || !out_v) return false;
    *out_k = {};
    *out_v = {};

    std::string line = Trim(line_in);
    if (line.empty()) return false;
    if (StartsWith(line, "#") || StartsWith(line, "//")) return false;

    // Strip trailing comments.
    const std::size_t hash_pos = line.find('#');
    const std::size_t slash_pos = line.find("//");
    std::size_t cut = std::string::npos;
    if (hash_pos != std::string::npos) cut = hash_pos;
    if (slash_pos != std::string::npos) cut = (cut == std::string::npos) ? slash_pos : std::min(cut, slash_pos);
    if (cut != std::string::npos) {
        line = Trim(line.substr(0, cut));
        if (line.empty()) return false;
    }

    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) return false;
    std::string k = Trim(line.substr(0, eq));
    std::string v = Trim(line.substr(eq + 1));
    if (k.empty()) return false;

    // Remove optional quotes.
    if (v.size() >= 2) {
        const char q0 = v.front();
        const char q1 = v.back();
        if ((q0 == '"' && q1 == '"') || (q0 == '\'' && q1 == '\'')) {
            v = v.substr(1, v.size() - 2);
        }
    }

    *out_k = std::move(k);
    *out_v = std::move(v);
    return true;
}

static bool ReadMetaFile(const std::filesystem::path& path,
                         std::unordered_map<std::string, std::string>* out,
                         std::string* err) {
    if (!out) {
        if (err) *err = "ReadMetaFile: out is null.";
        return false;
    }
    out->clear();
    std::ifstream in(path);
    if (!in) {
        if (err) *err = "ReadMetaFile: failed to open " + path.string();
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        std::string k, v;
        if (!ParseKeyValueLine(line, &k, &v)) continue;
        (*out)[k] = v;
    }
    return true;
}

static bool WriteMetaFile(const std::filesystem::path& path,
                          const std::vector<std::pair<std::string, std::string>>& kv,
                          std::string* err) {
    std::ofstream out(path);
    if (!out) {
        if (err) *err = "WriteMetaFile: failed to open " + path.string();
        return false;
    }
    for (const auto& [k, v] : kv) {
        out << k << "=" << v << "\n";
    }
    out.flush();
    if (!out) {
        if (err) *err = "WriteMetaFile: write failed: " + path.string();
        return false;
    }
    return true;
}

static std::uint64_t Mix64(std::uint64_t h, std::uint64_t v) {
    h ^= v;
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
}

static std::string ToHexU64(std::uint64_t v) {
    std::ostringstream oss;
    oss << "0x" << std::hex << v;
    return oss.str();
}

static bool ParseU64Auto(const std::string& s, std::uint64_t* out) {
    if (!out) return false;
    std::string t = Trim(s);
    if (t.empty()) return false;
    std::uint64_t v = 0;
    if (t.size() > 2 && (t[0] == '0') && (t[1] == 'x' || t[1] == 'X')) {
        std::istringstream iss(t);
        iss >> std::hex >> v;
        if (!iss) return false;
    } else {
        try {
            v = static_cast<std::uint64_t>(std::stoull(t));
        } catch (...) {
            return false;
        }
    }
    *out = v;
    return true;
}

static bool ParseI32(const std::string& s, int* out) {
    if (!out) return false;
    try {
        *out = std::stoi(Trim(s));
        return true;
    } catch (...) {
        return false;
    }
}

static std::uint64_t ComputeUnitCentroidsHash64(const stlq::ColMajorMatrix<float>& C_root0) {
    const int d = C_root0.rows;
    const int nlist = C_root0.cols;
    std::uint64_t h = 0x9e3779b97f4a7c15ULL;
    h = Mix64(h, 1ULL);  // version
    h = Mix64(h, static_cast<std::uint64_t>(d));
    h = Mix64(h, static_cast<std::uint64_t>(nlist));

    // Hash normalized float bits for stability.
    for (int cid = 0; cid < nlist; ++cid) {
        const float* c = C_root0.Col(cid);
        double ss = 0.0;
        for (int r = 0; r < d; ++r) {
            const auto v = static_cast<double>(c[r]);
            ss += v * v;
        }
        const float inv_norm = 1.0f / std::sqrt(std::max(static_cast<float>(ss), 1e-20f));
        for (int r = 0; r < d; ++r) {
            const float v = c[r] * inv_norm;
            std::uint32_t u = 0;
            static_assert(sizeof(float) == sizeof(std::uint32_t));
            std::memcpy(&u, &v, sizeof(u));
            h = Mix64(h, static_cast<std::uint64_t>(u));
        }
    }
    return h;
}

}  // namespace

struct IvfProbeHnswIndex::Impl {
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
};

IvfProbeHnswIndex::IvfProbeHnswIndex() = default;
IvfProbeHnswIndex::~IvfProbeHnswIndex() = default;

std::filesystem::path IvfProbeHnswIndex::DefaultIndexDirFromLinkageListDir(const std::filesystem::path& linkage_list_dir) {
    // linkage_list_dir is typically: <out_dir>/linkage_list
    return linkage_list_dir.parent_path() / "ivf_hnsw";
}

bool IvfProbeHnswIndex::LoadOrBuildUnitIP(const ColMajorMatrix<float>& C_root0,
                                          const std::filesystem::path& index_dir,
                                          int M,
                                          int ef_construction,
                                          int build_threads,
                                          bool* out_built,
                                          std::string* err) {
    if (out_built) *out_built = false;
    if (C_root0.rows <= 0 || C_root0.cols <= 0) {
        if (err) *err = "IvfProbeHnswIndex::LoadOrBuildUnitIP: invalid C_root0 shape.";
        return false;
    }

    const int d = C_root0.rows;
    const int nlist = C_root0.cols;
    const int M_eff = std::max(2, M);
    const int efc_eff = std::max(8, ef_construction);
    const int nth = (build_threads > 0) ? build_threads : std::max(1, omp_get_max_threads());

    const std::filesystem::path index_path = index_dir / "index.bin";
    const std::filesystem::path meta_path = index_dir / "meta.txt";

    std::error_code ec;
    const bool has_index = std::filesystem::exists(index_path, ec) && !ec;
    const bool has_meta = std::filesystem::exists(meta_path, ec) && !ec;

    const std::uint64_t cur_hash = ComputeUnitCentroidsHash64(C_root0);

    auto try_load = [&]() -> bool {
        if (!has_index || !has_meta) return false;
        std::unordered_map<std::string, std::string> meta;
        std::string meta_err;
        if (!ReadMetaFile(meta_path, &meta, &meta_err)) {
            return false;
        }

        auto get = [&](const char* k) -> std::string {
            auto it = meta.find(k);
            return (it == meta.end()) ? std::string() : it->second;
        };

        int meta_version = 0;
        int meta_d = 0;
        int meta_nlist = 0;
        int meta_M = 0;
        int meta_efc = 0;
        std::uint64_t meta_hash = 0;

        if (!ParseI32(get("version"), &meta_version) || meta_version != 1) return false;
        if (!ParseI32(get("d"), &meta_d) || meta_d != d) return false;
        if (!ParseI32(get("nlist"), &meta_nlist) || meta_nlist != nlist) return false;
        if (!ParseI32(get("M"), &meta_M) || meta_M != M_eff) return false;
        if (!ParseI32(get("ef_construction"), &meta_efc) || meta_efc != efc_eff) return false;
        if (!ParseU64Auto(get("centroids_unit_hash"), &meta_hash) || meta_hash != cur_hash) return false;
        if (Trim(get("space")) != "ip_unit") return false;

        try {
            impl_ = std::make_unique<Impl>();
            impl_->space = std::make_unique<hnswlib::InnerProductSpace>(d);
            impl_->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                impl_->space.get(), index_path.string(), /*nmslib=*/false,
                /*max_elements=*/static_cast<std::size_t>(nlist));
        } catch (...) {
            impl_.reset();
            return false;
        }

        d_ = d;
        nlist_ = nlist;
        M_ = M_eff;
        ef_construction_ = efc_eff;
        centroids_unit_hash_ = cur_hash;
        return true;
    };

    if (try_load()) {
        return true;
    }

    // Rebuild (no index, no meta, or meta mismatch).
    std::filesystem::create_directories(index_dir, ec);
    if (ec) {
        if (err) *err = "IvfProbeHnswIndex: cannot create directory: " + index_dir.string();
        return false;
    }

    // Build unit-normalized centroid matrix (col-major, one vector per centroid).
    std::vector<float> C_unit(static_cast<std::size_t>(d) * static_cast<std::size_t>(nlist), 0.0f);
    for (int cid = 0; cid < nlist; ++cid) {
        const float* c = C_root0.Col(cid);
        float* cu = C_unit.data() + static_cast<std::size_t>(cid) * static_cast<std::size_t>(d);
        double ss = 0.0;
        for (int r = 0; r < d; ++r) {
            const auto v = static_cast<double>(c[r]);
            ss += v * v;
        }
        const float inv_norm = 1.0f / std::sqrt(std::max(static_cast<float>(ss), 1e-20f));
        for (int r = 0; r < d; ++r) {
            cu[r] = c[r] * inv_norm;
        }
    }

    try {
        impl_ = std::make_unique<Impl>();
        impl_->space = std::make_unique<hnswlib::InnerProductSpace>(d);
        impl_->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            impl_->space.get(), static_cast<std::size_t>(nlist),
            static_cast<std::size_t>(M_eff), static_cast<std::size_t>(efc_eff),
            /*random_seed=*/100u);

        #pragma omp parallel for schedule(static) num_threads(nth)
        for (int cid = 0; cid < nlist; ++cid) {
            const float* cu = C_unit.data() + static_cast<std::size_t>(cid) * static_cast<std::size_t>(d);
            impl_->index->addPoint(reinterpret_cast<const void*>(cu), static_cast<hnswlib::labeltype>(cid));
        }
        impl_->index->saveIndex(index_path.string());
    } catch (const std::exception& e) {
        if (err) *err = std::string("IvfProbeHnswIndex: build/save failed: ") + e.what();
        impl_.reset();
        return false;
    } catch (...) {
        if (err) *err = "IvfProbeHnswIndex: build/save failed.";
        impl_.reset();
        return false;
    }

    std::vector<std::pair<std::string, std::string>> meta_kv;
    meta_kv.emplace_back("version", "1");
    meta_kv.emplace_back("space", "ip_unit");
    meta_kv.emplace_back("d", std::to_string(d));
    meta_kv.emplace_back("nlist", std::to_string(nlist));
    meta_kv.emplace_back("M", std::to_string(M_eff));
    meta_kv.emplace_back("ef_construction", std::to_string(efc_eff));
    meta_kv.emplace_back("centroids_unit_hash", ToHexU64(cur_hash));

    std::string meta_err;
    if (!WriteMetaFile(meta_path, meta_kv, &meta_err)) {
        if (err) *err = meta_err.empty() ? ("IvfProbeHnswIndex: failed to write meta: " + meta_path.string()) : meta_err;
        return false;
    }

    d_ = d;
    nlist_ = nlist;
    M_ = M_eff;
    ef_construction_ = efc_eff;
    centroids_unit_hash_ = cur_hash;
    if (out_built) *out_built = true;
    return true;
}

bool IvfProbeHnswIndex::SetEfSearch(int ef_search, std::string* err) {
    if (!impl_ || !impl_->index) {
        if (err) *err = "IvfProbeHnswIndex::SetEfSearch: index not loaded.";
        return false;
    }
    const int ef = std::max(ef_search, 8);
    try {
        impl_->index->setEf(static_cast<std::size_t>(ef));
        return true;
    } catch (const std::exception& e) {
        if (err) *err = std::string("IvfProbeHnswIndex::SetEfSearch failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "IvfProbeHnswIndex::SetEfSearch failed.";
        return false;
    }
}

bool IvfProbeHnswIndex::SearchTopK(const float* query,
                                  int k,
                                  int* out_ids,
                                  float* out_dist,
                                  std::string* err) const {
    if (!impl_ || !impl_->index || !impl_->space) {
        if (err) *err = "IvfProbeHnswIndex::SearchTopK: index not loaded.";
        return false;
    }
    if (!query || !out_ids) {
        if (err) *err = "IvfProbeHnswIndex::SearchTopK: null pointer.";
        return false;
    }
    const int want = std::max(1, std::min(k, nlist_));

    try {
        auto pq = impl_->index->searchKnn(reinterpret_cast<const void*>(query), static_cast<std::size_t>(want));
        // searchKnn returns a max-heap (worst on top). Pop then reverse.
        int got = 0;
        if (out_dist) {
            while (!pq.empty() && got < want) {
                const auto& top = pq.top();
                out_dist[got] = top.first;
                out_ids[got] = static_cast<int>(top.second);
                pq.pop();
                ++got;
            }
            std::reverse(out_ids, out_ids + got);
            std::reverse(out_dist, out_dist + got);
            for (int i = got; i < want; ++i) {
                out_ids[i] = -1;
                out_dist[i] = std::numeric_limits<float>::infinity();
            }
        } else {
            while (!pq.empty() && got < want) {
                out_ids[got] = static_cast<int>(pq.top().second);
                pq.pop();
                ++got;
            }
            std::reverse(out_ids, out_ids + got);
            for (int i = got; i < want; ++i) {
                out_ids[i] = -1;
            }
        }
        return true;
    } catch (const std::exception& e) {
        if (err) *err = std::string("IvfProbeHnswIndex::SearchTopK failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "IvfProbeHnswIndex::SearchTopK failed.";
        return false;
    }
}

}  // namespace stlq::ivf
