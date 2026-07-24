#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stlq/io/linkage_list_store.h"

namespace {

std::string GetArg(int argc, char** argv, const std::string& key, const std::string& def) {
    const std::string prefix = key + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i] ? argv[i] : "");
        if (a == key) {
            if (i + 1 < argc && argv[i + 1]) return argv[i + 1];
            return def;
        }
        if (a.rfind(prefix, 0) == 0) return a.substr(prefix.size());
    }
    return def;
}

int GetArgI32(int argc, char** argv, const std::string& key, int def) {
    const std::string v = GetArg(argc, argv, key, "");
    return v.empty() ? def : std::atoi(v.c_str());
}

bool HasLinkageMeta(const std::filesystem::path& dir) {
    std::error_code ec;
    return std::filesystem::exists(dir / "meta.bin", ec) && !ec;
}

double Quantile(std::vector<double> xs, double q) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    const double pos = std::clamp(q, 0.0, 1.0) * static_cast<double>(xs.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    if (lo == hi) return xs[lo];
    const double w = pos - static_cast<double>(lo);
    return xs[lo] * (1.0 - w) + xs[hi] * w;
}

double Mean(const std::vector<double>& xs) {
    if (xs.empty()) return 0.0;
    return std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

double Gini(std::vector<double> xs) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    const double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
    if (sum <= 0.0) return 0.0;
    long double weighted = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        weighted += static_cast<long double>(i + 1) * static_cast<long double>(xs[i]);
    }
    const long double n = static_cast<long double>(xs.size());
    return static_cast<double>((2.0L * weighted) / (n * static_cast<long double>(sum)) - (n + 1.0L) / n);
}

void AddHist(std::vector<std::uint64_t>* hist, int idx, std::uint64_t delta = 1) {
    if (!hist || idx < 0) return;
    if (static_cast<std::size_t>(idx) >= hist->size()) hist->resize(static_cast<std::size_t>(idx) + 1, 0);
    (*hist)[static_cast<std::size_t>(idx)] += delta;
}

struct ClusterSummary {
    int cid = -1;
    std::uint32_t n_real = 0;
    std::uint32_t n_virt = 0;
    std::uint32_t n_root = 0;
    int max_depth = 0;
    double mean_depth = 0.0;
    std::uint32_t max_child_degree = 0;
    std::uint32_t max_subtree_real = 0;
    double degree_gini = 0.0;
    double linked_ratio = 0.0;
    double real_leaf_ratio = 0.0;
    double real_single_child_ratio = 0.0;
    double real_branching_ratio = 0.0;
    std::uint32_t root_component_count = 0;
    double root_component_mean = 0.0;
    double root_component_p90 = 0.0;
    std::uint32_t root_component_max = 0;
    std::uint32_t virtual_root_component_count = 0;
    double virtual_root_component_mean = 0.0;
    double virtual_root_component_p90 = 0.0;
};

struct DepthAgg {
    std::uint64_t count = 0;
    std::uint64_t parent_super_root = 0;
    std::uint64_t parent_virtual = 0;
    std::uint64_t parent_real_prev_depth = 0;
    std::uint64_t parent_other = 0;
    std::uint64_t leaf = 0;
    std::uint64_t single_child = 0;
    std::uint64_t branching = 0;
    double degree_sum = 0.0;
    double subtree_sum = 0.0;
};

struct Stats {
    std::uint64_t nlist = 0;
    std::uint64_t total_real = 0;
    std::uint64_t total_virt = 0;
    std::uint64_t total_root_real = 0;
    std::uint64_t total_linked_real = 0;
    std::uint64_t empty_clusters = 0;
    std::uint64_t invalid_clusters = 0;
    std::uint64_t nonempty_clusters = 0;

    std::vector<std::uint64_t> depth_hist;
    std::vector<std::uint64_t> child_degree_hist;
    std::vector<std::uint64_t> subtree_real_hist;
    std::vector<std::uint64_t> parent_depth_delta_hist;  // index = child_depth - parent_depth, capped by actual delta.

    std::uint64_t parent_super_root = 0;
    std::uint64_t parent_virtual = 0;
    std::uint64_t parent_real_same_depth = 0;
    std::uint64_t parent_real_prev_depth = 0;
    std::uint64_t parent_real_skip_depth = 0;
    std::uint64_t parent_real_deeper_or_invalid = 0;

    std::uint64_t real_leaf_nodes = 0;
    std::uint64_t real_single_child_nodes = 0;
    std::uint64_t real_branching_nodes = 0;
    std::uint64_t all_leaf_nodes = 0;
    std::uint64_t all_single_child_nodes = 0;
    std::uint64_t all_branching_nodes = 0;

    std::vector<double> cluster_mean_depths;
    std::vector<double> cluster_degree_ginis;
    std::vector<double> node_degrees;
    std::vector<double> node_subtrees;
    std::vector<double> root_component_subtrees;
    std::vector<double> real_root_component_subtrees;
    std::vector<double> virtual_root_component_subtrees;
    std::vector<DepthAgg> depth_aggs;
    std::vector<ClusterSummary> clusters;
};

struct CsvWriters {
    bool enabled = false;
    std::ofstream clusters;
    std::ofstream nodes;
    std::ofstream edges;
};

std::string ParentSourceLabel(int child_depth,
                              int parent_local,
                              int n_virt,
                              int n_real,
                              const std::vector<int>& real_depth) {
    if (parent_local < 0) return "super_root";
    if (parent_local < n_virt) return "virtual";
    if (parent_local >= n_virt + n_real) return "invalid";
    const int parent_real = parent_local - n_virt;
    const int parent_depth = real_depth[static_cast<std::size_t>(parent_real)];
    const int delta = child_depth - parent_depth;
    if (delta == 0) return "real_same_depth";
    if (delta == 1) return "real_prev_depth";
    if (delta > 1) return "real_skip_depth";
    return "real_deeper_or_invalid";
}

bool OpenCsvWriters(const std::string& csv_dir, CsvWriters* out, std::string* err) {
    if (!out) return false;
    *out = {};
    if (csv_dir.empty()) return true;
    std::error_code ec;
    std::filesystem::create_directories(csv_dir, ec);
    if (ec) {
        if (err) *err = "failed to create csv_dir: " + csv_dir + ": " + ec.message();
        return false;
    }
    const std::filesystem::path dir(csv_dir);
    out->clusters.open(dir / "clusters.csv", std::ios::binary | std::ios::trunc);
    out->nodes.open(dir / "nodes.csv", std::ios::binary | std::ios::trunc);
    out->edges.open(dir / "edges.csv", std::ios::binary | std::ios::trunc);
    if (!out->clusters || !out->nodes || !out->edges) {
        if (err) *err = "failed to open graph quality csv files under: " + csv_dir;
        return false;
    }
    out->enabled = true;
    out->clusters << "cid,n_real,n_virtual,n_root,linked_ratio,max_depth,mean_depth,"
                     "max_child_degree,max_subtree_real,degree_gini,"
                     "real_leaf_ratio,real_single_child_ratio,real_branching_ratio,"
                     "root_component_count,root_component_mean,root_component_p90,root_component_max,"
                     "virtual_root_component_count,virtual_root_component_mean,virtual_root_component_p90\n";
    out->nodes << "cid,local_id,node_kind,real_pos,global_id,depth,parent_local_id,parent_kind,parent_depth,child_degree,subtree_real\n";
    out->edges << "cid,child_local_id,child_global_id,child_depth,parent_local_id,parent_kind,parent_real_pos,parent_global_id,parent_depth,depth_delta\n";
    return true;
}

void WriteCsvRows(CsvWriters* csv,
                  int cid,
                  int n_real,
                  int n_virt,
                  const std::vector<std::uint32_t>& real_ids,
                  const std::vector<std::uint32_t>& parent,
                  const std::vector<int>& real_depth,
                  const std::vector<std::uint32_t>& degree,
                  const std::vector<std::uint32_t>& subtree,
                  const ClusterSummary& cs) {
    if (!csv || !csv->enabled) return;
    csv->clusters << cid << ","
                  << cs.n_real << ","
                  << cs.n_virt << ","
                  << cs.n_root << ","
                  << cs.linked_ratio << ","
                  << cs.max_depth << ","
                  << cs.mean_depth << ","
                  << cs.max_child_degree << ","
                  << cs.max_subtree_real << ","
                  << cs.degree_gini << ","
                  << cs.real_leaf_ratio << ","
                  << cs.real_single_child_ratio << ","
                  << cs.real_branching_ratio << ","
                  << cs.root_component_count << ","
                  << cs.root_component_mean << ","
                  << cs.root_component_p90 << ","
                  << cs.root_component_max << ","
                  << cs.virtual_root_component_count << ","
                  << cs.virtual_root_component_mean << ","
                  << cs.virtual_root_component_p90 << "\n";

    for (int local = 0; local < n_virt; ++local) {
        csv->nodes << cid << ","
                   << local << ",virtual,-1,-1,-1,-1,super_root,-1,"
                   << degree[static_cast<std::size_t>(local)] << ","
                   << subtree[static_cast<std::size_t>(local)] << "\n";
    }
    for (int i = 0; i < n_real; ++i) {
        const int local = n_virt + i;
        const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
        const int p_local = (p1 == 0) ? -1 : static_cast<int>(p1) - 1;
        int p_depth = -1;
        int p_real = -1;
        std::uint32_t p_gid = 0;
        const std::string p_kind = ParentSourceLabel(real_depth[static_cast<std::size_t>(i)],
                                                     p_local, n_virt, n_real, real_depth);
        if (p_local >= n_virt && p_local < n_virt + n_real) {
            p_real = p_local - n_virt;
            p_depth = real_depth[static_cast<std::size_t>(p_real)];
            p_gid = real_ids[static_cast<std::size_t>(p_real)];
        }
        csv->nodes << cid << ","
                   << local << ",real,"
                   << i << ","
                   << real_ids[static_cast<std::size_t>(i)] << ","
                   << real_depth[static_cast<std::size_t>(i)] << ","
                   << p_local << ","
                   << p_kind << ","
                   << p_depth << ","
                   << degree[static_cast<std::size_t>(local)] << ","
                   << subtree[static_cast<std::size_t>(local)] << "\n";
        if (p1 != 0) {
            const int delta = (p_depth >= 0) ? (real_depth[static_cast<std::size_t>(i)] - p_depth) : -1;
            csv->edges << cid << ","
                       << local << ","
                       << real_ids[static_cast<std::size_t>(i)] << ","
                       << real_depth[static_cast<std::size_t>(i)] << ","
                       << p_local << ","
                       << p_kind << ","
                       << p_real << ","
                       << ((p_real >= 0) ? std::to_string(p_gid) : std::string("-1")) << ","
                       << p_depth << ","
                       << delta << "\n";
        }
    }
}

std::vector<int> BuildRealDepth(const std::vector<std::uint32_t>& depth_offsets, int n_real) {
    std::vector<int> real_depth(static_cast<std::size_t>(n_real), 0);
    for (std::size_t dep = 0; dep + 1 < depth_offsets.size(); ++dep) {
        const auto lo = static_cast<int>(depth_offsets[dep]);
        const auto hi = static_cast<int>(depth_offsets[dep + 1]);
        for (int i = std::max(0, lo); i < std::min(n_real, hi); ++i) {
            real_depth[static_cast<std::size_t>(i)] = static_cast<int>(dep);
        }
    }
    return real_depth;
}

DepthAgg& EnsureDepthAgg(std::vector<DepthAgg>* xs, int depth) {
    if (depth < 0) depth = 0;
    if (static_cast<std::size_t>(depth) >= xs->size()) xs->resize(static_cast<std::size_t>(depth) + 1);
    return (*xs)[static_cast<std::size_t>(depth)];
}

bool Analyze(const stlq::io::LinkageListReader& reader, Stats* out, CsvWriters* csv, std::string* err) {
    if (!out) return false;
    *out = {};
    out->nlist = static_cast<std::uint64_t>(reader.nlist());

    for (int cid = 0; cid < reader.nlist(); ++cid) {
        std::uint64_t real_lo = 0, real_hi = 0, virt_lo = 0, virt_hi = 0, depth_lo = 0, depth_hi = 0;
        if (!reader.ReadClusterSpan(cid, &real_lo, &real_hi, &virt_lo, &virt_hi, &depth_lo, &depth_hi, err)) {
            return false;
        }
        const int n_real = static_cast<int>(real_hi - real_lo);
        const int n_virt = static_cast<int>(virt_hi - virt_lo);
        if (n_real <= 0) {
            out->empty_clusters += 1;
            continue;
        }

        std::vector<std::uint32_t> real_ids;
        std::vector<std::uint32_t> parent;
        std::vector<std::uint32_t> depth_offsets;
        std::vector<std::uint8_t> codes;
        std::vector<float> coeffs;
        std::vector<std::uint8_t> code0;
        std::vector<float> a0;
        std::vector<std::uint8_t> virt_codes;
        std::vector<float> virt_coeffs;
        std::vector<float> virt_a0;
        if (!reader.ReadCluster(cid, &real_ids, &parent, &depth_offsets, &codes, &coeffs,
                                &code0, &a0, &virt_codes, &virt_coeffs, &virt_a0, err)) {
            return false;
        }
        if (static_cast<int>(real_ids.size()) != n_real ||
            static_cast<int>(parent.size()) != n_real ||
            depth_offsets.size() < 2 ||
            depth_offsets.back() != static_cast<std::uint32_t>(n_real)) {
            out->invalid_clusters += 1;
            continue;
        }

        const std::vector<int> real_depth = BuildRealDepth(depth_offsets, n_real);
        std::vector<std::uint32_t> degree(static_cast<std::size_t>(n_real + n_virt), 0);
        std::vector<std::uint32_t> subtree(static_cast<std::size_t>(n_real + n_virt), 0);
        std::uint64_t depth_sum = 0;
        int max_depth = 0;
        const std::uint32_t n_root = depth_offsets[1] - depth_offsets[0];

        for (int i = 0; i < n_real; ++i) {
            const int child_local = n_virt + i;
            subtree[static_cast<std::size_t>(child_local)] = 1;
            const int dep = real_depth[static_cast<std::size_t>(i)];
            max_depth = std::max(max_depth, dep);
            depth_sum += static_cast<std::uint64_t>(dep);
            AddHist(&out->depth_hist, dep);
            DepthAgg& da = EnsureDepthAgg(&out->depth_aggs, dep);
            da.count += 1;

            const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
            if (p1 == 0) {
                out->parent_super_root += 1;
                da.parent_super_root += 1;
                continue;
            }
            const int p_local = static_cast<int>(p1) - 1;
            if (p_local < 0 || p_local >= n_real + n_virt) {
                out->parent_real_deeper_or_invalid += 1;
                da.parent_other += 1;
                continue;
            }
            degree[static_cast<std::size_t>(p_local)] += 1;
            if (p_local < n_virt) {
                out->parent_virtual += 1;
                da.parent_virtual += 1;
                continue;
            }
            const int p_real = p_local - n_virt;
            const int p_dep = real_depth[static_cast<std::size_t>(p_real)];
            const int delta = dep - p_dep;
            if (delta == 0) {
                out->parent_real_same_depth += 1;
                da.parent_other += 1;
            } else if (delta == 1) {
                out->parent_real_prev_depth += 1;
                da.parent_real_prev_depth += 1;
            } else if (delta > 1) {
                out->parent_real_skip_depth += 1;
                da.parent_other += 1;
            } else {
                out->parent_real_deeper_or_invalid += 1;
                da.parent_other += 1;
            }
            AddHist(&out->parent_depth_delta_hist, std::max(0, delta));
        }

        for (int i = n_real - 1; i >= 0; --i) {
            const std::uint32_t p1 = parent[static_cast<std::size_t>(i)];
            if (p1 == 0) continue;
            const int child_local = n_virt + i;
            const int p_local = static_cast<int>(p1) - 1;
            if (p_local >= 0 && p_local < n_real + n_virt) {
                subtree[static_cast<std::size_t>(p_local)] += subtree[static_cast<std::size_t>(child_local)];
            }
        }

        std::uint32_t max_degree = 0;
        std::uint32_t max_subtree = 0;
        std::uint32_t cluster_real_leaf = 0;
        std::uint32_t cluster_real_single_child = 0;
        std::uint32_t cluster_real_branching = 0;
        std::vector<double> cluster_root_components;
        std::vector<double> cluster_virtual_root_components;
        std::vector<double> cluster_degrees;
        cluster_degrees.reserve(static_cast<std::size_t>(n_real + n_virt));
        for (int local = 0; local < n_real + n_virt; ++local) {
            const std::uint32_t deg = degree[static_cast<std::size_t>(local)];
            const std::uint32_t sub = subtree[static_cast<std::size_t>(local)];
            max_degree = std::max(max_degree, deg);
            max_subtree = std::max(max_subtree, sub);
            if (deg == 0) out->all_leaf_nodes += 1;
            else if (deg == 1) out->all_single_child_nodes += 1;
            else out->all_branching_nodes += 1;
            AddHist(&out->child_degree_hist, static_cast<int>(deg));
            AddHist(&out->subtree_real_hist, static_cast<int>(sub));
            out->node_degrees.push_back(static_cast<double>(deg));
            out->node_subtrees.push_back(static_cast<double>(sub));
            cluster_degrees.push_back(static_cast<double>(deg));
        }

        for (int local = 0; local < n_virt; ++local) {
            const std::uint32_t sub = subtree[static_cast<std::size_t>(local)];
            if (sub > 0) {
                out->root_component_subtrees.push_back(static_cast<double>(sub));
                out->virtual_root_component_subtrees.push_back(static_cast<double>(sub));
                cluster_root_components.push_back(static_cast<double>(sub));
                cluster_virtual_root_components.push_back(static_cast<double>(sub));
            }
        }
        for (int i = 0; i < n_real; ++i) {
            const int local = n_virt + i;
            const std::uint32_t deg = degree[static_cast<std::size_t>(local)];
            const std::uint32_t sub = subtree[static_cast<std::size_t>(local)];
            DepthAgg& da = EnsureDepthAgg(&out->depth_aggs, real_depth[static_cast<std::size_t>(i)]);
            da.degree_sum += static_cast<double>(deg);
            da.subtree_sum += static_cast<double>(sub);
            if (deg == 0) {
                out->real_leaf_nodes += 1;
                cluster_real_leaf += 1;
                da.leaf += 1;
            } else if (deg == 1) {
                out->real_single_child_nodes += 1;
                cluster_real_single_child += 1;
                da.single_child += 1;
            } else {
                out->real_branching_nodes += 1;
                cluster_real_branching += 1;
                da.branching += 1;
            }
            if (parent[static_cast<std::size_t>(i)] == 0) {
                out->root_component_subtrees.push_back(static_cast<double>(sub));
                out->real_root_component_subtrees.push_back(static_cast<double>(sub));
                cluster_root_components.push_back(static_cast<double>(sub));
            }
        }

        ClusterSummary cs;
        cs.cid = cid;
        cs.n_real = static_cast<std::uint32_t>(n_real);
        cs.n_virt = static_cast<std::uint32_t>(n_virt);
        cs.n_root = n_root;
        cs.max_depth = max_depth;
        cs.mean_depth = static_cast<double>(depth_sum) / static_cast<double>(n_real);
        cs.max_child_degree = max_degree;
        cs.max_subtree_real = max_subtree;
        cs.degree_gini = Gini(cluster_degrees);
        cs.linked_ratio = (n_real > 0) ? 1.0 - static_cast<double>(n_root) / static_cast<double>(n_real) : 0.0;
        cs.real_leaf_ratio = static_cast<double>(cluster_real_leaf) / static_cast<double>(n_real);
        cs.real_single_child_ratio = static_cast<double>(cluster_real_single_child) / static_cast<double>(n_real);
        cs.real_branching_ratio = static_cast<double>(cluster_real_branching) / static_cast<double>(n_real);
        cs.root_component_count = static_cast<std::uint32_t>(cluster_root_components.size());
        cs.root_component_mean = Mean(cluster_root_components);
        cs.root_component_p90 = Quantile(cluster_root_components, 0.90);
        cs.root_component_max = cluster_root_components.empty()
                                    ? 0
                                    : static_cast<std::uint32_t>(*std::max_element(cluster_root_components.begin(),
                                                                                  cluster_root_components.end()));
        cs.virtual_root_component_count = static_cast<std::uint32_t>(cluster_virtual_root_components.size());
        cs.virtual_root_component_mean = Mean(cluster_virtual_root_components);
        cs.virtual_root_component_p90 = Quantile(cluster_virtual_root_components, 0.90);

        out->clusters.push_back(cs);
        out->cluster_mean_depths.push_back(cs.mean_depth);
        out->cluster_degree_ginis.push_back(cs.degree_gini);
        out->total_real += static_cast<std::uint64_t>(n_real);
        out->total_virt += static_cast<std::uint64_t>(n_virt);
        out->total_root_real += n_root;
        out->total_linked_real += static_cast<std::uint64_t>(n_real) - n_root;
        out->nonempty_clusters += 1;

        WriteCsvRows(csv, cid, n_real, n_virt, real_ids, parent, real_depth, degree, subtree, cs);
    }
    return true;
}

void PrintTopClusters(std::ostringstream* oss,
                      std::vector<ClusterSummary> clusters,
                      int topk,
                      const std::string& title,
                      const std::string& key) {
    if (!oss || topk <= 0) return;
    if (key == "degree") {
        std::sort(clusters.begin(), clusters.end(), [](const auto& a, const auto& b) {
            if (a.max_child_degree != b.max_child_degree) return a.max_child_degree > b.max_child_degree;
            return a.cid < b.cid;
        });
    } else if (key == "subtree") {
        std::sort(clusters.begin(), clusters.end(), [](const auto& a, const auto& b) {
            if (a.max_subtree_real != b.max_subtree_real) return a.max_subtree_real > b.max_subtree_real;
            return a.cid < b.cid;
        });
    } else if (key == "gini") {
        std::sort(clusters.begin(), clusters.end(), [](const auto& a, const auto& b) {
            if (a.degree_gini != b.degree_gini) return a.degree_gini > b.degree_gini;
            return a.cid < b.cid;
        });
    } else {
        std::sort(clusters.begin(), clusters.end(), [](const auto& a, const auto& b) {
            if (a.mean_depth != b.mean_depth) return a.mean_depth > b.mean_depth;
            return a.cid < b.cid;
        });
    }
    if (static_cast<int>(clusters.size()) > topk) clusters.resize(static_cast<std::size_t>(topk));
    *oss << "\n[" << title << "]\n";
    for (const auto& c : clusters) {
        *oss << "cid=" << c.cid
             << " n_real=" << c.n_real
             << " n_virt=" << c.n_virt
             << " n_root=" << c.n_root
             << " linked_ratio=" << c.linked_ratio
             << " max_depth=" << c.max_depth
             << " mean_depth=" << c.mean_depth
             << " max_child_degree=" << c.max_child_degree
             << " max_subtree_real=" << c.max_subtree_real
             << " degree_gini=" << c.degree_gini
             << " leaf_ratio=" << c.real_leaf_ratio
             << " branching_ratio=" << c.real_branching_ratio
             << " root_component_p90=" << c.root_component_p90
             << " root_component_max=" << c.root_component_max
             << " virtual_root_components=" << c.virtual_root_component_count
             << "\n";
    }
}

void PrintClusterGroupSummary(std::ostringstream* oss,
                              const std::vector<ClusterSummary>& clusters,
                              const std::string& title,
                              bool want_virtual_clusters) {
    if (!oss) return;
    std::uint64_t n_cluster = 0;
    std::uint64_t n_real = 0;
    std::uint64_t n_virt = 0;
    std::uint64_t n_root = 0;
    std::uint64_t root_components = 0;
    std::uint64_t virtual_root_components = 0;
    long double weighted_mean_depth = 0.0L;
    long double weighted_leaf_ratio = 0.0L;
    long double weighted_branching_ratio = 0.0L;
    long double weighted_degree_gini = 0.0L;
    std::vector<double> mean_depths;
    std::vector<double> max_depths;
    std::vector<double> max_subtrees;
    std::vector<double> root_component_p90s;
    std::vector<double> root_component_maxes;

    for (const ClusterSummary& c : clusters) {
        const bool is_virtual_cluster = c.n_virt > 0;
        if (is_virtual_cluster != want_virtual_clusters) continue;
        ++n_cluster;
        n_real += c.n_real;
        n_virt += c.n_virt;
        n_root += c.n_root;
        root_components += c.root_component_count;
        virtual_root_components += c.virtual_root_component_count;
        weighted_mean_depth += static_cast<long double>(c.mean_depth) * static_cast<long double>(c.n_real);
        weighted_leaf_ratio += static_cast<long double>(c.real_leaf_ratio) * static_cast<long double>(c.n_real);
        weighted_branching_ratio += static_cast<long double>(c.real_branching_ratio) * static_cast<long double>(c.n_real);
        weighted_degree_gini += static_cast<long double>(c.degree_gini) * static_cast<long double>(c.n_real);
        mean_depths.push_back(c.mean_depth);
        max_depths.push_back(static_cast<double>(c.max_depth));
        max_subtrees.push_back(static_cast<double>(c.max_subtree_real));
        root_component_p90s.push_back(c.root_component_p90);
        root_component_maxes.push_back(static_cast<double>(c.root_component_max));
    }

    const double denom = n_real ? static_cast<double>(n_real) : 0.0;
    *oss << "\n[" << title << "]\n";
    *oss << "clusters=" << n_cluster << "\n";
    *oss << "real_nodes=" << n_real << "\n";
    *oss << "virtual_nodes=" << n_virt << "\n";
    *oss << "root_real=" << n_root << "\n";
    *oss << "linked_ratio=" << ((n_real > 0) ? 1.0 - static_cast<double>(n_root) / static_cast<double>(n_real) : 0.0) << "\n";
    *oss << "weighted_mean_depth=" << ((n_real > 0) ? static_cast<double>(weighted_mean_depth / denom) : 0.0) << "\n";
    *oss << "cluster_mean_depth_p50=" << Quantile(mean_depths, 0.50) << "\n";
    *oss << "cluster_mean_depth_p90=" << Quantile(mean_depths, 0.90) << "\n";
    *oss << "cluster_max_depth_p90=" << Quantile(max_depths, 0.90) << "\n";
    *oss << "weighted_leaf_ratio=" << ((n_real > 0) ? static_cast<double>(weighted_leaf_ratio / denom) : 0.0) << "\n";
    *oss << "weighted_branching_ratio=" << ((n_real > 0) ? static_cast<double>(weighted_branching_ratio / denom) : 0.0) << "\n";
    *oss << "weighted_degree_gini=" << ((n_real > 0) ? static_cast<double>(weighted_degree_gini / denom) : 0.0) << "\n";
    *oss << "root_component_count=" << root_components << "\n";
    *oss << "root_component_mean=" << ((root_components > 0) ? static_cast<double>(n_real) / static_cast<double>(root_components) : 0.0) << "\n";
    *oss << "cluster_root_component_p90_p50=" << Quantile(root_component_p90s, 0.50) << "\n";
    *oss << "cluster_root_component_p90_p90=" << Quantile(root_component_p90s, 0.90) << "\n";
    *oss << "cluster_root_component_max_p90=" << Quantile(root_component_maxes, 0.90) << "\n";
    *oss << "cluster_max_subtree_p90=" << Quantile(max_subtrees, 0.90) << "\n";
    *oss << "virtual_root_component_count=" << virtual_root_components << "\n";
}

std::string FormatReport(const std::string& linkage_list_dir,
                         const stlq::io::LinkageListReader& reader,
                         const Stats& s,
                         int topk) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "[linkage_graph_quality] linkage_list_dir = " << linkage_list_dir << "\n";
    oss << "[linkage_graph_quality] nlist = " << reader.nlist() << "\n";
    oss << "[linkage_graph_quality] total_real = " << s.total_real << "\n";
    oss << "[linkage_graph_quality] total_virtual = " << s.total_virt << "\n";
    oss << "[linkage_graph_quality] nonempty_clusters = " << s.nonempty_clusters << "\n";
    oss << "[linkage_graph_quality] empty_clusters = " << s.empty_clusters << "\n";
    oss << "[linkage_graph_quality] invalid_clusters = " << s.invalid_clusters << "\n";
    oss << "[linkage_graph_quality] root_real = " << s.total_root_real << "\n";
    oss << "[linkage_graph_quality] linked_real = " << s.total_linked_real << "\n";
    oss << "[linkage_graph_quality] linked_ratio = "
        << ((s.total_real > 0) ? static_cast<double>(s.total_linked_real) / static_cast<double>(s.total_real) : 0.0) << "\n";
    oss << "[linkage_graph_quality] mean_cluster_depth = " << Mean(s.cluster_mean_depths) << "\n";
    oss << "[linkage_graph_quality] p50_cluster_depth = " << Quantile(s.cluster_mean_depths, 0.50) << "\n";
    oss << "[linkage_graph_quality] p90_cluster_depth = " << Quantile(s.cluster_mean_depths, 0.90) << "\n";
    oss << "[linkage_graph_quality] p99_cluster_depth = " << Quantile(s.cluster_mean_depths, 0.99) << "\n";
    oss << "[linkage_graph_quality] node_degree_mean = " << Mean(s.node_degrees) << "\n";
    oss << "[linkage_graph_quality] node_degree_p95 = " << Quantile(s.node_degrees, 0.95) << "\n";
    oss << "[linkage_graph_quality] node_degree_p99 = " << Quantile(s.node_degrees, 0.99) << "\n";
    oss << "[linkage_graph_quality] node_degree_gini_global = " << Gini(s.node_degrees) << "\n";
    oss << "[linkage_graph_quality] subtree_real_mean = " << Mean(s.node_subtrees) << "\n";
    oss << "[linkage_graph_quality] subtree_real_p95 = " << Quantile(s.node_subtrees, 0.95) << "\n";
    oss << "[linkage_graph_quality] subtree_real_p99 = " << Quantile(s.node_subtrees, 0.99) << "\n";
    oss << "[linkage_graph_quality] subtree_real_gini_global = " << Gini(s.node_subtrees) << "\n";
    oss << "[linkage_graph_quality] real_leaf_nodes = " << s.real_leaf_nodes << "\n";
    oss << "[linkage_graph_quality] real_single_child_nodes = " << s.real_single_child_nodes << "\n";
    oss << "[linkage_graph_quality] real_branching_nodes = " << s.real_branching_nodes << "\n";
    oss << "[linkage_graph_quality] all_leaf_nodes = " << s.all_leaf_nodes << "\n";
    oss << "[linkage_graph_quality] all_single_child_nodes = " << s.all_single_child_nodes << "\n";
    oss << "[linkage_graph_quality] all_branching_nodes = " << s.all_branching_nodes << "\n";
    oss << "[linkage_graph_quality] real_leaf_ratio = "
        << ((s.total_real > 0) ? static_cast<double>(s.real_leaf_nodes) / static_cast<double>(s.total_real) : 0.0) << "\n";
    oss << "[linkage_graph_quality] real_branching_ratio = "
        << ((s.total_real > 0) ? static_cast<double>(s.real_branching_nodes) / static_cast<double>(s.total_real) : 0.0) << "\n";
    oss << "[linkage_graph_quality] root_component_count = " << s.root_component_subtrees.size() << "\n";
    oss << "[linkage_graph_quality] root_component_mean = " << Mean(s.root_component_subtrees) << "\n";
    oss << "[linkage_graph_quality] root_component_p50 = " << Quantile(s.root_component_subtrees, 0.50) << "\n";
    oss << "[linkage_graph_quality] root_component_p90 = " << Quantile(s.root_component_subtrees, 0.90) << "\n";
    oss << "[linkage_graph_quality] root_component_p99 = " << Quantile(s.root_component_subtrees, 0.99) << "\n";
    oss << "[linkage_graph_quality] root_component_max = "
        << (s.root_component_subtrees.empty() ? 0.0 : *std::max_element(s.root_component_subtrees.begin(), s.root_component_subtrees.end())) << "\n";
    oss << "[linkage_graph_quality] real_root_component_count = " << s.real_root_component_subtrees.size() << "\n";
    oss << "[linkage_graph_quality] virtual_root_component_count = " << s.virtual_root_component_subtrees.size() << "\n";
    oss << "[linkage_graph_quality] virtual_root_component_mean = " << Mean(s.virtual_root_component_subtrees) << "\n";
    oss << "[linkage_graph_quality] virtual_root_component_p90 = " << Quantile(s.virtual_root_component_subtrees, 0.90) << "\n";

    PrintClusterGroupSummary(&oss, s.clusters, "cluster_group_no_virtual_nodes", false);
    PrintClusterGroupSummary(&oss, s.clusters, "cluster_group_with_virtual_nodes", true);

    const double parent_total = static_cast<double>(s.parent_super_root + s.parent_virtual +
                                                   s.parent_real_same_depth + s.parent_real_prev_depth +
                                                   s.parent_real_skip_depth + s.parent_real_deeper_or_invalid);
    auto parent_line = [&](const char* name, std::uint64_t v) {
        oss << name << " count=" << v << " ratio=" << ((parent_total > 0.0) ? static_cast<double>(v) / parent_total : 0.0) << "\n";
    };
    oss << "\n[parent_source]\n";
    parent_line("super_root", s.parent_super_root);
    parent_line("virtual", s.parent_virtual);
    parent_line("real_same_depth", s.parent_real_same_depth);
    parent_line("real_prev_depth", s.parent_real_prev_depth);
    parent_line("real_skip_depth", s.parent_real_skip_depth);
    parent_line("real_deeper_or_invalid", s.parent_real_deeper_or_invalid);

    oss << "\n[node_depth_hist]\n";
    for (std::size_t i = 0; i < s.depth_hist.size(); ++i) {
        if (s.depth_hist[i] == 0) continue;
        oss << "depth=" << i << " count=" << s.depth_hist[i]
            << " ratio=" << ((s.total_real > 0) ? static_cast<double>(s.depth_hist[i]) / static_cast<double>(s.total_real) : 0.0)
            << "\n";
    }

    oss << "\n[parent_depth_delta_hist]\n";
    for (std::size_t i = 0; i < s.parent_depth_delta_hist.size(); ++i) {
        if (s.parent_depth_delta_hist[i] == 0) continue;
        oss << "delta=" << i << " count=" << s.parent_depth_delta_hist[i] << "\n";
    }

    oss << "\n[depth_layer_quality]\n";
    oss << "depth,count,root_ratio,virtual_parent_ratio,real_prev_parent_ratio,other_parent_ratio,"
           "leaf_ratio,single_child_ratio,branching_ratio,mean_child_degree,mean_subtree_real\n";
    for (std::size_t i = 0; i < s.depth_aggs.size(); ++i) {
        const DepthAgg& d = s.depth_aggs[i];
        if (d.count == 0) continue;
        const double n = static_cast<double>(d.count);
        oss << i << ","
            << d.count << ","
            << static_cast<double>(d.parent_super_root) / n << ","
            << static_cast<double>(d.parent_virtual) / n << ","
            << static_cast<double>(d.parent_real_prev_depth) / n << ","
            << static_cast<double>(d.parent_other) / n << ","
            << static_cast<double>(d.leaf) / n << ","
            << static_cast<double>(d.single_child) / n << ","
            << static_cast<double>(d.branching) / n << ","
            << d.degree_sum / n << ","
            << d.subtree_sum / n << "\n";
    }

    oss << "\n[child_degree_hist_nonzero]\n";
    for (std::size_t i = 1; i < s.child_degree_hist.size(); ++i) {
        if (s.child_degree_hist[i] == 0) continue;
        oss << "degree=" << i << " nodes=" << s.child_degree_hist[i] << "\n";
    }

    PrintTopClusters(&oss, s.clusters, topk, "top_max_child_degree_clusters", "degree");
    PrintTopClusters(&oss, s.clusters, topk, "top_max_subtree_clusters", "subtree");
    PrintTopClusters(&oss, s.clusters, topk, "top_degree_gini_clusters", "gini");
    PrintTopClusters(&oss, s.clusters, topk, "top_mean_depth_clusters", "depth");
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::string linkage_list_dir = GetArg(argc, argv, "--linkage_list_dir", "");
    const std::string out_dir = GetArg(argc, argv, "--out_dir", "");
    const std::string report_path = GetArg(argc, argv, "--report", "");
    const std::string csv_dir = GetArg(argc, argv, "--csv_dir", "");
    const int topk = std::max(0, GetArgI32(argc, argv, "--topk", 10));

    if (linkage_list_dir.empty() && !out_dir.empty()) {
        const std::filesystem::path p(out_dir);
        linkage_list_dir = HasLinkageMeta(p) ? p.string() : (p / "linkage_list").string();
    }
    if (linkage_list_dir.empty()) {
        std::cerr << "Usage: report_linkage_graph_quality --linkage_list_dir <dir>\n"
                     "  OR: report_linkage_graph_quality --out_dir <run_root>\n"
                     "  [--topk 10] [--report <path>] [--csv_dir <dir>]\n";
        return 1;
    }

    stlq::io::LinkageListReader reader;
    std::string err;
    if (!reader.Open(linkage_list_dir, &err)) {
        std::cerr << err << "\n";
        return 1;
    }

    CsvWriters csv;
    if (!OpenCsvWriters(csv_dir, &csv, &err)) {
        std::cerr << err << "\n";
        return 1;
    }
    Stats stats;
    if (!Analyze(reader, &stats, &csv, &err)) {
        std::cerr << err << "\n";
        return 1;
    }
    const std::string report = FormatReport(linkage_list_dir, reader, stats, topk);
    std::cout << report;
    if (!report_path.empty()) {
        std::ofstream out(report_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::cerr << "report_linkage_graph_quality: failed to open report path: " << report_path << "\n";
            return 1;
        }
        out.write(report.data(), static_cast<std::streamsize>(report.size()));
        if (!out) {
            std::cerr << "report_linkage_graph_quality: failed to write report: " << report_path << "\n";
            return 1;
        }
        std::cout << "[linkage_graph_quality] wrote report to " << report_path << "\n";
    }
    if (csv.enabled) {
        std::cout << "[linkage_graph_quality] wrote csv files to " << csv_dir << "\n";
    }
    return 0;
}
