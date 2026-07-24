#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
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
        const std::string a = argv[i] ? argv[i] : "";
        if (a == key) {
            if (i + 1 < argc && argv[i + 1]) return argv[i + 1];
            return def;
        }
        if (a.rfind(prefix, 0) == 0) {
            return a.substr(prefix.size());
        }
    }
    return def;
}

int GetArgI32(int argc, char** argv, const std::string& key, int def) {
    const std::string v = GetArg(argc, argv, key, "");
    return v.empty() ? def : std::atoi(v.c_str());
}

double GetArgF64(int argc, char** argv, const std::string& key, double def) {
    const std::string v = GetArg(argc, argv, key, "");
    return v.empty() ? def : std::atof(v.c_str());
}

bool HasMagicMeta(const std::string& dir) {
    std::ifstream in((std::filesystem::path(dir) / "meta.bin").string(), std::ios::binary);
    return in.is_open();
}

struct ClusterDepthSummary {
    int cid = -1;
    std::uint32_t n_real = 0;
    std::uint32_t n_root = 0;
    int max_depth = 0;
    double mean_depth = 0.0;
};

struct ReportStats {
    std::uint64_t total_real = 0;
    std::uint64_t total_root = 0;
    std::uint64_t nonempty_clusters = 0;
    std::uint64_t empty_clusters = 0;
    std::uint64_t invalid_clusters = 0;
    int global_max_depth = 0;
    double global_mean_depth = 0.0;
    double mean_cluster_mean_depth = 0.0;
    double p50_cluster_mean_depth = 0.0;
    double p90_cluster_mean_depth = 0.0;
    double p99_cluster_mean_depth = 0.0;
    double max_cluster_mean_depth = 0.0;
    std::vector<std::uint64_t> node_depth_hist;
    std::vector<ClusterDepthSummary> deepest_clusters;
    std::vector<ClusterDepthSummary> highest_mean_clusters;
    std::vector<std::pair<double, std::uint64_t>> cluster_mean_hist;
};

double QuantileFromSorted(const std::vector<double>& xs, double q) {
    if (xs.empty()) return 0.0;
    const double qq = std::clamp(q, 0.0, 1.0);
    const double pos = qq * static_cast<double>(xs.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    if (lo == hi) return xs[lo];
    const double w = pos - static_cast<double>(lo);
    return xs[lo] * (1.0 - w) + xs[hi] * w;
}

bool ReadDepthOffsetsAt(std::ifstream* in,
                        std::uint64_t depth_lo,
                        std::uint32_t depth_len,
                        std::vector<std::uint32_t>* depth_offsets,
                        std::string* err) {
    if (!in || !depth_offsets) {
        if (err) *err = "ReadDepthOffsetsAt: null input.";
        return false;
    }
    depth_offsets->clear();
    if (depth_len == 0) return true;
    const std::uint64_t off = depth_lo * sizeof(std::uint32_t);
    in->seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (!(*in)) {
        if (err) *err = "ReadDepthOffsetsAt: seek failed.";
        return false;
    }
    depth_offsets->resize(depth_len);
    in->read(reinterpret_cast<char*>(depth_offsets->data()),
             static_cast<std::streamsize>(static_cast<std::size_t>(depth_len) * sizeof(std::uint32_t)));
    if (!(*in)) {
        if (err) *err = "ReadDepthOffsetsAt: read failed.";
        return false;
    }
    return true;
}

ReportStats AnalyzeLinkageDepth(const stlq::io::LinkageListReader& reader,
                              const std::string& linkage_list_dir,
                              double mean_bucket_width,
                              int topk,
                              std::string* err) {
    ReportStats out;
    const int nlist = reader.nlist();
    std::ifstream depth_in((std::filesystem::path(linkage_list_dir) / "depth_offsets.u32").string(), std::ios::binary);
    if (!depth_in.is_open()) {
        if (err) *err = "AnalyzeLinkageDepth: failed to open depth_offsets.u32.";
        return out;
    }

    std::vector<std::uint32_t> depth_offsets;
    std::vector<double> cluster_means;
    cluster_means.reserve(static_cast<std::size_t>(nlist));
    std::unordered_map<long long, std::uint64_t> mean_hist_counts;
    std::vector<ClusterDepthSummary> clusters;
    clusters.reserve(static_cast<std::size_t>(nlist));
    double depth_sum_total = 0.0;

    for (int cid = 0; cid < nlist; ++cid) {
        std::uint64_t real_lo = 0, real_hi = 0, virt_lo = 0, virt_hi = 0, depth_lo = 0, depth_hi = 0;
        if (!reader.ReadClusterSpan(cid, &real_lo, &real_hi, &virt_lo, &virt_hi, &depth_lo, &depth_hi, err)) {
            return out;
        }
        const std::uint64_t n_real64 = real_hi - real_lo;
        if (n_real64 == 0) {
            out.empty_clusters += 1;
            continue;
        }
        if (n_real64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
            if (err) *err = "AnalyzeLinkageDepth: cluster too large.";
            return out;
        }
        const std::uint32_t n_real = static_cast<std::uint32_t>(n_real64);
        const std::uint32_t depth_len = static_cast<std::uint32_t>(depth_hi - depth_lo);
        if (!ReadDepthOffsetsAt(&depth_in, depth_lo, depth_len, &depth_offsets, err)) {
            return out;
        }
        if (depth_offsets.size() < 2 || depth_offsets.back() != n_real) {
            out.invalid_clusters += 1;
            continue;
        }

        std::uint64_t cluster_depth_sum = 0;
        std::uint32_t n_root = depth_offsets[1] - depth_offsets[0];
        int max_depth = 0;
        if (out.node_depth_hist.size() < depth_offsets.size() - 1) {
            out.node_depth_hist.resize(depth_offsets.size() - 1, 0);
        }
        for (std::size_t dep = 0; dep + 1 < depth_offsets.size(); ++dep) {
            const std::uint32_t lo = depth_offsets[dep];
            const std::uint32_t hi = depth_offsets[dep + 1];
            if (hi < lo) {
                out.invalid_clusters += 1;
                cluster_depth_sum = 0;
                n_root = 0;
                max_depth = 0;
                break;
            }
            const std::uint32_t cnt = hi - lo;
            if (cnt == 0) continue;
            max_depth = static_cast<int>(dep);
            out.node_depth_hist[dep] += cnt;
            cluster_depth_sum += static_cast<std::uint64_t>(dep) * static_cast<std::uint64_t>(cnt);
        }
        if (n_root == 0 && n_real > 0 && depth_offsets[1] == depth_offsets[0]) {
            out.invalid_clusters += 1;
            continue;
        }

        const double mean_depth = static_cast<double>(cluster_depth_sum) / static_cast<double>(n_real);
        cluster_means.push_back(mean_depth);
        depth_sum_total += static_cast<double>(cluster_depth_sum);
        out.total_real += n_real;
        out.total_root += n_root;
        out.nonempty_clusters += 1;
        out.global_max_depth = std::max(out.global_max_depth, max_depth);

        ClusterDepthSummary s;
        s.cid = cid;
        s.n_real = n_real;
        s.n_root = n_root;
        s.max_depth = max_depth;
        s.mean_depth = mean_depth;
        clusters.push_back(s);

        if (mean_bucket_width > 0.0) {
            const long long bucket = static_cast<long long>(std::floor(mean_depth / mean_bucket_width));
            mean_hist_counts[bucket] += 1;
        }
    }

    if (out.total_real > 0) {
        out.global_mean_depth = depth_sum_total / static_cast<double>(out.total_real);
    }
    if (!cluster_means.empty()) {
        std::sort(cluster_means.begin(), cluster_means.end());
        out.mean_cluster_mean_depth =
            std::accumulate(cluster_means.begin(), cluster_means.end(), 0.0) / static_cast<double>(cluster_means.size());
        out.p50_cluster_mean_depth = QuantileFromSorted(cluster_means, 0.50);
        out.p90_cluster_mean_depth = QuantileFromSorted(cluster_means, 0.90);
        out.p99_cluster_mean_depth = QuantileFromSorted(cluster_means, 0.99);
        out.max_cluster_mean_depth = cluster_means.back();
    }

    if (mean_bucket_width > 0.0) {
        out.cluster_mean_hist.reserve(mean_hist_counts.size());
        for (const auto& kv : mean_hist_counts) {
            out.cluster_mean_hist.push_back({static_cast<double>(kv.first) * mean_bucket_width, kv.second});
        }
        std::sort(out.cluster_mean_hist.begin(), out.cluster_mean_hist.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    std::sort(clusters.begin(), clusters.end(),
              [](const ClusterDepthSummary& a, const ClusterDepthSummary& b) {
                  if (a.max_depth != b.max_depth) return a.max_depth > b.max_depth;
                  if (a.mean_depth != b.mean_depth) return a.mean_depth > b.mean_depth;
                  return a.cid < b.cid;
              });
    if (topk > 0 && static_cast<int>(clusters.size()) > topk) clusters.resize(static_cast<std::size_t>(topk));
    out.deepest_clusters = clusters;

    std::sort(clusters.begin(), clusters.end(),
              [](const ClusterDepthSummary& a, const ClusterDepthSummary& b) {
                  if (a.mean_depth != b.mean_depth) return a.mean_depth > b.mean_depth;
                  if (a.max_depth != b.max_depth) return a.max_depth > b.max_depth;
                  return a.cid < b.cid;
              });
    if (topk > 0 && static_cast<int>(clusters.size()) > topk) clusters.resize(static_cast<std::size_t>(topk));
    out.highest_mean_clusters = clusters;

    return out;
}

std::string FormatReport(const std::string& linkage_list_dir,
                         const stlq::io::LinkageListReader& reader,
                         const ReportStats& stats,
                         double mean_bucket_width,
                         int topk) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "[linkage_depth] linkage_list_dir = " << linkage_list_dir << "\n";
    oss << "[linkage_depth] nlist = " << reader.nlist() << "\n";
    oss << "[linkage_depth] total_real = " << stats.total_real << "\n";
    oss << "[linkage_depth] nonempty_clusters = " << stats.nonempty_clusters << "\n";
    oss << "[linkage_depth] empty_clusters = " << stats.empty_clusters << "\n";
    oss << "[linkage_depth] invalid_clusters = " << stats.invalid_clusters << "\n";
    oss << "[linkage_depth] total_root_real = " << stats.total_root << "\n";
    oss << "[linkage_depth] global_mean_depth = " << stats.global_mean_depth << "\n";
    oss << "[linkage_depth] global_max_depth = " << stats.global_max_depth << "\n";
    oss << "[linkage_depth] mean_cluster_mean_depth = " << stats.mean_cluster_mean_depth << "\n";
    oss << "[linkage_depth] p50_cluster_mean_depth = " << stats.p50_cluster_mean_depth << "\n";
    oss << "[linkage_depth] p90_cluster_mean_depth = " << stats.p90_cluster_mean_depth << "\n";
    oss << "[linkage_depth] p99_cluster_mean_depth = " << stats.p99_cluster_mean_depth << "\n";
    oss << "[linkage_depth] max_cluster_mean_depth = " << stats.max_cluster_mean_depth << "\n";

    oss << "\n[node_depth_hist]\n";
    for (std::size_t dep = 0; dep < stats.node_depth_hist.size(); ++dep) {
        const std::uint64_t cnt = stats.node_depth_hist[dep];
        if (cnt == 0) continue;
        const double frac = (stats.total_real > 0)
            ? static_cast<double>(cnt) / static_cast<double>(stats.total_real)
            : 0.0;
        oss << "depth=" << dep
            << " count=" << cnt
            << " ratio=" << frac
            << "\n";
    }

    if (!stats.cluster_mean_hist.empty()) {
        oss << "\n[cluster_mean_depth_hist]\n";
        oss << "bucket_width=" << mean_bucket_width << "\n";
        for (const auto& kv : stats.cluster_mean_hist) {
            const double lo = kv.first;
            const double hi = lo + mean_bucket_width;
            const double frac = (stats.nonempty_clusters > 0)
                ? static_cast<double>(kv.second) / static_cast<double>(stats.nonempty_clusters)
                : 0.0;
            oss << "[" << lo << ", " << hi << ")"
                << " clusters=" << kv.second
                << " ratio=" << frac
                << "\n";
        }
    }

    if (topk > 0 && !stats.deepest_clusters.empty()) {
        oss << "\n[top_max_depth_clusters]\n";
        for (const auto& s : stats.deepest_clusters) {
            oss << "cid=" << s.cid
                << " n_real=" << s.n_real
                << " n_root=" << s.n_root
                << " max_depth=" << s.max_depth
                << " mean_depth=" << s.mean_depth
                << "\n";
        }
    }
    if (topk > 0 && !stats.highest_mean_clusters.empty()) {
        oss << "\n[top_mean_depth_clusters]\n";
        for (const auto& s : stats.highest_mean_clusters) {
            oss << "cid=" << s.cid
                << " n_real=" << s.n_real
                << " n_root=" << s.n_root
                << " max_depth=" << s.max_depth
                << " mean_depth=" << s.mean_depth
                << "\n";
        }
    }

    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::string linkage_list_dir = GetArg(argc, argv, "--linkage_list_dir", "");
    const std::string out_dir = GetArg(argc, argv, "--out_dir", "");
    const std::string report_path = GetArg(argc, argv, "--report", "");
    const double mean_bucket_width = std::max(0.0, GetArgF64(argc, argv, "--mean_bucket_width", 0.25));
    const int topk = std::max(0, GetArgI32(argc, argv, "--topk", 10));

    if (linkage_list_dir.empty() && !out_dir.empty()) {
        if (HasMagicMeta(out_dir)) {
            linkage_list_dir = out_dir;
        } else {
            linkage_list_dir = (std::filesystem::path(out_dir) / "linkage_list").string();
        }
    }

    if (linkage_list_dir.empty()) {
        std::cerr << "Usage: report_linkage_depth --linkage_list_dir <dir>\n"
                     "  OR: report_linkage_depth --out_dir <dir>\n"
                     "  [--mean_bucket_width 0.25] [--topk 10] [--report <path>]\n";
        return 1;
    }

    stlq::io::LinkageListReader reader;
    std::string err;
    if (!reader.Open(linkage_list_dir, &err)) {
        std::cerr << err << "\n";
        return 1;
    }

    ReportStats stats = AnalyzeLinkageDepth(reader, linkage_list_dir, mean_bucket_width, topk, &err);
    if (!err.empty()) {
        std::cerr << err << "\n";
        return 1;
    }

    const std::string report = FormatReport(linkage_list_dir, reader, stats, mean_bucket_width, topk);
    std::cout << report;

    if (!report_path.empty()) {
        std::ofstream out(report_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            std::cerr << "report_linkage_depth: failed to open report path: " << report_path << "\n";
            return 1;
        }
        out.write(report.data(), static_cast<std::streamsize>(report.size()));
        if (!out) {
            std::cerr << "report_linkage_depth: failed to write report: " << report_path << "\n";
            return 1;
        }
        std::cout << "[linkage_depth] wrote report to " << report_path << "\n";
    }
    return 0;
}
