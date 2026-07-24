#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Args {
    std::string csv;
    std::string report;
};

std::string GetArg(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i] ? argv[i] : "");
        if (a == key && i + 1 < argc && argv[i + 1]) return argv[i + 1];
        const std::string prefix = key + "=";
        if (a.rfind(prefix, 0) == 0) return a.substr(prefix.size());
    }
    return def;
}

bool ParseArgs(int argc, char** argv, Args* out) {
    if (!out) return false;
    out->csv = GetArg(argc, argv, "--csv", "");
    out->report = GetArg(argc, argv, "--report", "");
    if (out->csv.empty()) return false;
    if (out->report.empty()) {
        const std::filesystem::path p(out->csv);
        out->report = (p.parent_path() / (p.stem().string() + "_summary.txt")).string();
    }
    return true;
}

std::vector<std::string> SplitComma(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, ',')) out.push_back(cur);
    if (!s.empty() && s.back() == ',') out.emplace_back();
    return out;
}

int ToInt(const std::vector<std::string>& row,
          const std::unordered_map<std::string, int>& h,
          const std::string& key,
          int def = 0) {
    auto it = h.find(key);
    if (it == h.end() || it->second < 0 || it->second >= static_cast<int>(row.size())) return def;
    return std::atoi(row[static_cast<std::size_t>(it->second)].c_str());
}

double ToDouble(const std::vector<std::string>& row,
                const std::unordered_map<std::string, int>& h,
                const std::string& key,
                double def = 0.0) {
    auto it = h.find(key);
    if (it == h.end() || it->second < 0 || it->second >= static_cast<int>(row.size())) return def;
    return std::atof(row[static_cast<std::size_t>(it->second)].c_str());
}

bool HasField(const std::unordered_map<std::string, int>& h, const std::string& key) {
    return h.find(key) != h.end();
}

std::string ToString(const std::vector<std::string>& row,
                     const std::unordered_map<std::string, int>& h,
                     const std::string& key) {
    auto it = h.find(key);
    if (it == h.end() || it->second < 0 || it->second >= static_cast<int>(row.size())) return {};
    return row[static_cast<std::size_t>(it->second)];
}

struct NumStats {
    std::uint64_t n = 0;
    double sum = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    std::vector<double> sample;

    void Add(double v) {
        if (!std::isfinite(v)) return;
        ++n;
        sum += v;
        min = std::min(min, v);
        max = std::max(max, v);
        sample.push_back(v);
    }

    double Mean() const { return n ? sum / static_cast<double>(n) : 0.0; }

    double Quantile(double q) {
        if (sample.empty()) return 0.0;
        std::sort(sample.begin(), sample.end());
        const double pos = std::clamp(q, 0.0, 1.0) * static_cast<double>(sample.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
        const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
        if (lo == hi) return sample[lo];
        const double w = pos - static_cast<double>(lo);
        return sample[lo] * (1.0 - w) + sample[hi] * w;
    }
};

struct FeatureGroup {
    std::uint64_t n = 0;
    std::map<std::string, NumStats> nums;
    std::map<std::string, std::uint64_t> parent_kind;

    void Add(const std::vector<std::string>& row, const std::unordered_map<std::string, int>& h) {
        ++n;
        static const char* kFields[] = {
            "depth",
            "child_degree",
            "subtree_real",
            "cluster_n_real",
            "cluster_n_virtual",
            "cluster_linked_ratio",
            "cluster_mean_depth",
            "cluster_max_depth",
            "cluster_max_child_degree",
            "cluster_max_subtree_real",
            "cluster_degree_gini",
            "adc_minus_exact",
            "query_norm2",
            "raw_norm2",
            "center_norm2",
            "recon_norm2",
            "q_dot_raw",
            "q_dot_center",
            "q_dot_recon",
            "raw_dot_center",
            "raw_dot_recon",
            "q_center_distance",
            "raw_center_distance",
            "recon_distance",
            "recon_minus_exact",
            "residual_norm2",
            "exact_rank_in_shortlist",
        };
        for (const char* f : kFields) {
            if (HasField(h, f)) nums[f].Add(ToDouble(row, h, f));
        }
        if (HasField(h, "parent_kind")) parent_kind[ToString(row, h, "parent_kind")] += 1;
    }
};

struct ClusterAgg {
    int cid = -1;
    std::uint64_t gt_seen = 0;
    std::uint64_t gt_rank1 = 0;
    std::uint64_t gt_rank_late = 0;
    NumStats gt_adc_rank;
    NumStats gt_recon_minus_exact;
    NumStats gt_residual_norm2;
    NumStats gt_exact_rank;
    double cluster_mean_depth = 0.0;
    double cluster_degree_gini = 0.0;
    double cluster_linked_ratio = 0.0;
    int cluster_n_real = 0;
    int cluster_n_virtual = 0;
};

struct BucketStats {
    std::uint64_t n = 0;
    NumStats exact_rank;
    NumStats adc_minus_exact;
    NumStats recon_minus_exact;
    NumStats residual_norm2;
};

struct QueryPairStats {
    bool has_adc_top1 = false;
    bool has_exact_top1 = false;
    int q = -1;
    int adc_top1_exact_rank = -1;
    double adc_top1_adc = std::numeric_limits<double>::quiet_NaN();
    double adc_top1_exact = std::numeric_limits<double>::quiet_NaN();
    double adc_top1_recon = std::numeric_limits<double>::quiet_NaN();
    double adc_top1_recon_minus_exact = std::numeric_limits<double>::quiet_NaN();
    double adc_top1_residual = std::numeric_limits<double>::quiet_NaN();
    double adc_top1_q_center = std::numeric_limits<double>::quiet_NaN();
    double adc_top1_raw_center = std::numeric_limits<double>::quiet_NaN();
    double exact_top1_adc = std::numeric_limits<double>::quiet_NaN();
    double exact_top1_exact = std::numeric_limits<double>::quiet_NaN();
    double exact_top1_recon = std::numeric_limits<double>::quiet_NaN();
    double exact_top1_recon_minus_exact = std::numeric_limits<double>::quiet_NaN();
    double exact_top1_residual = std::numeric_limits<double>::quiet_NaN();
    double exact_top1_q_center = std::numeric_limits<double>::quiet_NaN();
    double exact_top1_raw_center = std::numeric_limits<double>::quiet_NaN();
};

struct PairGapStats {
    std::uint64_t n = 0;
    NumStats adc_gap;
    NumStats exact_gap;
    NumStats recon_gap;
    NumStats recon_minus_exact_gap;
    NumStats residual_gap;
    NumStats q_center_gap;
    NumStats raw_center_gap;
    NumStats adc_top1_exact_rank;
};

std::string DepthBucket(int depth) {
    if (depth < 0) return "missing";
    if (depth == 0) return "0";
    if (depth == 1) return "1";
    if (depth == 2) return "2";
    if (depth <= 4) return "3-4";
    if (depth <= 8) return "5-8";
    return "9+";
}

std::string VirtualRatioBucket(double n_virt, double n_real) {
    if (n_real <= 0.0) return "missing";
    const double r = n_virt / n_real;
    if (r <= 0.0) return "0";
    if (r < 0.05) return "(0,0.05)";
    if (r < 0.15) return "[0.05,0.15)";
    return ">=0.15";
}

std::string ValueBucket(double v, const std::vector<double>& bounds) {
    if (!std::isfinite(v)) return "missing";
    double lo = -std::numeric_limits<double>::infinity();
    for (double hi : bounds) {
        if (v < hi) {
            std::ostringstream os;
            os << "[" << lo << "," << hi << ")";
            return os.str();
        }
        lo = hi;
    }
    std::ostringstream os;
    os << "[" << lo << ",inf)";
    return os.str();
}

void AddBucket(std::map<std::string, BucketStats>* buckets,
               const std::string& key,
               int exact_rank,
               double adc_minus_exact,
               double recon_minus_exact,
               double residual_norm2) {
    BucketStats& b = (*buckets)[key];
    ++b.n;
    b.exact_rank.Add(static_cast<double>(exact_rank));
    b.adc_minus_exact.Add(adc_minus_exact);
    b.recon_minus_exact.Add(recon_minus_exact);
    b.residual_norm2.Add(residual_norm2);
}

void AddPairGap(PairGapStats* out, const QueryPairStats& q) {
    if (!out || !q.has_adc_top1 || !q.has_exact_top1 || q.adc_top1_exact_rank <= 1) return;
    ++out->n;
    out->adc_gap.Add(q.exact_top1_adc - q.adc_top1_adc);
    out->exact_gap.Add(q.adc_top1_exact - q.exact_top1_exact);
    out->recon_gap.Add(q.adc_top1_recon - q.exact_top1_recon);
    out->recon_minus_exact_gap.Add(q.adc_top1_recon_minus_exact - q.exact_top1_recon_minus_exact);
    out->residual_gap.Add(q.adc_top1_residual - q.exact_top1_residual);
    out->q_center_gap.Add(q.adc_top1_q_center - q.exact_top1_q_center);
    out->raw_center_gap.Add(q.adc_top1_raw_center - q.exact_top1_raw_center);
    out->adc_top1_exact_rank.Add(static_cast<double>(q.adc_top1_exact_rank));
}

void PrintNumStats(std::ostream& os, const std::string& name, NumStats& s) {
    os << name
       << " n=" << s.n
       << " mean=" << s.Mean()
       << " p50=" << s.Quantile(0.50)
       << " p90=" << s.Quantile(0.90)
       << " min=" << (s.n ? s.min : 0.0)
       << " max=" << (s.n ? s.max : 0.0)
       << "\n";
}

void PrintBuckets(std::ostream& os, const std::string& name, std::map<std::string, BucketStats>& buckets) {
    os << "\n[" << name << "]\n";
    for (auto& kv : buckets) {
        BucketStats& b = kv.second;
        os << "bucket=" << kv.first
           << " n=" << b.n
           << " exact_rank_mean=" << b.exact_rank.Mean()
           << " exact_rank_p50=" << b.exact_rank.Quantile(0.50)
           << " exact_rank_p90=" << b.exact_rank.Quantile(0.90)
           << " adc_minus_exact_mean=" << b.adc_minus_exact.Mean()
           << " recon_minus_exact_mean=" << b.recon_minus_exact.Mean()
           << " residual_norm2_mean=" << b.residual_norm2.Mean()
           << "\n";
    }
}

void PrintPairGap(std::ostream& os, const std::string& name, PairGapStats& s) {
    os << "\n[" << name << "]\n";
    os << "n=" << s.n << "\n";
    PrintNumStats(os, "adc_gap_exactTop1_minus_adcTop1", s.adc_gap);
    PrintNumStats(os, "exact_gap_adcTop1_minus_exactTop1", s.exact_gap);
    PrintNumStats(os, "recon_gap_adcTop1_minus_exactTop1", s.recon_gap);
    PrintNumStats(os, "recon_minus_exact_gap_adcTop1_minus_exactTop1", s.recon_minus_exact_gap);
    PrintNumStats(os, "residual_gap_adcTop1_minus_exactTop1", s.residual_gap);
    PrintNumStats(os, "q_center_distance_gap_adcTop1_minus_exactTop1", s.q_center_gap);
    PrintNumStats(os, "raw_center_distance_gap_adcTop1_minus_exactTop1", s.raw_center_gap);
    PrintNumStats(os, "adc_top1_exact_rank", s.adc_top1_exact_rank);
}

void PrintClusterList(std::ostream& os,
                      const std::string& name,
                      std::vector<ClusterAgg> cluster_vec,
                      bool sort_by_late_count,
                      bool has_cluster_graph_fields) {
    std::sort(cluster_vec.begin(), cluster_vec.end(), [sort_by_late_count](const ClusterAgg& a, const ClusterAgg& b) {
        if (sort_by_late_count) {
            if (a.gt_rank_late != b.gt_rank_late) return a.gt_rank_late > b.gt_rank_late;
            if (a.gt_seen != b.gt_seen) return a.gt_seen > b.gt_seen;
        } else {
            const double ar = a.gt_seen ? static_cast<double>(a.gt_rank_late) / static_cast<double>(a.gt_seen) : 0.0;
            const double br = b.gt_seen ? static_cast<double>(b.gt_rank_late) / static_cast<double>(b.gt_seen) : 0.0;
            if (ar != br) return ar > br;
            if (a.gt_seen != b.gt_seen) return a.gt_seen > b.gt_seen;
        }
        return a.cid < b.cid;
    });

    os << "\n[" << name << "]\n";
    int printed = 0;
    for (auto& c : cluster_vec) {
        if (c.gt_seen < 20) continue;
        const double late_ratio = static_cast<double>(c.gt_rank_late) / static_cast<double>(c.gt_seen);
        os << "cid=" << c.cid
           << " gt_seen=" << c.gt_seen
           << " gt_rank1=" << c.gt_rank1
           << " gt_late=" << c.gt_rank_late
           << " late_ratio=" << late_ratio
           << " gt_adc_rank_mean=" << c.gt_adc_rank.Mean()
           << " gt_adc_rank_p50=" << c.gt_adc_rank.Quantile(0.50)
           << " gt_adc_rank_p90=" << c.gt_adc_rank.Quantile(0.90)
           << " gt_recon_minus_exact_mean=" << c.gt_recon_minus_exact.Mean()
           << " gt_residual_norm2_mean=" << c.gt_residual_norm2.Mean();
        if (has_cluster_graph_fields) {
            os << " n_real=" << c.cluster_n_real
               << " n_virtual=" << c.cluster_n_virtual
               << " linked_ratio=" << c.cluster_linked_ratio
               << " mean_depth=" << c.cluster_mean_depth
               << " degree_gini=" << c.cluster_degree_gini;
        }
        os << "\n";
        if (++printed >= 20) break;
    }
}

void PrintGroup(std::ostream& os, const std::string& name, FeatureGroup& g) {
    os << "\n[" << name << "]\n";
    os << "n=" << g.n << "\n";
    for (auto& kv : g.nums) PrintNumStats(os, kv.first, kv.second);
    if (!g.parent_kind.empty()) {
        os << "parent_kind:";
        for (const auto& kv : g.parent_kind) os << " " << kv.first << "=" << kv.second;
        os << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!ParseArgs(argc, argv, &args)) {
        std::cerr << "Usage: report_stlq_candidate_feature_summary --csv <candidate_features.csv> [--report <txt>]\n";
        return 1;
    }

    std::ifstream in(args.csv);
    if (!in) {
        std::cerr << "failed to open CSV: " << args.csv << "\n";
        return 1;
    }

    std::string header_line;
    if (!std::getline(in, header_line)) {
        std::cerr << "empty CSV: " << args.csv << "\n";
        return 1;
    }
    const auto header = SplitComma(header_line);
    std::unordered_map<std::string, int> h;
    for (int i = 0; i < static_cast<int>(header.size()); ++i) h[header[static_cast<std::size_t>(i)]] = i;
    const bool has_cluster_graph_fields =
        HasField(h, "cluster_mean_depth") &&
        HasField(h, "cluster_degree_gini") &&
        HasField(h, "cluster_linked_ratio") &&
        HasField(h, "cluster_n_real") &&
        HasField(h, "cluster_n_virtual");

    std::uint64_t rows = 0;
    std::uint64_t nq = 0;
    int current_q = -1;
    int max_rank_seen = 0;
    std::map<int, std::uint64_t> gt_rank_hist;
    std::map<int, std::uint64_t> exact_rank_hist_for_adc_top1;
    std::map<int, std::uint64_t> adc_rank_hist_for_exact_top1;
    FeatureGroup gt_rank1;
    FeatureGroup gt_late;
    FeatureGroup adc_top1;
    FeatureGroup exact_top1;
    FeatureGroup inversion_rows;
    std::unordered_map<int, ClusterAgg> clusters;
    NumStats adc_minus_exact_all;
    NumStats adc_minus_exact_adc_top1;
    NumStats adc_minus_exact_exact_top1;
    std::map<std::string, BucketStats> wrong_top1_by_parent_kind;
    std::map<std::string, BucketStats> wrong_top1_by_depth;
    std::map<std::string, BucketStats> wrong_top1_by_virtual_ratio;
    std::map<std::string, BucketStats> wrong_top1_by_recon_bias;
    std::map<std::string, BucketStats> wrong_top1_by_residual_norm2;
    std::vector<QueryPairStats> query_pairs;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto row = SplitComma(line);
        const int q = ToInt(row, h, "query_id", -1);
        const int rank = ToInt(row, h, "rank", -1);
        const int exact_rank = ToInt(row, h, "exact_rank_in_shortlist", -1);
        const int is_gt = ToInt(row, h, "is_gt1", 0);
        const int gt_rank = ToInt(row, h, "gt1_adc_rank", -1);
        const int cid = ToInt(row, h, "cid", -1);
        const double adc_minus_exact = ToDouble(row, h, "adc_minus_exact");
        const double adc_distance = ToDouble(row, h, "adc_distance", std::numeric_limits<double>::quiet_NaN());
        const double exact_distance = ToDouble(row, h, "exact_distance", std::numeric_limits<double>::quiet_NaN());
        const double recon_distance = ToDouble(row, h, "recon_distance", std::numeric_limits<double>::quiet_NaN());
        const double recon_minus_exact = ToDouble(row, h, "recon_minus_exact", std::numeric_limits<double>::quiet_NaN());
        const double residual_norm2 = ToDouble(row, h, "residual_norm2", std::numeric_limits<double>::quiet_NaN());
        const double q_center_distance = ToDouble(row, h, "q_center_distance", std::numeric_limits<double>::quiet_NaN());
        const double raw_center_distance = ToDouble(row, h, "raw_center_distance", std::numeric_limits<double>::quiet_NaN());

        ++rows;
        max_rank_seen = std::max(max_rank_seen, rank);
        adc_minus_exact_all.Add(adc_minus_exact);
        if (q != current_q) {
            current_q = q;
            ++nq;
            gt_rank_hist[gt_rank] += 1;
        }
        if (q >= 0) {
            if (q >= static_cast<int>(query_pairs.size())) query_pairs.resize(static_cast<std::size_t>(q + 1));
            QueryPairStats& qp = query_pairs[static_cast<std::size_t>(q)];
            qp.q = q;
            if (rank == 1) {
                qp.has_adc_top1 = true;
                qp.adc_top1_exact_rank = exact_rank;
                qp.adc_top1_adc = adc_distance;
                qp.adc_top1_exact = exact_distance;
                qp.adc_top1_recon = recon_distance;
                qp.adc_top1_recon_minus_exact = recon_minus_exact;
                qp.adc_top1_residual = residual_norm2;
                qp.adc_top1_q_center = q_center_distance;
                qp.adc_top1_raw_center = raw_center_distance;
            }
            if (exact_rank == 1) {
                qp.has_exact_top1 = true;
                qp.exact_top1_adc = adc_distance;
                qp.exact_top1_exact = exact_distance;
                qp.exact_top1_recon = recon_distance;
                qp.exact_top1_recon_minus_exact = recon_minus_exact;
                qp.exact_top1_residual = residual_norm2;
                qp.exact_top1_q_center = q_center_distance;
                qp.exact_top1_raw_center = raw_center_distance;
            }
        }

        if (rank == 1) {
            adc_top1.Add(row, h);
            exact_rank_hist_for_adc_top1[exact_rank] += 1;
            adc_minus_exact_adc_top1.Add(adc_minus_exact);
        }
        if (exact_rank == 1) {
            exact_top1.Add(row, h);
            adc_rank_hist_for_exact_top1[rank] += 1;
            adc_minus_exact_exact_top1.Add(adc_minus_exact);
        }
        if (rank == 1 && exact_rank > 1) {
            inversion_rows.Add(row, h);
            AddBucket(&wrong_top1_by_parent_kind,
                      HasField(h, "parent_kind") ? ToString(row, h, "parent_kind") : "missing",
                      exact_rank,
                      adc_minus_exact,
                      recon_minus_exact,
                      residual_norm2);
            AddBucket(&wrong_top1_by_depth,
                      DepthBucket(ToInt(row, h, "depth", -1)),
                      exact_rank,
                      adc_minus_exact,
                      recon_minus_exact,
                      residual_norm2);
            AddBucket(&wrong_top1_by_virtual_ratio,
                      VirtualRatioBucket(ToDouble(row, h, "cluster_n_virtual", 0.0),
                                         ToDouble(row, h, "cluster_n_real", 0.0)),
                      exact_rank,
                      adc_minus_exact,
                      recon_minus_exact,
                      residual_norm2);
            AddBucket(&wrong_top1_by_recon_bias,
                      ValueBucket(recon_minus_exact, {-20000.0, -10000.0, -5000.0, 0.0, 5000.0, 10000.0}),
                      exact_rank,
                      adc_minus_exact,
                      recon_minus_exact,
                      residual_norm2);
            AddBucket(&wrong_top1_by_residual_norm2,
                      ValueBucket(residual_norm2, {5000.0, 10000.0, 15000.0, 20000.0, 30000.0}),
                      exact_rank,
                      adc_minus_exact,
                      recon_minus_exact,
                      residual_norm2);
        }

        if (is_gt) {
            if (gt_rank == 1) gt_rank1.Add(row, h);
            else if (gt_rank > 1) gt_late.Add(row, h);

            ClusterAgg& c = clusters[cid];
            c.cid = cid;
            c.gt_seen += 1;
            if (gt_rank == 1) c.gt_rank1 += 1;
            else if (gt_rank > 1) c.gt_rank_late += 1;
            c.gt_adc_rank.Add(static_cast<double>(gt_rank));
            c.gt_recon_minus_exact.Add(recon_minus_exact);
            c.gt_residual_norm2.Add(residual_norm2);
            c.gt_exact_rank.Add(static_cast<double>(exact_rank));
            if (has_cluster_graph_fields) {
                c.cluster_mean_depth = ToDouble(row, h, "cluster_mean_depth");
                c.cluster_degree_gini = ToDouble(row, h, "cluster_degree_gini");
                c.cluster_linked_ratio = ToDouble(row, h, "cluster_linked_ratio");
                c.cluster_n_real = ToInt(row, h, "cluster_n_real");
                c.cluster_n_virtual = ToInt(row, h, "cluster_n_virtual");
            }
        }
    }

    std::ostringstream report;
    report << std::fixed << std::setprecision(6);
    report << "[candidate_feature_summary]\n";
    report << "csv=" << args.csv << "\n";
    report << "rows=" << rows << "\n";
    report << "nq=" << nq << "\n";
    report << "candidate_k=" << max_rank_seen << "\n";

    auto recall_at = [&](int k) {
        std::uint64_t hit = 0;
        for (const auto& kv : gt_rank_hist) {
            if (kv.first > 0 && kv.first <= k) hit += kv.second;
        }
        return nq ? static_cast<double>(hit) / static_cast<double>(nq) : 0.0;
    };
    report << "\n[recall_from_gt_rank]\n";
    for (int k : {1, 2, 5, 10, 20, 50, 100}) {
        if (k <= max_rank_seen) report << "r@" << k << "=" << recall_at(k) << "\n";
    }
    report << "missing_gt_in_shortlist=" << gt_rank_hist[-1] << "\n";

    report << "\n[gt_rank_hist]\n";
    for (const auto& kv : gt_rank_hist) {
        if (kv.first < 0 || kv.first > max_rank_seen) continue;
        report << "rank=" << kv.first << " count=" << kv.second
               << " ratio=" << (nq ? static_cast<double>(kv.second) / static_cast<double>(nq) : 0.0)
               << "\n";
    }

    report << "\n[adc_rank_of_exact_top1]\n";
    for (const auto& kv : adc_rank_hist_for_exact_top1) {
        report << "adc_rank=" << kv.first << " count=" << kv.second << "\n";
    }

    report << "\n[exact_rank_of_adc_top1]\n";
    for (const auto& kv : exact_rank_hist_for_adc_top1) {
        report << "exact_rank=" << kv.first << " count=" << kv.second << "\n";
    }

    report << "\n[adc_minus_exact]\n";
    PrintNumStats(report, "all_candidates", adc_minus_exact_all);
    PrintNumStats(report, "adc_top1", adc_minus_exact_adc_top1);
    PrintNumStats(report, "exact_top1", adc_minus_exact_exact_top1);

    PrintGroup(report, "gt_candidate_rank1", gt_rank1);
    PrintGroup(report, "gt_candidate_rank2_to_k", gt_late);
    PrintGroup(report, "adc_top1_candidate", adc_top1);
    PrintGroup(report, "exact_top1_candidate", exact_top1);
    PrintGroup(report, "adc_top1_but_not_exact_top1", inversion_rows);

    PrintBuckets(report, "wrong_adc_top1_by_parent_kind", wrong_top1_by_parent_kind);
    PrintBuckets(report, "wrong_adc_top1_by_depth", wrong_top1_by_depth);
    PrintBuckets(report, "wrong_adc_top1_by_virtual_ratio", wrong_top1_by_virtual_ratio);
    PrintBuckets(report, "wrong_adc_top1_by_recon_minus_exact", wrong_top1_by_recon_bias);
    PrintBuckets(report, "wrong_adc_top1_by_residual_norm2", wrong_top1_by_residual_norm2);

    PairGapStats pair_gap;
    for (const auto& qp : query_pairs) AddPairGap(&pair_gap, qp);
    PrintPairGap(report, "wrong_query_pair_gap_adcTop1_vs_rawExactTop1", pair_gap);

    std::vector<ClusterAgg> cluster_vec;
    cluster_vec.reserve(clusters.size());
    for (const auto& kv : clusters) cluster_vec.push_back(kv.second);
    PrintClusterList(report, "worst_clusters_by_late_ratio_min20", cluster_vec, false, has_cluster_graph_fields);
    PrintClusterList(report, "worst_clusters_by_late_count_min20", cluster_vec, true, has_cluster_graph_fields);

    const std::filesystem::path report_path(args.report);
    if (report_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(report_path.parent_path(), ec);
    }
    std::ofstream out(report_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "failed to open report: " << report_path << "\n";
        return 1;
    }
    const std::string s = report.str();
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
    if (!out) {
        std::cerr << "failed to write report: " << report_path << "\n";
        return 1;
    }
    std::cout << s;
    std::cout << "[candidate_feature_summary] wrote report to " << report_path.string() << "\n";
    return 0;
}
