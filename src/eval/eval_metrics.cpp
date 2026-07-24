#include "stlq/eval/eval_metrics.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace stlq {

namespace {

constexpr int kReportPoints[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 35, 50, 100, 150, 200, 500, 1000, 2000, 5000, 10000
};

bool ContainsId(const int* ids, int len, int target) {
    for (int i = 0; i < len; ++i) {
        if (ids[i] == target) return true;
    }
    return false;
}

bool ContainsId(const std::vector<int>& ids, int target) {
    return std::find(ids.begin(), ids.end(), target) != ids.end();
}

std::vector<float> ComputeRecallCurveTop1(const std::vector<int>& ground_truth,
                                          const ColMajorMatrix<int>& indices) {
    const int k = indices.rows;
    const int nquery = indices.cols;
    if (k <= 0 || nquery <= 0 || ground_truth.size() != static_cast<std::size_t>(nquery)) {
        return {};
    }
    std::vector<int> ranks(static_cast<std::size_t>(nquery), k + 1);
    for (int q = 0; q < nquery; ++q) {
        const int g = ground_truth[static_cast<std::size_t>(q)];
        const int* row = indices.Col(q);
        for (int i = 0; i < k; ++i) {
            if (row[i] == g) {
                ranks[static_cast<std::size_t>(q)] = i + 1;
                break;
            }
        }
    }
    std::sort(ranks.begin(), ranks.end());

    std::vector<float> recall(static_cast<std::size_t>(k), 0.0f);
    int count = 0;
    std::size_t idx = 0;
    for (int i = 1; i <= k; ++i) {
        while (idx < ranks.size() && ranks[idx] <= i) {
            ++count;
            ++idx;
        }
        recall[static_cast<std::size_t>(i - 1)] =
            static_cast<float>(count) / static_cast<float>(nquery);
    }
    return recall;
}

double IdealDcgAtK(int relevant_count) {
    double idcg = 0.0;
    for (int i = 0; i < relevant_count; ++i) {
        idcg += 1.0 / std::log2(static_cast<double>(i + 2));
    }
    return idcg;
}

std::string FormatPercentLine(const std::string& name, int gt_k, int result_k, float value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (name == "r") {
        oss << "r@" << result_k << " = " << (100.0f * value);
    } else {
        oss << "top" << gt_k << "_r@" << result_k << " = " << (100.0f * value);
    }
    return oss.str();
}

std::string FormatNdcgLine(int gt_k, int result_k, float value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << "ndcg" << gt_k << "@" << result_k << " = " << value;
    return oss.str();
}

}  // namespace

bool ParseEvalMetricMode(int raw, EvalMetricMode* out) {
    if (!out) return false;
    if (raw < 0 || raw > 2) return false;
    *out = static_cast<EvalMetricMode>(raw);
    return true;
}

int EvalMetricModeToInt(EvalMetricMode mode) {
    return static_cast<int>(mode);
}

bool NeedsFullGroundtruth(EvalMetricMode mode) {
    return mode != EvalMetricMode::kTop1Recall;
}

std::vector<int> ResolveEvalMetricGtKs(const std::vector<int>& raw_gt_ks,
                                       int max_gt_k) {
    std::vector<int> resolved;
    resolved.reserve(raw_gt_ks.size());
    for (int v : raw_gt_ks) {
        if (v >= 1 && v <= max_gt_k) {
            resolved.push_back(v);
        }
    }
    if (resolved.empty()) {
        if (10 <= max_gt_k) resolved.push_back(10);
        if (50 <= max_gt_k) resolved.push_back(50);
        if (resolved.empty() && max_gt_k > 1) resolved.push_back(max_gt_k);
    }
    std::sort(resolved.begin(), resolved.end());
    resolved.erase(std::unique(resolved.begin(), resolved.end()), resolved.end());
    return resolved;
}

bool ComputeEvalMetricBundle(const std::vector<int>& gt_first,
                             const ColMajorMatrix<int>* gt_topk,
                             const ColMajorMatrix<int>& indices,
                             EvalMetricMode mode,
                             const std::vector<int>& gt_ks,
                             EvalMetricBundle* out,
                             std::string* err) {
    if (!out) {
        if (err) *err = "ComputeEvalMetricBundle: out is null.";
        return false;
    }
    *out = {};
    out->mode = mode;
    out->nq = indices.cols;
    out->result_k = indices.rows;
    if (indices.rows <= 0 || indices.cols <= 0) {
        if (err) *err = "ComputeEvalMetricBundle: empty indices.";
        return false;
    }
    if (gt_first.size() != static_cast<std::size_t>(indices.cols)) {
        if (err) *err = "ComputeEvalMetricBundle: gt_first size mismatch.";
        return false;
    }

    out->recall_curve = ComputeRecallCurveTop1(gt_first, indices);
    if (mode == EvalMetricMode::kTop1Recall) {
        return true;
    }

    if (!gt_topk) {
        if (err) *err = "ComputeEvalMetricBundle: gt_topk is required for metric_mode>=1.";
        return false;
    }
    if (gt_topk->cols != indices.cols || gt_topk->rows <= 0) {
        if (err) *err = "ComputeEvalMetricBundle: gt_topk shape mismatch.";
        return false;
    }

    const int nq = indices.cols;
    const int result_k = indices.rows;
    out->gt_k = gt_topk->rows;
    const std::vector<int> resolved_gt_ks = ResolveEvalMetricGtKs(gt_ks, gt_topk->rows);
    if (resolved_gt_ks.empty()) {
        if (err) *err = "ComputeEvalMetricBundle: no valid metric_gt_topks.";
        return false;
    }

    out->gt_curves.clear();
    out->gt_curves.reserve(resolved_gt_ks.size());
    for (int gt_k : resolved_gt_ks) {
        EvalGtMetricCurve curve;
        curve.gt_k = gt_k;
        curve.recall_curve.assign(static_cast<std::size_t>(result_k), 0.0f);
        if (mode == EvalMetricMode::kTopKNdcg) {
            curve.ndcg_curve.assign(static_cast<std::size_t>(result_k), 0.0f);
        }
        out->gt_curves.push_back(std::move(curve));
    }

    for (int q = 0; q < nq; ++q) {
        const int* pred = indices.Col(q);
        const int* gt = gt_topk->Col(q);

        for (EvalGtMetricCurve& curve : out->gt_curves) {
            const int gt_k = curve.gt_k;
            std::vector<int> gt_set;
            gt_set.reserve(static_cast<std::size_t>(gt_k));
            for (int i = 0; i < gt_k; ++i) {
                const int gt_id = gt[i];
                if (gt_id >= 0 && !ContainsId(gt_set, gt_id)) {
                    gt_set.push_back(gt_id);
                }
            }
            const int effective_gt_k = static_cast<int>(gt_set.size());

            int intersection = 0;
            double dcg = 0.0;
            std::vector<int> pred_seen;
            pred_seen.reserve(static_cast<std::size_t>(result_k));
            for (int k = 1; k <= result_k; ++k) {
                const int pred_id = pred[k - 1];
                const bool first_occ = !ContainsId(pred_seen, pred_id);
                if (first_occ) pred_seen.push_back(pred_id);
                const bool relevant =
                    first_occ && ContainsId(gt_set.data(), static_cast<int>(gt_set.size()), pred_id);
                if (relevant) {
                    ++intersection;
                    if (mode == EvalMetricMode::kTopKNdcg) {
                        dcg += 1.0 / std::log2(static_cast<double>(k + 1));
                    }
                }

                curve.recall_curve[static_cast<std::size_t>(k - 1)] +=
                    static_cast<float>(static_cast<double>(intersection) /
                                       static_cast<double>(std::max(1, effective_gt_k)));

                if (mode == EvalMetricMode::kTopKNdcg) {
                    const int ideal_relevant = std::min(effective_gt_k, k);
                    const double denom = IdealDcgAtK(ideal_relevant);
                    const double ndcg = (denom > 0.0) ? (dcg / denom) : 0.0;
                    curve.ndcg_curve[static_cast<std::size_t>(k - 1)] += static_cast<float>(ndcg);
                }
            }
        }
    }

    for (EvalGtMetricCurve& curve : out->gt_curves) {
        for (float& v : curve.recall_curve) {
            v /= static_cast<float>(nq);
        }
        for (float& v : curve.ndcg_curve) {
            v /= static_cast<float>(nq);
        }
    }
    return true;
}

float EvalMetricAtOrNeg(const std::vector<float>& curve, int k) {
    if (k <= 0 || static_cast<std::size_t>(k) > curve.size()) return -1.0f;
    return curve[static_cast<std::size_t>(k - 1)];
}

std::string FormatEvalMetricSummary(const EvalMetricBundle& metrics) {
    std::ostringstream oss;
    oss << std::fixed;
    bool first = true;
    const int ks[] = {1, 10, 50, 100};

    auto append_sep = [&]() {
        if (!first) oss << ' ';
        first = false;
    };

    for (int k : ks) {
        const float v = EvalMetricAtOrNeg(metrics.recall_curve, k);
        if (v < 0.0f) continue;
        append_sep();
        oss << std::setprecision(4) << "r@" << k << "=" << v;
    }
    for (const EvalGtMetricCurve& curve : metrics.gt_curves) {
        for (int k : ks) {
            if (k < curve.gt_k) continue;
            const float v = EvalMetricAtOrNeg(curve.recall_curve, k);
            if (v < 0.0f) continue;
            append_sep();
            oss << std::setprecision(4) << "top" << curve.gt_k << "_r@" << k << "=" << v;
        }
        if (metrics.mode == EvalMetricMode::kTopKNdcg) {
            for (int k : ks) {
                if (k < curve.gt_k) continue;
                const float v = EvalMetricAtOrNeg(curve.ndcg_curve, k);
                if (v < 0.0f) continue;
                append_sep();
                oss << std::setprecision(4) << "ndcg" << curve.gt_k << "@" << k << "=" << v;
            }
        }
    }
    return oss.str();
}

std::vector<std::string> BuildEvalMetricReportLines(const EvalMetricBundle& metrics) {
    std::vector<std::string> lines;
    lines.reserve(128);

    for (int k : kReportPoints) {
        const float v = EvalMetricAtOrNeg(metrics.recall_curve, k);
        if (v >= 0.0f) {
            lines.push_back(FormatPercentLine("r", 0, k, v));
        }
    }

    for (const EvalGtMetricCurve& curve : metrics.gt_curves) {
        for (int k : kReportPoints) {
            if (k < curve.gt_k) continue;
            const float v = EvalMetricAtOrNeg(curve.recall_curve, k);
            if (v >= 0.0f) {
                lines.push_back(FormatPercentLine("top", curve.gt_k, k, v));
            }
        }
        if (metrics.mode == EvalMetricMode::kTopKNdcg) {
            for (int k : kReportPoints) {
                if (k < curve.gt_k) continue;
                const float v = EvalMetricAtOrNeg(curve.ndcg_curve, k);
                if (v >= 0.0f) {
                    lines.push_back(FormatNdcgLine(curve.gt_k, k, v));
                }
            }
        }
    }
    return lines;
}

}  // namespace stlq
