#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "stlq/eval/eval_metrics.h"

namespace {

bool NearlyEqual(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

stlq::ColMajorMatrix<int> MakeColMajor(const std::vector<std::vector<int>>& cols) {
    const int n = static_cast<int>(cols.size());
    const int k = n > 0 ? static_cast<int>(cols.front().size()) : 0;
    stlq::ColMajorMatrix<int> out(k, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) {
            out(i, j) = cols[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
        }
    }
    return out;
}

const stlq::EvalGtMetricCurve* FindGtCurve(const stlq::EvalMetricBundle& metrics, int gt_k) {
    for (const auto& curve : metrics.gt_curves) {
        if (curve.gt_k == gt_k) return &curve;
    }
    return nullptr;
}

bool CheckExact() {
    const std::vector<int> gt1 = {10};
    const auto gt_topk = MakeColMajor({{10, 20, 30}});
    const auto pred = MakeColMajor({{10, 20, 30}});
    stlq::EvalMetricBundle metrics;
    std::string err;
    if (!stlq::ComputeEvalMetricBundle(gt1, &gt_topk, pred,
                                         stlq::EvalMetricMode::kTopKNdcg,
                                         std::vector<int>{3}, &metrics, &err)) {
        return false;
    }
    const auto* curve = FindGtCurve(metrics, 3);
    return curve &&
           NearlyEqual(metrics.recall_curve[0], 1.0f) &&
           NearlyEqual(curve->recall_curve[2], 1.0f) &&
           NearlyEqual(curve->ndcg_curve[2], 1.0f);
}

bool CheckPartial() {
    const std::vector<int> gt1 = {10};
    const auto gt_topk = MakeColMajor({{10, 20, 30}});
    const auto pred = MakeColMajor({{10, 99, 20}});
    stlq::EvalMetricBundle metrics;
    std::string err;
    if (!stlq::ComputeEvalMetricBundle(gt1, &gt_topk, pred,
                                         stlq::EvalMetricMode::kTopKNdcg,
                                         std::vector<int>{3}, &metrics, &err)) {
        return false;
    }
    const auto* curve = FindGtCurve(metrics, 3);
    return curve &&
           NearlyEqual(curve->recall_curve[2], 2.0f / 3.0f) &&
           (curve->ndcg_curve[2] > 0.0f && curve->ndcg_curve[2] < 1.0f);
}

bool CheckMiss() {
    const std::vector<int> gt1 = {10};
    const auto gt_topk = MakeColMajor({{10, 20, 30}});
    const auto pred = MakeColMajor({{99, 98, 97}});
    stlq::EvalMetricBundle metrics;
    std::string err;
    if (!stlq::ComputeEvalMetricBundle(gt1, &gt_topk, pred,
                                         stlq::EvalMetricMode::kTopKNdcg,
                                         std::vector<int>{3}, &metrics, &err)) {
        return false;
    }
    const auto* curve = FindGtCurve(metrics, 3);
    return curve &&
           NearlyEqual(curve->recall_curve[2], 0.0f) &&
           NearlyEqual(curve->ndcg_curve[2], 0.0f);
}

bool CheckDuplicate() {
    const std::vector<int> gt1 = {10};
    const auto gt_topk = MakeColMajor({{10, 20, 30}});
    const auto pred = MakeColMajor({{10, 10, 10}});
    stlq::EvalMetricBundle metrics;
    std::string err;
    if (!stlq::ComputeEvalMetricBundle(gt1, &gt_topk, pred,
                                         stlq::EvalMetricMode::kTopKRecall,
                                         std::vector<int>{3}, &metrics, &err)) {
        return false;
    }
    const auto* curve = FindGtCurve(metrics, 3);
    return curve && NearlyEqual(curve->recall_curve[2], 1.0f / 3.0f);
}

bool CheckEmpty() {
    stlq::ColMajorMatrix<int> pred;
    std::vector<int> gt1;
    stlq::EvalMetricBundle metrics;
    std::string err;
    return !stlq::ComputeEvalMetricBundle(gt1, nullptr, pred,
                                            stlq::EvalMetricMode::kTop1Recall,
                                            std::vector<int>(), &metrics, &err) &&
           !err.empty();
}

bool CheckZeroIdcg() {
    const std::vector<int> gt1 = {-1};
    const auto gt_topk = MakeColMajor({{-1, -1, -1}});
    const auto pred = MakeColMajor({{10, 20, 30}});
    stlq::EvalMetricBundle metrics;
    std::string err;
    if (!stlq::ComputeEvalMetricBundle(gt1, &gt_topk, pred,
                                         stlq::EvalMetricMode::kTopKNdcg,
                                         std::vector<int>{3}, &metrics, &err)) {
        return false;
    }
    const auto* curve = FindGtCurve(metrics, 3);
    return curve &&
           NearlyEqual(curve->recall_curve[2], 0.0f) &&
           NearlyEqual(curve->ndcg_curve[2], 0.0f);
}

bool CheckDedupInvalidGt() {
    const std::vector<int> gt1 = {10};
    const auto gt_topk_bad = MakeColMajor({{10, 10, -1}});
    const auto gt_topk_good = MakeColMajor({{10, 20, 30}});
    const auto pred = MakeColMajor({{10, 99, 98}});

    stlq::EvalMetricBundle bad_metrics;
    stlq::EvalMetricBundle good_metrics;
    std::string err;
    if (!stlq::ComputeEvalMetricBundle(gt1, &gt_topk_bad, pred,
                                         stlq::EvalMetricMode::kTopKNdcg,
                                         std::vector<int>{3}, &bad_metrics, &err)) {
        return false;
    }
    if (!stlq::ComputeEvalMetricBundle(gt1, &gt_topk_good, pred,
                                         stlq::EvalMetricMode::kTopKNdcg,
                                         std::vector<int>{1}, &good_metrics, &err)) {
        return false;
    }

    const auto* bad_curve = FindGtCurve(bad_metrics, 3);
    const auto* good_curve = FindGtCurve(good_metrics, 1);
    return bad_curve && good_curve &&
           NearlyEqual(bad_curve->recall_curve[2], 1.0f) &&
           NearlyEqual(bad_curve->ndcg_curve[2], 1.0f) &&
           NearlyEqual(good_curve->recall_curve[0], 1.0f) &&
           NearlyEqual(bad_metrics.recall_curve[0], good_metrics.recall_curve[0]);
}

}  // namespace

int main() {
    if (!CheckExact()) return 1;
    if (!CheckPartial()) return 2;
    if (!CheckMiss()) return 3;
    if (!CheckDuplicate()) return 4;
    if (!CheckEmpty()) return 5;
    if (!CheckZeroIdcg()) return 6;
    if (!CheckDedupInvalidGt()) return 7;
    std::cout << "test_eval_metrics: OK\n";
    return 0;
}
