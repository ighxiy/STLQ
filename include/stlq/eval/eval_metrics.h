#pragma once

#include <string>
#include <vector>

#include "stlq/common/types.h"

namespace stlq {

enum class EvalMetricMode : int {
    kTop1Recall = 0,
    kTopKRecall = 1,
    kTopKNdcg = 2,
};

struct EvalGtMetricCurve {
    int gt_k = 0;
    std::vector<float> recall_curve;
    std::vector<float> ndcg_curve;
};

struct EvalMetricBundle {
    EvalMetricMode mode = EvalMetricMode::kTop1Recall;
    int nq = 0;
    int result_k = 0;
    int gt_k = 0;
    std::vector<float> recall_curve;
    std::vector<EvalGtMetricCurve> gt_curves;
};

bool ParseEvalMetricMode(int raw, EvalMetricMode* out);
int EvalMetricModeToInt(EvalMetricMode mode);
bool NeedsFullGroundtruth(EvalMetricMode mode);

std::vector<int> ResolveEvalMetricGtKs(const std::vector<int>& raw_gt_ks,
                                       int max_gt_k);

bool ComputeEvalMetricBundle(const std::vector<int>& gt_first,
                             const ColMajorMatrix<int>* gt_topk,
                             const ColMajorMatrix<int>& indices,
                             EvalMetricMode mode,
                             const std::vector<int>& gt_ks,
                             EvalMetricBundle* out,
                             std::string* err);

float EvalMetricAtOrNeg(const std::vector<float>& curve, int k);
std::string FormatEvalMetricSummary(const EvalMetricBundle& metrics);
std::vector<std::string> BuildEvalMetricReportLines(const EvalMetricBundle& metrics);

}  // namespace stlq
