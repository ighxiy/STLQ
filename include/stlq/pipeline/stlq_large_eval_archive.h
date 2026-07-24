#pragma once

#include "stlq/common/config.h"
#include "stlq/eval/eval_metrics.h"
#include "stlq/eval/recall_linkage_disk.h"

#include <string>
#include <vector>

namespace stlq {

bool ArchiveLargeLinkageDiskEvalResult(const std::string& run_root,
                                     const std::string& archive_label,
                                     const EvalMetricBundle& metrics,
                                     int nq,
                                     bool use_coeff_codec,
                                     const Config& cfg,
                                     const DiskLinkageEvalTiming& timing,
                                     const DiskLinkageEvalSession& session,
                                     const std::vector<double>& repeat_core_secs,
                                     std::string* error);

}  // namespace stlq
