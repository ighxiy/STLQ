#pragma once

#include <string>

#include "stlq/common/config.h"
#include "stlq/io/base_list_store.h"
#include "stlq/io/ivf_lists.h"
#include "stlq/ivf/ivf_scan.h"
#include "stlq/common/types.h"

namespace stlq {

struct DiskIvfEvalTiming {
    double wall_sec = 0.0;       // total accounted wall time
    double core_wall_sec = 0.0;  // qt_rotate + qt_gemm + probe_sel + scan+topk
    // INPUT: set by caller before calling EvaluateRecallBaseIvfFromDiskTimed.
    double qt_rotate_wall_sec = 0.0;
    double qt_gemm_wall_sec = 0.0;
    double probe_sel_wall_sec = 0.0;
    double scan_topk_wall_sec = 0.0;
    double gemm_root_wall_sec = 0.0;
    double gemm_small_wall_sec = 0.0;
};

// IVF base recall evaluation that scans list-order contiguous base encodings from disk.
// This is designed for large-scale (1e8/1e9) where holding base encodings in RAM is infeasible.
bool EvaluateRecallBaseIvfFromDisk(const Config& cfg,
                                   const Dataset& query_dataset_inmem,
                                   const io::IvfListsReader& lists,
                                   const io::BaseListReader& base_list,
                                   const TrainResult& train,
                                   RecallResult* out,
                                   std::string* err);

// Same as EvaluateRecallBaseIvfFromDisk, but also returns timing counters.
bool EvaluateRecallBaseIvfFromDiskTimed(const Config& cfg,
                                        const Dataset& query_dataset_inmem,
                                        const io::IvfListsReader& lists,
                                        const io::BaseListReader& base_list,
                                        const TrainResult& train,
                                        RecallResult* out,
                                        DiskIvfEvalTiming* timing,
                                        std::string* err);

}  // namespace stlq
