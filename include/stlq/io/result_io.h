#pragma once

#include <string>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/common/types.h"

namespace stlq::io {

// Returns `base_path` with an appended `_YYYYMMDD` suffix before extension.
// If the resulting file already exists, appends `_01`, `_02`, ... until unique.
// Example: `train.h5` -> `train_20260101.h5` (or `train_20260101_01.h5` if exists).
std::string MakeUniqueDatedPath(const std::string& base_path, std::string* error);

// Builds a dated path for reading using config suffixes (underscores are auto-added):
// - date: YYYYMMDD (if empty => today)
// - seq: XXX digits (if empty => no seq)
std::string MakeDatedPathForLoad(const std::string& base_path,
                                 const std::string& load_date,
                                 const std::string& load_seq,
                                 std::string* error);

int ComputeMaxDepthFromParent(const std::vector<int>& parent);

// Julia reference: save_c_R / save_c_Rv (demos/utils.jl:314-347).
bool SaveTrainResults(const std::string& path,
                      const Config& config,
                      const TrainResult& result,
                      std::string* error);

// Profiling-only checkpoint: saves only C_root+R (C_one is intentionally absent).
// This is meant for resuming init-linkage without re-running RVQ/basic on very large datasets.
bool SaveTrainResultsPreInitLinkage(const std::string& path,
                                 const Config& config,
                                 const TrainResult& result,
                                 const std::string& exp_root_hint,
                                 std::string* error);

bool LoadTrainResults(const std::string& base_path,
                      const std::string& load_date,
                      const std::string& load_seq,
                      int m,
                      TrainResult* result,
                      std::string* error);

// Julia reference: save_base_results_hq / save_base_results_stlq (demos/utils.jl:350-379).
bool SaveBaseResultsHq(const std::string& path,
                       const Config& config,
                       const BaseEncoding& base,
                       float base_error,
                       int n_base_real,
                       float beam_error,
                       const VirtualEncoding* virt,
                       std::string* error);

bool SaveBaseResultsSTLQ(const std::string& path,
                           const Config& config,
                           const BaseEncoding& base,
                           const std::vector<int>& parent,
                           const std::vector<int>* cluster_id,
                           float base_error,
                           int n_base_real,
                           float linkage_ratio,
                           int linkage_max_depth,
                           float linkage_mean_depth,
                           const std::vector<bool>* is_bad_cluster,
                           const VirtualEncoding* virt,
                           std::string* error);

bool LoadBaseResultsHq(const std::string& base_path,
                       const std::string& load_date,
                       const std::string& load_seq,
                       BaseEncoding* base,
                       int* n_base_real,
                       VirtualEncoding* virt,
                       std::string* error);

bool LoadBaseResultsSTLQ(const std::string& base_path,
                           const std::string& load_date,
                           const std::string& load_seq,
                           const IOConfig& io_cfg,
                           BaseEncoding* base,
                           std::vector<int>* parent,
                           std::vector<int>* cluster_id,
                           int* n_base_real,
                           VirtualEncoding* virt,
                           std::string* error);

}  // namespace stlq::io
