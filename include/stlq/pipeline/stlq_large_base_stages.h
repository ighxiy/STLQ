#pragma once

#include "stlq/common/config.h"
#include "stlq/eval/eval_metrics.h"
#include "stlq/io/base_list_store.h"
#include "stlq/io/base_store.h"
#include "stlq/io/dataset_reader_factory.h"
#include "stlq/io/ivf_lists.h"
#include "stlq/quantizer/encoder.h"
#include "stlq/quantizer/precomp_large_root.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace stlq {

enum class BaseFormat { kBvecs, kFvecs, kFbin };

struct LargeBaseReaderState {
    BaseFormat format = BaseFormat::kFvecs;
    bool is_u8 = false;
    io::DatasetVectorReader reader;
    int d = 0;
    std::uint64_t n = 0;
    std::string path;
};

struct LargeRunState {
    int expected_nlist = 0;
    std::uint64_t base_ntotal_effective = 0;
    io::BaseBasicStoreConfig store_cfg;
    std::uint64_t base_basic_hash = 0;
    std::uint64_t base_list_hash_identity = 0;
    std::uint64_t linkage_list_hash_identity = 0;
    std::uint64_t run_hash_identity = 0;
    std::uint64_t coeff_codec_hash = 0;
    bool eval_only = false;
    std::string run_tag;
    std::string run_root;
    std::string run_tmp_root;
    std::string effective_run_root;
    std::string effective_run_tmp_root;
};

struct LargeBaseBasicStageState {
    std::string out_dir;
    std::string meta_path;
    std::string hash_path;
    bool rebuilt = false;
};

struct LargeIvfStageState {
    bool need_ivf = false;
    io::IvfListsReader ivf;
    io::BaseBasicReader base_basic;
};

struct LargeBaseListStageState {
    bool need_base_list = false;
    io::BaseListReader base_list;
    std::string base_list_dir;
    std::uint64_t base_list_hash = 0;
};

struct LargeDiskEvalQueryState {
    Dataset query_dataset;
    Dataset query_dataset_base;
    ColMajorMatrix<int> query_gt_topk;
    double base_qt_rotate_sec = 0.0;
    double linkage_qt_rotate_sec = 0.0;
};

std::string FormatHexU64(std::uint64_t v);

int OpenLargeBaseReaderStage(const Config& config, LargeBaseReaderState& base_reader, std::string& error);
int PrepareLargeRunStateStage(const Config& config, const TrainResult& train_result, const LargeBaseReaderState& base_reader, LargeRunState& run_state);
int WriteLargeRunArtifactsStage(const Config& config, const TrainResult& train_result, const LargeBaseReaderState& base_reader, const LargeRunState& run_state, const std::string& frozen_original_cfg, const std::filesystem::path& frozen_original_cfg_name, std::string& error);
int ValidateLargeRunGuardrailsStage(const Config& config);
int BuildLargePrecompRootStage(const Config& config, const TrainResult& train_result, StreamKernelProvider* stream_kernels, PrecompLargeRoot& pre_lr, PrecompLargeRoot*& pre_lr_ptr, std::string& error);
int RunLargeBaseBasicStage(const Config& config, const TrainResult& train_result, LargeBaseReaderState& base_reader, LargeRunState& run_state, Precomp& pre_root, bool& pre_root_ready, PrecompLargeRoot* pre_lr_ptr, StreamKernelProvider* stream_kernels, LargeBaseBasicStageState& base_basic_state, std::string& error);
int RunLargeIvfStage(const Config& config, const LargeRunState& run_state, const LargeBaseBasicStageState& base_basic_state, bool do_eval_base, LargeIvfStageState& ivf_state, std::string& error);
int RunLargeBaseListStage(const Config& config, const LargeRunState& run_state, const LargeBaseBasicStageState& base_basic_state, LargeIvfStageState& ivf_state, bool do_eval_base, LargeBaseListStageState& base_list_state, std::string& error);
bool BuildLargeEvalMetricsForIndices(const Config& config, EvalMetricMode eval_metric_mode, const ColMajorMatrix<int>& query_gt_topk, const std::vector<int>& gt_first, const ColMajorMatrix<int>& indices, EvalMetricBundle* metrics_out);
int PrepareLargeDiskEvalQueriesStage(const Config& config, const TrainResult& train_result, EvalMetricMode eval_metric_mode, bool do_eval_base, bool do_eval_linkage, LargeDiskEvalQueryState& eval_query_state, std::string& error);
int RunLargeBaseDiskEvalStage(const Config& config, const TrainResult& train_result, EvalMetricMode eval_metric_mode, const io::IvfListsReader& ivf, const io::BaseListReader& base_list, LargeDiskEvalQueryState& eval_query_state, std::string& error);
int RunLargeBaseListCleanupStage(const Config& config, const LargeBaseBasicStageState& base_basic_state, const LargeBaseListStageState& base_list_state);

}  // namespace stlq
