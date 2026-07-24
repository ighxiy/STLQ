#pragma once

#include <cstdint>
#include <string>

#include "stlq/common/config.h"
#include "stlq/common/types.h"
#include "stlq/io/bvecs_reader.h"
#include "stlq/io/fbin_reader.h"
#include "stlq/io/fvecs_reader.h"
#include "stlq/io/ibin_reader.h"
#include "stlq/io/ivecs_reader.h"

namespace stlq::io {

bool LoadExperimentData(const DatasetConfig& config, Dataset* dataset, std::string* error);
bool LoadExperimentData(const DatasetConfig& config, bool load_train, Dataset* dataset, std::string* error);

// Split loaders for out-of-core scaling.
// Julia reference: datasets are logically separate (train/base/query/gt).
bool LoadTrainSet(const DatasetConfig& config, ColMajorMatrix<float>* Xt, std::string* error);
bool LoadBaseSet(const DatasetConfig& config, ColMajorMatrix<float>* Xb, std::string* error);
bool LoadQuerySet(const DatasetConfig& config,
                  ColMajorMatrix<float>* Xq,
                  std::vector<int>* gt_first,
                  std::string* error);
bool LoadGroundtruthTopK(const DatasetConfig& config,
                         int nq,
                         int topk,
                         ColMajorMatrix<int>* gt_topk,
                         std::string* error);
bool ReadGroundtruthTopKFile(const std::string& path,
                             int nq,
                             int topk,
                             bool add_one,
                             ColMajorMatrix<int>* gt_topk,
                             std::string* error);

// Large-scale streaming openers (bvecs/ivecs).
bool OpenTrainBvecsReader(const Config& config, BvecsReader* out, std::string* error);
bool OpenBaseBvecsReader(const Config& config, BvecsReader* out, std::string* error);
bool OpenQueryBvecsReader(const Config& config, BvecsReader* out, std::string* error);
bool OpenGroundtruthIvecsReader(const Config& config, IvecsReader* out, std::string* error);
bool OpenGroundtruthIbinReader(const Config& config, IbinReader* out, std::string* error);

// Streaming readers for `.fvecs` (float32) datasets (e.g., SIFT1M).
bool OpenTrainFvecsReader(const Config& config, FvecsReader* out, std::string* error);
bool OpenBaseFvecsReader(const Config& config, FvecsReader* out, std::string* error);
bool OpenQueryFvecsReader(const Config& config, FvecsReader* out, std::string* error);

// Streaming readers for `.fbin` (Yandex float32 flat binary) datasets (e.g., Deep10M).
bool OpenTrainFbinReader(const Config& config, FbinReader* out, std::string* error);
bool OpenBaseFbinReader(const Config& config, FbinReader* out, std::string* error);
bool OpenQueryFbinReader(const Config& config, FbinReader* out, std::string* error);

// Fast uint8 -> float conversion for `.bvecs` blocks.
void ConvertBvecsToF32(const ColMajorMatrix<std::uint8_t>& in, ColMajorMatrix<float>* out);

// Reservoir sample K vectors from train file by sequential scan (large.enabled training).
// - K>=ntrain: reads all ntrain vectors sequentially.
// - If config.dataset.ntrain_set is true, `ntrain` will be clamped by it.
// - If the file contains fewer than ntrain vectors, ntrain is shrunk with a warning.
bool ReservoirSampleTrainF32(const Config& cfg,
                             std::int64_t ntrain,
                             int K,
                             ColMajorMatrix<float>* Xt_sample,
                             std::string* err);

bool ReservoirSampleTrainU8(const Config& cfg,
                            std::int64_t ntrain,
                            int K,
                            ColMajorMatrix<std::uint8_t>* Xt_u8_sample,
                            std::string* err);

bool ReservoirSampleTrainFbin(const Config& cfg,
                              std::int64_t ntrain,
                              int K,
                              ColMajorMatrix<float>* Xt_sample,
                              std::string* err);

// Convenience: always returns float32 sample; bvecs will be converted to f32.
bool ReservoirSampleTrainToF32(const Config& cfg,
                               std::int64_t ntrain,
                               int K,
                               ColMajorMatrix<float>* Xt_sample,
                               std::string* err);

}  // namespace stlq::io
