#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "stlq/common/config.h"
#include "stlq/common/types.h"
#include "stlq/io/bvecs_reader.h"
#include "stlq/io/fbin_reader.h"
#include "stlq/io/fvecs_reader.h"
#include "stlq/io/ibin_reader.h"
#include "stlq/io/ivecs_reader.h"

namespace stlq::io {

enum class DatasetRole {
    kTrain = 0,
    kBase = 1,
    kQuery = 2,
};

enum class VectorFileFormat {
    kAuto = 0,
    kBvecs = 1,
    kFvecs = 2,
    kFbin = 3,
};

enum class GroundtruthFileFormat {
    kAuto = 0,
    kIvecs = 1,
    kIbin = 2,
};

struct DatasetVectorReader {
    VectorFileFormat format = VectorFileFormat::kAuto;
    bool is_u8 = false;
    BvecsReader bvecs;
    FvecsReader fvecs;
    FbinReader fbin;
    int d = 0;
    std::uint64_t n = 0;
    std::string path;
};

struct DatasetGroundtruthReader {
    GroundtruthFileFormat format = GroundtruthFileFormat::kAuto;
    IvecsReader ivecs;
    IbinReader ibin;
    int k = 0;
    std::uint64_t n = 0;
    std::string path;
};

const char* DatasetRoleName(DatasetRole role);
const char* VectorFileFormatName(VectorFileFormat format);
const char* GroundtruthFileFormatName(GroundtruthFileFormat format);

bool OpenDatasetVectorReader(const Config& config,
                             DatasetRole role,
                             DatasetVectorReader* out,
                             std::string* error);

bool ReadDatasetVectorBlockF32(const DatasetVectorReader& reader,
                               std::uint64_t start,
                               std::uint32_t count,
                               ColMajorMatrix<float>* out,
                               std::string* error);

bool ReadDatasetVectorBlockU8(const DatasetVectorReader& reader,
                              std::uint64_t start,
                              std::uint32_t count,
                              ColMajorMatrix<std::uint8_t>* out,
                              std::string* error);

bool ReadDatasetVectorByIdsF32(const DatasetVectorReader& reader,
                               const std::vector<std::uint32_t>& ids,
                               ColMajorMatrix<float>* out,
                               std::string* error);

bool ReadDatasetVectorByIdsU8(const DatasetVectorReader& reader,
                              const std::vector<std::uint32_t>& ids,
                              ColMajorMatrix<std::uint8_t>* out,
                              std::string* error);

bool OpenDatasetGroundtruthReader(const Config& config,
                                  DatasetGroundtruthReader* out,
                                  std::string* error);

bool ReadDatasetGroundtruthFirst(const Config& config,
                                 const DatasetGroundtruthReader& reader,
                                 int* n_io,
                                 std::vector<int>* out,
                                 std::string* error);

bool ReadDatasetGroundtruthTopK(const Config& config,
                                const DatasetGroundtruthReader& reader,
                                int nq,
                                int topk,
                                ColMajorMatrix<int>* out,
                                std::string* error);

bool LoadQuerySet(const Config& config,
                  ColMajorMatrix<float>* Xq,
                  std::vector<int>* gt_first,
                  std::string* error);

bool LoadGroundtruthTopK(const Config& config,
                         int nq,
                         int topk,
                         ColMajorMatrix<int>* gt_topk,
                         std::string* error);

}  // namespace stlq::io
