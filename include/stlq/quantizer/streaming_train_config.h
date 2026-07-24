#pragma once

#include <filesystem>

#include "stlq/common/config.h"
#include "stlq/io/base_store.h"
#include "stlq/io/linkage_list_store.h"
#include "stlq/quantizer/spkmeans.h"
#include "stlq/common/types.h"

namespace stlq {

struct StreamingTrainWorkspacePaths {
    std::filesystem::path exp_root_run;
    std::filesystem::path train_basic_dir;
    std::filesystem::path train_ivf_dir;
    std::filesystem::path train_list_dir;
    std::filesystem::path train_linkage_list_dir;
    std::filesystem::path train_linkage_init_list_dir;
};

KmeansConfig MakeStreamingTrainKmeansConfig(const Config& config, bool train_is_u8);

Config MakeTrainBasicEncodeConfig(const Config& config, std::uint64_t ntrain);

io::BaseBasicStoreConfig MakeTrainBasicStoreConfig(const Config& config,
                                                   const std::filesystem::path& train_basic_dir,
                                                   int d,
                                                   bool is_u8);

io::LinkageListStoreConfig MakeTrainLinkageListStoreConfig(const Config& config,
                                                       const std::filesystem::path& train_linkage_list_dir,
                                                       int nlist);

Config MakeTrainLinkageConfig(const Config& config);

StreamingTrainWorkspacePaths ResolveStreamingTrainWorkspacePaths(const Config& config,
                                                                 const std::filesystem::path& exp_root,
                                                                 const TrainResult& result);

void EnsureStreamingTrainWorkspaceDirs(const StreamingTrainWorkspacePaths& paths);

}  // namespace stlq
