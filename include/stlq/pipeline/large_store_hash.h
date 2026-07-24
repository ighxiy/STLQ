#pragma once

#include <cstdint>
#include <string>

#include "stlq/common/config.h"
#include "stlq/io/base_store.h"
#include "stlq/common/types.h"

namespace stlq::app {

std::string StoreHashPath(const std::string& dir);
std::string CodecHashPath(const std::string& linkage_list_dir);

bool ReadU64File(const std::string& path, std::uint64_t* out);
bool WriteU64FileHex(const std::string& path, std::uint64_t v, std::string* error);
bool ReadNorm2SourceHash(const std::string& linkage_list_dir,
                         bool use_coeff_codec,
                         std::uint64_t* out);

std::uint64_t ComputeBaseBasicStoreHash(const Config& cfg,
                                        const TrainResult& tr,
                                        const io::BaseBasicStoreConfig& store_cfg,
                                        bool base_is_u8);

std::uint64_t ComputeBaseListStoreHash(const Config& cfg,
                                       std::uint64_t base_basic_hash,
                                       int nlist,
                                       std::uint64_t ntotal);

std::uint64_t ComputeLinkageListStoreHash(const Config& cfg,
                                        const TrainResult& tr,
                                        std::uint64_t base_list_hash,
                                        int nlist);

std::uint64_t ComputeLinkageCoeffCodecHash(const Config& cfg,
                                         std::uint64_t linkage_list_hash);

}  // namespace stlq::app
