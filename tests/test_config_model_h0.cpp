#include "stlq/common/config.h"
#include "stlq/pipeline/large_store_hash.h"
#include "stlq/pipeline/app_utils.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

std::filesystem::path WriteCfg(const std::string& name, const std::string& body) {
    const auto path = std::filesystem::current_path() / name;
    std::ofstream out(path, std::ios::trunc);
    out << body;
    return path;
}

stlq::Config LoadAndNormalize(const std::filesystem::path& path) {
    stlq::Config cfg;
    std::string err;
    const bool ok = stlq::LoadConfigFile(path.string(), &cfg, &err);
    if (!ok) {
        std::cerr << err << "\n";
        std::exit(1);
    }
    stlq::NormalizeHVec(&cfg);
    return cfg;
}

}  // namespace

int main() {
    const auto legacy_path = WriteCfg(
        "test_model_h0_legacy.cfg",
        "model.m = 10\n"
        "model.h_vec = [4096,256,256,256,256,256,256,256,256,256]\n");
    const auto shorthand_path = WriteCfg(
        "test_model_h0_shorthand.cfg",
        "model.m = 10\n"
        "model.h0 = 4096\n");

    const stlq::Config legacy = LoadAndNormalize(legacy_path);
    const stlq::Config shorthand = LoadAndNormalize(shorthand_path);

    const std::vector<int> expected{4096, 256, 256, 256, 256, 256, 256, 256, 256, 256};
    assert(legacy.model.h_vec == expected);
    assert(shorthand.model.h_vec == expected);
    assert(shorthand.model.h_vec == legacy.model.h_vec);

    stlq::TrainResult train;
    train.R = stlq::ColMajorMatrix<float>(2, 2);
    train.R(0, 0) = 1.0f;
    train.R(1, 1) = 1.0f;
    train.C_root.books.resize(1);
    train.C_root.books[0] = stlq::ColMajorMatrix<float>(2, 1);
    train.C_root.books[0](0, 0) = 0.25f;
    train.C_root.books[0](1, 0) = 0.75f;

    stlq::io::BaseBasicStoreConfig store_cfg;
    store_cfg.d = 2;
    store_cfg.m = legacy.model.m;
    store_cfg.h_vec = legacy.model.h_vec;

    assert(stlq::app::ComputeBaseBasicStoreHash(legacy, train, store_cfg, false) ==
           stlq::app::ComputeBaseBasicStoreHash(shorthand, train, store_cfg, false));

    std::error_code ec;
    std::filesystem::remove(legacy_path, ec);
    std::filesystem::remove(shorthand_path, ec);

    std::cout << "test_config_model_h0: OK\n";
    return 0;
}
