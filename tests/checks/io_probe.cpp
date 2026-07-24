#include "stlq/common/config.h"
#include "stlq/io/dataset_io.h"
#include "stlq/common/logger.h"

#include <cstdint>
#include <cstdlib>
#include <string>

namespace {

std::string GetArg(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) {
            return argv[i + 1];
        }
    }
    return def;
}

std::int64_t GetArgI64(int argc, char** argv, const std::string& key, std::int64_t def) {
    const std::string s = GetArg(argc, argv, key, "");
    if (s.empty()) {
        return def;
    }
    return static_cast<std::int64_t>(std::strtoll(s.c_str(), nullptr, 10));
}

}  // namespace

int main(int argc, char** argv) {
    using namespace stlq;

    const std::string root = GetArg(argc, argv, "--root", "dataset");
    const std::string name = GetArg(argc, argv, "--name", "SIFT1B");
    const std::int64_t nbase_for_gt = GetArgI64(argc, argv, "--nbase_for_gt", 1'000'000'000LL);
    const std::int64_t nprobe = GetArgI64(argc, argv, "--nprobe", 4);

    Config cfg = DefaultConfig(true);
    cfg.dataset.name = name;
    cfg.dataset.data_root = root;
    cfg.dataset.nbase = static_cast<int>(std::min<std::int64_t>(nbase_for_gt, 2'000'000'000LL));

    std::string err;

    io::BvecsReader train;
    if (io::OpenTrainBvecsReader(cfg, &train, &err)) {
        LogInfo("Train: " + train.path() + " d=" + std::to_string(train.d()) +
                " n=" + std::to_string(train.n()));
    } else {
        LogWarn("Train open failed: " + err);
        err.clear();
    }

    io::BvecsReader base;
    if (!io::OpenBaseBvecsReader(cfg, &base, &err)) {
        LogError("Base open failed: " + err);
        return 1;
    }
    LogInfo("Base: " + base.path() + " d=" + std::to_string(base.d()) +
            " n=" + std::to_string(base.n()));
    stlq::ColMajorMatrix<std::uint8_t> xb;
    if (!base.ReadBlock(0, static_cast<std::uint32_t>(nprobe), &xb, &err)) {
        LogError("Base read failed: " + err);
        return 1;
    }
    if (xb.cols > 0) {
        const std::uint8_t* v0 = xb.Col(0);
        LogInfo("Base[0] first8 bytes: " + std::to_string(v0[0]) + " " + std::to_string(v0[1]) +
                " " + std::to_string(v0[2]) + " " + std::to_string(v0[3]) + " " +
                std::to_string(v0[4]) + " " + std::to_string(v0[5]) + " " +
                std::to_string(v0[6]) + " " + std::to_string(v0[7]));
    }

    io::BvecsReader query;
    if (!io::OpenQueryBvecsReader(cfg, &query, &err)) {
        LogError("Query open failed: " + err);
        return 1;
    }
    LogInfo("Query: " + query.path() + " d=" + std::to_string(query.d()) +
            " n=" + std::to_string(query.n()));

    io::IvecsReader gt;
    if (!io::OpenGroundtruthIvecsReader(cfg, &gt, &err)) {
        LogError("GT open failed: " + err);
        return 1;
    }
    LogInfo("GT: " + gt.path() + " k=" + std::to_string(gt.k()) +
            " n=" + std::to_string(gt.n()));
    stlq::ColMajorMatrix<std::int32_t> gt0;
    if (!gt.ReadBlock(0, 1, &gt0, &err)) {
        LogError("GT read failed: " + err);
        return 1;
    }
    if (gt0.cols > 0) {
        const std::int32_t* ids = gt0.Col(0);
        const int k_print = std::min(gt0.rows, 10);
        std::string msg = "GT[0] top" + std::to_string(k_print) + " ids:";
        for (int i = 0; i < k_print; ++i) {
            msg += " " + std::to_string(ids[i]);
        }
        LogInfo(msg);
    }

    LogInfo("IO probe done.");
    return 0;
}
