#include "stlq/io/result_io.h"

#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace {

stlq::ColMajorMatrix<float> MakeMatrix(int rows, int cols, float base) {
    stlq::ColMajorMatrix<float> out(rows, cols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            out(i, j) = base + static_cast<float>(10 * j + i);
        }
    }
    return out;
}

stlq::ColMajorMatrix<stlq::FullCode> MakeCodeMatrix(int rows, int cols, int base) {
    stlq::ColMajorMatrix<stlq::FullCode> out(rows, cols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            out(i, j) = static_cast<stlq::FullCode>(base + 10 * j + i);
        }
    }
    return out;
}

void Require(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "test_train_result_bin_io: " << message << "\n";
        std::exit(1);
    }
}

void ExpectMatrixEq(const stlq::ColMajorMatrix<float>& a,
                    const stlq::ColMajorMatrix<float>& b) {
    Require(a.rows == b.rows, "matrix row mismatch");
    Require(a.cols == b.cols, "matrix col mismatch");
    Require(a.data.size() == b.data.size(), "matrix size mismatch");
    for (std::size_t i = 0; i < a.data.size(); ++i) {
        Require(std::fabs(a.data[i] - b.data[i]) < 1e-6f, "matrix value mismatch");
    }
}

void ExpectMatrixEq(const stlq::ColMajorMatrix<stlq::FullCode>& a,
                    const stlq::ColMajorMatrix<stlq::FullCode>& b) {
    Require(a.rows == b.rows, "code matrix row mismatch");
    Require(a.cols == b.cols, "code matrix col mismatch");
    Require(a.data == b.data, "code matrix value mismatch");
}

stlq::TrainResult MakeFullResult() {
    stlq::TrainResult result;
    result.C_root.d = 3;
    result.C_root.books = {MakeMatrix(3, 4, 1.0f), MakeMatrix(3, 2, 100.0f)};
    result.C_root.h_vec = {4, 2};
    result.C_one.d = 3;
    result.C_one.books = {MakeMatrix(3, 3, 200.0f), MakeMatrix(3, 2, 300.0f)};
    result.C_one.h_vec = {3, 2};
    result.R = MakeMatrix(3, 3, 400.0f);
    result.is_bad_cluster = {false, true, false, true};
    result.init_mse = 1.25f;
    result.beam_mse = 2.5f;
    result.icm_mse = 3.75f;
    result.linkage_mse_mean = 4.25f;
    result.linkage_ratio = 0.5f;
    result.linkage_max_depth = 7;
    result.linkage_mean_depth = 2.25f;
    result.R_iters = 11;
    result.final_linkaged_mse_rot = 5.5f;
    result.exp_root_hint = "exp-root";
    return result;
}

stlq::Config MakeConfig(const std::filesystem::path& train_file) {
    stlq::Config config;
    config.dataset.name = "TEST";
    config.dataset.ntrain = 8;
    config.model.m = 2;
    config.model.h_vec = {4, 2};
    config.io.result_format = "bin";
    config.io.train_file = train_file.string();
    config.io.train_file_set = true;
    return config;
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::current_path() / "test_train_result_bin_io_tmp";
    fs::remove_all(root);
    fs::create_directories(root);

    const fs::path full_path = root / "train_full.stlqbin";
    stlq::Config config = MakeConfig(full_path);
    const stlq::TrainResult full = MakeFullResult();
    std::string error;
    Require(stlq::io::SaveTrainResults(full_path.string(), config, full, &error), "save full result failed");

    stlq::TrainResult loaded;
    Require(stlq::io::LoadTrainResults(full_path.string(), "raw", "", 2, &loaded, &error), "load full result failed");
    Require(loaded.checkpoint_stage == 0, "full checkpoint stage mismatch");
    Require(loaded.C_root.books.size() == full.C_root.books.size(), "full C_root count mismatch");
    Require(loaded.C_one.books.size() == full.C_one.books.size(), "full C_one count mismatch");
    for (std::size_t i = 0; i < full.C_root.books.size(); ++i) {
        ExpectMatrixEq(loaded.C_root.books[i], full.C_root.books[i]);
    }
    for (std::size_t i = 0; i < full.C_one.books.size(); ++i) {
        ExpectMatrixEq(loaded.C_one.books[i], full.C_one.books[i]);
    }
    ExpectMatrixEq(loaded.R, full.R);
    Require(loaded.is_bad_cluster == full.is_bad_cluster, "is_bad_cluster mismatch");
    Require(loaded.R_iters == full.R_iters, "R_iters mismatch");
    Require(loaded.linkage_max_depth == full.linkage_max_depth, "linkage_max_depth mismatch");
    Require(loaded.exp_root_hint == full.exp_root_hint, "exp_root_hint mismatch");
    Require(fs::exists(full_path.string() + ".checkpoint.txt"), "missing checkpoint sidecar");
    Require(fs::exists(full_path.string() + ".config_snapshot.txt"), "missing config sidecar");

    const fs::path pre_path = root / "train_pre.stlqbin";
    stlq::Config pre_config = MakeConfig(pre_path);
    stlq::TrainResult pre = full;
    pre.C_one = {};
    Require(stlq::io::SaveTrainResultsPreInitLinkage(pre_path.string(), pre_config, pre, "pre-exp-root", &error),
            "save pre-init-linkage result failed");

    stlq::TrainResult pre_loaded;
    Require(stlq::io::LoadTrainResults(pre_path.string(), "raw", "", 2, &pre_loaded, &error),
            "load pre-init-linkage result failed");
    Require(pre_loaded.checkpoint_stage == 1, "pre checkpoint stage mismatch");
    Require(pre_loaded.C_root.books.size() == pre.C_root.books.size(), "pre C_root count mismatch");
    Require(pre_loaded.C_one.books.empty(), "pre C_one should be empty");
    Require(pre_loaded.C_one.d == pre.C_root.d, "pre C_one d mismatch");
    Require(pre_loaded.exp_root_hint == "pre-exp-root", "pre exp_root_hint mismatch");
    ExpectMatrixEq(pre_loaded.R, pre.R);

    stlq::BaseEncoding base;
    base.B = MakeCodeMatrix(2, 5, 1);
    base.a = MakeMatrix(2, 5, 10.0f);
    stlq::VirtualEncoding virt;
    virt.m = 2;
    virt.nlist = 2;
    virt.B_by_cluster = {MakeCodeMatrix(2, 1, 40), MakeCodeMatrix(2, 2, 50)};
    virt.a_by_cluster = {MakeMatrix(2, 1, 60.0f), MakeMatrix(2, 2, 70.0f)};

    const fs::path base_path = root / "base_hq.stlqbin";
    stlq::Config base_config = MakeConfig(base_path);
    Require(stlq::io::SaveBaseResultsHq(base_path.string(), base_config, base, 1.0f, 5, 2.0f, &virt, &error),
            "save base hq bin failed");
    stlq::BaseEncoding base_loaded;
    stlq::VirtualEncoding virt_loaded;
    int n_base_real = -1;
    Require(stlq::io::LoadBaseResultsHq(base_path.string(), "raw", "", &base_loaded, &n_base_real, &virt_loaded, &error),
            "load base hq bin failed");
    ExpectMatrixEq(base_loaded.B, base.B);
    ExpectMatrixEq(base_loaded.a, base.a);
    Require(n_base_real == 5, "base hq n_base_real mismatch");
    Require(virt_loaded.nlist == virt.nlist, "base hq virtual nlist mismatch");
    ExpectMatrixEq(virt_loaded.B_by_cluster[1], virt.B_by_cluster[1]);
    ExpectMatrixEq(virt_loaded.a_by_cluster[1], virt.a_by_cluster[1]);

    const fs::path linkage_path = root / "base_linkage.stlqbin";
    stlq::Config linkage_config = MakeConfig(linkage_path);
    const std::vector<int> parent = {0, 1, 1, 3, 4};
    const std::vector<int> cluster_id = {0, 0, 1, 1, 1};
    const std::vector<bool> is_bad = {false, true};
    Require(stlq::io::SaveBaseResultsSTLQ(linkage_path.string(), linkage_config, base, parent, &cluster_id,
                                          1.5f, 5, 0.6f, 4, 1.25f, &is_bad, &virt, &error),
            "save base linkage bin failed");
    stlq::BaseEncoding linkage_loaded;
    stlq::VirtualEncoding linkage_virt_loaded;
    std::vector<int> parent_loaded;
    std::vector<int> cluster_id_loaded;
    int linkage_n_base_real = -1;
    Require(stlq::io::LoadBaseResultsSTLQ(linkage_path.string(), "raw", "", linkage_config.io,
                                          &linkage_loaded, &parent_loaded, &cluster_id_loaded,
                                          &linkage_n_base_real, &linkage_virt_loaded, &error),
            "load base linkage bin failed");
    ExpectMatrixEq(linkage_loaded.B, base.B);
    ExpectMatrixEq(linkage_loaded.a, base.a);
    Require(parent_loaded == parent, "linkage parent mismatch");
    Require(cluster_id_loaded == cluster_id, "linkage cluster_id mismatch");
    Require(linkage_n_base_real == 5, "linkage n_base_real mismatch");
    ExpectMatrixEq(linkage_virt_loaded.B_by_cluster[0], virt.B_by_cluster[0]);
    ExpectMatrixEq(linkage_virt_loaded.a_by_cluster[0], virt.a_by_cluster[0]);

    fs::remove_all(root);
    std::cout << "test_train_result_bin_io: OK\n";
    return 0;
}
