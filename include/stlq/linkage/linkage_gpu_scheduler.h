#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "stlq/cuda/cuda_stream_kernels_pool.h"
#include "stlq/common/types.h"

namespace stlq {

enum class LinkageGpuSchedulerTaskSource : std::uint8_t {
    GoodInner = 0,
    GoodSameFrozen = 1,
    GoodDynBefore = 2,
    GoodDynWindow = 3,
    BadFrozen = 4,
    BadDynWindow = 5,
    InitInner = 6,
    InitDynBefore = 7,
    InitDynWindow = 8,
    InitSameSingle = 9,
};

using LinkageGpuSchedulerClusterHandle = std::uint64_t;

struct LinkageGpuSchedulerStats {
    std::uint64_t many_tasks_submitted = 0;
    std::uint64_t single_tasks_submitted = 0;
    std::uint64_t many_launches = 0;
    std::uint64_t single_launches = 0;
    std::uint64_t many_direct_launches = 0;
    std::uint64_t many_direct_tasks = 0;
    std::uint64_t tasks_merged_total = 0;
    std::uint64_t tasks_merged_max = 0;
    std::uint64_t nodes_total = 0;
    std::uint64_t nodes_max = 0;
    std::uint64_t pairs_total = 0;
    std::uint64_t pairs_max = 0;
    std::uint64_t cross_cluster_launches = 0;
    std::uint64_t update_ops_applied = 0;
    double queue_wait_total_s = 0.0;
    double queue_wait_max_s = 0.0;
};

struct LinkageGpuSchedulerManyNodesResult {
    int B = 0;
    int m = 0;
    bool root_codes_u32 = false;
    std::vector<int> best_parent_local;
    std::vector<float> best_cost;
    std::vector<RootCode> best_code0_root;
    std::vector<FullCode> best_B;
    std::vector<float> best_a;
};

struct LinkageGpuSchedulerSingleNodeResult {
    int best_parent_local = -1;
    float best_cost = 0.0f;
    bool root_codes_u32 = false;
    RootCode best_code0_root = 0;
    std::vector<FullCode> best_B;
    std::vector<float> best_a;
};

struct LinkageGpuSchedulerManyNodesTask {
    LinkageGpuSchedulerClusterHandle cluster = 0;
    LinkageGpuSchedulerTaskSource source = LinkageGpuSchedulerTaskSource::GoodInner;
    std::uint64_t required_rfull_generation = 0;

    const Precomp* pre_one = nullptr;
    const CodebookPack* codebook_large_root = nullptr;

    int d = 0;
    int B = 0;
    int m = 0;
    std::shared_ptr<const std::vector<float>> X_block_shared;
    std::size_t X_block_shared_offset = 0;
    std::vector<float> X_block;                  // d x B col-major
    std::vector<int> cand_parent_local_flat;     // Npairs
    std::vector<int> pair_node;                  // Npairs
    std::vector<int> cand_offsets;               // B + 1
    std::shared_ptr<const std::vector<std::uint64_t>> node_sample_id_base_shared;
    std::size_t node_sample_id_base_shared_offset = 0;
    std::vector<std::uint64_t> node_sample_id_base;  // B
    std::vector<int> pair_t_override;            // optional, Npairs

    std::uint32_t seed = 0;
    int icm_iters = 0;
    int ils_iters = 0;
    int perturb_k = 0;
};

struct LinkageGpuSchedulerSingleNodeTask {
    LinkageGpuSchedulerClusterHandle cluster = 0;
    std::uint64_t required_rfull_generation = 0;

    const Precomp* pre_one = nullptr;
    const CodebookPack* codebook_large_root = nullptr;

    int d = 0;
    int m = 0;
    std::vector<float> xi;                // d
    std::vector<int> cand_parent_local;   // Kp

    std::uint32_t seed = 0;
    std::uint64_t sample_id_offset = 0;
    int icm_iters = 0;
    int ils_iters = 0;
    int perturb_k = 0;
};

class LinkageGpuScheduler {
public:
    explicit LinkageGpuScheduler(const CudaPoolConfig& cfg);
    ~LinkageGpuScheduler();

    LinkageGpuScheduler(const LinkageGpuScheduler&) = delete;
    LinkageGpuScheduler& operator=(const LinkageGpuScheduler&) = delete;

    LinkageGpuSchedulerClusterHandle RegisterClusterRfull(const float* R_full,
                                                        int d,
                                                        int n_cols,
                                                        std::string* err);
    bool UnregisterCluster(LinkageGpuSchedulerClusterHandle cluster, std::string* err);

    std::uint64_t SubmitClusterRfullUpdates(LinkageGpuSchedulerClusterHandle cluster,
                                            const int* cols,
                                            const float* r_cols,
                                            int d,
                                            int B,
                                            std::string* err);

    std::shared_future<LinkageGpuSchedulerManyNodesResult>
    SubmitManyNodesTask(LinkageGpuSchedulerManyNodesTask&& task, std::string* err);

    std::shared_future<LinkageGpuSchedulerSingleNodeResult>
    SubmitImmediateSingleNodeTask(LinkageGpuSchedulerSingleNodeTask&& task, std::string* err);

    LinkageGpuSchedulerStats GetStats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stlq
