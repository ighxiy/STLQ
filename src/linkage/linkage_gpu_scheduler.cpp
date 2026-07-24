#include "stlq/linkage/linkage_gpu_scheduler.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#if defined(STLQ_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#include "stlq/linkage/eval_candidates_cuda.h"
#include "stlq/common/logger.h"

namespace stlq {

namespace {

using Clock = std::chrono::steady_clock;

double NowSeconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

constexpr int kManyBatchTargetTasks = 4;
constexpr std::uint64_t kManyBatchTargetNodes = 256;
constexpr std::uint64_t kManyBatchTargetPairs = 8192;
constexpr auto kManyBatchGatherWindow = std::chrono::microseconds(250);
constexpr std::uint64_t kManyDirectMaxNodes = 8;
constexpr std::uint64_t kManyDirectMaxPairs = 512;
constexpr int kSingleBatchTargetTasks = 8;
constexpr std::uint64_t kSingleBatchTargetPairs = 256;
constexpr auto kSingleBatchGatherWindow = std::chrono::microseconds(250);

struct DeviceFloatBuffer {
    float* ptr = nullptr;
    std::size_t size = 0;

    void Reset() {
#if defined(STLQ_ENABLE_CUDA)
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
#endif
        size = 0;
    }

    ~DeviceFloatBuffer() { Reset(); }
};

struct ClusterRfullUpdate {
    std::uint64_t generation = 0;
    int d = 0;
    int B = 0;
    std::vector<int> cols;
    std::vector<float> r_cols;
};

struct ClusterState {
    LinkageGpuSchedulerClusterHandle handle = 0;
    int d = 0;
    int n_cols = 0;
    DeviceFloatBuffer d_rfull;
    std::vector<float> h_rfull;
    std::uint64_t submitted_generation = 0;
    std::uint64_t applied_generation = 0;
    std::deque<ClusterRfullUpdate> pending_updates;
    bool live = true;
    std::uint64_t queued_many = 0;
    std::uint64_t queued_single = 0;
};

struct RegisterRequest {
    LinkageGpuSchedulerClusterHandle handle = 0;
    int d = 0;
    int n_cols = 0;
    std::vector<float> h_rfull;
    std::promise<bool> promise;
};

struct UnregisterRequest {
    LinkageGpuSchedulerClusterHandle handle = 0;
    std::promise<bool> promise;
};

struct UpdateRequest {
    LinkageGpuSchedulerClusterHandle handle = 0;
    ClusterRfullUpdate update;
};

struct ManyTaskRequest {
    LinkageGpuSchedulerManyNodesTask task;
    std::promise<LinkageGpuSchedulerManyNodesResult> promise;
    double submit_ts = 0.0;
};

struct SingleTaskRequest {
    LinkageGpuSchedulerSingleNodeTask task;
    std::promise<LinkageGpuSchedulerSingleNodeResult> promise;
    double submit_ts = 0.0;
};

using IngressItem = std::variant<RegisterRequest, UnregisterRequest, UpdateRequest, ManyTaskRequest, SingleTaskRequest>;

std::string InvalidClusterError(LinkageGpuSchedulerClusterHandle handle) {
    return "LinkageGpuScheduler: invalid cluster handle " + std::to_string(handle) + ".";
}

bool SchedulerDebugVerifyEnabled() {
    const char* v = std::getenv("STLQ_SCHED_VERIFY");
    return v && *v && std::strcmp(v, "0") != 0;
}

bool SameFloatVec(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

const float* ManyTaskXPtr(const LinkageGpuSchedulerManyNodesTask& task) {
    if (task.X_block_shared) {
        return task.X_block_shared->data() + task.X_block_shared_offset;
    }
    return task.X_block.data();
}

const std::uint64_t* ManyTaskSampleIdBasePtr(const LinkageGpuSchedulerManyNodesTask& task) {
    if (task.node_sample_id_base_shared) {
        return task.node_sample_id_base_shared->data() + task.node_sample_id_base_shared_offset;
    }
    return task.node_sample_id_base.data();
}

std::vector<int> CanonicalPairTFromOffsets(const std::vector<int>& offsets) {
    if (offsets.empty()) return {};
    const int B = static_cast<int>(offsets.size()) - 1;
    std::vector<int> out(static_cast<std::size_t>(offsets.back()), 0);
    for (int i = 0; i < B; ++i) {
        const int start = offsets[static_cast<std::size_t>(i)];
        const int stop = offsets[static_cast<std::size_t>(i + 1)];
        for (int p = start; p < stop; ++p) {
            out[static_cast<std::size_t>(p)] = p - start;
        }
    }
    return out;
}

void VerifySingleTaskAgainstLegacy(CudaCtx* ctx,
                                   const LinkageGpuSchedulerSingleNodeTask& task,
                                   const ClusterState& cluster,
                                   const LinkageGpuSchedulerSingleNodeResult& got) {
    if (!task.codebook_large_root) {
        return;
    }
    const bool need_root_u32 =
        !task.codebook_large_root->books.empty() && task.codebook_large_root->books.front().cols > 256;
    std::string err;
    if (!CudaLinkageUploadClusterRfull(ctx, cluster.h_rfull.data(), cluster.d, cluster.n_cols, &err)) {
        throw std::runtime_error("LinkageGpuScheduler verify: upload current cluster R_full failed: " + err);
    }
    const int Kp = static_cast<int>(task.cand_parent_local.size());
    std::vector<int> pair_node(static_cast<std::size_t>(Kp), 0);
    const int offsets[2] = {0, Kp};
    std::vector<int> want_parent(1, -1);
    std::vector<float> want_cost(1, std::numeric_limits<float>::infinity());
    std::vector<float> want_a(static_cast<std::size_t>(task.m), 0.0f);
    bool ok = false;
    std::vector<FullCode> want_B(static_cast<std::size_t>(task.m), 0);
    std::vector<std::uint32_t> want_B_u32;
    if (!need_root_u32) {
        ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRoot(
            ctx,
            *task.codebook_large_root,
            task.xi.data(),
            task.d,
            /*B=*/1,
            task.cand_parent_local.data(),
            pair_node.data(),
            offsets,
            Kp,
            task.icm_iters,
            task.ils_iters,
            task.perturb_k,
            task.seed,
            &task.sample_id_offset,
            want_parent.data(),
            want_cost.data(),
            want_B.data(),
            want_a.data(),
            &err);
    } else {
        want_B_u32.assign(static_cast<std::size_t>(task.m), 0u);
        ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32(
            ctx,
            *task.codebook_large_root,
            task.xi.data(),
            task.d,
            /*B=*/1,
            task.cand_parent_local.data(),
            pair_node.data(),
            offsets,
            Kp,
            task.icm_iters,
            task.ils_iters,
            task.perturb_k,
            task.seed,
            &task.sample_id_offset,
            want_parent.data(),
            want_cost.data(),
            want_B_u32.data(),
            want_a.data(),
            &err);
    }
    if (!ok) {
        throw std::runtime_error("LinkageGpuScheduler verify: legacy single-node eval failed: " + err);
    }
    if (got.best_parent_local != want_parent[0] ||
        std::fabs(got.best_cost - want_cost[0]) > 1e-4f ||
        !SameFloatVec(got.best_a, want_a, 1e-4f)) {
        throw std::runtime_error("LinkageGpuScheduler verify: scheduler single-node parent/cost/a mismatch.");
    }
    if (!need_root_u32) {
        if (got.root_codes_u32 || got.best_B != want_B) {
            throw std::runtime_error("LinkageGpuScheduler verify: scheduler single-node best_B mismatch.");
        }
    } else {
        if (!got.root_codes_u32 ||
            got.best_code0_root != static_cast<RootCode>(want_B_u32[0]) ||
            got.best_B.size() != static_cast<std::size_t>(task.m) ||
            (!got.best_B.empty() && got.best_B[0] != static_cast<FullCode>(0))) {
            throw std::runtime_error("LinkageGpuScheduler verify: scheduler single-node root-code mismatch.");
        }
        for (int l = 1; l < task.m; ++l) {
            if (got.best_B[static_cast<std::size_t>(l)] !=
                static_cast<FullCode>(want_B_u32[static_cast<std::size_t>(l)])) {
                throw std::runtime_error("LinkageGpuScheduler verify: scheduler single-node tail-code mismatch.");
            }
        }
    }
}

void VerifyManyBatchAgainstLegacy(
    CudaCtx* ctx,
    const std::vector<std::shared_ptr<ManyTaskRequest>>& reqs,
    const std::vector<std::shared_ptr<ClusterState>>& clusters,
    const std::vector<int>& task_node_offsets,
    const std::vector<int>& best_parent_super,
    const std::vector<float>& best_cost_super,
    const std::vector<FullCode>& best_B_super,
    const std::vector<float>& best_a_super,
    const std::vector<std::uint32_t>* best_B_super_u32) {
    constexpr float kTol = 1e-4f;
    for (std::size_t idx = 0; idx < reqs.size(); ++idx) {
        const auto& task = reqs[idx]->task;
        const auto& cluster = *clusters[idx];
        if (task.pre_one) {
            const int* pair_t_ptr = task.pair_t_override.empty() ? nullptr : task.pair_t_override.data();
            std::string err;
            std::vector<int> want_parent(static_cast<std::size_t>(task.B), -1);
            std::vector<float> want_cost(static_cast<std::size_t>(task.B), std::numeric_limits<float>::infinity());
            std::vector<float> want_a(static_cast<std::size_t>(task.B) * static_cast<std::size_t>(task.m), 0.0f);
            std::vector<FullCode> want_B(static_cast<std::size_t>(task.B) * static_cast<std::size_t>(task.m), 0);
            const bool ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseEx(
                ctx,
                *task.pre_one,
                ManyTaskXPtr(task),
                task.d,
                task.B,
                task.cand_parent_local_flat.data(),
                task.pair_node.data(),
                task.cand_offsets.data(),
                static_cast<int>(task.cand_parent_local_flat.size()),
                task.icm_iters,
                task.ils_iters,
                task.perturb_k,
                task.seed,
                ManyTaskSampleIdBasePtr(task),
                pair_t_ptr,
                want_parent.data(),
                want_cost.data(),
                want_B.data(),
                want_a.data(),
                &err);
            if (!ok) {
                throw std::runtime_error("LinkageGpuScheduler verify: legacy many-nodes pre_one eval failed: " + err);
            }
            const int node_base = task_node_offsets[idx];
            for (int i = 0; i < task.B; ++i) {
                if (best_parent_super[static_cast<std::size_t>(node_base + i)] != want_parent[static_cast<std::size_t>(i)] ||
                    std::fabs(best_cost_super[static_cast<std::size_t>(node_base + i)] -
                              want_cost[static_cast<std::size_t>(i)]) > kTol) {
                    throw std::runtime_error("LinkageGpuScheduler verify: merged pre_one best parent/cost mismatch.");
                }
            }
            const std::size_t base = static_cast<std::size_t>(node_base) * static_cast<std::size_t>(task.m);
            if (!std::equal(want_B.begin(), want_B.end(), best_B_super.begin() + base)) {
                throw std::runtime_error("LinkageGpuScheduler verify: merged pre_one best_B mismatch.");
            }
            for (std::size_t i = 0; i < want_a.size(); ++i) {
                if (std::fabs(want_a[i] - best_a_super[base + i]) > kTol) {
                    throw std::runtime_error("LinkageGpuScheduler verify: merged pre_one best_a mismatch.");
                }
            }
            continue;
        }
        const bool need_root_u32 =
            task.codebook_large_root && !task.codebook_large_root->books.empty() &&
            task.codebook_large_root->books.front().cols > 256;
        std::string err;
        if (!CudaLinkageUploadClusterRfull(ctx, cluster.h_rfull.data(), cluster.d, cluster.n_cols, &err)) {
            throw std::runtime_error("LinkageGpuScheduler verify: upload current cluster R_full failed: " + err);
        }
        std::vector<int> want_parent(static_cast<std::size_t>(task.B), -1);
        std::vector<float> want_cost(static_cast<std::size_t>(task.B), std::numeric_limits<float>::infinity());
        std::vector<float> want_a(static_cast<std::size_t>(task.B) * static_cast<std::size_t>(task.m), 0.0f);
        const int* pair_t_ptr = task.pair_t_override.empty() ? nullptr : task.pair_t_override.data();
        bool ok = false;
        std::vector<FullCode> want_B;
        std::vector<std::uint32_t> want_B_u32;
        if (!need_root_u32) {
            want_B.assign(static_cast<std::size_t>(task.B) * static_cast<std::size_t>(task.m), 0);
            ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootEx(
                ctx,
                *task.codebook_large_root,
                ManyTaskXPtr(task),
                task.d,
                task.B,
                task.cand_parent_local_flat.data(),
                task.pair_node.data(),
                task.cand_offsets.data(),
                static_cast<int>(task.cand_parent_local_flat.size()),
                task.icm_iters,
                task.ils_iters,
                task.perturb_k,
                task.seed,
                ManyTaskSampleIdBasePtr(task),
                pair_t_ptr,
                want_parent.data(),
                want_cost.data(),
                want_B.data(),
                want_a.data(),
                &err);
        } else {
            want_B_u32.assign(static_cast<std::size_t>(task.B) * static_cast<std::size_t>(task.m), 0u);
            ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootExU32(
                ctx,
                *task.codebook_large_root,
                ManyTaskXPtr(task),
                task.d,
                task.B,
                task.cand_parent_local_flat.data(),
                task.pair_node.data(),
                task.cand_offsets.data(),
                static_cast<int>(task.cand_parent_local_flat.size()),
                task.icm_iters,
                task.ils_iters,
                task.perturb_k,
                task.seed,
                ManyTaskSampleIdBasePtr(task),
                pair_t_ptr,
                want_parent.data(),
                want_cost.data(),
                want_B_u32.data(),
                want_a.data(),
                &err);
        }
        if (!ok) {
            throw std::runtime_error("LinkageGpuScheduler verify: legacy many-nodes eval failed: " + err);
        }
        const int node_base = task_node_offsets[idx];
        for (int i = 0; i < task.B; ++i) {
            if (best_parent_super[static_cast<std::size_t>(node_base + i)] != want_parent[static_cast<std::size_t>(i)] ||
                std::fabs(best_cost_super[static_cast<std::size_t>(node_base + i)] -
                          want_cost[static_cast<std::size_t>(i)]) > kTol) {
                throw std::runtime_error("LinkageGpuScheduler verify: merged best parent/cost mismatch.");
            }
        }
        const std::size_t base = static_cast<std::size_t>(node_base) * static_cast<std::size_t>(task.m);
        if (!need_root_u32) {
            if (!std::equal(want_B.begin(), want_B.end(), best_B_super.begin() + base)) {
                throw std::runtime_error("LinkageGpuScheduler verify: merged best_B mismatch.");
            }
        } else {
            if (!best_B_super_u32) {
                throw std::runtime_error("LinkageGpuScheduler verify: missing merged U32 codes.");
            }
            if (!std::equal(want_B_u32.begin(), want_B_u32.end(), best_B_super_u32->begin() + base)) {
                throw std::runtime_error("LinkageGpuScheduler verify: merged best_B_u32 mismatch.");
            }
        }
        for (std::size_t i = 0; i < want_a.size(); ++i) {
            if (std::fabs(want_a[i] - best_a_super[base + i]) > kTol) {
                throw std::runtime_error("LinkageGpuScheduler verify: merged best_a mismatch.");
            }
        }
    }
}

template <typename T>
void SetPromiseError(std::promise<T>* promise, const std::string& err) {
    try {
        throw std::runtime_error(err);
    } catch (...) {
        promise->set_exception(std::current_exception());
    }
}

}  // namespace

struct LinkageGpuScheduler::Impl {
    explicit Impl(const CudaPoolConfig& cfg_in)
        : cfg(cfg_in), pool(1, cfg_in), worker(&Impl::WorkerLoop, this) {}

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stopping = true;
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    CudaPoolConfig cfg;
    CudaStreamKernelsPool pool;

    mutable std::mutex mu;
    std::condition_variable cv;
    bool stopping = false;
    bool worker_ready = false;
    std::thread worker;
    CudaCtx* worker_ctx = nullptr;

    std::uint64_t next_cluster_handle = 1;
    std::unordered_map<LinkageGpuSchedulerClusterHandle, std::shared_ptr<ClusterState>> clusters;
    std::deque<IngressItem> ingress;
    std::deque<RegisterRequest> pending_register;
    std::deque<UnregisterRequest> pending_unregister;
    std::deque<std::shared_ptr<ManyTaskRequest>> pending_many;
    std::deque<std::shared_ptr<SingleTaskRequest>> pending_single;
    LinkageGpuSchedulerStats stats;

    bool AnyPendingUpdatesUnsafe() const {
        for (const auto& entry : clusters) {
            if (!entry.second->pending_updates.empty()) {
                return true;
            }
        }
        return false;
    }

    bool HasPendingUnsafe() const {
        return !ingress.empty() || !pending_register.empty() || !pending_unregister.empty() ||
               !pending_many.empty() || !pending_single.empty() || AnyPendingUpdatesUnsafe();
    }

    std::shared_ptr<ClusterState> FindClusterUnsafe(LinkageGpuSchedulerClusterHandle handle) const {
        auto it = clusters.find(handle);
        if (it == clusters.end()) {
            return nullptr;
        }
        return it->second;
    }

    void DrainIngressUnsafe() {
        while (!ingress.empty()) {
            IngressItem item = std::move(ingress.front());
            ingress.pop_front();
            std::visit(
                [&](auto&& op) {
                    using T = std::decay_t<decltype(op)>;
                    if constexpr (std::is_same_v<T, RegisterRequest>) {
                        pending_register.push_back(std::move(op));
                    } else if constexpr (std::is_same_v<T, UnregisterRequest>) {
                        pending_unregister.push_back(std::move(op));
                    } else if constexpr (std::is_same_v<T, UpdateRequest>) {
                        auto cluster = FindClusterUnsafe(op.handle);
                        if (cluster && cluster->live) {
                            cluster->pending_updates.push_back(std::move(op.update));
                        }
                    } else if constexpr (std::is_same_v<T, ManyTaskRequest>) {
                        auto req = std::make_shared<ManyTaskRequest>(std::move(op));
                        auto cluster = FindClusterUnsafe(req->task.cluster);
                        if (!cluster || !cluster->live) {
                            SetPromiseError(&req->promise, InvalidClusterError(req->task.cluster));
                            return;
                        }
                        cluster->queued_many += 1;
                        pending_many.push_back(std::move(req));
                    } else if constexpr (std::is_same_v<T, SingleTaskRequest>) {
                        auto req = std::make_shared<SingleTaskRequest>(std::move(op));
                        auto cluster = FindClusterUnsafe(req->task.cluster);
                        if (!cluster || !cluster->live) {
                            SetPromiseError(&req->promise, InvalidClusterError(req->task.cluster));
                            return;
                        }
                        cluster->queued_single += 1;
                        pending_single.push_back(std::move(req));
                    }
                },
                std::move(item));
        }
    }

    void EnsureWorkerCtx() {
        if (worker_ctx) {
            return;
        }
        worker_ctx = pool.Acquire();
        if (!worker_ctx) {
            throw std::runtime_error("LinkageGpuScheduler: failed to acquire worker CUDA context.");
        }
    }

    void UploadFullClusterRfull(ClusterState& cluster) {
#if defined(STLQ_ENABLE_CUDA)
        if (cluster.n_cols <= 0) {
            cluster.d_rfull.Reset();
            return;
        }
        const std::size_t elems = static_cast<std::size_t>(cluster.d) * static_cast<std::size_t>(cluster.n_cols);
        if (cluster.d_rfull.size != elems) {
            cluster.d_rfull.Reset();
            cluster.d_rfull.size = elems;
            cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&cluster.d_rfull.ptr), elems * sizeof(float));
            if (st != cudaSuccess) {
                cluster.d_rfull.ptr = nullptr;
                cluster.d_rfull.size = 0;
                throw std::runtime_error(std::string("LinkageGpuScheduler: cudaMalloc(d_rfull) failed: ") +
                                         cudaGetErrorString(st));
            }
        }
        if (!cluster.h_rfull.empty()) {
            cudaError_t st = cudaMemcpyAsync(cluster.d_rfull.ptr,
                                             cluster.h_rfull.data(),
                                             elems * sizeof(float),
                                             cudaMemcpyHostToDevice,
                                             worker_ctx->stream);
            if (st != cudaSuccess) {
                throw std::runtime_error(std::string("LinkageGpuScheduler: cudaMemcpyAsync(initial R_full) failed: ") +
                                         cudaGetErrorString(st));
            }
            st = cudaStreamSynchronize(worker_ctx->stream);
            if (st != cudaSuccess) {
                throw std::runtime_error(std::string("LinkageGpuScheduler: cudaStreamSynchronize(initial R_full) failed: ") +
                                         cudaGetErrorString(st));
            }
        }
#else
        (void)cluster;
        throw std::runtime_error("LinkageGpuScheduler: CUDA disabled.");
#endif
    }

    void ApplyOneUpdate(ClusterState& cluster, ClusterRfullUpdate update) {
#if defined(STLQ_ENABLE_CUDA)
        if (update.d != cluster.d) {
            throw std::runtime_error("LinkageGpuScheduler: R_full update dimension mismatch.");
        }
        if (update.B < 0 || static_cast<int>(update.cols.size()) != update.B) {
            throw std::runtime_error("LinkageGpuScheduler: invalid update column count.");
        }
        if (static_cast<int>(update.r_cols.size()) != update.B * cluster.d) {
            throw std::runtime_error("LinkageGpuScheduler: invalid packed update size.");
        }
        for (int idx = 0; idx < update.B; ++idx) {
            const int col = update.cols[static_cast<std::size_t>(idx)];
            if (col < 0 || col >= cluster.n_cols) {
                throw std::runtime_error("LinkageGpuScheduler: R_full update col out of range.");
            }
            std::memcpy(cluster.h_rfull.data() + static_cast<std::size_t>(col) * static_cast<std::size_t>(cluster.d),
                        update.r_cols.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(cluster.d),
                        sizeof(float) * static_cast<std::size_t>(cluster.d));
            const cudaError_t st =
                cudaMemcpyAsync(cluster.d_rfull.ptr + static_cast<std::size_t>(col) * static_cast<std::size_t>(cluster.d),
                                update.r_cols.data() + static_cast<std::size_t>(idx) * static_cast<std::size_t>(cluster.d),
                                sizeof(float) * static_cast<std::size_t>(cluster.d),
                                cudaMemcpyHostToDevice,
                                worker_ctx->stream);
            if (st != cudaSuccess) {
                throw std::runtime_error(std::string("LinkageGpuScheduler: cudaMemcpyAsync(R_full update) failed: ") +
                                         cudaGetErrorString(st));
            }
        }
        const cudaError_t sync_st = cudaStreamSynchronize(worker_ctx->stream);
        if (sync_st != cudaSuccess) {
            throw std::runtime_error(std::string("LinkageGpuScheduler: cudaStreamSynchronize(R_full update) failed: ") +
                                     cudaGetErrorString(sync_st));
        }
#else
        (void)cluster;
        (void)update;
        throw std::runtime_error("LinkageGpuScheduler: CUDA disabled.");
#endif
    }

    void ExecuteRegister(RegisterRequest op) {
        auto cluster = std::make_shared<ClusterState>();
        cluster->handle = op.handle;
        cluster->d = op.d;
        cluster->n_cols = op.n_cols;
        cluster->h_rfull = std::move(op.h_rfull);
        UploadFullClusterRfull(*cluster);
        {
            std::lock_guard<std::mutex> lock(mu);
            clusters.emplace(cluster->handle, std::move(cluster));
        }
        op.promise.set_value(true);
    }

    bool ClusterHasPendingTasksUnsafe(LinkageGpuSchedulerClusterHandle handle) const {
        for (const auto& req : pending_many) {
            if (req->task.cluster == handle) {
                return true;
            }
        }
        for (const auto& req : pending_single) {
            if (req->task.cluster == handle) {
                return true;
            }
        }
        return false;
    }

    void ExecuteUnregister(UnregisterRequest op) {
        std::shared_ptr<ClusterState> cluster;
        {
            std::lock_guard<std::mutex> lock(mu);
            cluster = FindClusterUnsafe(op.handle);
            if (!cluster) {
                op.promise.set_value(true);
                return;
            }
            if (!cluster->pending_updates.empty() || cluster->queued_many > 0 || cluster->queued_single > 0 ||
                ClusterHasPendingTasksUnsafe(op.handle)) {
                pending_unregister.push_back(std::move(op));
                return;
            }
            cluster->live = false;
            clusters.erase(op.handle);
        }
        cluster->d_rfull.Reset();
        op.promise.set_value(true);
    }

    bool ApplyOnePendingUpdate() {
        std::shared_ptr<ClusterState> cluster;
        ClusterRfullUpdate update;
        {
            std::lock_guard<std::mutex> lock(mu);
            for (auto& entry : clusters) {
                if (!entry.second->pending_updates.empty()) {
                    cluster = entry.second;
                    update = std::move(cluster->pending_updates.front());
                    cluster->pending_updates.pop_front();
                    break;
                }
            }
        }
        if (!cluster) {
            return false;
        }
        ApplyOneUpdate(*cluster, update);
        {
            std::lock_guard<std::mutex> lock(mu);
            cluster->applied_generation = std::max(cluster->applied_generation, update.generation);
            stats.update_ops_applied += 1;
        }
        return true;
    }

    std::shared_ptr<ManyTaskRequest> PopReadyManyTask() {
        std::lock_guard<std::mutex> lock(mu);
        for (auto it = pending_many.begin(); it != pending_many.end(); ++it) {
            const auto cluster = FindClusterUnsafe((*it)->task.cluster);
            if (!cluster || !cluster->live) {
                SetPromiseError(&(*it)->promise, InvalidClusterError((*it)->task.cluster));
                pending_many.erase(it);
                return nullptr;
            }
            if (cluster->applied_generation >= (*it)->task.required_rfull_generation) {
                auto out = *it;
                pending_many.erase(it);
                return out;
            }
        }
        return nullptr;
    }

    static bool CompatibleManyTasks(const LinkageGpuSchedulerManyNodesTask& a,
                                    const LinkageGpuSchedulerManyNodesTask& b) {
        return a.pre_one == b.pre_one &&
               a.codebook_large_root == b.codebook_large_root &&
               a.d == b.d &&
               a.B >= 0 && b.B >= 0 &&
               a.m == b.m &&
               a.icm_iters == b.icm_iters &&
               a.ils_iters == b.ils_iters &&
               a.perturb_k == b.perturb_k;
    }

    struct ReadyManyBatch {
        std::vector<std::shared_ptr<ManyTaskRequest>> reqs;
        std::vector<std::shared_ptr<ClusterState>> clusters;
    };

    struct ManyBatchShape {
        std::uint64_t tasks = 0;
        std::uint64_t nodes = 0;
        std::uint64_t pairs = 0;
    };

    static ManyBatchShape ComputeManyBatchShape(const ReadyManyBatch& batch) {
        ManyBatchShape shape;
        shape.tasks = static_cast<std::uint64_t>(batch.reqs.size());
        for (const auto& req : batch.reqs) {
            shape.nodes += static_cast<std::uint64_t>(std::max(0, req->task.B));
            shape.pairs += static_cast<std::uint64_t>(req->task.cand_parent_local_flat.size());
        }
        return shape;
    }

    static bool ShouldWaitForMoreManyTasks(const ReadyManyBatch& batch) {
        if (batch.reqs.empty()) {
            return false;
        }
        const ManyBatchShape shape = ComputeManyBatchShape(batch);
        return shape.tasks < static_cast<std::uint64_t>(kManyBatchTargetTasks) &&
               shape.nodes < kManyBatchTargetNodes &&
               shape.pairs < kManyBatchTargetPairs;
    }

    static bool ShouldExecuteManyDirect(const ReadyManyBatch& batch) {
        if (batch.reqs.size() != 1) {
            return false;
        }
        const ManyBatchShape shape = ComputeManyBatchShape(batch);
        return shape.nodes <= kManyDirectMaxNodes &&
               shape.pairs <= kManyDirectMaxPairs;
    }

    void AppendCompatibleReadyManyTasksUnsafe(ReadyManyBatch* batch) {
        if (!batch) {
            return;
        }
        for (auto it = pending_many.begin(); it != pending_many.end();) {
            const auto cluster = FindClusterUnsafe((*it)->task.cluster);
            if (!cluster || !cluster->live) {
                SetPromiseError(&(*it)->promise, InvalidClusterError((*it)->task.cluster));
                it = pending_many.erase(it);
                continue;
            }
            if (cluster->applied_generation < (*it)->task.required_rfull_generation) {
                ++it;
                continue;
            }
            if (batch->reqs.empty()) {
                batch->reqs.push_back(*it);
                batch->clusters.push_back(cluster);
                it = pending_many.erase(it);
                continue;
            }
            if (CompatibleManyTasks(batch->reqs.front()->task, (*it)->task)) {
                batch->reqs.push_back(*it);
                batch->clusters.push_back(cluster);
                it = pending_many.erase(it);
                continue;
            }
            ++it;
        }
    }

    ReadyManyBatch PopReadyManyBatch() {
        ReadyManyBatch batch;
        std::lock_guard<std::mutex> lock(mu);
        AppendCompatibleReadyManyTasksUnsafe(&batch);
        return batch;
    }

    void GrowReadyManyBatch(ReadyManyBatch* batch) {
        if (!batch || batch->reqs.empty() || !ShouldWaitForMoreManyTasks(*batch)) {
            return;
        }
        const auto deadline = Clock::now() + kManyBatchGatherWindow;
        while (ShouldWaitForMoreManyTasks(*batch)) {
            std::unique_lock<std::mutex> lock(mu);
            if (stopping) {
                return;
            }
            if (!pending_register.empty() || !pending_unregister.empty()) {
                return;
            }
            cv.wait_until(lock, deadline, [&]() { return stopping || !ingress.empty(); });
            DrainIngressUnsafe();
            if (stopping) {
                return;
            }
            if (!pending_register.empty() || !pending_unregister.empty()) {
                return;
            }
            AppendCompatibleReadyManyTasksUnsafe(batch);
            if (Clock::now() >= deadline) {
                return;
            }
        }
    }

    static bool CompatibleSingleTasks(const LinkageGpuSchedulerSingleNodeTask& a,
                                      const LinkageGpuSchedulerSingleNodeTask& b) {
        return a.pre_one == b.pre_one &&
               a.codebook_large_root == b.codebook_large_root &&
               a.d == b.d &&
               a.m == b.m &&
               a.icm_iters == b.icm_iters &&
               a.ils_iters == b.ils_iters &&
               a.perturb_k == b.perturb_k;
    }

    struct ReadySingleBatch {
        std::vector<std::shared_ptr<SingleTaskRequest>> reqs;
        std::vector<std::shared_ptr<ClusterState>> clusters;
    };

    static bool ShouldWaitForMoreSingleTasks(const ReadySingleBatch& batch) {
        if (batch.reqs.empty()) {
            return false;
        }
        std::uint64_t pairs = 0;
        for (const auto& req : batch.reqs) {
            pairs += static_cast<std::uint64_t>(req->task.cand_parent_local.size());
        }
        return batch.reqs.size() < static_cast<std::size_t>(kSingleBatchTargetTasks) &&
               pairs < kSingleBatchTargetPairs;
    }

    void AppendCompatibleReadySingleTasksUnsafe(ReadySingleBatch* batch) {
        if (!batch) {
            return;
        }
        for (auto it = pending_single.begin(); it != pending_single.end();) {
            const auto cluster = FindClusterUnsafe((*it)->task.cluster);
            if (!cluster || !cluster->live) {
                SetPromiseError(&(*it)->promise, InvalidClusterError((*it)->task.cluster));
                it = pending_single.erase(it);
                continue;
            }
            if (cluster->applied_generation < (*it)->task.required_rfull_generation) {
                ++it;
                continue;
            }
            if (batch->reqs.empty()) {
                batch->reqs.push_back(*it);
                batch->clusters.push_back(cluster);
                it = pending_single.erase(it);
                continue;
            }
            if (CompatibleSingleTasks(batch->reqs.front()->task, (*it)->task)) {
                batch->reqs.push_back(*it);
                batch->clusters.push_back(cluster);
                it = pending_single.erase(it);
                continue;
            }
            ++it;
        }
    }

    ReadySingleBatch PopReadySingleBatch() {
        ReadySingleBatch batch;
        std::lock_guard<std::mutex> lock(mu);
        AppendCompatibleReadySingleTasksUnsafe(&batch);
        return batch;
    }

    void GrowReadySingleBatch(ReadySingleBatch* batch) {
        if (!batch || batch->reqs.empty() || !ShouldWaitForMoreSingleTasks(*batch)) {
            return;
        }
        const auto deadline = Clock::now() + kSingleBatchGatherWindow;
        while (ShouldWaitForMoreSingleTasks(*batch)) {
            std::unique_lock<std::mutex> lock(mu);
            if (stopping) {
                return;
            }
            if (!pending_register.empty() || !pending_unregister.empty()) {
                return;
            }
            cv.wait_until(lock, deadline, [&]() { return stopping || !ingress.empty(); });
            DrainIngressUnsafe();
            if (stopping) {
                return;
            }
            if (!pending_register.empty() || !pending_unregister.empty()) {
                return;
            }
            AppendCompatibleReadySingleTasksUnsafe(batch);
            if (Clock::now() >= deadline) {
                return;
            }
        }
    }

    int ApplyPendingUpdatesBurst() {
        int applied = 0;
        while (ApplyOnePendingUpdate()) {
            ++applied;
        }
        return applied;
    }

    std::shared_ptr<SingleTaskRequest> PopReadySingleTask() {
        std::lock_guard<std::mutex> lock(mu);
        for (auto it = pending_single.begin(); it != pending_single.end(); ++it) {
            const auto cluster = FindClusterUnsafe((*it)->task.cluster);
            if (!cluster || !cluster->live) {
                SetPromiseError(&(*it)->promise, InvalidClusterError((*it)->task.cluster));
                pending_single.erase(it);
                return nullptr;
            }
            if (cluster->applied_generation >= (*it)->task.required_rfull_generation) {
                auto out = *it;
                pending_single.erase(it);
                return out;
            }
        }
        return nullptr;
    }

    void UpdateQueueWaitStats(double submit_ts) {
        const double wait_s = std::max(0.0, NowSeconds() - submit_ts);
        std::lock_guard<std::mutex> lock(mu);
        stats.queue_wait_total_s += wait_s;
        stats.queue_wait_max_s = std::max(stats.queue_wait_max_s, wait_s);
    }

    void ExecuteManyTask(const std::shared_ptr<ManyTaskRequest>& req) {
        UpdateQueueWaitStats(req->submit_ts);
        std::shared_ptr<ClusterState> cluster;
        {
            std::lock_guard<std::mutex> lock(mu);
            cluster = FindClusterUnsafe(req->task.cluster);
        }
        if (!cluster || !cluster->live) {
            SetPromiseError(&req->promise, InvalidClusterError(req->task.cluster));
            return;
        }
        if (!req->task.pre_one && !req->task.codebook_large_root) {
            SetPromiseError(&req->promise, "LinkageGpuScheduler: many-nodes task missing evaluator description.");
            return;
        }

        LinkageGpuSchedulerManyNodesResult result;
        result.B = req->task.B;
        result.m = req->task.m;
        result.root_codes_u32 = false;
        result.best_parent_local.resize(static_cast<std::size_t>(req->task.B));
        result.best_cost.resize(static_cast<std::size_t>(req->task.B));
        result.best_B.resize(static_cast<std::size_t>(req->task.B) * static_cast<std::size_t>(req->task.m));
        result.best_a.resize(static_cast<std::size_t>(req->task.B) * static_cast<std::size_t>(req->task.m));

        std::string err;
        bool ok = false;
        const bool use_pre_one = (req->task.pre_one != nullptr);
        const bool need_root_u32 =
            !use_pre_one &&
            req->task.codebook_large_root &&
            !req->task.codebook_large_root->books.empty() &&
            req->task.codebook_large_root->books.front().cols > 256;
        std::vector<int> task_node_offsets = {0, req->task.B};
        std::vector<int> task_pair_offsets = {0, static_cast<int>(req->task.cand_parent_local_flat.size())};
        std::vector<const float*> task_rfull_ptrs = {cluster->d_rfull.ptr};
        std::vector<std::uint32_t> task_seeds = {req->task.seed};
        std::vector<int> pair_task(req->task.cand_parent_local_flat.size(), 0);
        std::vector<int> pair_t_override;
        if (req->task.pair_t_override.empty()) {
            pair_t_override = CanonicalPairTFromOffsets(req->task.cand_offsets);
        } else {
            pair_t_override = req->task.pair_t_override;
        }
        std::vector<std::uint32_t> best_B_u32;
        if (need_root_u32) {
            best_B_u32.resize(static_cast<std::size_t>(req->task.B) * static_cast<std::size_t>(req->task.m), 0u);
            result.root_codes_u32 = true;
            result.best_code0_root.resize(static_cast<std::size_t>(req->task.B), 0);
        }
        ok = use_pre_one
                 ? EvaluateParentCandidatesManyTasksBatchCudaDeviceRfullWithNodeSampleIdBaseEx(
                       worker_ctx,
                       *req->task.pre_one,
                       ManyTaskXPtr(req->task),
                       req->task.d,
                       req->task.B,
                       /*task_count=*/1,
                       task_node_offsets.data(),
                       task_pair_offsets.data(),
                       task_rfull_ptrs.data(),
                       task_seeds.data(),
                       req->task.cand_parent_local_flat.data(),
                       req->task.pair_node.data(),
                       pair_task.data(),
                       req->task.cand_offsets.data(),
                       static_cast<int>(req->task.cand_parent_local_flat.size()),
                       ManyTaskSampleIdBasePtr(req->task),
                       pair_t_override.data(),
                       req->task.icm_iters,
                       req->task.ils_iters,
                       req->task.perturb_k,
                       result.best_parent_local.data(),
                       result.best_cost.data(),
                       result.best_B.data(),
                       result.best_a.data(),
                       &err)
                 : need_root_u32
                       ? EvaluateParentCandidatesManyTasksBatchCudaDeviceRfullMetaOnlyU32(
                             worker_ctx,
                             *req->task.codebook_large_root,
                             ManyTaskXPtr(req->task),
                             req->task.d,
                             req->task.B,
                             /*task_count=*/1,
                             task_node_offsets.data(),
                             task_pair_offsets.data(),
                             task_rfull_ptrs.data(),
                             task_seeds.data(),
                             req->task.cand_parent_local_flat.data(),
                             req->task.pair_node.data(),
                             pair_task.data(),
                             req->task.cand_offsets.data(),
                             static_cast<int>(req->task.cand_parent_local_flat.size()),
                             ManyTaskSampleIdBasePtr(req->task),
                             pair_t_override.data(),
                             req->task.icm_iters,
                             req->task.ils_iters,
                             req->task.perturb_k,
                             result.best_parent_local.data(),
                             result.best_cost.data(),
                             best_B_u32.data(),
                             result.best_a.data(),
                             &err)
                       : EvaluateParentCandidatesManyTasksBatchCudaDeviceRfullMetaOnly(
                             worker_ctx,
                             *req->task.codebook_large_root,
                             ManyTaskXPtr(req->task),
                             req->task.d,
                             req->task.B,
                             /*task_count=*/1,
                             task_node_offsets.data(),
                             task_pair_offsets.data(),
                             task_rfull_ptrs.data(),
                             task_seeds.data(),
                             req->task.cand_parent_local_flat.data(),
                             req->task.pair_node.data(),
                             pair_task.data(),
                             req->task.cand_offsets.data(),
                             static_cast<int>(req->task.cand_parent_local_flat.size()),
                             ManyTaskSampleIdBasePtr(req->task),
                             pair_t_override.data(),
                             req->task.icm_iters,
                             req->task.ils_iters,
                             req->task.perturb_k,
                             result.best_parent_local.data(),
                             result.best_cost.data(),
                             result.best_B.data(),
                             result.best_a.data(),
                             &err);
        if (!ok) {
            SetPromiseError(&req->promise, "LinkageGpuScheduler: many-nodes eval failed: " + err);
            return;
        }
        if (need_root_u32) {
            for (int i = 0; i < req->task.B; ++i) {
                const std::size_t base =
                    static_cast<std::size_t>(i) * static_cast<std::size_t>(req->task.m);
                result.best_code0_root[static_cast<std::size_t>(i)] =
                    static_cast<RootCode>(best_B_u32[base]);
                result.best_B[base] = static_cast<FullCode>(0);
                for (int l = 1; l < req->task.m; ++l) {
                    result.best_B[base + static_cast<std::size_t>(l)] =
                        static_cast<FullCode>(best_B_u32[base + static_cast<std::size_t>(l)]);
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(mu);
            cluster->queued_many -= 1;
            stats.many_launches += 1;
            stats.many_direct_launches += 1;
            stats.many_direct_tasks += 1;
            stats.tasks_merged_total += 1;
            stats.tasks_merged_max = std::max<std::uint64_t>(stats.tasks_merged_max, 1);
            stats.nodes_total += static_cast<std::uint64_t>(std::max(0, req->task.B));
            stats.nodes_max = std::max<std::uint64_t>(stats.nodes_max, static_cast<std::uint64_t>(std::max(0, req->task.B)));
            stats.pairs_total += static_cast<std::uint64_t>(req->task.cand_parent_local_flat.size());
            stats.pairs_max = std::max<std::uint64_t>(stats.pairs_max,
                                                      static_cast<std::uint64_t>(req->task.cand_parent_local_flat.size()));
        }
        req->promise.set_value(std::move(result));
    }

    void ExecuteManyBatch(ReadyManyBatch batch) {
        if (batch.reqs.empty()) {
            return;
        }
        UpdateQueueWaitStats(batch.reqs.front()->submit_ts);
        for (std::size_t i = 1; i < batch.reqs.size(); ++i) {
            UpdateQueueWaitStats(batch.reqs[i]->submit_ts);
        }

        const auto& first = batch.reqs.front()->task;

        for (const auto& req : batch.reqs) {
            if (!req->task.pre_one && !req->task.codebook_large_root) {
                SetPromiseError(&req->promise, "LinkageGpuScheduler: many-nodes task missing evaluator description.");
                return;
            }
        }

        const int task_count = static_cast<int>(batch.reqs.size());
        int B_total = 0;
        int Npairs = 0;
        for (const auto& req : batch.reqs) {
            B_total += req->task.B;
            Npairs += static_cast<int>(req->task.cand_parent_local_flat.size());
        }

        const int d = first.d;
        const int m = first.m;
        const bool use_pre_one = (first.pre_one != nullptr);
        const bool need_root_u32 =
            !use_pre_one &&
            !first.codebook_large_root->books.empty() &&
            first.codebook_large_root->books.front().cols > 256;

        std::vector<int> task_node_offsets(static_cast<std::size_t>(task_count + 1), 0);
        std::vector<int> task_pair_offsets(static_cast<std::size_t>(task_count + 1), 0);
        std::vector<float> X_super(static_cast<std::size_t>(d) * static_cast<std::size_t>(B_total), 0.0f);
        std::vector<int> cand_parent_local_flat(static_cast<std::size_t>(Npairs), 0);
        std::vector<int> pair_node_super(static_cast<std::size_t>(Npairs), 0);
        std::vector<int> pair_task_super(static_cast<std::size_t>(Npairs), 0);
        std::vector<int> cand_offsets_super(static_cast<std::size_t>(B_total + 1), 0);
        std::vector<std::uint64_t> node_sample_id_base_super(static_cast<std::size_t>(B_total), 0);
        std::vector<int> pair_t_override_super(static_cast<std::size_t>(Npairs), 0);
        std::vector<const float*> task_rfull_ptrs_host(static_cast<std::size_t>(task_count), nullptr);
        std::vector<std::uint32_t> task_seeds_host(static_cast<std::size_t>(task_count), 0u);

        int node_base = 0;
        int pair_base = 0;
        for (int t = 0; t < task_count; ++t) {
            const auto& req = batch.reqs[static_cast<std::size_t>(t)];
            const auto& cluster = batch.clusters[static_cast<std::size_t>(t)];
            const auto& task = req->task;
            task_node_offsets[static_cast<std::size_t>(t)] = node_base;
            task_pair_offsets[static_cast<std::size_t>(t)] = pair_base;
            task_rfull_ptrs_host[static_cast<std::size_t>(t)] = cluster->d_rfull.ptr;
            task_seeds_host[static_cast<std::size_t>(t)] = task.seed;

            const std::size_t x_elems =
                static_cast<std::size_t>(d) * static_cast<std::size_t>(task.B);
            std::memcpy(X_super.data() + static_cast<std::size_t>(d) * static_cast<std::size_t>(node_base),
                        ManyTaskXPtr(task),
                        x_elems * sizeof(float));
            std::copy(task.cand_parent_local_flat.begin(),
                      task.cand_parent_local_flat.end(),
                      cand_parent_local_flat.begin() + static_cast<std::size_t>(pair_base));
            std::memcpy(node_sample_id_base_super.data() + static_cast<std::size_t>(node_base),
                        ManyTaskSampleIdBasePtr(task),
                        sizeof(std::uint64_t) * static_cast<std::size_t>(task.B));

            std::vector<int> canonical_pair_t;
            const std::vector<int>* pair_t_src = &task.pair_t_override;
            if (task.pair_t_override.empty()) {
                canonical_pair_t = CanonicalPairTFromOffsets(task.cand_offsets);
                pair_t_src = &canonical_pair_t;
            }
            std::copy(pair_t_src->begin(),
                      pair_t_src->end(),
                      pair_t_override_super.begin() + static_cast<std::size_t>(pair_base));

            for (int i = 0; i < task.B; ++i) {
                cand_offsets_super[static_cast<std::size_t>(node_base + i)] =
                    pair_base + task.cand_offsets[static_cast<std::size_t>(i)];
            }
            cand_offsets_super[static_cast<std::size_t>(node_base + task.B)] =
                pair_base + task.cand_offsets[static_cast<std::size_t>(task.B)];

            const int local_pairs = static_cast<int>(task.cand_parent_local_flat.size());
            for (int p = 0; p < local_pairs; ++p) {
                pair_node_super[static_cast<std::size_t>(pair_base + p)] =
                    node_base + task.pair_node[static_cast<std::size_t>(p)];
                pair_task_super[static_cast<std::size_t>(pair_base + p)] = t;
            }

            node_base += task.B;
            pair_base += local_pairs;
        }
        task_node_offsets[static_cast<std::size_t>(task_count)] = B_total;
        task_pair_offsets[static_cast<std::size_t>(task_count)] = Npairs;

        std::vector<int> best_parent_super(static_cast<std::size_t>(B_total), -1);
        std::vector<float> best_cost_super(static_cast<std::size_t>(B_total),
                                           std::numeric_limits<float>::infinity());
        std::vector<float> best_a_super(static_cast<std::size_t>(B_total) * static_cast<std::size_t>(m), 0.0f);
        std::vector<FullCode> best_B_super(static_cast<std::size_t>(B_total) * static_cast<std::size_t>(m), 0);
        std::vector<std::uint32_t> best_B_super_u32;
        if (need_root_u32) {
            best_B_super_u32.resize(static_cast<std::size_t>(B_total) * static_cast<std::size_t>(m), 0u);
        }

        std::string err;
        const bool ok = use_pre_one
                            ? EvaluateParentCandidatesManyTasksBatchCudaDeviceRfullWithNodeSampleIdBaseEx(
                                  worker_ctx,
                                  *first.pre_one,
                                  X_super.data(),
                                  d,
                                  B_total,
                                  task_count,
                                  task_node_offsets.data(),
                                  task_pair_offsets.data(),
                                  task_rfull_ptrs_host.data(),
                                  task_seeds_host.data(),
                                  cand_parent_local_flat.data(),
                                  pair_node_super.data(),
                                  pair_task_super.data(),
                                  cand_offsets_super.data(),
                                  Npairs,
                                  node_sample_id_base_super.data(),
                                  pair_t_override_super.data(),
                                  first.icm_iters,
                                  first.ils_iters,
                                  first.perturb_k,
                                  best_parent_super.data(),
                                  best_cost_super.data(),
                                  best_B_super.data(),
                                  best_a_super.data(),
                                  &err)
                            : need_root_u32
                            ? EvaluateParentCandidatesManyTasksBatchCudaDeviceRfullMetaOnlyU32(
                                  worker_ctx,
                                  *first.codebook_large_root,
                                  X_super.data(),
                                  d,
                                  B_total,
                                  task_count,
                                  task_node_offsets.data(),
                                  task_pair_offsets.data(),
                                  task_rfull_ptrs_host.data(),
                                  task_seeds_host.data(),
                                  cand_parent_local_flat.data(),
                                  pair_node_super.data(),
                                  pair_task_super.data(),
                                  cand_offsets_super.data(),
                                  Npairs,
                                  node_sample_id_base_super.data(),
                                  pair_t_override_super.data(),
                                  first.icm_iters,
                                  first.ils_iters,
                                  first.perturb_k,
                                  best_parent_super.data(),
                                  best_cost_super.data(),
                                  best_B_super_u32.data(),
                                  best_a_super.data(),
                                  &err)
                            : EvaluateParentCandidatesManyTasksBatchCudaDeviceRfullMetaOnly(
                                  worker_ctx,
                                  *first.codebook_large_root,
                                  X_super.data(),
                                  d,
                                  B_total,
                                  task_count,
                                  task_node_offsets.data(),
                                  task_pair_offsets.data(),
                                  task_rfull_ptrs_host.data(),
                                  task_seeds_host.data(),
                                  cand_parent_local_flat.data(),
                                  pair_node_super.data(),
                                  pair_task_super.data(),
                                  cand_offsets_super.data(),
                                  Npairs,
                                  node_sample_id_base_super.data(),
                                  pair_t_override_super.data(),
                                  first.icm_iters,
                                  first.ils_iters,
                                  first.perturb_k,
                                  best_parent_super.data(),
                                  best_cost_super.data(),
                                  best_B_super.data(),
                                  best_a_super.data(),
                                  &err);
        if (!ok) {
            for (auto& req : batch.reqs) {
                SetPromiseError(&req->promise, "LinkageGpuScheduler: merged many-nodes eval failed: " + err);
            }
            return;
        }

        if (SchedulerDebugVerifyEnabled()) {
            VerifyManyBatchAgainstLegacy(worker_ctx,
                                         batch.reqs,
                                         batch.clusters,
                                         task_node_offsets,
                                         best_parent_super,
                                         best_cost_super,
                                         best_B_super,
                                         best_a_super,
                                         need_root_u32 ? &best_B_super_u32 : nullptr);
        }

        for (int t = 0; t < task_count; ++t) {
            const auto& req = batch.reqs[static_cast<std::size_t>(t)];
            const int task_B = req->task.B;
            const int task_node_base = task_node_offsets[static_cast<std::size_t>(t)];
            LinkageGpuSchedulerManyNodesResult result;
            result.B = task_B;
            result.m = m;
            result.root_codes_u32 = need_root_u32;
            result.best_parent_local.resize(static_cast<std::size_t>(task_B));
            result.best_cost.resize(static_cast<std::size_t>(task_B));
            result.best_B.resize(static_cast<std::size_t>(task_B) * static_cast<std::size_t>(m), 0);
            result.best_a.resize(static_cast<std::size_t>(task_B) * static_cast<std::size_t>(m), 0.0f);
            if (need_root_u32) {
                result.best_code0_root.resize(static_cast<std::size_t>(task_B), 0);
            }
            for (int i = 0; i < task_B; ++i) {
                result.best_parent_local[static_cast<std::size_t>(i)] =
                    best_parent_super[static_cast<std::size_t>(task_node_base + i)];
                result.best_cost[static_cast<std::size_t>(i)] =
                    best_cost_super[static_cast<std::size_t>(task_node_base + i)];
                const std::size_t src_base =
                    static_cast<std::size_t>(task_node_base + i) * static_cast<std::size_t>(m);
                const std::size_t dst_base =
                    static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
                for (int l = 0; l < m; ++l) {
                    result.best_a[dst_base + static_cast<std::size_t>(l)] =
                        best_a_super[src_base + static_cast<std::size_t>(l)];
                    if (!need_root_u32) {
                        result.best_B[dst_base + static_cast<std::size_t>(l)] =
                            best_B_super[src_base + static_cast<std::size_t>(l)];
                    } else if (l == 0) {
                        result.best_code0_root[static_cast<std::size_t>(i)] =
                            static_cast<RootCode>(best_B_super_u32[src_base]);
                        result.best_B[dst_base] = static_cast<FullCode>(0);
                    } else {
                        result.best_B[dst_base + static_cast<std::size_t>(l)] =
                            static_cast<FullCode>(best_B_super_u32[src_base + static_cast<std::size_t>(l)]);
                    }
                }
            }
            req->promise.set_value(std::move(result));
        }

        std::uint64_t distinct_clusters = 0;
        {
            std::unordered_map<LinkageGpuSchedulerClusterHandle, bool> seen;
            for (const auto& req : batch.reqs) {
                seen.emplace(req->task.cluster, true);
            }
            distinct_clusters = static_cast<std::uint64_t>(seen.size());
        }
        {
            std::lock_guard<std::mutex> lock(mu);
            for (const auto& cluster : batch.clusters) {
                cluster->queued_many -= 1;
            }
            stats.many_launches += 1;
            stats.tasks_merged_total += static_cast<std::uint64_t>(task_count);
            stats.tasks_merged_max = std::max<std::uint64_t>(stats.tasks_merged_max,
                                                             static_cast<std::uint64_t>(task_count));
            stats.nodes_total += static_cast<std::uint64_t>(std::max(0, B_total));
            stats.nodes_max = std::max<std::uint64_t>(stats.nodes_max,
                                                      static_cast<std::uint64_t>(std::max(0, B_total)));
            stats.pairs_total += static_cast<std::uint64_t>(Npairs);
            stats.pairs_max = std::max<std::uint64_t>(stats.pairs_max,
                                                      static_cast<std::uint64_t>(Npairs));
            if (distinct_clusters > 1) {
                stats.cross_cluster_launches += 1;
            }
        }
    }

    void ExecuteSingleTask(const std::shared_ptr<SingleTaskRequest>& req) {
        UpdateQueueWaitStats(req->submit_ts);
        std::shared_ptr<ClusterState> cluster;
        {
            std::lock_guard<std::mutex> lock(mu);
            cluster = FindClusterUnsafe(req->task.cluster);
        }
        if (!cluster || !cluster->live) {
            SetPromiseError(&req->promise, InvalidClusterError(req->task.cluster));
            return;
        }
        if (!req->task.pre_one && !req->task.codebook_large_root) {
            SetPromiseError(&req->promise,
                            "LinkageGpuScheduler: single-node task missing evaluator description.");
            return;
        }

        LinkageGpuSchedulerSingleNodeResult result;
        result.best_B.resize(static_cast<std::size_t>(req->task.m));
        result.best_a.resize(static_cast<std::size_t>(req->task.m));
        std::string err;
        bool ok = false;
        if (req->task.codebook_large_root) {
            const bool need_root_u32 =
                !req->task.codebook_large_root->books.empty() &&
                req->task.codebook_large_root->books.front().cols > 256;
            if (!need_root_u32) {
                const int Kp = static_cast<int>(req->task.cand_parent_local.size());
                std::vector<int> task_node_offsets = {0, 1};
                std::vector<int> task_pair_offsets = {0, Kp};
                std::vector<const float*> task_rfull_ptrs = {cluster->d_rfull.ptr};
                std::vector<std::uint32_t> task_seeds = {req->task.seed};
                std::vector<int> pair_node(static_cast<std::size_t>(Kp), 0);
                std::vector<int> pair_task(static_cast<std::size_t>(Kp), 0);
                std::vector<int> cand_offsets = {0, Kp};
                std::vector<std::uint64_t> node_sample_id_base = {req->task.sample_id_offset};
                std::vector<int> pair_t_override(static_cast<std::size_t>(Kp), 0);
                for (int t = 0; t < Kp; ++t) {
                    pair_t_override[static_cast<std::size_t>(t)] = t;
                }

                std::vector<int> best_parent(1, -1);
                std::vector<float> best_cost(1, std::numeric_limits<float>::infinity());
                std::vector<float> best_a(static_cast<std::size_t>(req->task.m), 0.0f);
                std::vector<FullCode> best_B(static_cast<std::size_t>(req->task.m), 0);
                ok = EvaluateParentCandidatesManyTasksBatchCudaDeviceRfullMetaOnly(
                    worker_ctx,
                    *req->task.codebook_large_root,
                    req->task.xi.data(),
                    req->task.d,
                    /*B_total=*/1,
                    /*task_count=*/1,
                    task_node_offsets.data(),
                    task_pair_offsets.data(),
                    task_rfull_ptrs.data(),
                    task_seeds.data(),
                    req->task.cand_parent_local.data(),
                    pair_node.data(),
                    pair_task.data(),
                    cand_offsets.data(),
                    Kp,
                    node_sample_id_base.data(),
                    pair_t_override.data(),
                    req->task.icm_iters,
                    req->task.ils_iters,
                    req->task.perturb_k,
                    best_parent.data(),
                    best_cost.data(),
                    best_B.data(),
                    best_a.data(),
                    &err);
                if (ok) {
                    result.best_parent_local = best_parent[0];
                    result.best_cost = best_cost[0];
                    result.best_B = std::move(best_B);
                    result.best_a = std::move(best_a);
                }
            } else {
                std::string upload_err;
                if (!CudaLinkageUploadClusterRfull(worker_ctx, cluster->h_rfull.data(), cluster->d, cluster->n_cols, &upload_err)) {
                    SetPromiseError(&req->promise, "LinkageGpuScheduler: upload current cluster R_full failed: " + upload_err);
                    return;
                }
                const int Kp = static_cast<int>(req->task.cand_parent_local.size());
                std::vector<int> pair_node(static_cast<std::size_t>(Kp), 0);
                const int cand_offsets[2] = {0, Kp};
                int best_parent = -1;
                float best_cost = std::numeric_limits<float>::infinity();
                std::vector<std::uint32_t> best_B_u32(static_cast<std::size_t>(req->task.m), 0u);
                std::vector<float> best_a(static_cast<std::size_t>(req->task.m), 0.0f);
                ok = EvaluateParentCandidatesManyNodesBatchCudaDeviceRfullWithNodeSampleIdBaseLargeRootU32(
                    worker_ctx,
                    *req->task.codebook_large_root,
                    req->task.xi.data(),
                    req->task.d,
                    /*B=*/1,
                    req->task.cand_parent_local.data(),
                    pair_node.data(),
                    cand_offsets,
                    Kp,
                    req->task.icm_iters,
                    req->task.ils_iters,
                    req->task.perturb_k,
                    req->task.seed,
                    &req->task.sample_id_offset,
                    &best_parent,
                    &best_cost,
                    best_B_u32.data(),
                    best_a.data(),
                    &err);
                if (ok) {
                    result.best_parent_local = best_parent;
                    result.best_cost = best_cost;
                    result.root_codes_u32 = true;
                    result.best_code0_root = static_cast<RootCode>(best_B_u32[0]);
                    result.best_B.assign(static_cast<std::size_t>(req->task.m), 0);
                    for (int l = 1; l < req->task.m; ++l) {
                        result.best_B[static_cast<std::size_t>(l)] =
                            static_cast<FullCode>(best_B_u32[static_cast<std::size_t>(l)]);
                    }
                    result.best_a = std::move(best_a);
                }
            }
        } else {
            if (!CudaLinkageUploadClusterRfull(worker_ctx, cluster->h_rfull.data(), cluster->d, cluster->n_cols, &err)) {
                SetPromiseError(&req->promise, "LinkageGpuScheduler: upload current cluster R_full failed: " + err);
                return;
            }
            ok = EvaluateParentCandidatesBatchCudaDeviceRfull(
                worker_ctx,
                *req->task.pre_one,
                req->task.xi.data(),
                req->task.d,
                req->task.cand_parent_local.data(),
                static_cast<int>(req->task.cand_parent_local.size()),
                req->task.icm_iters,
                req->task.ils_iters,
                req->task.perturb_k,
                req->task.seed,
                req->task.sample_id_offset,
                &result.best_parent_local,
                &result.best_cost,
                &result.best_B,
                &result.best_a,
                &err);
        }
        if (!ok) {
            SetPromiseError(&req->promise, "LinkageGpuScheduler: single-node eval failed: " + err);
            return;
        }

        if (SchedulerDebugVerifyEnabled()) {
            VerifySingleTaskAgainstLegacy(worker_ctx, req->task, *cluster, result);
        }

        {
            std::lock_guard<std::mutex> lock(mu);
            cluster->queued_single -= 1;
            stats.single_launches += 1;
        }
        req->promise.set_value(std::move(result));
    }

    void ExecuteSingleBatch(ReadySingleBatch batch) {
        if (batch.reqs.empty()) {
            return;
        }
        if (batch.reqs.size() == 1) {
            ExecuteSingleTask(batch.reqs.front());
            return;
        }
        for (const auto& req : batch.reqs) {
            UpdateQueueWaitStats(req->submit_ts);
        }

        const auto& first = batch.reqs.front()->task;
        const int task_count = static_cast<int>(batch.reqs.size());
        const int d = first.d;
        const int m = first.m;
        const bool use_pre_one = (first.pre_one != nullptr);
        const bool need_root_u32 =
            !use_pre_one &&
            first.codebook_large_root &&
            !first.codebook_large_root->books.empty() &&
            first.codebook_large_root->books.front().cols > 256;

        int Kp_total = 0;
        for (const auto& req : batch.reqs) {
            if (!req->task.pre_one && !req->task.codebook_large_root) {
                SetPromiseError(&req->promise, "LinkageGpuScheduler: single-node task missing evaluator description.");
                return;
            }
            Kp_total += static_cast<int>(req->task.cand_parent_local.size());
        }

        std::vector<int> task_node_offsets(static_cast<std::size_t>(task_count + 1), 0);
        std::vector<int> task_pair_offsets(static_cast<std::size_t>(task_count + 1), 0);
        std::vector<float> X_super(static_cast<std::size_t>(d) * static_cast<std::size_t>(task_count), 0.0f);
        std::vector<int> cand_parent_local_flat(static_cast<std::size_t>(Kp_total), 0);
        std::vector<int> pair_node_super(static_cast<std::size_t>(Kp_total), 0);
        std::vector<int> pair_task_super(static_cast<std::size_t>(Kp_total), 0);
        std::vector<int> cand_offsets_super(static_cast<std::size_t>(task_count + 1), 0);
        std::vector<std::uint64_t> node_sample_id_base_super(static_cast<std::size_t>(task_count), 0);
        std::vector<int> pair_t_override_super(static_cast<std::size_t>(Kp_total), 0);
        std::vector<const float*> task_rfull_ptrs_host(static_cast<std::size_t>(task_count), nullptr);
        std::vector<std::uint32_t> task_seeds_host(static_cast<std::size_t>(task_count), 0u);

        int pair_base = 0;
        for (int t = 0; t < task_count; ++t) {
            const auto& req = batch.reqs[static_cast<std::size_t>(t)];
            const auto& cluster = batch.clusters[static_cast<std::size_t>(t)];
            const auto& task = req->task;
            task_node_offsets[static_cast<std::size_t>(t)] = t;
            task_pair_offsets[static_cast<std::size_t>(t)] = pair_base;
            task_rfull_ptrs_host[static_cast<std::size_t>(t)] = cluster->d_rfull.ptr;
            task_seeds_host[static_cast<std::size_t>(t)] = task.seed;
            std::memcpy(X_super.data() + static_cast<std::size_t>(d) * static_cast<std::size_t>(t),
                        task.xi.data(),
                        static_cast<std::size_t>(d) * sizeof(float));
            std::copy(task.cand_parent_local.begin(),
                      task.cand_parent_local.end(),
                      cand_parent_local_flat.begin() + static_cast<std::size_t>(pair_base));
            node_sample_id_base_super[static_cast<std::size_t>(t)] = task.sample_id_offset;
            cand_offsets_super[static_cast<std::size_t>(t)] = pair_base;
            const int Kp = static_cast<int>(task.cand_parent_local.size());
            for (int p = 0; p < Kp; ++p) {
                pair_node_super[static_cast<std::size_t>(pair_base + p)] = t;
                pair_task_super[static_cast<std::size_t>(pair_base + p)] = t;
                pair_t_override_super[static_cast<std::size_t>(pair_base + p)] = p;
            }
            pair_base += Kp;
            cand_offsets_super[static_cast<std::size_t>(t + 1)] = pair_base;
        }
        task_node_offsets[static_cast<std::size_t>(task_count)] = task_count;
        task_pair_offsets[static_cast<std::size_t>(task_count)] = Kp_total;

        std::vector<int> best_parent_super(static_cast<std::size_t>(task_count), -1);
        std::vector<float> best_cost_super(static_cast<std::size_t>(task_count),
                                           std::numeric_limits<float>::infinity());
        std::vector<float> best_a_super(static_cast<std::size_t>(task_count) * static_cast<std::size_t>(m), 0.0f);
        std::vector<FullCode> best_B_super(static_cast<std::size_t>(task_count) * static_cast<std::size_t>(m), 0);
        std::vector<std::uint32_t> best_B_super_u32;
        if (need_root_u32) {
            best_B_super_u32.resize(static_cast<std::size_t>(task_count) * static_cast<std::size_t>(m), 0u);
        }

        std::string err;
        const bool ok = use_pre_one
                            ? EvaluateParentCandidatesManyTasksBatchCudaDeviceRfullWithNodeSampleIdBaseEx(
                                  worker_ctx,
                                  *first.pre_one,
                                  X_super.data(),
                                  d,
                                  task_count,
                                  task_count,
                                  task_node_offsets.data(),
                                  task_pair_offsets.data(),
                                  task_rfull_ptrs_host.data(),
                                  task_seeds_host.data(),
                                  cand_parent_local_flat.data(),
                                  pair_node_super.data(),
                                  pair_task_super.data(),
                                  cand_offsets_super.data(),
                                  Kp_total,
                                  node_sample_id_base_super.data(),
                                  pair_t_override_super.data(),
                                  first.icm_iters,
                                  first.ils_iters,
                                  first.perturb_k,
                                  best_parent_super.data(),
                                  best_cost_super.data(),
                                  best_B_super.data(),
                                  best_a_super.data(),
                                  &err)
                            : need_root_u32
                            ? EvaluateParentCandidatesManyTasksBatchCudaDeviceRfullMetaOnlyU32(
                                  worker_ctx,
                                  *first.codebook_large_root,
                                  X_super.data(),
                                  d,
                                  task_count,
                                  task_count,
                                  task_node_offsets.data(),
                                  task_pair_offsets.data(),
                                  task_rfull_ptrs_host.data(),
                                  task_seeds_host.data(),
                                  cand_parent_local_flat.data(),
                                  pair_node_super.data(),
                                  pair_task_super.data(),
                                  cand_offsets_super.data(),
                                  Kp_total,
                                  node_sample_id_base_super.data(),
                                  pair_t_override_super.data(),
                                  first.icm_iters,
                                  first.ils_iters,
                                  first.perturb_k,
                                  best_parent_super.data(),
                                  best_cost_super.data(),
                                  best_B_super_u32.data(),
                                  best_a_super.data(),
                                  &err)
                            : EvaluateParentCandidatesManyTasksBatchCudaDeviceRfullMetaOnly(
                                  worker_ctx,
                                  *first.codebook_large_root,
                                  X_super.data(),
                                  d,
                                  task_count,
                                  task_count,
                                  task_node_offsets.data(),
                                  task_pair_offsets.data(),
                                  task_rfull_ptrs_host.data(),
                                  task_seeds_host.data(),
                                  cand_parent_local_flat.data(),
                                  pair_node_super.data(),
                                  pair_task_super.data(),
                                  cand_offsets_super.data(),
                                  Kp_total,
                                  node_sample_id_base_super.data(),
                                  pair_t_override_super.data(),
                                  first.icm_iters,
                                  first.ils_iters,
                                  first.perturb_k,
                                  best_parent_super.data(),
                                  best_cost_super.data(),
                                  best_B_super.data(),
                                  best_a_super.data(),
                                  &err);
        if (!ok) {
            for (auto& req : batch.reqs) {
                SetPromiseError(&req->promise, "LinkageGpuScheduler: merged single-node eval failed: " + err);
            }
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mu);
            for (const auto& cluster : batch.clusters) {
                cluster->queued_single -= 1;
            }
            stats.single_launches += 1;
        }

        for (int t = 0; t < task_count; ++t) {
            LinkageGpuSchedulerSingleNodeResult result;
            result.best_parent_local = best_parent_super[static_cast<std::size_t>(t)];
            result.best_cost = best_cost_super[static_cast<std::size_t>(t)];
            result.root_codes_u32 = need_root_u32;
            result.best_B.resize(static_cast<std::size_t>(m), 0);
            result.best_a.resize(static_cast<std::size_t>(m), 0.0f);
            const std::size_t base = static_cast<std::size_t>(t) * static_cast<std::size_t>(m);
            for (int l = 0; l < m; ++l) {
                result.best_a[static_cast<std::size_t>(l)] = best_a_super[base + static_cast<std::size_t>(l)];
                if (!need_root_u32) {
                    result.best_B[static_cast<std::size_t>(l)] = best_B_super[base + static_cast<std::size_t>(l)];
                } else if (l == 0) {
                    result.best_code0_root = static_cast<RootCode>(best_B_super_u32[base]);
                } else {
                    result.best_B[static_cast<std::size_t>(l)] =
                        static_cast<FullCode>(best_B_super_u32[base + static_cast<std::size_t>(l)]);
                }
            }
            batch.reqs[static_cast<std::size_t>(t)]->promise.set_value(std::move(result));
        }
    }

    void WorkerLoop() {
        try {
            EnsureWorkerCtx();
            {
                std::lock_guard<std::mutex> lock(mu);
                worker_ready = true;
            }
            cv.notify_all();

            while (true) {
                RegisterRequest reg;
                UnregisterRequest unreg;
                std::shared_ptr<ManyTaskRequest> many;
                std::shared_ptr<SingleTaskRequest> single;
                bool have_reg = false;
                bool have_unreg = false;
                bool did_update = false;

                {
                    std::unique_lock<std::mutex> lock(mu);
                    cv.wait(lock, [&]() { return stopping || HasPendingUnsafe(); });
                    if (stopping && !HasPendingUnsafe()) {
                        break;
                    }
                    DrainIngressUnsafe();
                    if (!pending_register.empty()) {
                        reg = std::move(pending_register.front());
                        pending_register.pop_front();
                        have_reg = true;
                    }
                }
                if (have_reg) {
                    ExecuteRegister(std::move(reg));
                    continue;
                }

                did_update = ApplyPendingUpdatesBurst() > 0;

                auto many_batch = PopReadyManyBatch();
                if (!many_batch.reqs.empty()) {
                    if (ShouldExecuteManyDirect(many_batch)) {
                        ExecuteManyTask(many_batch.reqs.front());
                        continue;
                    }
                    GrowReadyManyBatch(&many_batch);
                    ExecuteManyBatch(std::move(many_batch));
                    continue;
                }

                auto single_batch = PopReadySingleBatch();
                if (!single_batch.reqs.empty()) {
                    GrowReadySingleBatch(&single_batch);
                    ExecuteSingleBatch(std::move(single_batch));
                    continue;
                }

                {
                    std::lock_guard<std::mutex> lock(mu);
                    if (!pending_unregister.empty()) {
                        unreg = std::move(pending_unregister.front());
                        pending_unregister.pop_front();
                        have_unreg = true;
                    }
                }
                if (have_unreg) {
                    ExecuteUnregister(std::move(unreg));
                    continue;
                }

                if (did_update) {
                    continue;
                }
            }
        } catch (const std::exception& e) {
            std::deque<std::shared_ptr<ManyTaskRequest>> many_to_fail;
            std::deque<std::shared_ptr<SingleTaskRequest>> single_to_fail;
            std::deque<RegisterRequest> register_to_fail;
            std::deque<UnregisterRequest> unregister_to_fail;
            {
                std::lock_guard<std::mutex> lock(mu);
                worker_ready = true;
                stopping = true;
                many_to_fail.swap(pending_many);
                single_to_fail.swap(pending_single);
                register_to_fail.swap(pending_register);
                unregister_to_fail.swap(pending_unregister);
            }
            const std::string err = std::string("LinkageGpuScheduler worker failed: ") + e.what();
            for (auto& req : many_to_fail) {
                SetPromiseError(&req->promise, err);
            }
            for (auto& req : single_to_fail) {
                SetPromiseError(&req->promise, err);
            }
            for (auto& req : register_to_fail) {
                SetPromiseError(&req.promise, err);
            }
            for (auto& req : unregister_to_fail) {
                SetPromiseError(&req.promise, err);
            }
        }

        if (worker_ctx) {
            pool.Release(worker_ctx);
            worker_ctx = nullptr;
        }
    }
};

LinkageGpuScheduler::LinkageGpuScheduler(const CudaPoolConfig& cfg)
    : impl_(std::make_unique<Impl>(cfg)) {
    std::unique_lock<std::mutex> lock(impl_->mu);
    impl_->cv.wait(lock, [&]() { return impl_->worker_ready; });
}

LinkageGpuScheduler::~LinkageGpuScheduler() = default;

LinkageGpuSchedulerClusterHandle LinkageGpuScheduler::RegisterClusterRfull(const float* R_full,
                                                                       int d,
                                                                       int n_cols,
                                                                       std::string* err) {
    if (d <= 0 || n_cols < 0) {
        if (err) {
            *err = "LinkageGpuScheduler::RegisterClusterRfull: invalid shape.";
        }
        return 0;
    }
    if (n_cols > 0 && !R_full) {
        if (err) {
            *err = "LinkageGpuScheduler::RegisterClusterRfull: R_full is null.";
        }
        return 0;
    }

    RegisterRequest request;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
    request.handle = impl_->next_cluster_handle++;
    }
    request.d = d;
    request.n_cols = n_cols;
    const std::size_t total = static_cast<std::size_t>(d) * static_cast<std::size_t>(n_cols);
    if (total > 0) {
        request.h_rfull.assign(R_full, R_full + total);
    }
    auto future = request.promise.get_future();
    const LinkageGpuSchedulerClusterHandle handle = request.handle;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->ingress.emplace_back(std::move(request));
    }
    impl_->cv.notify_all();
    try {
        future.get();
        return handle;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return 0;
    }
}

bool LinkageGpuScheduler::UnregisterCluster(LinkageGpuSchedulerClusterHandle cluster, std::string* err) {
    UnregisterRequest request;
    request.handle = cluster;
    auto future = request.promise.get_future();
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->ingress.emplace_back(std::move(request));
    }
    impl_->cv.notify_all();
    try {
        future.get();
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

std::uint64_t LinkageGpuScheduler::SubmitClusterRfullUpdates(LinkageGpuSchedulerClusterHandle cluster,
                                                           const int* cols,
                                                           const float* r_cols,
                                                           int d,
                                                           int B,
                                                           std::string* err) {
    if (d <= 0 || B < 0) {
        if (err) {
            *err = "LinkageGpuScheduler::SubmitClusterRfullUpdates: invalid shape.";
        }
        return 0;
    }
    if (B > 0 && (!cols || !r_cols)) {
        if (err) {
            *err = "LinkageGpuScheduler::SubmitClusterRfullUpdates: null input.";
        }
        return 0;
    }

    UpdateRequest request;
    request.handle = cluster;
    request.update.d = d;
    request.update.B = B;
    if (B > 0) {
        request.update.cols.assign(cols, cols + static_cast<std::size_t>(B));
        request.update.r_cols.assign(r_cols, r_cols + static_cast<std::size_t>(B) * static_cast<std::size_t>(d));
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        const auto cluster_state = impl_->FindClusterUnsafe(cluster);
        if (!cluster_state || !cluster_state->live) {
            if (err) {
                *err = InvalidClusterError(cluster);
            }
            return 0;
        }
        request.update.generation = ++cluster_state->submitted_generation;
    }
    const std::uint64_t generation = request.update.generation;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->ingress.emplace_back(std::move(request));
    }
    impl_->cv.notify_all();
    return generation;
}

std::shared_future<LinkageGpuSchedulerManyNodesResult>
LinkageGpuScheduler::SubmitManyNodesTask(LinkageGpuSchedulerManyNodesTask&& task, std::string* err) {
    ManyTaskRequest request;
    request.task = std::move(task);
    request.submit_ts = NowSeconds();
    auto future = request.promise.get_future().share();
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        const auto cluster = impl_->FindClusterUnsafe(request.task.cluster);
        if (!cluster || !cluster->live) {
            const std::string msg = InvalidClusterError(request.task.cluster);
            if (err) {
                *err = msg;
            }
            SetPromiseError(&request.promise, msg);
            return future;
        }
        impl_->stats.many_tasks_submitted += 1;
        impl_->ingress.emplace_back(std::move(request));
    }
    impl_->cv.notify_all();
    return future;
}

std::shared_future<LinkageGpuSchedulerSingleNodeResult>
LinkageGpuScheduler::SubmitImmediateSingleNodeTask(LinkageGpuSchedulerSingleNodeTask&& task, std::string* err) {
    SingleTaskRequest request;
    request.task = std::move(task);
    request.submit_ts = NowSeconds();
    auto future = request.promise.get_future().share();
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        const auto cluster = impl_->FindClusterUnsafe(request.task.cluster);
        if (!cluster || !cluster->live) {
            const std::string msg = InvalidClusterError(request.task.cluster);
            if (err) {
                *err = msg;
            }
            SetPromiseError(&request.promise, msg);
            return future;
        }
        impl_->stats.single_tasks_submitted += 1;
        impl_->ingress.emplace_back(std::move(request));
    }
    impl_->cv.notify_all();
    return future;
}

LinkageGpuSchedulerStats LinkageGpuScheduler::GetStats() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->stats;
}

}  // namespace stlq




