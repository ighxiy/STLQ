#include "stlq/ivf/ivf_scan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#include <omp.h>

#include "stlq/core/blas.h"
#include "stlq/ivf/cluster_select.h"
#include "stlq/common/logger.h"
#include "stlq/quantizer/codebook_meta.h"
#include "stlq/common/timer.h"

extern "C" {

void quantize_norms_complete_mkl_uint8(
    uint8_t* B,
    const float* a,
    float** codebooks,
    int* codebook_sizes,
    int M,
    int n_base,
    int d,
    int h_norms,
    int max_iter,
    int* assignments,
    float* norm_centers);

void quantize_norms_complete_mkl_uint16(
    uint16_t* B,
    const float* a,
    float** codebooks,
    int* codebook_sizes,
    int M,
    int n_base,
    int d,
    int h_norms,
    int max_iter,
    int* assignments,
    float* norm_centers);

}  // extern "C"

namespace stlq {

namespace {

std::vector<float> ComputeIvfInvNormFromFirstBook(const CodebookMeta& meta, int nlist) {
    std::vector<float> inv_norm(static_cast<std::size_t>(std::max(0, nlist)), 1.0f);
    if (nlist <= 0 || meta.ptrs.empty() || meta.sizes.empty() || meta.d <= 0) {
        return inv_norm;
    }
    const float* book0 = meta.ptrs[0];
    const int cols0 = meta.sizes[0];
    const int limit = std::min(nlist, cols0);
    for (int cid = 0; cid < limit; ++cid) {
        const float* col = book0 + static_cast<std::size_t>(cid) * meta.d;
        double ss = 0.0;
        for (int r = 0; r < meta.d; ++r) {
            const double v = static_cast<double>(col[r]);
            ss += v * v;
        }
        inv_norm[static_cast<std::size_t>(cid)] =
            1.0f / std::sqrt(std::max(static_cast<float>(ss), 1e-20f));
    }
    return inv_norm;
}

static_assert(sizeof(int32_t) == sizeof(int), "Expected 32-bit int.");

template <typename InCodeT, typename OutCodeT>
bool ConvertCodes(const ColMajorMatrix<InCodeT>& B, std::vector<OutCodeT>* out, std::string* error) {
    out->resize(B.data.size());
    const OutCodeT max_val = std::numeric_limits<OutCodeT>::max();
    for (std::size_t i = 0; i < B.data.size(); ++i) {
        const InCodeT val = B.data[i];
        if (val > max_val) {
            if (error) {
                *error = "Code value exceeds storage type.";
            }
            return false;
        }
        (*out)[i] = static_cast<OutCodeT>(val);
    }
    return true;
}

float ComputeRecallAtKMatrix(const std::vector<int>& ground_truth,
                             const ColMajorMatrix<int>& indices) {
    const int k = indices.rows;
    const int nquery = indices.cols;
    if (k <= 0 || nquery <= 0) {
        return 0.0f;
    }
    if (ground_truth.size() != static_cast<std::size_t>(nquery)) {
        return 0.0f;
    }
    int hits = 0;
    for (int qi = 0; qi < nquery; ++qi) {
        const int g = ground_truth[static_cast<std::size_t>(qi)];
        const int* row = indices.Col(qi);
        bool found = false;
        for (int j = 0; j < k; ++j) {
            if (row[j] == g) {
                found = true;
                break;
            }
        }
        if (found) {
            ++hits;
        }
    }
    return static_cast<float>(hits) / static_cast<float>(nquery);
}

// Deterministic ordering: (dist, id). This avoids nondeterministic tie handling
// when many candidates share identical distances.
struct ByDistThenIdLess {
    bool operator()(const std::pair<float, int32_t>& a,
                    const std::pair<float, int32_t>& b) const {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    }
};

template <typename CodeT>
inline float AdcDistanceOne(const CodeT* __restrict code,
                            const float* __restrict coeff,
                            const float* __restrict xCq_col,
                            const int* __restrict offsets,
                            int m,
                            float dbnorm) {
    float acc = 0.0f;
    for (int cb = 0; cb < m; ++cb) {
        const int flat = offsets[cb] + static_cast<int>(code[cb]);
        acc += coeff[cb] * xCq_col[flat];
    }
    return dbnorm - 2.0f * acc;
}

template <typename CodeT>
void AdcScanAllTopK(const CodeT* __restrict codes,
                    const float* __restrict a,
                    const float* __restrict dbnorms,
                    const float* __restrict xCq_col,
                    const int* __restrict offsets,
                    int m,
                    int n_base,
                    int k,
                    float* __restrict out_dists,
                    int* __restrict out_indices,
                    std::vector<std::pair<float, int32_t>>* heap) {
    heap->clear();
    heap->reserve(static_cast<std::size_t>(k));
    for (int i = 0; i < n_base; ++i) {
        const CodeT* code = codes + static_cast<std::size_t>(i) * m;
        const float* coeff = a + static_cast<std::size_t>(i) * m;
        const float dist = AdcDistanceOne(code, coeff, xCq_col, offsets, m, dbnorms[i]);

        if (static_cast<int>(heap->size()) < k) {
            heap->emplace_back(dist, static_cast<int32_t>(i));
            if (static_cast<int>(heap->size()) == k) {
                std::make_heap(heap->begin(), heap->end(), ByDistThenIdLess{});
            }
            continue;
        }
        if (dist > heap->front().first ||
            (dist == heap->front().first && static_cast<int32_t>(i) >= heap->front().second)) {
            continue;
        }
        std::pop_heap(heap->begin(), heap->end(), ByDistThenIdLess{});
        heap->back() = {dist, static_cast<int32_t>(i)};
        std::push_heap(heap->begin(), heap->end(), ByDistThenIdLess{});
    }

    if (!heap->empty()) {
        std::sort_heap(heap->begin(), heap->end(), ByDistThenIdLess{});
    }
    const int got = static_cast<int>(heap->size());
    for (int j = 0; j < got; ++j) {
        out_dists[j] = (*heap)[static_cast<std::size_t>(j)].first;
        out_indices[j] = static_cast<int>((*heap)[static_cast<std::size_t>(j)].second);
    }
    for (int j = got; j < k; ++j) {
        out_dists[j] = std::numeric_limits<float>::infinity();
        out_indices[j] = -1;
    }
}

template <typename CodeT>
void AdcScanCandidatesTopK(const CodeT* __restrict codes,
                           const float* __restrict a,
                           const float* __restrict dbnorms,
                           const float* __restrict xCq_col,
                           const int* __restrict offsets,
                           int m,
                           const int* __restrict candidates,
                           int n_candidates,
                           int k,
                           float* __restrict out_dists,
                           int* __restrict out_indices,
                           std::vector<std::pair<float, int32_t>>* heap) {
    heap->clear();
    heap->reserve(static_cast<std::size_t>(k));
    for (int t = 0; t < n_candidates; ++t) {
        const int i = candidates[t];
        const CodeT* code = codes + static_cast<std::size_t>(i) * m;
        const float* coeff = a + static_cast<std::size_t>(i) * m;
        const float dist = AdcDistanceOne(code, coeff, xCq_col, offsets, m, dbnorms[i]);

        if (static_cast<int>(heap->size()) < k) {
            heap->emplace_back(dist, static_cast<int32_t>(i));
            if (static_cast<int>(heap->size()) == k) {
                std::make_heap(heap->begin(), heap->end(), ByDistThenIdLess{});
            }
            continue;
        }
        if (dist > heap->front().first ||
            (dist == heap->front().first && static_cast<int32_t>(i) >= heap->front().second)) {
            continue;
        }
        std::pop_heap(heap->begin(), heap->end(), ByDistThenIdLess{});
        heap->back() = {dist, static_cast<int32_t>(i)};
        std::push_heap(heap->begin(), heap->end(), ByDistThenIdLess{});
    }

    if (!heap->empty()) {
        std::sort_heap(heap->begin(), heap->end(), ByDistThenIdLess{});
    }
    const int got = static_cast<int>(heap->size());
    for (int j = 0; j < got; ++j) {
        out_dists[j] = (*heap)[static_cast<std::size_t>(j)].first;
        out_indices[j] = static_cast<int>((*heap)[static_cast<std::size_t>(j)].second);
    }
    for (int j = got; j < k; ++j) {
        out_dists[j] = std::numeric_limits<float>::infinity();
        out_indices[j] = -1;
    }
}

}  // namespace

bool EvaluateRecallBase(const Config& config,
                        const Dataset& dataset,
                        const CodebookPack& codebooks,
                        const BaseEncoding& base,
                        RecallResult* result,
                        std::string* error) {
    if (!result) {
        return false;
    }
    if (dataset.gt.size() != static_cast<std::size_t>(dataset.Xq.cols)) {
        if (error) {
            *error = "Ground truth size does not match query count.";
        }
        return false;
    }

    const int n_queries = dataset.Xq.cols;
    const int n_base = base.B.cols;
    const int m = base.B.rows;
    int k = config.dataset.k;
    if (k <= 0) {
        if (error) {
            *error = "Invalid k in config.";
        }
        return false;
    }
    k = std::min(k, n_base);

    CodebookMeta meta = BuildCodebookMeta(GatherBooks(codebooks));
    if (meta.m == 0 || meta.d == 0) {
        if (error) {
            *error = "Codebooks are empty.";
        }
        return false;
    }
    if (meta.m != m) {
        if (error) {
            *error = "Codebook count does not match encoding rows.";
        }
        return false;
    }

    const int h_norms = config.base.encode.hnorms;
    const int max_iter = 60;

    std::vector<int> assignments(n_base, 0);
    std::vector<float> centers(h_norms, 0.0f);
    std::vector<float> dbnorms(n_base, 0.0f);

    result->dists = ColMajorMatrix<float>(k, n_queries);
    result->indices = ColMajorMatrix<int>(k, n_queries);

    Timer norm_timer;
    if (meta.max_h <= 256) {
        std::vector<uint8_t> codes;
        if (!ConvertCodes(base.B, &codes, error)) {
            return false;
        }
        quantize_norms_complete_mkl_uint8(codes.data(), base.a.data.data(), meta.ptrs.data(),
                                          meta.sizes.data(), m, n_base, meta.d, h_norms,
                                          max_iter, assignments.data(), centers.data());

        for (int i = 0; i < n_base; ++i) {
            int idx = assignments[i];
            // if (idx < 0 || idx >= h_norms) {
            //     idx = 0;
            // }
            dbnorms[i] = centers[idx];
        }

        ColMajorMatrix<float> xCq(meta.total_cols, n_queries);
        {
            BlasSetThreads(std::max(1, omp_get_max_threads()));
            Timer gemm_timer;
            Gemm(true, false, 1.0f, meta.flat, dataset.Xq, 0.0f, &xCq);
            BlasSetThreads(1);
            LogInfo("Full-scan ADC query GEMM time: " + std::to_string(gemm_timer.ElapsedSeconds()) + "s");
        }

        Timer scan_timer;
        #pragma omp parallel default(none) shared(codes, base, dbnorms, xCq, meta, result) firstprivate(m, n_base, k, n_queries)
        {
            std::vector<std::pair<float, int32_t>> work;
            #pragma omp for schedule(static)
            for (int qi = 0; qi < n_queries; ++qi) {
                AdcScanAllTopK<uint8_t>(codes.data(), base.a.data.data(), dbnorms.data(),
                                        xCq.Col(qi), meta.offsets.data(),
                                        m, n_base, k,
                                        result->dists.Col(qi), result->indices.Col(qi),
                                        &work);
            }
        }
        LogInfo("Full-scan ADC scan+topk time: " + std::to_string(scan_timer.ElapsedSeconds()) + "s");
    } else {
        std::vector<uint16_t> codes;
        if (!ConvertCodes(base.B, &codes, error)) {
            return false;
        }
        quantize_norms_complete_mkl_uint16(codes.data(), base.a.data.data(), meta.ptrs.data(),
                                           meta.sizes.data(), m, n_base, meta.d, h_norms,
                                           max_iter, assignments.data(), centers.data());

        for (int i = 0; i < n_base; ++i) {
            int idx = assignments[i];
            // if (idx < 0 || idx >= h_norms) {
            //     idx = 0;
            // }
            dbnorms[i] = centers[idx];
        }

        ColMajorMatrix<float> xCq(meta.total_cols, n_queries);
        {
            BlasSetThreads(std::max(1, omp_get_max_threads()));
            Timer gemm_timer;
            Gemm(true, false, 1.0f, meta.flat, dataset.Xq, 0.0f, &xCq);
            BlasSetThreads(1);
            LogInfo("Full-scan ADC query GEMM time: " + std::to_string(gemm_timer.ElapsedSeconds()) + "s");
        }

        Timer scan_timer;
        #pragma omp parallel default(none) shared(codes, base, dbnorms, xCq, meta, result) firstprivate(m, n_base, k, n_queries)
        {
            std::vector<std::pair<float, int32_t>> work;
            #pragma omp for schedule(static)
            for (int qi = 0; qi < n_queries; ++qi) {
                AdcScanAllTopK<uint16_t>(codes.data(), base.a.data.data(), dbnorms.data(),
                                         xCq.Col(qi), meta.offsets.data(),
                                         m, n_base, k,
                                         result->dists.Col(qi), result->indices.Col(qi),
                                         &work);
            }
        }
        LogInfo("Full-scan ADC scan+topk time: " + std::to_string(scan_timer.ElapsedSeconds()) + "s");
    }

    LogInfo("Full-scan ADC norm-quant time: " + std::to_string(norm_timer.ElapsedSeconds()) + "s");
    result->recall = ComputeRecallAtKMatrix(dataset.gt, result->indices);
    return true;
}

namespace {

template <typename CodeT>
bool EvaluateRecallBaseIvfImpl(const Config& config,
                               const Dataset& dataset,
                               const CodebookPack& codebooks,
                               const BaseEncoding& base,
                               const CodebookMeta& meta,
                               int k,
                               RecallResult* result,
                               std::string* error) {
    const int n_queries = dataset.Xq.cols;
    const int n_base = base.B.cols;
    const int m = base.B.rows;

    const int nlist = codebooks.books.empty() ? 0 : codebooks.books.front().cols;
    if (nlist <= 0) {
        if (error) {
            *error = "EvaluateRecallBaseIvf: invalid nlist.";
        }
        return false;
    }
    if (codebooks.books.front().rows != meta.d) {
        if (error) {
            *error = "EvaluateRecallBaseIvf: codebook dimension mismatch.";
        }
        return false;
    }

    const int h_norms = config.base.encode.hnorms;
    const int max_iter = 60;

    std::vector<CodeT> codes;
    if (!ConvertCodes(base.B, &codes, error)) {
        return false;
    }

    Timer norm_timer;
    std::vector<int> assignments(n_base, 0);
    std::vector<float> centers(h_norms, 0.0f);
    if constexpr (std::is_same<CodeT, uint8_t>::value) {
        quantize_norms_complete_mkl_uint8(codes.data(), base.a.data.data(),
                                          const_cast<float**>(meta.ptrs.data()),
                                          const_cast<int*>(meta.sizes.data()),
                                          m, n_base, meta.d, h_norms,
                                          max_iter, assignments.data(), centers.data());
    } else {
        quantize_norms_complete_mkl_uint16(codes.data(), base.a.data.data(),
                                           const_cast<float**>(meta.ptrs.data()),
                                           const_cast<int*>(meta.sizes.data()),
                                           m, n_base, meta.d, h_norms,
                                           max_iter, assignments.data(), centers.data());
    }

    std::vector<float> dbnorms(n_base, 0.0f);
    for (int i = 0; i < n_base; ++i) {
        int idx = assignments[i];
        // if (idx < 0 || idx >= h_norms) {
        //     idx = 0;
        // }
        dbnorms[i] = centers[idx];
    }

    std::vector<std::vector<int>> lists(static_cast<std::size_t>(nlist));
    for (int i = 0; i < n_base; ++i) {
        int cid = static_cast<int>(base.B(0, i));
        // if (cid < 0 || cid >= nlist) continue;
        lists[static_cast<std::size_t>(cid)].push_back(i);
    }

    const int nprobe_cfg = std::max(1, config.eval.base_nprobe);
    const int nprobe_cap = std::min(nprobe_cfg, nlist);

    result->dists = ColMajorMatrix<float>(k, n_queries);
    result->indices = ColMajorMatrix<int>(k, n_queries);

    ColMajorMatrix<float> xCq(meta.total_cols, n_queries);
    {
        BlasSetThreads(std::max(1, omp_get_max_threads()));
        Timer gemm_timer;
        Gemm(true, false, 1.0f, meta.flat, dataset.Xq, 0.0f, &xCq);
        BlasSetThreads(1);
        LogInfo("IVF ADC query GEMM time: " + std::to_string(gemm_timer.ElapsedSeconds()) + "s");
    }

    const std::vector<float> ivf_inv_norm = ComputeIvfInvNormFromFirstBook(meta, nlist);

    Timer scan_timer;
    #pragma omp parallel default(none) shared(xCq, lists, codes, base, dbnorms, meta, result, ivf_inv_norm) firstprivate(n_queries, nlist, nprobe_cap, m, n_base, k)
    {
        std::vector<std::pair<float, int32_t>> topk_buf;
        std::vector<int> best_ids(static_cast<std::size_t>(nprobe_cap), 0);
        std::vector<float> best_scores(static_cast<std::size_t>(nprobe_cap),
                                       -std::numeric_limits<float>::infinity());
        std::vector<float> probe_scores(static_cast<std::size_t>(nlist),
                                        -std::numeric_limits<float>::infinity());
        std::vector<int> candidates;

        #pragma omp for schedule(static)
        for (int qi = 0; qi < n_queries; ++qi) {
            const float* xCq_col = xCq.Col(qi);

            for (int cid = 0; cid < nlist; ++cid) {
                probe_scores[static_cast<std::size_t>(cid)] =
                    xCq_col[cid] * ivf_inv_norm[static_cast<std::size_t>(cid)];
            }

            const int nprobe =
                ivf::SelectTopClustersByScore(probe_scores.data(), nlist, nprobe_cap,
                                              best_ids.data(), best_scores.data());

            std::size_t total = 0;
            for (int i = 0; i < nprobe; ++i) {
                total += lists[static_cast<std::size_t>(best_ids[static_cast<std::size_t>(i)])].size();
            }
            candidates.clear();
            candidates.reserve(total);
            for (int i = 0; i < nprobe; ++i) {
                const auto& list = lists[static_cast<std::size_t>(best_ids[static_cast<std::size_t>(i)])];
                candidates.insert(candidates.end(), list.begin(), list.end());
            }

            float* out_d = result->dists.Col(qi);
            int* out_i = result->indices.Col(qi);

            if (static_cast<int>(candidates.size()) < k) {
                // If probes return too few candidates (tiny lists), fall back to full scan.
                AdcScanAllTopK<CodeT>(codes.data(), base.a.data.data(), dbnorms.data(),
                                      xCq_col, meta.offsets.data(),
                                      m, n_base, k, out_d, out_i, &topk_buf);
            } else {
                AdcScanCandidatesTopK<CodeT>(codes.data(), base.a.data.data(), dbnorms.data(),
                                             xCq_col, meta.offsets.data(),
                                             m, candidates.data(),
                                             static_cast<int>(candidates.size()),
                                             k, out_d, out_i, &topk_buf);
            }
        }
    }

    LogInfo("IVF ADC scan+topk time: " + std::to_string(scan_timer.ElapsedSeconds()) + "s");
    LogInfo("IVF ADC norm-quant time: " + std::to_string(norm_timer.ElapsedSeconds()) + "s");

    result->recall = ComputeRecallAtKMatrix(dataset.gt, result->indices);
    return true;
}

}  // namespace

bool EvaluateRecallBaseIvf(const Config& config,
                           const Dataset& dataset,
                           const CodebookPack& codebooks,
                           const BaseEncoding& base,
                           RecallResult* result,
                           std::string* error) {
    if (!result) {
        return false;
    }
    if (dataset.gt.size() != static_cast<std::size_t>(dataset.Xq.cols)) {
        if (error) {
            *error = "Ground truth size does not match query count.";
        }
        return false;
    }

    const int n_base = base.B.cols;
    const int m = base.B.rows;
    int k = std::max(1, config.dataset.k);
    k = std::min(k, n_base);

    CodebookMeta meta = BuildCodebookMeta(GatherBooks(codebooks));
    if (meta.m == 0 || meta.d == 0) {
        if (error) {
            *error = "Codebooks are empty.";
        }
        return false;
    }
    if (meta.m != m) {
        if (error) {
            *error = "Codebook count does not match encoding rows.";
        }
        return false;
    }

    if (meta.max_h <= 256) {
        return EvaluateRecallBaseIvfImpl<uint8_t>(config, dataset, codebooks, base, meta, k, result, error);
    }
    return EvaluateRecallBaseIvfImpl<uint16_t>(config, dataset, codebooks, base, meta, k, result, error);
}

}  // namespace stlq
