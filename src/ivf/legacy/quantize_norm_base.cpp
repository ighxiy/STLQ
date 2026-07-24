#include "blas_compat.h"

#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>

// 辅助函数：计算列优先矩阵的索引
inline int col_major_idx(int row, int col, int num_rows) {
    return row + col * num_rows;
}

extern "C" {
void kmeans_1d_mkl_optimized(float* data,
                            int n,
                            int k,
                            int max_iter,
                            int* assignments,
                            float* centers);

void precompute_codebook_dots_mkl(float** codebooks,
                                 const int* codebook_sizes,
                                 int M,
                                 int d,
                                 float* D);
}  // extern "C"

namespace {

template <typename CodeT>
void ComputeTrueNorms(const CodeT* B,
                      const float* len_rate_each,
                      const float* D,
                      const int* codebook_sizes,
                      int M,
                      int n_base,
                      float* norms) {
    std::vector<int> d_offsets(M * M);
    int offset = 0;
    for (int m = 0; m < M; ++m) {
        for (int l = m; l < M; ++l) {
            d_offsets[m * M + l] = offset;
            d_offsets[l * M + m] = offset;
            offset += codebook_sizes[m] * codebook_sizes[l];
        }
    }

    #pragma omp parallel default(none) shared(B, len_rate_each, D, codebook_sizes, M, n_base, norms, d_offsets)
    {
        std::vector<float> lambda_cache(M);
        std::vector<CodeT> code_cache(M);

        #pragma omp for schedule(static)
        for (int i = 0; i < n_base; ++i) {
            for (int mm = 0; mm < M; ++mm) {
                lambda_cache[mm] = len_rate_each[col_major_idx(mm, i, M)];
                code_cache[mm] = B[col_major_idx(mm, i, M)];
            }

            float temp = 0.0f;

            for (int mm = 0; mm < M; ++mm) {
                const int code_idx_m = static_cast<int>(code_cache[mm]);
                const float lambda_m = lambda_cache[mm];
                const int d_offset = d_offsets[mm * M + mm];
                const int km = codebook_sizes[mm];
                const float dot_val = D[d_offset + col_major_idx(code_idx_m, code_idx_m, km)];
                temp += lambda_m * lambda_m * dot_val;
            }

            for (int mm = 0; mm < M; ++mm) {
                const int code_idx_m = static_cast<int>(code_cache[mm]);
                const float lambda_m = lambda_cache[mm];
                for (int l = mm + 1; l < M; ++l) {
                    const int code_idx_l = static_cast<int>(code_cache[l]);
                    const float lambda_l = lambda_cache[l];
                    const int d_offset = d_offsets[mm * M + l];
                    const int km = codebook_sizes[mm];
                    const float dot_val = D[d_offset + col_major_idx(code_idx_m, code_idx_l, km)];
                    temp += 2.0f * lambda_m * lambda_l * dot_val;
                }
            }

            norms[i] = temp;
        }
    }
}

template <typename CodeT>
void QuantizeNormsComplete(CodeT* B,
                           const float* len_rate_each,
                           float** codebooks,
                           int* codebook_sizes,
                           int M,
                           int n_base,
                           int d,
                           int h_norms,
                           int max_iter,
                           int* assignments,
                           float* norm_centers) {
    SetBlasThreads(1);

    int d_total_size = 0;
    for (int m = 0; m < M; ++m) {
        for (int l = m; l < M; ++l) {
            d_total_size += codebook_sizes[m] * codebook_sizes[l];
        }
    }

    std::vector<float> D(static_cast<std::size_t>(d_total_size), 0.0f);
    std::vector<float> norms(static_cast<std::size_t>(n_base), 0.0f);

    SetBlasThreads(omp_get_max_threads());
    // D is stored column-major: each block is km×kl with leading dimension km.
    precompute_codebook_dots_mkl(codebooks, codebook_sizes, M, d, D.data());

    SetBlasThreads(1);
    ComputeTrueNorms<CodeT>(B, len_rate_each, D.data(), codebook_sizes, M, n_base, norms.data());

    kmeans_1d_mkl_optimized(norms.data(), n_base, h_norms, max_iter, assignments, norm_centers);
}

}  // namespace

extern "C" {

// 修正的1D K-means函数
void kmeans_1d_mkl_optimized(
    float* data,
    int n,
    int k,
    int max_iter,
    int* assignments,
    float* centers
) {
    // 初始化中心点：选择数据的分位数
    std::vector<float> sorted_data(data, data + n);
    std::sort(sorted_data.begin(), sorted_data.end());

    for (int i = 0; i < k; i++) {
        int idx = (i * n) / k;
        centers[i] = sorted_data[idx];
    }

    std::vector<int> counts(k);
    std::vector<float> new_centers(k);

    for (int iter = 0; iter < max_iter; iter++) {
        // 重置累积器
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(new_centers.begin(), new_centers.end(), 0.0f);

        // 分配数据点到最近的中心点
        #pragma omp parallel default(none) shared(data, centers, assignments, counts, new_centers) firstprivate(n, k)
        {
            // 每个线程的私有累积器
            std::vector<int> local_counts(k, 0);
            std::vector<float> local_centers(k, 0.0f);

            #pragma omp for
            for (int i = 0; i < n; i++) {
                float min_dist = std::abs(data[i] - centers[0]);
                int best_cluster = 0;

                for (int j = 1; j < k; j++) {
                    float dist = std::abs(data[i] - centers[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }

                assignments[i] = best_cluster;
                local_counts[best_cluster]++;
                local_centers[best_cluster] += data[i];
            }

            // 合并线程结果
            #pragma omp critical
            {
                for (int j = 0; j < k; j++) {
                    counts[j] += local_counts[j];
                    new_centers[j] += local_centers[j];
                }
            }
        }

        // 更新中心点
        bool converged = true;
        for (int j = 0; j < k; j++) {
            if (counts[j] > 0) {
                float new_center = new_centers[j] / counts[j];
                if (std::abs(new_center - centers[j]) > 1e-6f) {
                    converged = false;
                }
                centers[j] = new_center;
            }
        }
        
        if (converged) break;
    }
}

// 预计算码本点积 - 使用列优先布局
void precompute_codebook_dots_mkl(
    float** codebooks, const int* codebook_sizes, int M, int d, float* D
) {
    int offset = 0;
    for (int m = 0; m < M; m++) {
        for (int l = m; l < M; l++) {
            int km = codebook_sizes[m];
            int kl = codebook_sizes[l];
            
            // 使用正确的转置和步长参数
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                       km, kl, d,
                       1.0f, codebooks[m], d,   // lda = d (列优先存储)
                             codebooks[l], d,   // ldb = d (列优先存储)
                       0.0f, &D[offset], km);   // ldc = km (输出矩阵列优先)
            
            offset += km * kl;
        }
    }
}


// uint8版本的完整量化函数
void quantize_norms_complete_mkl_uint8(
    uint8_t* B,
    const float* len_rate_each,
    float** codebooks,
    int* codebook_sizes,
    int M,
    int n_base,
    int d,
    int h_norms,
    int max_iter,
    int* assignments,
    float* norm_centers
) {
    QuantizeNormsComplete<uint8_t>(B, len_rate_each, codebooks, codebook_sizes,
                                   M, n_base, d, h_norms, max_iter,
                                   assignments, norm_centers);
}

// uint16版本的完整量化函数
void quantize_norms_complete_mkl_uint16(
    uint16_t* B,
    const float* len_rate_each,
    float** codebooks,
    int* codebook_sizes,
    int M,
    int n_base,
    int d,
    int h_norms,
    int max_iter,
    int* assignments,
    float* norm_centers
) {
    QuantizeNormsComplete<uint16_t>(B, len_rate_each, codebooks, codebook_sizes,
                                    M, n_base, d, h_norms, max_iter,
                                    assignments, norm_centers);
}


} // extern "C"
