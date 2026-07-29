#include <mkl.h>
#include <mkl_cblas.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>


inline int col_major_idx(int row, int col, int num_rows) {
    return row + col * num_rows;
}


extern "C" {


void kmeans_1d_mkl_optimized(
    float* data,
    int n,
    int k,
    int max_iter,
    int* assignments,
    float* centers
) {
    
    std::vector<float> sorted_data(data, data + n);
    std::sort(sorted_data.begin(), sorted_data.end());
    
    for (int i = 0; i < k; i++) {
        int idx = (i * n) / k;
        centers[i] = sorted_data[idx];
    }
    
    std::vector<int> counts(k);
    std::vector<float> new_centers(k);
    
    for (int iter = 0; iter < max_iter; iter++) {
        
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(new_centers.begin(), new_centers.end(), 0.0f);
        
        
        #pragma omp parallel
        {
            
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
            
            
            #pragma omp critical
            {
                for (int j = 0; j < k; j++) {
                    counts[j] += local_counts[j];
                    new_centers[j] += local_centers[j];
                }
            }
        }
        
        
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


void precompute_codebook_dots_mkl(
    float** codebooks, int* codebook_sizes, int M, int d, float* D 
) {
    int offset = 0;
    for (int m = 0; m < M; m++) {
        for (int l = m; l < M; l++) {
            int km = codebook_sizes[m];
            int kl = codebook_sizes[l];
            
            
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                       km, kl, d,
                       1.0f, codebooks[m], d,   
                             codebooks[l], d,   
                       0.0f, &D[offset], km);   
            
            offset += km * kl;
        }
    }
}



void compute_true_norms_mkl_optimized_uint8(
    uint8_t* B, float* len_rate_each, float* D, int* codebook_sizes, int M, int n_base, float* norms 
) {
    
    std::vector<int> d_offsets(M * M);
    int offset = 0;
    for (int m = 0; m < M; m++) {
        for (int l = m; l < M; l++) {
            d_offsets[m * M + l] = offset;
            d_offsets[l * M + m] = offset;
            offset += codebook_sizes[m] * codebook_sizes[l];
        }
    }

    #pragma omp parallel
    {
        
        std::vector<float> lambda_cache(M);
        std::vector<uint8_t> code_cache(M);
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n_base; i++) {
            
            for (int m = 0; m < M; m++) {
                lambda_cache[m] = len_rate_each[col_major_idx(m, i, M)];
                code_cache[m] = B[col_major_idx(m, i, M)];
            }
            
            float temp = 0.0f;
            
            
            for (int m = 0; m < M; m++) {
                uint8_t code_idx_m = code_cache[m];
                float lambda_m = lambda_cache[m];
                
                int d_offset = d_offsets[m * M + m];
                int km = codebook_sizes[m];
                
                float dot_val = D[d_offset + col_major_idx(code_idx_m, code_idx_m, km)];
                
                temp += lambda_m * lambda_m * dot_val;
            }
            
            
            for (int m = 0; m < M; m++) {
                uint8_t code_idx_m = code_cache[m];
                float lambda_m = lambda_cache[m];
                
                for (int l = m + 1; l < M; l++) {
                    uint8_t code_idx_l = code_cache[l];
                    float lambda_l = lambda_cache[l];
                    
                    int d_offset = d_offsets[m * M + l];
                    int km = codebook_sizes[m];
                    
                    float dot_val = D[d_offset + col_major_idx(code_idx_m, code_idx_l, km)];
                    
                    temp += 2.0f * lambda_m * lambda_l * dot_val;
                }
            }
            
            norms[i] = temp;
        }
    }
}


void compute_true_norms_mkl_optimized_uint16(
    uint16_t* B,
    float* len_rate_each,
    float* D,
    int* codebook_sizes,
    int M,
    int n_base,
    float* norms
) {
    
    std::vector<int> d_offsets(M * M);
    int offset = 0;
    for (int m = 0; m < M; m++) {
        for (int l = m; l < M; l++) {
            d_offsets[m * M + l] = offset;
            d_offsets[l * M + m] = offset;
            offset += codebook_sizes[m] * codebook_sizes[l];
        }
    }
    
    #pragma omp parallel
    {
        
        std::vector<float> lambda_cache(M);
        std::vector<uint16_t> code_cache(M);
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n_base; i++) {
            
            for (int m = 0; m < M; m++) {
                lambda_cache[m] = len_rate_each[col_major_idx(m, i, M)];
                code_cache[m] = B[col_major_idx(m, i, M)];
            }
            
            float temp = 0.0f;
            
            
            for (int m = 0; m < M; m++) {
                uint16_t code_idx_m = code_cache[m];
                float lambda_m = lambda_cache[m];
                
                int d_offset = d_offsets[m * M + m];
                int km = codebook_sizes[m];
                float dot_val = D[d_offset + col_major_idx(code_idx_m, code_idx_m, km)];
                
                temp += lambda_m * lambda_m * dot_val;
            }
            
            
            for (int m = 0; m < M; m++) {
                uint16_t code_idx_m = code_cache[m];
                float lambda_m = lambda_cache[m];
                
                for (int l = m + 1; l < M; l++) {
                    uint16_t code_idx_l = code_cache[l];
                    float lambda_l = lambda_cache[l];
                    
                    int d_offset = d_offsets[m * M + l];
                    int km = codebook_sizes[m];
                    float dot_val = D[d_offset + col_major_idx(code_idx_m, code_idx_l, km)];
                    
                    temp += 2.0f * lambda_m * lambda_l * dot_val;
                }
            }
            
            norms[i] = temp;
        }
    }
}


void quantize_norms_complete_mkl_uint8(
    uint8_t* B,
    float* len_rate_each,
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
    
    mkl_set_num_threads(1);
    
    
    int d_total_size = 0;
    for (int m = 0; m < M; m++) {
        for (int l = m; l < M; l++) {
            d_total_size += codebook_sizes[m] * codebook_sizes[l];
        }
    }
    
    std::vector<float> D(d_total_size);
    std::vector<float> norms(n_base);
    
    
    mkl_set_num_threads(omp_get_max_threads());
    precompute_codebook_dots_mkl(codebooks, codebook_sizes, M, d, D.data());
    
    
    mkl_set_num_threads(1);
    compute_true_norms_mkl_optimized_uint8(B, len_rate_each, D.data(), 
                                          codebook_sizes, M, n_base, norms.data());
    
    kmeans_1d_mkl_optimized(norms.data(), n_base, h_norms, max_iter, 
                           assignments, norm_centers);
}


void quantize_norms_complete_mkl_uint16(
    uint16_t* B,
    float* len_rate_each,
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
    
    mkl_set_num_threads(1);
    
    
    int d_total_size = 0;
    for (int m = 0; m < M; m++) {
        for (int l = m; l < M; l++) {
            d_total_size += codebook_sizes[m] * codebook_sizes[l];
        }
    }
    
    std::vector<float> D(d_total_size);
    std::vector<float> norms(n_base);
    
    
    mkl_set_num_threads(omp_get_max_threads());
    precompute_codebook_dots_mkl(codebooks, codebook_sizes, M, d, D.data());
    
    
    mkl_set_num_threads(1);
    compute_true_norms_mkl_optimized_uint16(B, len_rate_each, D.data(), 
                                           codebook_sizes, M, n_base, norms.data());
    
    kmeans_1d_mkl_optimized(norms.data(), n_base, h_norms, max_iter, 
                           assignments, norm_centers);
}


} 

