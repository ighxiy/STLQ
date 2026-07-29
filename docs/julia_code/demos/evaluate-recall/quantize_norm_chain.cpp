#include <mkl.h>
#include <mkl_cblas.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <queue>
#include <unordered_map>



std::vector<int> compute_depths(const uint32_t* parent, int n) {
    std::vector<int> depths(n, 0);
    for (int i = 0; i < n; i++) {
        int cur = i;
        int depth = 0;
        while (parent[cur] != 0) {
            depth++;
            cur = parent[cur] - 1;
        }
        depths[i] = depth;
    }
    return depths;
}



inline int col_major_idx(int row, int col, int num_rows) {
    return row + col * num_rows;
}


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




template <typename T>
float compute_linkeded_dot_product_iterative_twocodebook(
    int start_idx,           
    int target_layer,        
    T target_code_idx,     
    T* B,              
    float* len_rate_each,    
    float* D,                
    int* codebook_sizes,     
    uint32_t* parent,        
    int M_split,                   
    int M, 
    const std::vector<int>& d_offsets  
) {
    float dot_val = 0.0f;
    int current = start_idx;

    while (true) {
        if (parent[current] != 0) {
            for (int m = 0; m < M_split; m++) {
                T code_idx_m = B[col_major_idx(m, current, M_split)];
                float lambda_m = len_rate_each[col_major_idx(m, current, M_split)];
                
                int layer1 = target_layer;
                int layer2 = m + M_split;
                T idx1 = target_code_idx;
                T idx2 = code_idx_m;
                if (layer1 > layer2) {
                    std::swap(layer1, layer2);
                    std::swap(idx1, idx2);
                }
                int kt = codebook_sizes[layer1];
                
                int d_off = d_offsets[layer1 * M + layer2];
                dot_val += lambda_m * D[d_off + col_major_idx(idx1, idx2, kt)];
            }
            current = parent[current] - 1;
        }
        else{
            for (int m = 0; m < M_split; m++) {
                T code_idx_m = B[col_major_idx(m, current, M_split)];
                float lambda_m = len_rate_each[col_major_idx(m, current, M_split)];
                
                int km= codebook_sizes[m];
                int d_off = d_offsets[m * M + target_layer];
                dot_val += lambda_m * D[d_off + col_major_idx(code_idx_m, target_code_idx, km)];
            }
            return dot_val;
        }
    }
    return dot_val;
}






template <typename T>
void compute_linkeded_norms_generic_twocodebook(
    T* B,
    float* len_rate_each,
    float* D,
    int* codebook_sizes,
    int M,
    int n_base,
    uint32_t* parent,      
    float* norms,          
    int max_depth          
) {
    
    std::vector<int> depths = compute_depths(parent, n_base);
    int M_split = int(M / 2);

    
    int real_max_depth = *std::max_element(depths.begin(), depths.end());
    std::vector<std::vector<int>> depth_groups(real_max_depth + 1);
    for (int i = 0; i < n_base; i++) {
        depth_groups[depths[i]].push_back(i);
    }
    
    
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
        
        std::vector<float> lambda_cache(M_split);
        std::vector<T> code_cache(M_split);
        
        
        for (int depth = 0; depth <= real_max_depth; depth++) {
            const auto& group = depth_groups[depth];
            if (group.empty()) continue;
            
            
            #pragma omp for schedule(static)
            for (int idx = 0; idx < group.size(); idx++) {
                int i = group[idx];
                
                if (depth == 0) {
                    
                    float temp = 0.0f;
                    
                    
                    for (int m = 0; m < M_split; m++) {
                        lambda_cache[m] = len_rate_each[col_major_idx(m, i, M_split)];
                        code_cache[m] = B[col_major_idx(m, i, M_split)];
                    }
                    
                    
                    for (int m = 0; m < M_split; m++) {
                        T code_idx_m = code_cache[m];
                        float lambda_m = lambda_cache[m];
                        
                        int d_offset = d_offsets[m * M + m];
                        int km = codebook_sizes[m];
                        float dot_val = D[d_offset + col_major_idx(code_idx_m, code_idx_m, km)];
                        
                        temp += lambda_m * lambda_m * dot_val;
                    }
                    
                    
                    for (int m = 0; m < M_split; m++) {
                        T code_idx_m = code_cache[m];
                        float lambda_m = lambda_cache[m];
                        
                        for (int l = m + 1; l < M_split; l++) {
                            T code_idx_l = code_cache[l];
                            float lambda_l = lambda_cache[l];
                            
                            int d_offset = d_offsets[m * M + l];
                            int km = codebook_sizes[m];
                            float dot_val = D[d_offset + col_major_idx(code_idx_m, code_idx_l, km)];
                            
                            temp += 2.0f * lambda_m * lambda_l * dot_val;
                        }
                    }
                    
                    norms[i] = temp;
                } else {
                    
                    uint32_t p_idx = parent[i] - 1;  
                    
                    
                    float residual_norm = 0.0f;
                    
                    
                    for (int m = 0; m < M_split; m++) {
                        lambda_cache[m] = len_rate_each[col_major_idx(m, i, M_split)];
                        code_cache[m] = B[col_major_idx(m, i, M_split)];
                    }
                    
                    
                    for (int m = 0; m < M_split; m++) {
                        T code_idx_m = code_cache[m];
                        float lambda_m = lambda_cache[m];

                        int m_offset = m+M_split;
                        int d_offset = d_offsets[m_offset * M + m_offset];
                        int km = codebook_sizes[m_offset];
                        float dot_val = D[d_offset + col_major_idx(code_idx_m, code_idx_m, km)];
                        
                        residual_norm += lambda_m * lambda_m * dot_val;
                    }
                    
                    
                    for (int m = 0; m < M_split; m++) {
                        T code_idx_m = code_cache[m];
                        float lambda_m = lambda_cache[m];
                        
                        for (int l = m + 1; l < M_split; l++) {
                            T code_idx_l = code_cache[l];
                            float lambda_l = lambda_cache[l];

                            int m_offset = m+M_split;
                            int d_offset = d_offsets[m_offset * M + l + M_split];
                            int km = codebook_sizes[m_offset];
                            float dot_val = D[d_offset + col_major_idx(code_idx_m, code_idx_l, km)];
                            
                            residual_norm += 2.0f * lambda_m * lambda_l * dot_val;
                        }
                    }
                    
                    
                    float dot_product = 0.0f;
                    for (int l = 0; l < M_split; l++) {
                        
                        T r_code_idx_l = code_cache[l];
                        float r_lambda_l = lambda_cache[l];
                        int l_offset = l + M_split;
                        
                        dot_product += 2.0f * r_lambda_l * compute_linkeded_dot_product_iterative_twocodebook(
                            p_idx, l_offset, r_code_idx_l,
                            B, len_rate_each, D, codebook_sizes, parent, M_split, M, d_offsets
                        );
                    }
                    
                    
                    norms[i] = norms[p_idx] + residual_norm + dot_product;
                }
            }
        }
    }
}


extern "C" {

void quantize_norms_complete_mkl_uint16_twocodebook(
    uint16_t* B,
    float* len_rate_each,
    float** codebooks,
    int* codebook_sizes,
    int M,
    int n_base,
    int d,
    int h_norms,
    int max_iter,
    uint32_t* parent,      
    int max_depth,          
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
    compute_linkeded_norms_generic_twocodebook<uint16_t>(
        B, len_rate_each, D.data(), codebook_sizes, M, n_base, parent, norms.data(), max_depth
    );
    kmeans_1d_mkl_optimized(norms.data(), n_base, h_norms, max_iter, 
                           assignments, norm_centers);
}




void quantize_norms_complete_mkl_uint8_twocodebook(
    uint8_t* B,
    float* len_rate_each,
    float** codebooks,
    int* codebook_sizes,
    int M,
    int n_base,
    int d,
    int h_norms,
    int max_iter,
    uint32_t* parent,      
    int max_depth,          
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
    compute_linkeded_norms_generic_twocodebook<uint8_t>(
        B, len_rate_each, D.data(), codebook_sizes, M, n_base, parent, norms.data(), max_depth
    );
    kmeans_1d_mkl_optimized(norms.data(), n_base, h_norms, max_iter, 
                           assignments, norm_centers);
}

} 
