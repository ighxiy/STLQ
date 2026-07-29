#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <cstdint>
#include <utility>
#include <vector>
#include <immintrin.h> 
#include <cstdlib>
#include <memory>
#include <limits>
#include<chrono>

#include<fstream>
#include<string>
#include <sstream>
#include <iomanip>
using namespace std;


void debug_print(const char* message) {
    FILE* log = fopen("debug.log", "a");
    if (log) {
        fprintf(log, "%s", message);
        fclose(log);
    }
}

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


template <typename AAA>
void linkeded_dot_products(
    std::vector<std::vector<int>>& depth_groups,  
    int real_max_depth,
    float* dot_products,          
    float* residual_dots,         
    const AAA* codes,         
    const float* coefficients,     
    const float* T,                
    const uint32_t* parent,        
    int n_base,                    
    int m,                         
    int h                          
) {
   
    
    

    for (int i = 0; i < n_base; i++) {
        float dot = 0.0f;
        #pragma omp simd
        for (int layer = 0; layer < m; layer++) {
            dot += coefficients[i * m + layer] * T[layer * h + codes[i * m + layer]];
        }
        dot_products[i] = dot;
    }
    
    
    memcpy(residual_dots, dot_products, n_base * sizeof(float));
    
    
    for (int depth = 1; depth <= real_max_depth; depth++) {
        const auto& group = depth_groups[depth];
        if (group.empty()) continue;
        int gs = group.size();

        for (int idx = 0; idx < gs; idx++) {
            int i = group[idx];
            
            dot_products[i] = residual_dots[i] + dot_products[parent[i] - 1];
        }
    }
}





constexpr int ALIGNMENT = 64;


template <size_t A, class T>
static inline T* assume_aligned(T* p) noexcept {
#if defined(__clang__) || defined(__GNUC__)
    return (T*)__builtin_assume_aligned(p, A);
#elif defined(_MSC_VER)
    __assume(((uintptr_t)p % A) == 0);
    return p;
#else
    return p;
#endif
}


static inline void* aligned_malloc_bytes(size_t bytes) {
    return aligned_alloc(ALIGNMENT, bytes);
}

static inline void aligned_free(void* p) { 
    free(p); 
}


static void compute_depths_memo(const uint32_t* parent, int n,
                                int* depths_out, int& real_max_depth) {
    std::fill(depths_out, depths_out + n, -1);
    real_max_depth = 0;
    
    
    const int initial_stack_cap = 256;
    int* stack = new int[initial_stack_cap];
    
    for (int i = 0; i < n; ++i) {
        if (depths_out[i] >= 0) continue;
        
        int stack_sz = 0;
        int stack_cap = initial_stack_cap;
        int v = i;
        
        while (true) {
            if (depths_out[v] >= 0) break;
            if (parent[v] == 0) { 
                depths_out[v] = 0; 
                break; 
            }
            
            
            if (stack_sz == stack_cap) {
                stack_cap *= 2;
                int* new_stack = new int[stack_cap];
                std::copy(stack, stack + stack_sz, new_stack);
                delete[] stack;
                stack = new_stack;
            }
            
            stack[stack_sz++] = v;
            v = static_cast<int>(parent[v]) - 1; 
        }
        
        int base = depths_out[v];
        for (int s = stack_sz - 1; s >= 0; --s) {
            int u = stack[s];
            depths_out[u] = ++base;
        }
        
        if (base > real_max_depth) real_max_depth = base;
    }
    
    delete[] stack;
}




void _linscan_linked_hq_u8_ex(
    float* __restrict dists,
    int32_t* __restrict indices, 
    const uint8_t* __restrict codes,
    const float* __restrict queries,
    const float* __restrict codebooks,
    const float* __restrict dbnorms,
    const float* __restrict coefficients,
    const uint32_t* __restrict parent,
    const int32_t* __restrict codebook_sizes, 
    const int32_t* __restrict codebook_offsets, 
    int n_queries,
    int n_base,
    int m,
    int d,
    int k,
    int max_depth
) {
    
    int total_cb_entries = 0;
    for (int i = 0; i < m; i++) {
        total_cb_entries += codebook_sizes[i];
    }

    
    int* depths = new int[n_base];
    int real_max_depth = 0;
    compute_depths_memo(parent, n_base, depths, real_max_depth);

    
    int* depth_counts = new int[real_max_depth + 2]();
    for (int i = 0; i < n_base; ++i) {
        depth_counts[depths[i] + 1]++;
    }

    
    for (int i = 1; i <= real_max_depth + 1; ++i) {
        depth_counts[i] += depth_counts[i - 1];
    }

    
    int* flat_groups = new int[n_base];
    {
        int* depth_pos = new int[real_max_depth + 1]();
        for (int i = 0; i < n_base; ++i) {
            int depth = depths[i];
            int pos = depth_counts[depth] + depth_pos[depth]++;
            flat_groups[pos] = i;
        }
        delete[] depth_pos;
    }

    
    #pragma omp parallel
    {
        
        float* T = static_cast<float*>(aligned_malloc_bytes(total_cb_entries * sizeof(float)));
        float* dot_products = static_cast<float*>(aligned_malloc_bytes(n_base * sizeof(float)));
        float* distances = static_cast<float*>(aligned_malloc_bytes(n_base * sizeof(float)));
        int32_t* all_indices = static_cast<int32_t*>(aligned_malloc_bytes(n_base * sizeof(int32_t)));
        
        
        T = assume_aligned<ALIGNMENT>(T);
        dot_products = assume_aligned<ALIGNMENT>(dot_products);
        distances = assume_aligned<ALIGNMENT>(distances);
        all_indices = assume_aligned<ALIGNMENT>(all_indices);

        #pragma omp for schedule(static)
        for (int q_idx = 0; q_idx < n_queries; ++q_idx) {
            const float* __restrict query = queries + static_cast<size_t>(q_idx) * d;
            query = assume_aligned<ALIGNMENT>(const_cast<float*>(query));

            
            int entry_idx = 0;
            for (int cb_idx = 0; cb_idx < m; cb_idx++) {
                int cb_size = codebook_sizes[cb_idx];
                int cb_offset = codebook_offsets[cb_idx];
                
                for (int j = 0; j < cb_size; j++) {
                    const float* __restrict centry = codebooks + (cb_offset + j) * d;
                    float sum = 0.0f;
                    
                    
                    #pragma omp simd reduction(+:sum) aligned(query, centry:ALIGNMENT)
                    for (int dim = 0; dim < d; ++dim) {
                        sum += query[dim] * centry[dim];
                    }
                    T[entry_idx++] = sum;
                }
            }

            
            #pragma omp simd aligned(dot_products:ALIGNMENT)
            for (int i = 0; i < n_base; ++i) {
                const uint8_t* __restrict code_ptr = codes + static_cast<size_t>(i) * m;
                const float* __restrict coeff_ptr = coefficients + static_cast<size_t>(i) * m;
                float sum = 0.0f;
                
                for (int layer = 0; layer < m; ++layer) {
                    int cb_offset = codebook_offsets[layer];
                    uint8_t code_val = code_ptr[layer];
                    
                    sum += coeff_ptr[layer] * T[cb_offset + code_val];
                }
                dot_products[i] = sum;
            }
 
            
            for (int depth = 1; depth <= real_max_depth; ++depth) {
                const int start = depth_counts[depth];
                const int end = depth_counts[depth + 1];
                const int size = end - start;

                
                if (depth + 1 <= real_max_depth) {
                    const int next_start = depth_counts[depth + 1];
                    __builtin_prefetch(flat_groups + next_start, 0, 3);
                }

                #pragma omp simd aligned(dot_products:ALIGNMENT)
                for (int t = 0; t < size; ++t) {
                    const int i = flat_groups[start + t];
                    const int p = static_cast<int>(parent[i]) - 1;
                    dot_products[i] += dot_products[p];
                }
            }

            
            #pragma omp simd aligned(distances, dot_products, dbnorms:ALIGNMENT)
            for (int i = 0; i < n_base; ++i) {
                distances[i] = dbnorms[i] - 2.0f * dot_products[i];
                all_indices[i] = i;
            }

            
            int32_t* first = all_indices;
            int32_t* mid = first + k;
            int32_t* last = first + n_base;
            
            auto cmp_dist = [&](int32_t a, int32_t b) noexcept {
                return distances[a] < distances[b];
            };
            
            std::nth_element(first, mid, last, cmp_dist);
            std::sort(first, mid, cmp_dist);

            
            for (int j = 0; j < k; ++j) {
                const int idx = all_indices[j];
                dists[static_cast<size_t>(q_idx) * k + j] = distances[idx];
                indices[static_cast<size_t>(q_idx) * k + j] = idx + 1;
            }
        }

        
        aligned_free(T);
        aligned_free(dot_products);
        aligned_free(distances);
        aligned_free(all_indices);
    }

    
    delete[] depths;
    delete[] depth_counts;
    delete[] flat_groups;
}

void _linscan_linked_hq_u8_ex_twocodebook(
    float* __restrict dists,
    int32_t* __restrict indices, 
    const uint8_t* __restrict codes,
    const float* __restrict queries,
    const float* __restrict codebooks,
    const float* __restrict dbnorms,
    const float* __restrict coefficients,
    const uint32_t* __restrict parent,
    const int32_t* __restrict codebook_sizes, 
    const int32_t* __restrict codebook_offsets, 
    int n_queries,
    int n_base,
    int m,
    int d,
    int k,
    int max_depth
) {
    
    int total_cb_entries = 0;
    for (int i = 0; i < m; i++) {
        total_cb_entries += codebook_sizes[i];
    }

    
    int* depths = new int[n_base];
    int real_max_depth = 0;
    compute_depths_memo(parent, n_base, depths, real_max_depth);

    
    int* depth_counts = new int[real_max_depth + 2]();
    for (int i = 0; i < n_base; ++i) {
        depth_counts[depths[i] + 1]++;
    }

    
    for (int i = 1; i <= real_max_depth + 1; ++i) {
        depth_counts[i] += depth_counts[i - 1];
    }

    
    int* flat_groups = new int[n_base];
    {
        int* depth_pos = new int[real_max_depth + 1]();
        for (int i = 0; i < n_base; ++i) {
            int depth = depths[i];
            int pos = depth_counts[depth] + depth_pos[depth]++;
            flat_groups[pos] = i;
        }
        delete[] depth_pos;
    }

    
    #pragma omp parallel
    {
        
        float* T = static_cast<float*>(aligned_malloc_bytes(total_cb_entries * sizeof(float)));
        float* dot_products = static_cast<float*>(aligned_malloc_bytes(n_base * sizeof(float)));
        float* distances = static_cast<float*>(aligned_malloc_bytes(n_base * sizeof(float)));
        int32_t* all_indices = static_cast<int32_t*>(aligned_malloc_bytes(n_base * sizeof(int32_t)));

        
        T = assume_aligned<ALIGNMENT>(T);
        dot_products = assume_aligned<ALIGNMENT>(dot_products);
        distances = assume_aligned<ALIGNMENT>(distances);
        all_indices = assume_aligned<ALIGNMENT>(all_indices);

        #pragma omp for schedule(static)
        for (int q_idx = 0; q_idx < n_queries; ++q_idx) {
            const float* __restrict query = queries + static_cast<size_t>(q_idx) * d;
            query = assume_aligned<ALIGNMENT>(const_cast<float*>(query));

            
            int entry_idx = 0;
            for (int cb_idx = 0; cb_idx < m; cb_idx++) {
                int cb_size = codebook_sizes[cb_idx];
                int cb_offset = codebook_offsets[cb_idx];

                for (int j = 0; j < cb_size; j++) {
                    const float* __restrict centry = codebooks + (cb_offset + j) * d;
                    float sum = 0.0f;

                    
                    #pragma omp simd reduction(+:sum) aligned(query, centry:ALIGNMENT)
                    for (int dim = 0; dim < d; ++dim) {
                        sum += query[dim] * centry[dim];
                    }
                    T[entry_idx++] = sum;
                }
            }

            int m_split = int(m / 2);
            
            #pragma omp simd aligned(dot_products:ALIGNMENT)
            for (int i = 0; i < n_base; ++i) {
                const uint8_t* __restrict code_ptr = codes + static_cast<size_t>(i) * m_split;
                const float* __restrict coeff_ptr = coefficients + static_cast<size_t>(i) * m_split;
                float sum = 0.0f;

                int offsets = 0;
                if (parent[i] != 0) {
                    offsets = m_split;
                }
                for (int layer = 0; layer < m_split; ++layer) {
                    int cb_offset = codebook_offsets[offsets+layer];
                    uint8_t code_val = code_ptr[layer];

                    sum += coeff_ptr[layer] * T[cb_offset + code_val];
                }
                dot_products[i] = sum;
            }

            
            for (int depth = 1; depth <= real_max_depth; ++depth) {
                const int start = depth_counts[depth];
                const int end = depth_counts[depth + 1];
                const int size = end - start;

                
                if (depth + 1 <= real_max_depth) {
                    const int next_start = depth_counts[depth + 1];
                    __builtin_prefetch(flat_groups + next_start, 0, 3);
                }

                #pragma omp simd aligned(dot_products:ALIGNMENT)
                for (int t = 0; t < size; ++t) {
                    const int i = flat_groups[start + t];
                    const int p = static_cast<int>(parent[i]) - 1;
                    dot_products[i] += dot_products[p];
                }
            }

            
            #pragma omp simd aligned(distances, dot_products, dbnorms:ALIGNMENT)
            for (int i = 0; i < n_base; ++i) {
                distances[i] = dbnorms[i] - 2.0f * dot_products[i];
                all_indices[i] = i;
            }

            
            int32_t* first = all_indices;
            int32_t* mid = first + k;
            int32_t* last = first + n_base;

            auto cmp_dist = [&](int32_t a, int32_t b) noexcept {
                return distances[a] < distances[b];
            };

            std::nth_element(first, mid, last, cmp_dist);
            std::sort(first, mid, cmp_dist);

            
            for (int j = 0; j < k; ++j) {
                const int idx = all_indices[j];
                dists[static_cast<size_t>(q_idx) * k + j] = distances[idx];
                indices[static_cast<size_t>(q_idx) * k + j] = idx + 1;
            }
        }

        
        aligned_free(T);
        aligned_free(dot_products);
        aligned_free(distances);
        aligned_free(all_indices);
    }

    
    delete[] depths;
    delete[] depth_counts;
    delete[] flat_groups;
}



void _linscan_linked_hq_u16_ex_twocodebook(
    float* __restrict dists,
    int32_t* __restrict indices, 
    const uint16_t* __restrict codes,
    const float* __restrict queries,
    const float* __restrict codebooks,
    const float* __restrict dbnorms,
    const float* __restrict coefficients,
    const uint32_t* __restrict parent,
    const int32_t* __restrict codebook_sizes, 
    const int32_t* __restrict codebook_offsets, 
    int n_queries,
    int n_base,
    int m,
    int d,
    int k,
    int max_depth
) {
    
    int total_cb_entries = 0;
    for (int i = 0; i < m; i++) {
        total_cb_entries += codebook_sizes[i];
    }

    
    int* depths = new int[n_base];
    int real_max_depth = 0;
    compute_depths_memo(parent, n_base, depths, real_max_depth);

    
    int* depth_counts = new int[real_max_depth + 2]();
    for (int i = 0; i < n_base; ++i) {
        depth_counts[depths[i] + 1]++;
    }

    
    for (int i = 1; i <= real_max_depth + 1; ++i) {
        depth_counts[i] += depth_counts[i - 1];
    }

    
    int* flat_groups = new int[n_base];
    {
        int* depth_pos = new int[real_max_depth + 1]();
        for (int i = 0; i < n_base; ++i) {
            int depth = depths[i];
            int pos = depth_counts[depth] + depth_pos[depth]++;
            flat_groups[pos] = i;
        }
        delete[] depth_pos;
    }

    
    #pragma omp parallel
    {
        
        float* T = static_cast<float*>(aligned_malloc_bytes(total_cb_entries * sizeof(float)));
        float* dot_products = static_cast<float*>(aligned_malloc_bytes(n_base * sizeof(float)));
        float* distances = static_cast<float*>(aligned_malloc_bytes(n_base * sizeof(float)));
        int32_t* all_indices = static_cast<int32_t*>(aligned_malloc_bytes(n_base * sizeof(int32_t)));

        
        T = assume_aligned<ALIGNMENT>(T);
        dot_products = assume_aligned<ALIGNMENT>(dot_products);
        distances = assume_aligned<ALIGNMENT>(distances);
        all_indices = assume_aligned<ALIGNMENT>(all_indices);

        #pragma omp for schedule(static)
        for (int q_idx = 0; q_idx < n_queries; ++q_idx) {
            const float* __restrict query = queries + static_cast<size_t>(q_idx) * d;
            query = assume_aligned<ALIGNMENT>(const_cast<float*>(query));

            
            int entry_idx = 0;
            for (int cb_idx = 0; cb_idx < m; cb_idx++) {
                int cb_size = codebook_sizes[cb_idx];
                int cb_offset = codebook_offsets[cb_idx];

                for (int j = 0; j < cb_size; j++) {
                    const float* __restrict centry = codebooks + (cb_offset + j) * d;
                    float sum = 0.0f;

                    
                    #pragma omp simd reduction(+:sum) aligned(query, centry:ALIGNMENT)
                    for (int dim = 0; dim < d; ++dim) {
                        sum += query[dim] * centry[dim];
                    }
                    T[entry_idx++] = sum;
                }
            }

            int m_split = int(m / 2);
            
            #pragma omp simd aligned(dot_products:ALIGNMENT)
            for (int i = 0; i < n_base; ++i) {
                const uint16_t* __restrict code_ptr = codes + static_cast<size_t>(i) * m_split;
                const float* __restrict coeff_ptr = coefficients + static_cast<size_t>(i) * m_split;
                float sum = 0.0f;

                int offsets = 0;
                if (parent[i] != 0) {
                    offsets = m_split;
                }
                for (int layer = 0; layer < m_split; ++layer) {
                    int cb_offset = codebook_offsets[offsets+layer];
                    uint16_t code_val = code_ptr[layer];

                    sum += coeff_ptr[layer] * T[cb_offset + code_val];
                }
                dot_products[i] = sum;
            }

            
            for (int depth = 1; depth <= real_max_depth; ++depth) {
                const int start = depth_counts[depth];
                const int end = depth_counts[depth + 1];
                const int size = end - start;

                
                if (depth + 1 <= real_max_depth) {
                    const int next_start = depth_counts[depth + 1];
                    __builtin_prefetch(flat_groups + next_start, 0, 3);
                }

                #pragma omp simd aligned(dot_products:ALIGNMENT)
                for (int t = 0; t < size; ++t) {
                    const int i = flat_groups[start + t];
                    const int p = static_cast<int>(parent[i]) - 1;
                    dot_products[i] += dot_products[p];
                }
            }

            
            #pragma omp simd aligned(distances, dot_products, dbnorms:ALIGNMENT)
            for (int i = 0; i < n_base; ++i) {
                distances[i] = dbnorms[i] - 2.0f * dot_products[i];
                all_indices[i] = i;
            }

            
            int32_t* first = all_indices;
            int32_t* mid = first + k;
            int32_t* last = first + n_base;

            auto cmp_dist = [&](int32_t a, int32_t b) noexcept {
                return distances[a] < distances[b];
            };

            std::nth_element(first, mid, last, cmp_dist);
            std::sort(first, mid, cmp_dist);

            
            for (int j = 0; j < k; ++j) {
                const int idx = all_indices[j];
                dists[static_cast<size_t>(q_idx) * k + j] = distances[idx];
                indices[static_cast<size_t>(q_idx) * k + j] = idx + 1;
            }
        }

        
        aligned_free(T);
        aligned_free(dot_products);
        aligned_free(distances);
        aligned_free(all_indices);
    }

    
    delete[] depths;
    delete[] depth_counts;
    delete[] flat_groups;
}



void _linscan_linked_hq_u16_ex(
    float* __restrict dists,
    int32_t* __restrict indices, 
    const uint16_t* __restrict codes,
    const float* __restrict queries,
    const float* __restrict codebooks,
    const float* __restrict dbnorms,
    const float* __restrict coefficients,
    const uint32_t* __restrict parent,
    const int32_t* __restrict codebook_sizes, 
    const int32_t* __restrict codebook_offsets, 
    int n_queries,
    int n_base,
    int m,
    int d,
    int k,
    int max_depth
) {
    
    int total_cb_entries = 0;
    for (int i = 0; i < m; i++) {
        total_cb_entries += codebook_sizes[i];
    }

    
    int* depths = new int[n_base];
    int real_max_depth = 0;
    compute_depths_memo(parent, n_base, depths, real_max_depth);

    
    int* depth_counts = new int[real_max_depth + 2]();
    for (int i = 0; i < n_base; ++i) {
        depth_counts[depths[i] + 1]++;
    }

    
    for (int i = 1; i <= real_max_depth + 1; ++i) {
        depth_counts[i] += depth_counts[i - 1];
    }

    
    int* flat_groups = new int[n_base];
    {
        int* depth_pos = new int[real_max_depth + 1]();
        for (int i = 0; i < n_base; ++i) {
            int depth = depths[i];
            int pos = depth_counts[depth] + depth_pos[depth]++;
            flat_groups[pos] = i;
        }
        delete[] depth_pos;
    }

    
    #pragma omp parallel
    {
        
        float* T = static_cast<float*>(aligned_malloc_bytes(total_cb_entries * sizeof(float)));
        float* dot_products = static_cast<float*>(aligned_malloc_bytes(n_base * sizeof(float)));
        float* distances = static_cast<float*>(aligned_malloc_bytes(n_base * sizeof(float)));
        int32_t* all_indices = static_cast<int32_t*>(aligned_malloc_bytes(n_base * sizeof(int32_t)));
        
        
        T = assume_aligned<ALIGNMENT>(T);
        dot_products = assume_aligned<ALIGNMENT>(dot_products);
        distances = assume_aligned<ALIGNMENT>(distances);
        all_indices = assume_aligned<ALIGNMENT>(all_indices);

        #pragma omp for schedule(static)
        for (int q_idx = 0; q_idx < n_queries; ++q_idx) {
            const float* __restrict query = queries + static_cast<size_t>(q_idx) * d;
            query = assume_aligned<ALIGNMENT>(const_cast<float*>(query));

            
            int entry_idx = 0;
            for (int cb_idx = 0; cb_idx < m; cb_idx++) {
                int cb_size = codebook_sizes[cb_idx];
                int cb_offset = codebook_offsets[cb_idx];
                
                for (int j = 0; j < cb_size; j++) {
                    const float* __restrict centry = codebooks + (cb_offset + j) * d;
                    float sum = 0.0f;
                    
                    
                    #pragma omp simd reduction(+:sum) aligned(query, centry:ALIGNMENT)
                    for (int dim = 0; dim < d; ++dim) {
                        sum += query[dim] * centry[dim];
                    }
                    T[entry_idx++] = sum;
                }
            }

            
            #pragma omp simd aligned(dot_products:ALIGNMENT)
            for (int i = 0; i < n_base; ++i) {
                const uint16_t* __restrict code_ptr = codes + static_cast<size_t>(i) * m;
                const float* __restrict coeff_ptr = coefficients + static_cast<size_t>(i) * m;
                float sum = 0.0f;
                
                for (int layer = 0; layer < m; ++layer) {
                    int cb_offset = codebook_offsets[layer];
                    uint16_t code_val = code_ptr[layer];
                    
                    sum += coeff_ptr[layer] * T[cb_offset + code_val];
                }
                dot_products[i] = sum;
            }
 
            
            for (int depth = 1; depth <= real_max_depth; ++depth) {
                const int start = depth_counts[depth];
                const int end = depth_counts[depth + 1];
                const int size = end - start;

                
                if (depth + 1 <= real_max_depth) {
                    const int next_start = depth_counts[depth + 1];
                    __builtin_prefetch(flat_groups + next_start, 0, 3);
                }

                #pragma omp simd aligned(dot_products:ALIGNMENT)
                for (int t = 0; t < size; ++t) {
                    const int i = flat_groups[start + t];
                    const int p = static_cast<int>(parent[i]) - 1;
                    dot_products[i] += dot_products[p];
                }
            }

            
            #pragma omp simd aligned(distances, dot_products, dbnorms:ALIGNMENT)
            for (int i = 0; i < n_base; ++i) {
                distances[i] = dbnorms[i] - 2.0f * dot_products[i];
                all_indices[i] = i;
            }

            
            int32_t* first = all_indices;
            int32_t* mid = first + k;
            int32_t* last = first + n_base;
            
            auto cmp_dist = [&](int32_t a, int32_t b) noexcept {
                return distances[a] < distances[b];
            };
            
            std::nth_element(first, mid, last, cmp_dist);
            std::sort(first, mid, cmp_dist);

            
            for (int j = 0; j < k; ++j) {
                const int idx = all_indices[j];
                dists[static_cast<size_t>(q_idx) * k + j] = distances[idx];
                indices[static_cast<size_t>(q_idx) * k + j] = idx + 1;
            }
        }

        
        aligned_free(T);
        aligned_free(dot_products);
        aligned_free(distances);
        aligned_free(all_indices);
    }

    
    delete[] depths;
    delete[] depth_counts;
    delete[] flat_groups;
}




extern "C"{

	void linscan_linked_hq_u8_ex(
	    float* dists, int32_t* idx,
	    uint8_t* codes, float* queries, float* codebooks,
	    float* dbnorms, float* coefficients, uint32_t* parent,
	    int32_t* codebook_sizes, int32_t* codebook_offsets,
	    int nqueries, int ncodes,
	    int m, int d, int knn, int max_depth){
	    
	    	_linscan_linked_hq_u8_ex(dists, idx, codes,  queries, codebooks, dbnorms, coefficients,  parent, codebook_sizes, codebook_offsets, nqueries, ncodes, m, d,  knn,  max_depth);
	    }
	       
	void linscan_linked_hq_u16_ex(
	    float* dists, int32_t* idx,
	    uint16_t* codes, float* queries, float* codebooks,
	    float* dbnorms, float* coefficients, uint32_t* parent,
	    int32_t* codebook_sizes, int32_t* codebook_offsets,
	    int nqueries, int ncodes,
	    int m, int d, int knn, int max_depth){
	    
	    	_linscan_linked_hq_u16_ex(dists, idx, codes,  queries, codebooks, dbnorms, coefficients,  parent, codebook_sizes, codebook_offsets, nqueries, ncodes, m, d,  knn,  max_depth);
	    }
	    



void linscan_linked_hq_u8_ex_twocodebook(
float* dists, int32_t* idx,
uint8_t* codes, float* queries, float* codebooks,
float* dbnorms, float* coefficients, uint32_t* parent,
int32_t* codebook_sizes, int32_t* codebook_offsets,
int nqueries, int ncodes,
int m, int d, int knn, int max_depth){

	    _linscan_linked_hq_u8_ex_twocodebook(dists, idx, codes,  queries, codebooks, dbnorms, coefficients,  parent, codebook_sizes, codebook_offsets, nqueries, ncodes, m, d,  knn,  max_depth);
	    }
	    
void linscan_linked_hq_u16_ex_twocodebook(
float* dists, int32_t* idx,
uint16_t* codes, float* queries, float* codebooks,
float* dbnorms, float* coefficients, uint32_t* parent,
int32_t* codebook_sizes, int32_t* codebook_offsets,
int nqueries, int ncodes,
int m, int d, int knn, int max_depth){

	    _linscan_linked_hq_u16_ex_twocodebook(dists, idx, codes,  queries, codebooks, dbnorms, coefficients,  parent, codebook_sizes, codebook_offsets, nqueries, ncodes, m, d,  knn,  max_depth);
	   }
	   



void linscan_linked_hq_u8_ex_twocodebook_virtual(
    float* __restrict dists,
    int32_t* __restrict indices,
    const uint8_t* __restrict codes,
    const float* __restrict queries,
    const float* __restrict codebooks,
    const float* __restrict dbnorms,
    const float* __restrict coefficients,
    const uint32_t* __restrict parent,
    const int32_t* __restrict codebook_sizes,
    const int32_t* __restrict codebook_offsets,
    int n_queries,
    int n_base_real,
    int n_base,
    int m,
    int d,
    int k
) {
    
    int total_cb_entries = 0;
    for (int i = 0; i < m; i++) {
        total_cb_entries += codebook_sizes[i];
    }

    
    int* depths = new int[n_base];
    int real_max_depth = 0;
    compute_depths_memo(parent, n_base, depths, real_max_depth);

    
    int* depth_counts = new int[real_max_depth + 2]();
    for (int i = 0; i < n_base; ++i) {
        depth_counts[depths[i] + 1]++;
    }

    
    for (int i = 1; i <= real_max_depth + 1; ++i) {
        depth_counts[i] += depth_counts[i - 1];
    }

    
    int* flat_groups = new int[n_base];
    {
        int* depth_pos = new int[real_max_depth + 1]();
        for (int i = 0; i < n_base; ++i) {
            int depth = depths[i];
            int pos = depth_counts[depth] + depth_pos[depth]++;
            flat_groups[pos] = i;
        }
        delete[] depth_pos;
    }

    
    #pragma omp parallel
    {
        
        float* T = static_cast<float*>(aligned_malloc_bytes(total_cb_entries * sizeof(float)));
        float* dot_products = static_cast<float*>(aligned_malloc_bytes(n_base * sizeof(float)));
        float* distances = static_cast<float*>(aligned_malloc_bytes(n_base_real * sizeof(float)));
        int32_t* all_indices = static_cast<int32_t*>(aligned_malloc_bytes(n_base_real * sizeof(int32_t)));

        
        T = assume_aligned<ALIGNMENT>(T);
        dot_products = assume_aligned<ALIGNMENT>(dot_products);
        distances = assume_aligned<ALIGNMENT>(distances);
        all_indices = assume_aligned<ALIGNMENT>(all_indices);

        #pragma omp for schedule(static)
        for (int q_idx = 0; q_idx < n_queries; ++q_idx) {
            const float* __restrict query = queries + static_cast<size_t>(q_idx) * d;
            query = assume_aligned<ALIGNMENT>(const_cast<float*>(query));

            
            int entry_idx = 0;
            for (int cb_idx = 0; cb_idx < m; cb_idx++) {
                int cb_size = codebook_sizes[cb_idx];
                int cb_offset = codebook_offsets[cb_idx];

                for (int j = 0; j < cb_size; j++) {
                    const float* __restrict centry = codebooks + (cb_offset + j) * d;
                    float sum = 0.0f;

                    
                    #pragma omp simd reduction(+:sum) aligned(query, centry:ALIGNMENT)
                    for (int dim = 0; dim < d; ++dim) {
                        sum += query[dim] * centry[dim];
                    }
                    T[entry_idx++] = sum;
                }
            }

            int m_split = int(m / 2);
            
            #pragma omp simd aligned(dot_products:ALIGNMENT)
            for (int i = 0; i < n_base; ++i) {
                const uint8_t* __restrict code_ptr = codes + static_cast<size_t>(i) * m_split;
                const float* __restrict coeff_ptr = coefficients + static_cast<size_t>(i) * m_split;
                float sum = 0.0f;

                int offsets = 0;
                if (parent[i] != 0) {
                    offsets = m_split;
                }
                for (int layer = 0; layer < m_split; ++layer) {
                    int cb_offset = codebook_offsets[offsets+layer];
                    uint8_t code_val = code_ptr[layer];

                    sum += coeff_ptr[layer] * T[cb_offset + code_val];
                }
                dot_products[i] = sum;
            }

            
            for (int depth = 1; depth <= real_max_depth; ++depth) {
                const int start = depth_counts[depth];
                const int end = depth_counts[depth + 1];
                const int size = end - start;

                
                if (depth + 1 <= real_max_depth) {
                    const int next_start = depth_counts[depth + 1];
                    __builtin_prefetch(flat_groups + next_start, 0, 3);
                }

                #pragma omp simd aligned(dot_products:ALIGNMENT)
                for (int t = 0; t < size; ++t) {
                    const int i = flat_groups[start + t];
                    const int p = static_cast<int>(parent[i]) - 1;
                    dot_products[i] += dot_products[p];
                }
            }

            
            #pragma omp simd aligned(distances, dot_products, dbnorms:ALIGNMENT)
            for (int i = 0; i < n_base_real; ++i) {
                float dist = dbnorms[i] - 2.0f * dot_products[i];

                distances[i]  = dist;
                all_indices[i] = i;
            }

            
            int32_t* first = all_indices;
            int32_t* mid = first + k;
            int32_t* last = first + n_base_real;

            auto cmp_dist = [&](int32_t a, int32_t b) noexcept {
                return distances[a] < distances[b];
            };

            std::nth_element(first, mid, last, cmp_dist);
            std::sort(first, mid, cmp_dist);

            
            for (int j = 0; j < k; ++j) {
                const int idx = all_indices[j];
                dists[static_cast<size_t>(q_idx) * k + j] = distances[idx];
                indices[static_cast<size_t>(q_idx) * k + j] = idx + 1;
            }
        }

        
        aligned_free(T);
        aligned_free(dot_products);
        aligned_free(distances);
        aligned_free(all_indices);
    }

    
    delete[] depths;
    delete[] depth_counts;
    delete[] flat_groups;
}



}
