#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <cstdint>
#include <utility>
#include <vector>
using namespace std;


void _linscan_hq_with_coeff_u16_ex(
    float* dists,         
    int*   idx,           
    uint16_t* codes,      
    float* queries,       
    float* codebooks,     
    float* dbnorms,       
    float* coefficients,  
    int* codebook_sizes,  
    int* codebook_offsets,
    int nqueries,         
    int ncodes,           
    int m,                
    int d,                
    int nn)               
{
    
    int total_cb_entries = 0;
    for (int i = 0; i < m; i++) {
        total_cb_entries += codebook_sizes[i];
    }
    
    int buffer_size = (int)1e7;
    int npairs = min(ncodes, buffer_size + nn);

    #pragma omp parallel for
    for(int i = 0; i < nqueries; i++) {       
        
        float* query  = queries + i * d;      		     
        float* tentry = new float[ total_cb_entries ](); 

        
        int entry_idx = 0;
        for (int cb_idx = 0; cb_idx < m; cb_idx++) {
            int cb_size = codebook_sizes[cb_idx];
            int cb_offset = codebook_offsets[cb_idx];
            
            for (int j = 0; j < cb_size; j++) {
                float* centry = codebooks + (cb_offset + j) * d;
                for (int k = 0; k < d; k++) {
                    tentry[entry_idx] -= 2 * query[k] * centry[k]; 
                }
                entry_idx++;
            }
        }

        std::pair<float, int>* pairs = new std::pair<float, int>[npairs]();

        
        uint16_t* code = codes;
        float* coeff = coefficients;  
        int normidx = 0;

        int from = 0;
        while (from < ncodes) {
            int offset = (from > 0) ? nn : 0;

            for (long j = offset; j < min(ncodes, from + buffer_size + (nn - offset)) - from + offset; 
                 j++, code += m, coeff += m, normidx++) {
                
                pairs[j].first = 0;
                for (int k = 0; k < m; k++) {  
                    float lambda_k = coeff[k];  
                    int cb_size = codebook_sizes[k];
                    int cb_offset = codebook_offsets[k];
                    
                    
                    float base_value = tentry[cb_offset + code[k]];
                    pairs[j].first += lambda_k * base_value;    
                }
                pairs[j].first += dbnorms[normidx]; 
                pairs[j].second = j + 1 + from - offset; 
            }

            from = min(ncodes, from + buffer_size + (nn - offset));

            
            std::partial_sort(pairs, pairs + nn, pairs + npairs);
        }

        for (long j = 0; j < nn; j++) {
            dists[i * nn + j] = pairs[j].first;
            idx[i * nn + j] = pairs[j].second;
        }
        
        delete[] pairs;
        delete[] tentry;
    }
}



void _linscan_hq_with_coeff_u8_ex(
    float* dists,         
    int*   idx,           
    uint8_t* codes,      
    float* queries,       
    float* codebooks,     
    float* dbnorms,       
    float* coefficients,  
    int* codebook_sizes,  
    int* codebook_offsets,
    int nqueries,         
    int ncodes,           
    int m,                
    int d,                
    int nn)               
{
    
    int total_cb_entries = 0;
    for (int i = 0; i < m; i++) {
        total_cb_entries += codebook_sizes[i];
    }
    
    int buffer_size = (int)1e7;
    int npairs = min(ncodes, buffer_size + nn);

    #pragma omp parallel for
    for(int i = 0; i < nqueries; i++) {       
        
        float* query  = queries + i * d;      		     
        float* tentry = new float[ total_cb_entries ](); 

        
        int entry_idx = 0;
        for (int cb_idx = 0; cb_idx < m; cb_idx++) {
            int cb_size = codebook_sizes[cb_idx];
            int cb_offset = codebook_offsets[cb_idx];
            
            for (int j = 0; j < cb_size; j++) {
                float* centry = codebooks + (cb_offset + j) * d;
                for (int k = 0; k < d; k++) {
                    tentry[entry_idx] -= 2 * query[k] * centry[k]; 
                }
                entry_idx++;
            }
        }

        std::pair<float, int>* pairs = new std::pair<float, int>[npairs]();

        
        uint8_t* code = codes;
        float* coeff = coefficients;  
        int normidx = 0;

        int from = 0;
        while (from < ncodes) {
            int offset = (from > 0) ? nn : 0;

            for (long j = offset; j < min(ncodes, from + buffer_size + (nn - offset)) - from + offset; 
                 j++, code += m, coeff += m, normidx++) {
                
                pairs[j].first = 0;
                for (int k = 0; k < m; k++) {  
                    float lambda_k = coeff[k];  
                    int cb_size = codebook_sizes[k];
                    int cb_offset = codebook_offsets[k];
                    
                    
                    float base_value = tentry[cb_offset + code[k]];
                    pairs[j].first += lambda_k * base_value;    
                }
                pairs[j].first += dbnorms[normidx]; 
                pairs[j].second = j + 1 + from - offset; 
            }

            from = min(ncodes, from + buffer_size + (nn - offset));

            
            std::partial_sort(pairs, pairs + nn, pairs + npairs);
        }

        for (long j = 0; j < nn; j++) {
            dists[i * nn + j] = pairs[j].first;
            idx[i * nn + j] = pairs[j].second;
        }
        
        delete[] pairs;
        delete[] tentry;
    }
}




extern "C"
{
 
        void linscan_hq_with_coeff_u16_ex( 
        float* dists, int* idx,
        uint16_t* codes, float* queries, float* codebooks, float* dbnorms, float* coefficients, int* codebook_sizes, int* codebook_offsets,
        int nqueries, int ncodes, int m, int d, int nn ) {

        _linscan_hq_with_coeff_u16_ex( dists, idx,
            codes, queries, codebooks, dbnorms, coefficients,codebook_sizes, codebook_offsets,
            nqueries, ncodes, m, d, nn );
     }   
            
        void linscan_hq_with_coeff_u8_ex( 
        float* dists, int* idx,
        uint8_t* codes, float* queries, float* codebooks, float* dbnorms, float* coefficients, int* codebook_sizes, int* codebook_offsets,
        int nqueries, int ncodes, int m, int d, int nn ) {

        _linscan_hq_with_coeff_u8_ex( dists, idx,
            codes, queries, codebooks, dbnorms, coefficients,codebook_sizes, codebook_offsets,
            nqueries, ncodes, m, d, nn );
    }
    
}
