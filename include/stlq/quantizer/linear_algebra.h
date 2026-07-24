#pragma once

#include <vector>

#include "stlq/common/types.h"

namespace stlq {

inline float& Mat(std::vector<float>& A, int n, int r, int c) {
    return A[static_cast<std::size_t>(r) * n + c];
}

inline const float& Mat(const std::vector<float>& A, int n, int r, int c) {
    return A[static_cast<std::size_t>(r) * n + c];
}

inline float GAt(const ColMajorMatrix<float>& G, int row, int col) {
    return G.data[static_cast<std::size_t>(col) * G.rows + row];
}

bool CholeskyUpper(std::vector<float>& A, int n);

void SolveCholUpper(const std::vector<float>& R,
                    int n,
                    const float* b,
                    float* x,
                    float* y);

void SolveCholUpperRaw(const float* R,
                       int n,
                       const float* b,
                       float* x,
                       float* y,
                       int stride);

}  // namespace stlq
