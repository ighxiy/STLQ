#include "stlq/quantizer/linear_algebra.h"

#include <cmath>

namespace stlq {

namespace {

constexpr float kEps = 1e-6f;

}  // namespace

bool CholeskyUpper(std::vector<float>& A, int n) {
    for (int k = 0; k < n; ++k) {
        float sum = Mat(A, n, k, k);
        for (int j = 0; j < k; ++j) {
            float v = Mat(A, n, j, k);
            sum -= v * v;
        }
        if (sum <= kEps) {
            return false;
        }
        float diag = std::sqrt(sum);
        Mat(A, n, k, k) = diag;
        for (int i = k + 1; i < n; ++i) {
            float s = Mat(A, n, k, i);
            for (int j = 0; j < k; ++j) {
                s -= Mat(A, n, j, k) * Mat(A, n, j, i);
            }
            Mat(A, n, k, i) = s / diag;
        }
    }
    return true;
}

void SolveCholUpper(const std::vector<float>& R,
                    int n,
                    const float* b,
                    float* x,
                    float* y) {
    for (int i = 0; i < n; ++i) {
        float s = b[i];
        for (int k = 0; k < i; ++k) {
            s -= Mat(R, n, k, i) * y[k];
        }
        y[i] = s / Mat(R, n, i, i);
    }
    for (int i = n - 1; i >= 0; --i) {
        float s = y[i];
        for (int k = i + 1; k < n; ++k) {
            s -= Mat(R, n, i, k) * x[k];
        }
        x[i] = s / Mat(R, n, i, i);
    }
}

void SolveCholUpperRaw(const float* R,
                       int n,
                       const float* b,
                       float* x,
                       float* y,
                       int stride) {
    for (int i = 0; i < n; ++i) {
        float s = b[i];
        for (int k = 0; k < i; ++k) {
            s -= R[static_cast<std::size_t>(k) * stride + i] * y[k];
        }
        y[i] = s / R[static_cast<std::size_t>(i) * stride + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        float s = y[i];
        for (int k = i + 1; k < n; ++k) {
            s -= R[static_cast<std::size_t>(i) * stride + k] * x[k];
        }
        x[i] = s / R[static_cast<std::size_t>(i) * stride + i];
    }
}

}  // namespace stlq
