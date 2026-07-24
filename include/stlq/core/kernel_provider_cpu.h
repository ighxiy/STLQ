#pragma once

#include <vector>

#include "stlq/core/kernel_provider.h"

namespace stlq {

class CpuStreamKernels final : public StreamKernelProvider {
public:
    void ConvertU8ToF32(const ColMajorMatrix<std::uint8_t>& in,
                        ColMajorMatrix<float>* out) override;

    void ConvertU8ToF32AndRotate(const ColMajorMatrix<std::uint8_t>& in,
                                 const ColMajorMatrix<float>& R,
                                 ColMajorMatrix<float>* out) override;

    void ConvertU8ToF32AndRotatePtr(const std::uint8_t* in,
                                   int ld_in,
                                   int rows,
                                   int cols,
                                   const ColMajorMatrix<float>& R,
                                   ColMajorMatrix<float>* out) override;

    void Gemm(bool transA, bool transB,
              float alpha,
              const ColMajorMatrix<float>& A,
              const ColMajorMatrix<float>& B,
              float beta,
              ColMajorMatrix<float>* C) override;

private:
    ColMajorMatrix<float> tmp_;
};

}  // namespace stlq
