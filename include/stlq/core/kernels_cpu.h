#pragma once

#include "stlq/core/kernels.h"

namespace stlq {

class CpuKernels : public KernelProvider {
public:
    void GemmXc(const ColMajorMatrix<float>& C_all,
                const ColMajorMatrix<float>& X,
                ColMajorMatrix<float>* xC) override;

    void UpdateRc(const ColMajorMatrix<float>& xC,
                  const ColMajorMatrix<float>& G,
                  const ColMajorMatrix<Index>& B,
                  const ColMajorMatrix<float>& a,
                  const std::vector<int>& offsets,
                  ColMajorMatrix<float>* rC) override;

    void SolveLeastSquares(const ColMajorMatrix<float>& X,
                           const ColMajorMatrix<float>& C_all,
                           const ColMajorMatrix<Index>& B,
                           const std::vector<int>& offsets,
                           ColMajorMatrix<float>* a) override;

    void IcmStep(const ColMajorMatrix<float>& X,
                 const ColMajorMatrix<float>& C_all,
                 const ColMajorMatrix<float>& G,
                 const std::vector<int>& offsets,
                 ColMajorMatrix<Index>* B,
                 ColMajorMatrix<float>* a) override;
};

}  // namespace stlq
