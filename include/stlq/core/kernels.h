#pragma once

#include "stlq/common/types.h"

namespace stlq {

class KernelProvider {
public:
    virtual ~KernelProvider() = default;

    virtual void GemmXc(const ColMajorMatrix<float>& C_all,
                        const ColMajorMatrix<float>& X,
                        ColMajorMatrix<float>* xC) = 0;

    virtual void UpdateRc(const ColMajorMatrix<float>& xC,
                          const ColMajorMatrix<float>& G,
                          const ColMajorMatrix<Index>& B,
                          const ColMajorMatrix<float>& a,
                          const std::vector<int>& offsets,
                          ColMajorMatrix<float>* rC) = 0;

    virtual void SolveLeastSquares(const ColMajorMatrix<float>& X,
                                   const ColMajorMatrix<float>& C_all,
                                   const ColMajorMatrix<Index>& B,
                                   const std::vector<int>& offsets,
                                   ColMajorMatrix<float>* a) = 0;

    virtual void IcmStep(const ColMajorMatrix<float>& X,
                         const ColMajorMatrix<float>& C_all,
                         const ColMajorMatrix<float>& G,
                         const std::vector<int>& offsets,
                         ColMajorMatrix<Index>* B,
                         ColMajorMatrix<float>* a) = 0;
};

}  // namespace stlq
