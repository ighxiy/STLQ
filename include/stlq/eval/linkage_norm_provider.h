#pragma once

#include <string>
#include <vector>

#include "stlq/eval/linkage_cluster_provider.h"
#include "stlq/quantizer/codebook_meta.h"
#include "stlq/common/types.h"

namespace stlq::eval {

class LinkageNormProviderLookup final : public IClusterNormProvider {
public:
    LinkageNormProviderLookup(const ColMajorMatrix<float>& C_root0,
                            const CodebookMeta& meta_root_small,
                            const CodebookMeta& meta_one);

    bool ComputeNorm2(const ClusterView& cv,
                      std::vector<float>* out_r_norm2,
                      std::string* err) override;

    const ColMajorMatrix<float>& D_rr() const { return D_rr_; }
    const ColMajorMatrix<float>& D_oo() const { return D_oo_; }
    const ColMajorMatrix<float>& D_ro() const { return D_ro_; }

private:
    const ColMajorMatrix<float>& C_root0_;
    const CodebookMeta& meta_root_small_;
    const CodebookMeta& meta_one_;

    // Offsets for root_small, indexed by layer l (0..m-1); valid for l>=1.
    std::vector<int> offsets_root_small_;

    // Global dot tables (Phase 0), column-major.
    ColMajorMatrix<float> D_rr_; // Hrs×Hrs
    ColMajorMatrix<float> D_oo_; // Ho×Ho
    ColMajorMatrix<float> D_ro_; // Hrs×Ho
};

}  // namespace stlq::eval
