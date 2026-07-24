#include "stlq/linkage/linkage_builder.h"
#include "stlq/linkage/linkage_builder_virtual.h"
#include "stlq/linkage/linkage_summary.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include "stlq/linkage/linkage_build_profile.h"
#include "stlq/linkage/linkage_finalize.h"
#include "stlq/core/blas.h"
#include "stlq/core/lapack.h"
#include "stlq/common/logger.h"
#include "stlq/knn/hnsw_cluster_knn.h"
#include "stlq/quantizer/encoder.h"
#include "stlq/quantizer/linear_algebra.h"
#if defined(STLQ_ENABLE_CUDA)
#include "stlq/cuda/cuda_stream_kernels_pool.h"
#include "stlq/linkage/eval_candidates_cuda.h"
#endif
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"

namespace stlq {

namespace {
#include "linkage_builder_common.inc"
#include "linkage_builder_one_for_init.inc"
#include "linkage_builder_two_inner_to_outer.inc"
#include "linkage_builder_two_multicenter.inc"
}

#include "linkage_builder_interface.inc"
#include "linkage_builder_interface_virtual.inc"

}  // namespace stlq
