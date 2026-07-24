#include "stlq/quantizer/encoder.h"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "stlq/core/lapack.h"
#include "stlq/core/blas.h"
#include "stlq/core/threading.h"
#if defined(STLQ_ENABLE_CUDA)
#include "stlq/core/kernel_provider_cuda_stream.h"
#endif
#include "stlq/linkage/linkage_builder.h"
#include "stlq/linkage/linkage_builder_virtual.h"
#include "stlq/linkage/linkage_reconstruction.h"
#include "stlq/linkage/linkage_summary.h"
#include "stlq/quantizer/beam_search.h"
#include "stlq/quantizer/icm.h"
#if defined(STLQ_ENABLE_CUDA)
#include "stlq/quantizer/icm_cuda.h"
#endif
#include "stlq/quantizer/least_squares.h"
#include "stlq/quantizer/linear_algebra.h"
#include "stlq/quantizer/reconstruct.h"
#include "stlq/quantizer/spkmeans.h"
#include "stlq/io/col_block_reader.h"
#include "stlq/io/dataset_io.h"
#include "stlq/quantizer/rvq_layer_store.h"
#include "stlq/linkage/virtual_augment.h"
#include "stlq/common/logger.h"
#include "stlq/common/timer.h"

#include "hnswlib/hnswlib.h"
#include "hnswlib/space_l2.h"

namespace stlq {

namespace {

#include "encoder_internal.inc"

}  // namespace

#include "encoder_precomp.inc"
#include "encoder_train.inc"
#include "encoder_encode_base.inc"

}  // namespace stlq
