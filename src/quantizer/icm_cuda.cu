#include "stlq/quantizer/icm_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>

#include <cuda_runtime.h>

#include "stlq/common/logger.h"
#include "stlq/common/timer.h"
#include "stlq/core/model_limits.h"
#include "stlq/quantizer/cost_utils.h"
#include "stlq/quantizer/icm.h"
#include "stlq/quantizer/precomp_large_root.h"

namespace stlq {

#include "icm_cuda_kernels.inc"
#include "icm_cuda_api.inc"
