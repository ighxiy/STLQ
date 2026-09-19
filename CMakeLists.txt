cmake_minimum_required(VERSION 3.16)
project(STLQ LANGUAGES C CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "STLQ: CMAKE_BUILD_TYPE not set; defaulting to Release.")
endif()

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:/W4>
                        $<$<COMPILE_LANGUAGE:CXX>:/W4>) # warning level 4
else()
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wall>
                        $<$<COMPILE_LANGUAGE:C>:-Wextra>
                        $<$<COMPILE_LANGUAGE:C>:-Wpedantic>
                        $<$<COMPILE_LANGUAGE:C>:-Wunused>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wunused>) # GCC/Clang
endif()

# ---------------------------------------------------------------------------
# SIMD / auto-vectorization diagnostic report (off by default)
#
# Enable with:  cmake -DSTLQ_VEC_REPORT=ON ...
#
# GCC/Clang: emits missed-vectorization notes per translation unit via
#   -fopt-info-vec-missed   (auto-vec only; quieter)
#   -fopt-info-vec-omp-missed (auto-vec + omp simd; more verbose)
#
# MSVC:  /Qvec-report:2  reports both vectorized and not-vectorized loops.
#        Line numbers in Release (/O2) are precise because MSVC reports before
#        the back-end transform, unlike GCC which reports after SSA lowering.
#        Tip: build with /Od (Debug, no optimizations) on MSVC for exact lines.
#
# Noise reduction tips:
#   - Most "missed" entries from <bits/stl_heap.h>, <bits/stl_algo.h>, etc.
#     are system-header internals; they are not actionable user code.
#   - Use -fopt-info-vec-missed (without "-omp") to suppress OMP barrier spam.
#   - On GCC, Debug (-O0) disables auto-vectorization so no notes are emitted;
#     use -O2 or -O3 with -g for accurate line numbers.
# ---------------------------------------------------------------------------
option(STLQ_VEC_REPORT "Emit compiler auto-vectorization diagnostic messages" OFF)
if(STLQ_VEC_REPORT)
    if(MSVC)
        # /Qvec-report:2 → report both vectorized (1) and blocked (2) loops.
        # /Qpar-report:2 → same for the parallel auto-parallelizer (/Qpar).
        # These flags apply only to CXX to avoid confusing the CUDA host path.
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Qvec-report:2>)
    else()
        # -fopt-info-vec-missed is quieter than -fopt-info-vec-omp-missed:
        # it skips the OMP-barrier "statement clobbers memory" noise that comes
        # from every #pragma omp parallel block even when the inner loop IS
        # vectorized.  Use -fopt-info-vec-omp-missed for full detail.
        add_compile_options(-fopt-info-vec-missed)
    endif()
endif()

# ---------------------------------------------------------------------------
# Link-time optimization / whole-program optimization
#
# Goals:
# - Enable LTO for C/C++ Release-style builds on Linux GCC and Windows MSVC.
# - Keep CUDA compilation behavior unchanged.
#   The current binary size problem is already dominated by embedded CUDA fatbins,
#   and host-side LTO is still useful for the C/C++ code.
# ---------------------------------------------------------------------------
option(STLQ_ENABLE_LTO "Enable C/C++ link-time optimization in Release-like builds" ON)

function(stlq_apply_lto target_name)
    if(NOT STLQ_ENABLE_LTO)
        return()
    endif()
    if(NOT TARGET ${target_name})
        return()
    endif()

    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>>:/GL>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>>:/GL>
        )
        target_link_options(${target_name} PRIVATE
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:/LTCG>
        )
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target_name} PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>>:-flto>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>>:-flto>
        )
        target_link_options(${target_name} PRIVATE
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:-flto>
        )
    endif()
endfunction()


set(_STLQ_USE_MKL_DEFAULT ON)
if(WIN32)
    # On Windows, MKL is typically used via MSVC; keep the default off to avoid surprise configure/link errors.
    set(_STLQ_USE_MKL_DEFAULT OFF)
endif()
option(STLQ_USE_MKL "Use Intel MKL for BLAS" ${_STLQ_USE_MKL_DEFAULT})
option(STLQ_ENABLE_CUDA "Enable CUDA stream kernels (cuBLAS)" ON)
option(STLQ_ENABLE_HDF5 "Enable HDF5 result IO compatibility" ON)
option(BUILD_TESTING "Build STLQ test executables" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
if(STLQ_ENABLE_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS C)
endif()

add_library(stlq_core
    src/stlq.cpp
    src/pipeline/app_utils.cpp
    src/pipeline/main_virtual_helpers.cpp
    src/pipeline/stlq_config_bootstrap.cpp
    src/pipeline/stlq_large_base_stages.cpp
    src/pipeline/stlq_entry_stages.cpp
    src/pipeline/stlq_train_quantizer_flow.cpp
    src/pipeline/stlq_large_base_flow.cpp
    src/pipeline/stlq_large_pipeline_flow.cpp
    src/pipeline/stlq_large_linkage_stages.cpp
    src/pipeline/stlq_large_eval_archive.cpp
    src/pipeline/stlq_large_eval_execute.cpp
    src/pipeline/stlq_large_eval_plan.cpp
    src/pipeline/stlq_large_eval_stages.cpp
    src/pipeline/stlq_non_large_pipeline_flow.cpp
    src/pipeline/large_store_hash.cpp
    src/pipeline/large_store_lifecycle.cpp
    src/common/config.cpp
    src/common/logger.cpp
    src/common/timer.cpp
    src/core/blas.cpp
    src/core/kernel_provider_cpu.cpp
    src/core/kernels_cpu.cpp
    src/io/dataset_io.cpp
    src/io/dataset_reader_factory.cpp
    src/io/bvecs_reader.cpp
    src/io/fvecs_reader.cpp
    src/io/fbin_reader.cpp
    src/io/ibin_reader.cpp
    src/io/ivecs_reader.cpp
    src/io/base_store.cpp
    src/io/base_list_store.cpp
    src/io/ivf_lists.cpp
    src/io/linkage_store_virtual.cpp
    src/io/linkage_list_store.cpp
    src/io/linkage_coeff_codec_store.cpp
    src/io/col_block_reader.cpp
    src/io/result_io.cpp
    src/quantizer/encoder.cpp
    src/quantizer/encoder_train_streaming.cpp
    src/quantizer/beam_search.cpp
    src/quantizer/icm.cpp
    src/quantizer/cost_utils.cpp
    src/quantizer/least_squares.cpp
    src/quantizer/linear_algebra.cpp
    src/quantizer/encode_base_streaming.cpp
    src/quantizer/precomp_large_root.cpp
    src/quantizer/reconstruct.cpp
    src/quantizer/rvq_layer_store.cpp
    src/quantizer/spkmeans.cpp
    src/quantizer/streaming_train_config.cpp
    src/quantizer/streaming_train_init_stages.cpp
    src/quantizer/streaming_train_iteration_stages.cpp
    src/quantizer/streaming_train_io.cpp
    src/quantizer/streaming_train_round_stages.cpp
    src/quantizer/update_codebooks_streaming.cpp
    src/linkage/virtual_augment.cpp
    src/linkage/hnsw_cluster_knn.cpp
    src/linkage/reference_forest.cpp
    src/linkage/linkage_builder.cpp
    src/linkage/linkage_base_streaming_builder.cpp
    src/linkage/linkage_coeff_codec_rebuild.cpp
    src/linkage/linkage_init_streaming_builder.cpp
    src/linkage/linkage_streaming_support.cpp
    src/linkage/linkage_reconstruction.cpp
    src/linkage/linkage_tools.cpp
    src/coeff/huffman_canonical.cpp
    src/coeff/coeff_cluster_quantizer.cpp
    src/coeff/coeff_cluster_codec.cpp
    src/succinct/parent_louds.cpp
    src/eval/recall_base.cpp
    src/eval/recall_linkage.cpp
    src/eval/recall_disk.cpp
    src/eval/eval_metrics.cpp
    src/eval/norm2_lut.cpp
    src/eval/query_table_builder.cpp
    src/eval/linkage_cluster_provider.cpp
    src/eval/linkage_norm_provider.cpp
    src/eval/recall_linkage_disk.cpp
    src/eval/recall_linkage_disk_parent_louds_native.cpp
    src/ivf/legacy/quantize_norm_base.cpp
    src/ivf/ivf_probe_hnsw_index.cpp
)
stlq_apply_lto(stlq_core)

target_include_directories(stlq_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(stlq_core SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(stlq_core PUBLIC
    OpenMP::OpenMP_CXX
    Threads::Threads
)

if(MSVC)
    # This project uses `#pragma omp simd` and some OpenMP 4+ constructs.
    # MSVC requires `/openmp:experimental` for `omp simd` to compile.
    # IMPORTANT: don't pass MSVC-only flags directly to nvcc when CUDA is enabled.
    target_compile_options(stlq_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
endif()

if(STLQ_ENABLE_CUDA)
    # CMake may fail to auto-detect a default CUDA architecture (common in some IDE toollinkages).
    # Provide a safe default for RTX 4090 (sm_89); override via:
    #   -DSTLQ_CUDA_ARCHITECTURES=89
    # or
    #   -DCMAKE_CUDA_ARCHITECTURES=89
    set(STLQ_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures (CMake CUDA_ARCHITECTURES format)")
    # Keep `CMAKE_CUDA_ARCHITECTURES` in sync with `STLQ_CUDA_ARCHITECTURES` across re-configures.
    # Otherwise, changing only `-DSTLQ_CUDA_ARCHITECTURES=...` won't take effect because the previous
    # cached `CMAKE_CUDA_ARCHITECTURES` value wins.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR
       "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "" OR
       NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "${STLQ_CUDA_ARCHITECTURES}")
        set(CMAKE_CUDA_ARCHITECTURES "${STLQ_CUDA_ARCHITECTURES}" CACHE STRING
            "CUDA arch list; set this explicitly if CMake can't detect a default." FORCE)
    endif()

    # NVCC needs to be able to spawn the host compiler (cl.exe) reliably on Windows.
    # Some setups have cl.exe on PATH (VsDevCmd), yet NVCC still fails to locate it unless
    # we provide an explicit host compiler path.
    if(WIN32 AND (NOT DEFINED CMAKE_CUDA_HOST_COMPILER OR "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL ""))
        # Prefer the already-detected C++ compiler when it is MSVC cl.exe.
        if(MSVC AND DEFINED CMAKE_CXX_COMPILER AND "${CMAKE_CXX_COMPILER}" MATCHES "cl\\.exe$")
            set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH
                "Host compiler used by NVCC (cl.exe)" FORCE)
        else()
            find_program(_STLQ_MSVC_CL_EXE cl.exe)
            if(_STLQ_MSVC_CL_EXE)
                set(CMAKE_CUDA_HOST_COMPILER "${_STLQ_MSVC_CL_EXE}" CACHE FILEPATH
                    "Host compiler used by NVCC (cl.exe)" FORCE)
            else()
                message(WARNING "CUDA enabled but cl.exe not found. "
                                "Run VsDevCmd.bat or set -DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>.")
            endif()
        endif()
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

		    target_sources(stlq_core PRIVATE
		        src/core/kernel_provider_cuda_stream.cu
		        src/cuda/cuda_stream_kernels_pool.cu
		        src/linkage/linkage_encode_cuda.cu
		        src/linkage/eval_candidates_cuda.cu
		        src/eval/linkage_norm_provider_cuda.cu
		        src/eval/recall_linkage_scan_cuda.cu
		        src/quantizer/beam_search_cuda.cu
		        src/quantizer/icm_cuda.cu
		    )
    set_target_properties(stlq_core PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # Workaround: on some Windows setups NVCC fails to locate cl.exe even when MSVC is active in the environment.
    # Force NVCC to use the same host compiler directory as `CMAKE_CXX_COMPILER`.
    if(WIN32 AND MSVC AND DEFINED CMAKE_CXX_COMPILER AND "${CMAKE_CXX_COMPILER}" MATCHES "cl\\.exe$")
        get_filename_component(_STLQ_MSVC_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        if(_STLQ_MSVC_BIN_DIR AND NOT _STLQ_MSVC_BIN_DIR STREQUAL "")
            target_compile_options(stlq_core PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--compiler-bindir=${_STLQ_MSVC_BIN_DIR}>
            )
        endif()
    endif()
    target_compile_definitions(stlq_core PUBLIC STLQ_ENABLE_CUDA=1)
    target_link_libraries(stlq_core PUBLIC CUDA::cudart CUDA::cublas)
else()
    # CPU-only build: provide stubs so translation units that contain unreachable CUDA fast paths
    # (guarded only by runtime checks) can still link successfully.
    target_sources(stlq_core PRIVATE
        src/core/kernel_provider_cuda_stream_stub.cpp
    )
endif()

if(STLQ_ENABLE_HDF5)
    target_sources(stlq_core PRIVATE src/io/hdf5_io.cpp)
    target_compile_definitions(stlq_core PRIVATE STLQ_ENABLE_HDF5=1)
    if(TARGET HDF5::HDF5)
        target_link_libraries(stlq_core PRIVATE HDF5::HDF5)
    else()
        target_include_directories(stlq_core PRIVATE ${HDF5_INCLUDE_DIRS})
        target_link_libraries(stlq_core PRIVATE ${HDF5_LIBRARIES})
    endif()
endif()

target_compile_definitions(stlq_core PUBLIC
    STLQ_COL_MAJOR=1
    $<$<BOOL:${STLQ_USE_MKL}>:STLQ_USE_MKL=1>
)

if(STLQ_USE_MKL)
    set(MKLROOT "" CACHE PATH "Root directory of Intel MKL")
    if(NOT MKLROOT AND DEFINED ENV{MKLROOT})
        set(MKLROOT "$ENV{MKLROOT}")
    endif()
    if(NOT MKLROOT)
        if(WIN32)
            set(MKLROOT "C:/Program Files (x86)/Intel/oneAPI/mkl/latest")
        else()
            set(MKLROOT "/opt/intel/oneapi/mkl/latest")
        endif()
    endif()

    find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKLROOT} PATH_SUFFIXES include)
    # Prefer MKL's unified runtime library when available (simplest across platforms/toollinkages).
    find_library(MKL_RT mkl_rt PATHS ${MKLROOT} PATH_SUFFIXES lib/intel64)
    if(NOT MKL_RT)
        find_library(MKL_INTEL_LP64 mkl_intel_lp64 PATHS ${MKLROOT} PATH_SUFFIXES lib/intel64)
        # GNU-threaded MKL is the common path on Linux; MSVC users should typically rely on mkl_rt.
        find_library(MKL_GNU_THREAD mkl_gnu_thread PATHS ${MKLROOT} PATH_SUFFIXES lib/intel64)
        find_library(MKL_CORE mkl_core PATHS ${MKLROOT} PATH_SUFFIXES lib/intel64)
    endif()

    if(NOT MKL_INCLUDE_DIR OR (NOT MKL_RT AND (NOT MKL_INTEL_LP64 OR NOT MKL_GNU_THREAD OR NOT MKL_CORE)))
        message(FATAL_ERROR "MKL not found. Set MKLROOT or disable STLQ_USE_MKL.")
    endif()
    if(MKL_RT)
        get_filename_component(MKL_LIB_DIR ${MKL_RT} DIRECTORY)
    else()
        get_filename_component(MKL_LIB_DIR ${MKL_CORE} DIRECTORY)
    endif()
    if(MKL_RT)
        message(STATUS "stlq_core BLAS: MKL (mkl_rt) MKLROOT=${MKLROOT}")
        message(STATUS "stlq_core BLAS: MKL include=${MKL_INCLUDE_DIR} lib=${MKL_RT}")
    else()
        message(STATUS "stlq_core BLAS: MKL (split libs) MKLROOT=${MKLROOT}")
        message(STATUS "stlq_core BLAS: MKL include=${MKL_INCLUDE_DIR} libs=${MKL_INTEL_LP64};${MKL_GNU_THREAD};${MKL_CORE}")
    endif()

    # Ensure MKL headers (mkl.h / mkl_lapacke.h) are visible to this target's compilation
    # across generators (notably Ninja).
    target_include_directories(stlq_core PRIVATE ${MKL_INCLUDE_DIR})
    target_include_directories(stlq_core INTERFACE ${MKL_INCLUDE_DIR})
    # Some IDE-bundled Ninja toollinkages fail to propagate include directories reliably.
    # Add an explicit include flag for C++ compilation as a last-resort to keep builds working.
    # IMPORTANT: Do not pass MSVC-style `/I...` directly to NVCC; some NVCC versions treat it as an input
    # file which then triggers:
    #   "nvcc fatal: A single input file is required for a non-link phase when an outputfile is specified"
    if(MSVC)
        target_compile_options(stlq_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/I${MKL_INCLUDE_DIR}>)
    else()
        target_compile_options(stlq_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-I${MKL_INCLUDE_DIR}>)
    endif()
    if(MKL_RT)
        target_link_libraries(stlq_core PUBLIC ${MKL_RT})
    else()
        target_link_libraries(stlq_core PUBLIC
            ${MKL_INTEL_LP64}
            ${MKL_GNU_THREAD}
            ${MKL_CORE}
        )
    endif()
    if(UNIX)
        target_link_libraries(stlq_core PUBLIC m dl)
    endif()
else()
    # OpenBLAS + system LAPACK (Ubuntu/Windows): avoid accidentally picking MKL via FindBLAS.
    #
    # Windows note:
    # - With CUDA enabled, CMake will typically use MSVC as the host toollinkage.
    # - Many users have a standalone OpenBLAS zip (include/lib/bin) that CMake can't auto-detect.
    #   Provide STLQ_OPENBLAS_ROOT as an escape hatch.
    set(STLQ_OPENBLAS_ROOT "" CACHE PATH "Root directory of OpenBLAS (optional override)")

    if(STLQ_OPENBLAS_ROOT)
        set(_OPENBLAS_INC "${STLQ_OPENBLAS_ROOT}/include")
        if(NOT EXISTS "${_OPENBLAS_INC}/cblas.h")
            message(FATAL_ERROR "OpenBLAS not found under STLQ_OPENBLAS_ROOT: missing include/cblas.h")
        endif()
        target_include_directories(stlq_core PUBLIC "${_OPENBLAS_INC}")

        set(_OPENBLAS_LIB "")
        if(WIN32)
            if(EXISTS "${STLQ_OPENBLAS_ROOT}/lib/libopenblas.lib")
                set(_OPENBLAS_LIB "${STLQ_OPENBLAS_ROOT}/lib/libopenblas.lib")
            elseif(EXISTS "${STLQ_OPENBLAS_ROOT}/lib/libopenblas.dll.a")
                set(_OPENBLAS_LIB "${STLQ_OPENBLAS_ROOT}/lib/libopenblas.dll.a")
            elseif(EXISTS "${STLQ_OPENBLAS_ROOT}/lib/libopenblas.a")
                set(_OPENBLAS_LIB "${STLQ_OPENBLAS_ROOT}/lib/libopenblas.a")
            endif()
        else()
            find_library(_OPENBLAS_LIB openblas PATHS "${STLQ_OPENBLAS_ROOT}" PATH_SUFFIXES lib NO_DEFAULT_PATH)
        endif()

        if(NOT _OPENBLAS_LIB)
            message(FATAL_ERROR "OpenBLAS library not found under STLQ_OPENBLAS_ROOT. Expected libopenblas.* in lib/.")
        endif()
        message(STATUS "stlq_core BLAS: OpenBLAS (STLQ_OPENBLAS_ROOT) inc=${_OPENBLAS_INC} lib=${_OPENBLAS_LIB}")
        target_link_libraries(stlq_core PUBLIC "${_OPENBLAS_LIB}")
        target_compile_definitions(stlq_core PUBLIC STLQ_USE_OPENBLAS=1)
    else()
        set(BLA_VENDOR OpenBLAS)
        find_package(BLAS REQUIRED)
        message(STATUS "stlq_core BLAS: FindBLAS vendor=OpenBLAS libs=${BLAS_LIBRARIES}")
        # Best-effort detection to enable openblas_set_num_threads at runtime.
        string(TOLOWER "${BLAS_LIBRARIES}" STLQ_BLAS_LIBS_LOWER)
        if(STLQ_BLAS_LIBS_LOWER MATCHES "openblas")
            target_compile_definitions(stlq_core PUBLIC STLQ_USE_OPENBLAS=1)
        endif()
        if(TARGET BLAS::BLAS)
            target_link_libraries(stlq_core PUBLIC BLAS::BLAS)
        else()
            target_link_libraries(stlq_core PUBLIC ${BLAS_LIBRARIES})
        endif()

        # cblas.h isn't provided by FindBLAS; try to locate headers so `<cblas.h>` builds.
        find_path(OPENBLAS_INCLUDE_DIR cblas.h)
        if(OPENBLAS_INCLUDE_DIR)
            target_include_directories(stlq_core PUBLIC ${OPENBLAS_INCLUDE_DIR})
        endif()
    endif()
    find_package(LAPACK QUIET)
    if(LAPACK_FOUND)
        if(TARGET LAPACK::LAPACK)
            target_link_libraries(stlq_core PUBLIC LAPACK::LAPACK)
        else()
            target_link_libraries(stlq_core PUBLIC ${LAPACK_LIBRARIES})
        endif()
    else()
        # Common on Windows/MinGW where OpenBLAS may ship LAPACK symbols but CMake can't locate a separate LAPACK package.
        message(STATUS "LAPACK not found via CMake; assuming BLAS provides LAPACK symbols (e.g. OpenBLAS build with LAPACK).")
    endif()

    # LAPACKE (C interface) is preferred when available.
    find_path(LAPACKE_INCLUDE_DIR lapacke.h)
    find_library(LAPACKE_LIB lapacke)
    if(LAPACKE_INCLUDE_DIR AND LAPACKE_LIB)
        target_include_directories(stlq_core PUBLIC ${LAPACKE_INCLUDE_DIR})
        target_link_libraries(stlq_core PUBLIC ${LAPACKE_LIB})
        target_compile_definitions(stlq_core PUBLIC STLQ_HAS_LAPACKE=1)
    endif()
endif()

add_executable(stlq_main
    cmd/stlq_main.cpp
)
target_link_libraries(stlq_main PRIVATE stlq_core)
stlq_apply_lto(stlq_main)

add_executable(export_stlq_gt_pairs cmd/export_stlq_gt_pairs.cpp)
target_link_libraries(export_stlq_gt_pairs PRIVATE stlq_core)
stlq_apply_lto(export_stlq_gt_pairs)

add_executable(export_stlq_cluster_angles cmd/export_stlq_cluster_angles.cpp)
target_link_libraries(export_stlq_cluster_angles PRIVATE stlq_core)
stlq_apply_lto(export_stlq_cluster_angles)

add_executable(export_stlq_query_adc cmd/export_stlq_query_adc.cpp)
target_link_libraries(export_stlq_query_adc PRIVATE stlq_core)
stlq_apply_lto(export_stlq_query_adc)

add_executable(export_stlq_candidate_features cmd/export_stlq_candidate_features.cpp)
target_link_libraries(export_stlq_candidate_features PRIVATE stlq_core)
stlq_apply_lto(export_stlq_candidate_features)

add_executable(report_stlq_candidate_feature_summary cmd/report_stlq_candidate_feature_summary.cpp)
target_link_libraries(report_stlq_candidate_feature_summary PRIVATE stlq_core)
stlq_apply_lto(report_stlq_candidate_feature_summary)

add_executable(report_linkage_depth cmd/report_linkage_depth.cpp)
target_link_libraries(report_linkage_depth PRIVATE stlq_core)
stlq_apply_lto(report_linkage_depth)

add_executable(report_linkage_graph_quality cmd/report_linkage_graph_quality.cpp)
target_link_libraries(report_linkage_graph_quality PRIVATE stlq_core)
stlq_apply_lto(report_linkage_graph_quality)

add_executable(report_stlq_oracle_rerank cmd/report_stlq_oracle_rerank.cpp)
target_link_libraries(report_stlq_oracle_rerank PRIVATE stlq_core)
stlq_apply_lto(report_stlq_oracle_rerank)

add_executable(bench_stlq_exact_rerank cmd/bench_stlq_exact_rerank.cpp)
target_link_libraries(bench_stlq_exact_rerank PRIVATE stlq_core)
stlq_apply_lto(bench_stlq_exact_rerank)

add_executable(report_linkage_int8_recon_mse cmd/report_linkage_int8_recon_mse.cpp)
target_link_libraries(report_linkage_int8_recon_mse PRIVATE stlq_core)
stlq_apply_lto(report_linkage_int8_recon_mse)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_reference_forest_and_subkmeans tests/test_reference_forest_and_subkmeans.cpp)
    target_link_libraries(test_reference_forest_and_subkmeans PRIVATE stlq_core)
    stlq_apply_lto(test_reference_forest_and_subkmeans)
    add_test(NAME test_reference_forest_and_subkmeans COMMAND test_reference_forest_and_subkmeans)

    add_executable(stlq_io_probe tests/checks/io_probe.cpp)
    target_link_libraries(stlq_io_probe PRIVATE stlq_core)
    stlq_apply_lto(stlq_io_probe)

    add_executable(stlq_parent_louds_probe tests/checks/parent_louds_probe.cpp)
    target_link_libraries(stlq_parent_louds_probe PRIVATE stlq_core)
    stlq_apply_lto(stlq_parent_louds_probe)

    add_executable(test_streaming_encode_base tests/test_streaming_encode_base.cpp)
    target_link_libraries(test_streaming_encode_base PRIVATE stlq_core)
    stlq_apply_lto(test_streaming_encode_base)
    add_test(NAME test_streaming_encode_base COMMAND test_streaming_encode_base)
    add_executable(test_eval_metrics tests/test_eval_metrics.cpp)
    target_link_libraries(test_eval_metrics PRIVATE stlq_core)
    stlq_apply_lto(test_eval_metrics)
    add_test(NAME test_eval_metrics COMMAND test_eval_metrics)
    add_executable(test_large_eval_plan tests/test_large_eval_plan.cpp)
    target_link_libraries(test_large_eval_plan PRIVATE stlq_core)
    stlq_apply_lto(test_large_eval_plan)
    add_test(NAME test_large_eval_plan COMMAND test_large_eval_plan)
    add_executable(test_large_eval_archive tests/test_large_eval_archive.cpp)
    target_link_libraries(test_large_eval_archive PRIVATE stlq_core)
    stlq_apply_lto(test_large_eval_archive)
    add_test(NAME test_large_eval_archive COMMAND test_large_eval_archive)
    add_executable(test_large_store_lifecycle tests/test_large_store_lifecycle.cpp)
    target_link_libraries(test_large_store_lifecycle PRIVATE stlq_core)
    stlq_apply_lto(test_large_store_lifecycle)
    add_test(NAME test_large_store_lifecycle COMMAND test_large_store_lifecycle)
    add_executable(test_config_model_h0 tests/test_config_model_h0.cpp)
    target_link_libraries(test_config_model_h0 PRIVATE stlq_core)
    stlq_apply_lto(test_config_model_h0)
    add_test(NAME test_config_model_h0 COMMAND test_config_model_h0 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    add_executable(test_train_result_bin_io tests/test_train_result_bin_io.cpp)
    target_link_libraries(test_train_result_bin_io PRIVATE stlq_core)
    stlq_apply_lto(test_train_result_bin_io)
    add_test(NAME test_train_result_bin_io COMMAND test_train_result_bin_io)
    add_executable(test_parent_louds_virtual_bfs tests/test_parent_louds_virtual_bfs.cpp)
    target_link_libraries(test_parent_louds_virtual_bfs PRIVATE stlq_core)
    stlq_apply_lto(test_parent_louds_virtual_bfs)
    add_test(NAME test_parent_louds_virtual_bfs COMMAND test_parent_louds_virtual_bfs)
    add_executable(test_linkage_list_parent_louds_matches_u32 tests/test_linkage_list_parent_louds_matches_u32.cpp)
    target_link_libraries(test_linkage_list_parent_louds_matches_u32 PRIVATE stlq_core)
    stlq_apply_lto(test_linkage_list_parent_louds_matches_u32)
    add_test(NAME test_linkage_list_parent_louds_matches_u32 COMMAND test_linkage_list_parent_louds_matches_u32)
    add_executable(test_linkage_list_parent_louds_without_u32 tests/test_linkage_list_parent_louds_without_u32.cpp)
    target_link_libraries(test_linkage_list_parent_louds_without_u32 PRIVATE stlq_core)
    stlq_apply_lto(test_linkage_list_parent_louds_without_u32)
    add_test(NAME test_linkage_list_parent_louds_without_u32 COMMAND test_linkage_list_parent_louds_without_u32)
    if(STLQ_ENABLE_CUDA)
        add_executable(stlq_gpu_sanity tests/checks/linkage_gpu_sanity.cpp)
        target_link_libraries(stlq_gpu_sanity PRIVATE stlq_core)
        stlq_apply_lto(stlq_gpu_sanity)
        add_executable(bench_gemm_cpu_vs_cublas tests/bench_gemm_cpu_vs_cublas.cpp)
        target_link_libraries(bench_gemm_cpu_vs_cublas PRIVATE stlq_core)
        stlq_apply_lto(bench_gemm_cpu_vs_cublas)
        add_executable(test_linkage_build_virtual_streaming_cuda_smoke tests/test_linkage_build_virtual_streaming_cuda_smoke.cpp)
        target_link_libraries(test_linkage_build_virtual_streaming_cuda_smoke PRIVATE stlq_core)
        stlq_apply_lto(test_linkage_build_virtual_streaming_cuda_smoke)
        add_test(NAME test_linkage_build_virtual_streaming_cuda_smoke COMMAND test_linkage_build_virtual_streaming_cuda_smoke)
        add_executable(test_cuda_linkage_candidates tests/test_cuda_linkage_candidates.cpp)
        target_link_libraries(test_cuda_linkage_candidates PRIVATE stlq_core)
        stlq_apply_lto(test_cuda_linkage_candidates)
        add_test(NAME test_cuda_linkage_candidates COMMAND test_cuda_linkage_candidates)
        add_executable(test_cuda_linkage_scan_lookup_topk tests/test_cuda_linkage_scan_lookup_topk.cpp)
        target_link_libraries(test_cuda_linkage_scan_lookup_topk PRIVATE stlq_core)
        stlq_apply_lto(test_cuda_linkage_scan_lookup_topk)
        add_test(NAME test_cuda_linkage_scan_lookup_topk COMMAND test_cuda_linkage_scan_lookup_topk)
        add_executable(test_icm_cuda_strict tests/test_icm_cuda_strict.cpp)
        target_link_libraries(test_icm_cuda_strict PRIVATE stlq_core)
        stlq_apply_lto(test_icm_cuda_strict)
        add_test(NAME test_icm_cuda_strict COMMAND test_icm_cuda_strict)
        add_executable(test_icm_cuda_large_root_strict tests/test_icm_cuda_large_root_strict.cpp)
        target_link_libraries(test_icm_cuda_large_root_strict PRIVATE stlq_core)
        stlq_apply_lto(test_icm_cuda_large_root_strict)
        add_test(NAME test_icm_cuda_large_root_strict COMMAND test_icm_cuda_large_root_strict)
        add_executable(test_beam_large_root_best_err_matches_cost tests/test_beam_large_root_best_err_matches_cost.cpp)
        target_link_libraries(test_beam_large_root_best_err_matches_cost PRIVATE stlq_core)
        stlq_apply_lto(test_beam_large_root_best_err_matches_cost)
        add_test(NAME test_beam_large_root_best_err_matches_cost COMMAND test_beam_large_root_best_err_matches_cost)
        add_executable(test_linkage_encode_forced_root_strict tests/test_linkage_encode_forced_root_strict.cpp)
        target_link_libraries(test_linkage_encode_forced_root_strict PRIVATE stlq_core)
        stlq_apply_lto(test_linkage_encode_forced_root_strict)
        add_test(NAME test_linkage_encode_forced_root_strict COMMAND test_linkage_encode_forced_root_strict)
        add_executable(test_linkage_encode_forced_root_device_xc_from_init tests/test_linkage_encode_forced_root_device_xc_from_init.cpp)
        target_link_libraries(test_linkage_encode_forced_root_device_xc_from_init PRIVATE stlq_core)
        stlq_apply_lto(test_linkage_encode_forced_root_device_xc_from_init)
        add_test(NAME test_linkage_encode_forced_root_device_xc_from_init COMMAND test_linkage_encode_forced_root_device_xc_from_init)
        add_executable(test_linkage_encode_forced_root_large_root_matches_full tests/test_linkage_encode_forced_root_large_root_matches_full.cpp)
        target_link_libraries(test_linkage_encode_forced_root_large_root_matches_full PRIVATE stlq_core)
        stlq_apply_lto(test_linkage_encode_forced_root_large_root_matches_full)
        add_test(NAME test_linkage_encode_forced_root_large_root_matches_full COMMAND test_linkage_encode_forced_root_large_root_matches_full)
        add_executable(test_cuda_linkage_many_nodes_determinism tests/test_cuda_linkage_many_nodes_determinism.cpp)
        target_link_libraries(test_cuda_linkage_many_nodes_determinism PRIVATE stlq_core)
        stlq_apply_lto(test_cuda_linkage_many_nodes_determinism)
        add_test(NAME test_cuda_linkage_many_nodes_determinism COMMAND test_cuda_linkage_many_nodes_determinism)
        add_executable(test_kmeans_anneal_regression tests/test_kmeans_anneal_regression.cpp)
        target_link_libraries(test_kmeans_anneal_regression PRIVATE stlq_core)
        stlq_apply_lto(test_kmeans_anneal_regression)
        add_test(NAME test_kmeans_anneal_regression COMMAND test_kmeans_anneal_regression)
        add_executable(test_cuda_rvq_metadata_per_slot tests/test_cuda_rvq_metadata_per_slot.cpp)
        target_link_libraries(test_cuda_rvq_metadata_per_slot PRIVATE stlq_core)
        stlq_apply_lto(test_cuda_rvq_metadata_per_slot)
        add_test(NAME test_cuda_rvq_metadata_per_slot COMMAND test_cuda_rvq_metadata_per_slot)
    endif()
endif()


if(STLQ_USE_MKL AND UNIX)
    # Make the executables runnable without sourcing oneAPI env scripts.
    set(_STLQ_RPATH_TARGETS stlq_main)
    if(TARGET stlq_io_probe)
        list(APPEND _STLQ_RPATH_TARGETS stlq_io_probe)
    endif()
    if(TARGET test_streaming_encode_base)
        list(APPEND _STLQ_RPATH_TARGETS test_streaming_encode_base)
    endif()
    if(TARGET test_train_result_bin_io)
        list(APPEND _STLQ_RPATH_TARGETS test_train_result_bin_io)
    endif()
    foreach(tgt ${_STLQ_RPATH_TARGETS})
        set_property(TARGET ${tgt} APPEND PROPERTY BUILD_RPATH ${MKL_LIB_DIR})
    endforeach()
endif()
