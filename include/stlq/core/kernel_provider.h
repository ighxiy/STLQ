#pragma once

#include <stdexcept>

#include "stlq/common/types.h"

namespace stlq {

struct DeviceMatF32View {
    const float* ptr = nullptr; // device pointer (CUDA) or nullptr (CPU/no-device)
    int rows = 0;
    int cols = 0;
    int ld = 0; // leading dimension (column-major)
};

struct DeviceVecI32View {
    const int* ptr = nullptr; // device pointer (CUDA) or nullptr (CPU/no-device)
    int n = 0;
};

struct DeviceVecF32View {
    const float* ptr = nullptr; // device pointer (CUDA) or nullptr (CPU/no-device)
    int n = 0;
};

// Kernel interface for streaming-heavy workloads (uint8 input + rotation + GEMM).
// Phase-2 GPU port can replace this provider while keeping the top-level pipeline unchanged.
class StreamKernelProvider {
public:
    virtual ~StreamKernelProvider() = default;

    // Used to keep fast CPU-only paths (pointer-based GEMM / tile conversions) while enabling
    // a CUDA provider to take a different (copy + device) code path.
    virtual bool IsGpu() const { return false; }

    virtual void ConvertU8ToF32(const ColMajorMatrix<std::uint8_t>& in,
                                ColMajorMatrix<float>* out) = 0;

    // out = R * float(in)  (column-major)
    virtual void ConvertU8ToF32AndRotate(const ColMajorMatrix<std::uint8_t>& in,
                                         const ColMajorMatrix<float>& R,
                                         ColMajorMatrix<float>* out) = 0;

    // out = R * float(in) where `in` is a column-major u8 tile (ld_in rows stride).
    // This avoids materializing a temporary `ColMajorMatrix<uint8_t>` when the caller already has
    // a contiguous pointer to a tile from a larger block.
    virtual void ConvertU8ToF32AndRotatePtr(const std::uint8_t* in,
                                           int ld_in,
                                           int rows,
                                           int cols,
                                           const ColMajorMatrix<float>& R,
                                           ColMajorMatrix<float>* out) = 0;

    virtual void Gemm(bool transA, bool transB,
                      float alpha,
                      const ColMajorMatrix<float>& A,
                      const ColMajorMatrix<float>& B,
                      float beta,
                      ColMajorMatrix<float>* C) = 0;

    // Optional: compute GEMM but keep output on device (no download).
    // Default implementation falls back to host `Gemm` (out->ptr remains nullptr).
    virtual void GemmDevice(bool transA, bool transB,
                            float alpha,
                            const ColMajorMatrix<float>& A,
                            const ColMajorMatrix<float>& B,
                            float beta,
                            DeviceMatF32View* out) {
        if (out) {
            out->ptr = nullptr;
            out->rows = 0;
            out->cols = 0;
            out->ld = 0;
        }
        ColMajorMatrix<float> tmp;
        Gemm(transA, transB, alpha, A, B, beta, &tmp);
    }

    // Optional: GEMM where B is specified as a host pointer (column-major with leading dim `ldB`).
    // This avoids materializing a temporary `ColMajorMatrix<float>` for a contiguous submatrix view.
    virtual void GemmDeviceHostPtrB(bool transA, bool transB,
                                    float alpha,
                                    const ColMajorMatrix<float>& A,
                                    const float* B,
                                    int ldB,
                                    int rowsB,
                                    int colsB,
                                    float beta,
                                    DeviceMatF32View* out) {
        if (!B || rowsB <= 0 || colsB <= 0 || ldB < rowsB) {
            if (out) {
                out->ptr = nullptr;
                out->rows = 0;
                out->cols = 0;
                out->ld = 0;
            }
            return;
        }
        ColMajorMatrix<float> Btmp(rowsB, colsB);
        for (int c = 0; c < colsB; ++c) {
            for (int r = 0; r < rowsB; ++r) {
                Btmp(r, c) = B[r + c * ldB];
            }
        }
        GemmDevice(transA, transB, alpha, A, Btmp, beta, out);
    }

    // Optional: ensure a persistent device copy exists for a host matrix, returning a device view.
    // This is useful when the same matrix (e.g. X in k-means assignment) is reused across many GEMMs.
    // Default implementation returns false (no persistent device cache).
    virtual bool EnsureDeviceF32(const ColMajorMatrix<float>& host,
                                 DeviceMatF32View* out) {
        if (out) {
            out->ptr = nullptr;
            out->rows = 0;
            out->cols = 0;
            out->ld = 0;
        }
        (void)host;
        return false;
    }

    // Optional: GEMM where B is already on device (column-major, leading dim `ldB`).
    // Default implementation throws (no device backend).
    virtual void GemmDeviceDevicePtrB(bool transA, bool transB,
                                      float alpha,
                                      const ColMajorMatrix<float>& A,
                                      const float* dB,
                                      int ldB,
                                      int rowsB,
                                      int colsB,
                                      float beta,
                                      DeviceMatF32View* out) {
        (void)transA;
        (void)transB;
        (void)alpha;
        (void)A;
        (void)dB;
        (void)ldB;
        (void)rowsB;
        (void)colsB;
        (void)beta;
        if (out) {
            out->ptr = nullptr;
            out->rows = 0;
            out->cols = 0;
            out->ld = 0;
        }
        throw std::runtime_error("StreamKernelProvider::GemmDeviceDevicePtrB: device GEMM not supported by this backend.");
    }

    // Optional: download a device matrix view into a host matrix.
    // Default implementation throws if `view.ptr` is non-null (no device backend).
    virtual void DownloadF32(const DeviceMatF32View& view,
                             ColMajorMatrix<float>* out) {
        if (view.ptr != nullptr) {
            throw std::runtime_error("StreamKernelProvider::DownloadF32: device view not supported by this backend.");
        }
        if (out) {
            out->rows = 0;
            out->cols = 0;
            out->data.clear();
        }
    }

    // Optional: return a device view for a host matrix if the backend already has a cached
    // device copy (e.g. rotated X tiles). Returns false when unavailable.
    virtual bool TryGetCachedDeviceF32(const ColMajorMatrix<float>& host,
                                       DeviceMatF32View* out) {
        if (out) {
            out->ptr = nullptr;
            out->rows = 0;
            out->cols = 0;
            out->ld = 0;
        }
        (void)host;
        return false;
    }

    // Optional: argmax per column for a device matrix view.
    // Produces host-side `assignments` (size = cols) and `best_values` (size = cols).
    // Default implementation throws when `scores.ptr` is non-null (no device backend).
    virtual void ArgmaxColsF32(const DeviceMatF32View& scores,
                               std::vector<int>* assignments,
                               std::vector<float>* best_values) {
        if (scores.ptr != nullptr) {
            throw std::runtime_error("StreamKernelProvider::ArgmaxColsF32: device argmax not supported by this backend.");
        }
        if (assignments) assignments->clear();
        if (best_values) best_values->clear();
    }

    // Optional: argmax per column for a device matrix view, leaving results on device.
    // The returned views are only valid until the next call that overwrites the backend's
    // internal argmax buffers (i.e., consume immediately on the same logical tile).
    //
    // Default implementation throws when `scores.ptr` is non-null (no device backend).
    virtual void ArgmaxColsF32Device(const DeviceMatF32View& scores,
                                     DeviceVecI32View* d_assign,
                                     DeviceVecF32View* d_best) {
        if (scores.ptr != nullptr) {
            throw std::runtime_error("StreamKernelProvider::ArgmaxColsF32Device: device argmax not supported by this backend.");
        }
        if (d_assign) {
            d_assign->ptr = nullptr;
            d_assign->n = 0;
        }
        if (d_best) {
            d_best->ptr = nullptr;
            d_best->n = 0;
        }
    }

    // Optional: synchronize backend work (used only for profiling).
    virtual void Sync() {}

    // Optional: synchronize only the backend's main compute stream (used for profiling).
    // CUDA implementation uses this to avoid blocking on any auxiliary copy/packing streams,
    // which can otherwise distort per-kernel timing (e.g. Argmax/GEMM timers in k-means).
    virtual void SyncCompute() { Sync(); }
};

}  // namespace stlq
