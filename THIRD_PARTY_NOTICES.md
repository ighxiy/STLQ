# Third-Party Notices

STLQ's own source code is distributed under the BSD 3-Clause License. See
`LICENSE`.

Some build configurations use third-party software. Those components keep
their own licenses; they are not relicensed as STLQ code.

## Vendored Source

| Component | Location | License | Notes |
|---|---|---|---|
| hnswlib | `third_party/hnswlib/` | Apache License 2.0 | Header-only HNSW implementation used by linkage candidate selection and IVF HNSW probing. The license text is included at `third_party/hnswlib/LICENSE`. |

Apache-2.0 is a permissive license and is generally compatible with distributing
STLQ under BSD 3-Clause, provided the Apache-2.0 license and notices are kept
with the vendored code.

## Optional External Dependencies

| Component | How STLQ Uses It | License Surface |
|---|---|---|
| Intel oneAPI MKL | Optional BLAS/LAPACK backend when `STLQ_USE_MKL=ON` | External proprietary/free-to-use Intel license. STLQ does not vendor MKL. Binary redistribution must follow Intel's terms. |
| OpenBLAS | Optional BLAS/LAPACK backend when `STLQ_USE_MKL=OFF` | BSD-style permissive license. STLQ does not vendor OpenBLAS. |
| HDF5 C library | Optional compatibility IO when `STLQ_ENABLE_HDF5=ON` | HDF5 license, BSD-style permissive with required notices. STLQ does not vendor HDF5. |
| NVIDIA CUDA Toolkit | Optional CUDA/cuBLAS backend when `STLQ_ENABLE_CUDA=ON` | NVIDIA toolkit/runtime licenses. STLQ does not vendor CUDA. Binary redistribution must follow NVIDIA's terms. |

## Practical Guidance

- Source releases should include `LICENSE`, this file, and
  `third_party/hnswlib/LICENSE`.
- Binary releases should also include notices for the external runtime
  libraries that are bundled with the binary, if any.
- Builds that rely only on system-installed MKL/OpenBLAS/HDF5/CUDA do not
  bundle those libraries, but users still need to satisfy their local license
  terms.
