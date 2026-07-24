#pragma once

#include <memory>
#include <string>
#include <vector>

#include "stlq/common/types.h"

namespace stlq::io {

enum class Hdf5MatrixLayout {
    // Store datasets with dims (rows, cols) using the in-memory ColMajorMatrix buffer as-is.
    // This is "native" for this C++ project, but Julia's HDF5.jl will typically see a transposed shape.
    kCxx = 0,
    // Store datasets with swapped dims (cols, rows) using the same in-memory buffer.
    // This makes Julia HDF5.jl read matrices with the expected (rows, cols) shape.
    kJulia = 1,
};

// Global default used by Hdf5Writer/Hdf5Reader when per-dataset metadata is missing.
void SetDefaultHdf5MatrixLayout(Hdf5MatrixLayout layout);
Hdf5MatrixLayout GetDefaultHdf5MatrixLayout();

class Hdf5Writer {
public:
    explicit Hdf5Writer(const std::string& path);
    ~Hdf5Writer();

    Hdf5Writer(Hdf5Writer&&) noexcept;
    Hdf5Writer& operator=(Hdf5Writer&&) noexcept;
    Hdf5Writer(const Hdf5Writer&) = delete;
    Hdf5Writer& operator=(const Hdf5Writer&) = delete;

    bool IsOpen() const;

    bool WriteMatrix(const std::string& name, const ColMajorMatrix<float>& matrix);
    bool WriteMatrix(const std::string& name, const ColMajorMatrix<Index>& matrix);
    bool WriteMatrix(const std::string& name, const ColMajorMatrix<Code>& matrix);
    bool WriteVector(const std::string& name, const std::vector<Index>& vec);
    bool WriteVector(const std::string& name, const std::vector<std::uint8_t>& vec);
    bool WriteVector(const std::string& name, const std::vector<int>& vec);
    bool WriteScalar(const std::string& name, int value);
    bool WriteScalar(const std::string& name, Index value);
    bool WriteScalar(const std::string& name, float value);
    bool WriteString(const std::string& name, const std::string& value);

private:
    struct Impl;
    bool EnsureGroups(const std::string& name) const;

    Hdf5MatrixLayout matrix_layout_ = Hdf5MatrixLayout::kJulia;
    std::unique_ptr<Impl> impl_;
};

class Hdf5Reader {
public:
    explicit Hdf5Reader(const std::string& path);
    ~Hdf5Reader();

    Hdf5Reader(Hdf5Reader&&) noexcept;
    Hdf5Reader& operator=(Hdf5Reader&&) noexcept;
    Hdf5Reader(const Hdf5Reader&) = delete;
    Hdf5Reader& operator=(const Hdf5Reader&) = delete;

    bool IsOpen() const;
    bool Has(const std::string& name) const;

    bool ReadMatrix(const std::string& name, ColMajorMatrix<float>* matrix);
    bool ReadMatrix(const std::string& name, ColMajorMatrix<Index>* matrix);
    bool ReadMatrix(const std::string& name, ColMajorMatrix<Code>* matrix);
    bool ReadVector(const std::string& name, std::vector<Index>* vec) const;
    bool ReadVector(const std::string& name, std::vector<std::uint8_t>* vec) const;
    bool ReadVector(const std::string& name, std::vector<int>* vec) const;
    bool ReadScalar(const std::string& name, int* value) const;
    bool ReadScalar(const std::string& name, Index* value) const;
    bool ReadScalar(const std::string& name, float* value) const;
    bool ReadString(const std::string& name, std::string* value) const;

private:
    struct Impl;
    Hdf5MatrixLayout default_layout_ = Hdf5MatrixLayout::kJulia;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stlq::io
